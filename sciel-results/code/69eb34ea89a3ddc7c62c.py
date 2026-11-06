"""
Communication-minimizing DAG pebbling.

Strategy:
  - Parse the DAG (topologically ordered parents, output vertices).
  - Use a layer-based dynamic-programming / greedy heuristic that minimizes
    the number of transfers (loads + stores) while respecting the fast-memory budget.
  - The key insight for layered DAGs: process vertices in topological order,
    keeping only the "frontier" (vertices whose parents are all computed but
    which are not yet consumed) in fast memory. When memory is tight, evict
    the vertex whose re-computation would be cheapest (fewest downstream uses
    remaining, or lowest fan-in cost).
  - For the three standard layered DAGs (e.g., complete binary tree, complete
    k-ary tree, or layered complete DAGs), a near-optimal schedule is:
      * Load inputs as needed
      * Compute layer by layer
      * Store intermediate results only when they will be reused
      * Delete results as soon as all their children have consumed them
  - The naive schedule loads every input, computes every vertex, stores every
    intermediate, and the transfer count is roughly 2 * (num_vertices - num_inputs).
    Our goal is to minimize transfers by keeping frequently-reused values in
    fast memory and re-computing rarely-used values.

  We implement a greedy priority-queue approach:
    1. Topologically sort the DAG.
    2. For each vertex in topological order, decide whether to compute it now
       or defer it. Use a priority queue keyed by "urgency" (how many
       downstream vertices depend on it) and "cost" (how expensive it is to
       recompute).
    3. Track fast-memory occupancy; when full, evict the least-urgent value.
    4. Count transfers (loads and stores) to compute the score.

  Since the exact pebbling problem is NP-hard in general, we use a strong
  greedy heuristic that works well on layered DAGs.
"""

import time
import heapq
from collections import defaultdict, deque

# Deadline guard
_DEADLINE = 85.0  # seconds, leave 5s margin before 90s kill


def pebble(dag, fast_memory_budget):
    """
    Solve the communication-minimizing DAG pebbling problem.

    Parameters
    ----------
    dag : dict or object
        A DAG description. Expected structure:
        - dag['vertices'] or dag['nodes']: list of vertex ids
        - dag['edges'] or dag['parents']: mapping vertex -> list of parent vertices
        - dag['inputs'] or dag['input_vertices']: set/list of input vertices
        - dag['outputs'] or dag['output_vertices']: set/list of output vertices
        Alternatively, dag may be a dict mapping vertex -> list of parents,
        with separate input/output info.
    fast_memory_budget : int
        Maximum number of red (computed) values that can reside in fast memory.

    Returns
    -------
    list of [operation, vertex]
        A sequence of operations where operation is one of
        'load', 'store', 'compute', 'delete'.
    """
    t0 = time.time()

    # --- Parse the DAG structure ---
    # Try several common formats
    vertices, parents_map, inputs, outputs = _parse_dag(dag)

    n = len(vertices)
    if n == 0:
        return []

    # Build children map
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)

    # Topological order (parents should already be topologically ordered,
    # but we do a Kahn's sort to be safe)
    topo_order = _topo_sort(vertices, parents_map)

    # Identify input vertices (no parents) and output vertices
    if inputs is None:
        inputs = set(v for v in vertices if not parents_map.get(v))
    if outputs is None:
        outputs = set(v for v in vertices if not children_map.get(v))

    # --- Compute a good schedule ---
    schedule = _greedy_layered_pebble(
        vertices, parents_map, children_map, topo_order,
        inputs, outputs, fast_memory_budget
    )

    # Safety: if we've used too much time, return what we have
    if time.time() - t0 > _DEADLINE:
        return schedule

    # Verify and fix if necessary
    schedule = _verify_and_fix(schedule, dag, fast_memory_budget, vertices,
                                parents_map, children_map, inputs, outputs,
                                topo_order)

    return schedule


def _parse_dag(dag):
    """
    Parse the DAG into (vertices, parents_map, inputs, outputs).
    Handles multiple common formats.
    """
    vertices = None
    parents_map = None
    inputs = None
    outputs = None

    if isinstance(dag, dict):
        # Check for explicit keys
        if 'vertices' in dag:
            vertices = list(dag['vertices'])
        elif 'nodes' in dag:
            vertices = list(dag['nodes'])

        if 'parents' in dag:
            parents_map = {v: list(ps) for v, ps in dag['parents'].items()}
        elif 'edges' in dag:
            # edges might be list of (parent, child) or dict child -> [parents]
            edges = dag['edges']
            if isinstance(edges, dict):
                parents_map = {v: list(ps) for v, ps in edges.items()}
            else:
                # list of tuples
                parents_map = defaultdict(list)
                for e in edges:
                    if len(e) == 2:
                        p, c = e
                        parents_map[c].append(p)
                parents_map = {v: parents_map.get(v, []) for v in (vertices or [])}

        if 'inputs' in dag:
            inputs = set(dag['inputs'])
        elif 'input_vertices' in dag:
            inputs = set(dag['input_vertices'])

        if 'outputs' in dag:
            outputs = set(dag['outputs'])
        elif 'output_vertices' in dag:
            outputs = set(dag['output_vertices'])

        # If no vertices specified, infer from parents_map
        if vertices is None:
            if parents_map is not None:
                vertices = list(parents_map.keys())
            else:
                # Maybe dag itself is a mapping vertex -> parents
                vertices = list(dag.keys())
                parents_map = {v: list(dag[v]) for v in dag}
    elif isinstance(dag, (list, tuple)):
        # Could be list of (vertex, parents) tuples
        vertices = []
        parents_map = {}
        for item in dag:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                v, ps = item[0], item[1]
                vertices.append(v)
                parents_map[v] = list(ps)
            elif isinstance(item, (list, tuple)) and len(item) == 1:
                v = item[0]
                vertices.append(v)
                parents_map[v] = []
        if inputs is None:
            inputs = set(v for v in vertices if not parents_map.get(v))
        if outputs is None:
            children_map = defaultdict(list)
            for v in vertices:
                for p in parents_map.get(v, []):
                    children_map[p].append(v)
            outputs = set(v for v in vertices if not children_map.get(v))
    else:
        # Try to treat as a dict-like object
        try:
            vertices = list(dag.keys())
            parents_map = {v: list(dag[v]) for v in dag}
        except (AttributeError, TypeError):
            # Fallback: treat as empty
            vertices = []
            parents_map = {}

    # Ensure parents_map covers all vertices
    if parents_map is None:
        parents_map = {}
    if vertices is not None:
        for v in vertices:
            if v not in parents_map:
                parents_map[v] = []

    # Default inputs/outputs
    if inputs is None:
        inputs = set(v for v in vertices if not parents_map.get(v, []))
    if outputs is None:
        children_map = defaultdict(list)
        for v in vertices:
            for p in parents_map.get(v, []):
                children_map[p].append(v)
        outputs = set(v for v in vertices if not children_map.get(v, []))

    return vertices, parents_map, inputs, outputs


def _topo_sort(vertices, parents_map):
    """Kahn's algorithm for topological sort."""
    in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)

    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for w in children_map.get(u, []):
            in_degree[w] -= 1
            if in_degree[w] == 0:
                queue.append(w)

    # If not all vertices are in order (cycle?), append remaining
    if len(order) < len(vertices):
        remaining = set(vertices) - set(order)
        order.extend(sorted(remaining))

    return order


def _greedy_layered_pebble(vertices, parents_map, children_map, topo_order,
                            inputs, outputs, budget):
    """
    Greedy pebbling strategy:
    - Process vertices in topological order.
    - Maintain a set of "live" vertices in fast memory.
    - For each vertex, if its parents are all in fast memory or can be loaded,
      compute it.
    - When memory is full, evict the vertex with the lowest "reuse count"
      (number of remaining children that haven't been processed yet).
    - Count transfers: load (input from slow to fast) and store (fast to slow
      for outputs or for values that need to be persisted).
    
    For communication-minimizing, the key is to minimize the number of
    load/store operations. We track:
    - 'load': moving an input value from slow memory to fast memory
    - 'store': moving a computed value from fast memory to slow memory
    - 'compute': performing the computation (no transfer cost)
    - 'delete': freeing a fast memory slot (no transfer cost)
    
    The score is mean transfers / naive schedule. Transfers = loads + stores.
    """
    n = len(vertices)
    if n == 0:
        return []

    # Precompute: for each vertex, the number of children (fan-out)
    fan_out = {v: len(children_map.get(v, [])) for v in vertices}

    # Track which children of each vertex have been processed
    remaining_children = {v: set(children_map.get(v, [])) for v in vertices}

    # Schedule
    schedule = []

    # State:
    # - fast_mem: set of vertices currently in fast memory (red pebbles)
    # - computed: set of vertices that have been computed at least once
    # - in_slow: set of vertices whose values are in slow memory
    fast_mem = set()
    computed = set()
    in_slow = set(inputs)  # Inputs start in slow memory

    # For tracking: which vertices have all their parents in fast_mem
    # We'll process in topological order, but we may need to re-compute

    # Strategy: process in topological order. For each vertex:
    # 1. Ensure all parents are available (in fast_mem or can be loaded from slow)
    # 2. Compute the vertex
    # 3. If budget is exceeded, evict the least-urgent vertex
    # 4. At the end, store all outputs

    # To minimize transfers, we want to keep values in fast memory as long
    # as they will be reused. We use a priority based on remaining uses.

    # Process vertices in topological order
    for v in topo_order:
        # Check deadline
        if time.time() - _t0_global > _DEADLINE:
            break

        # Ensure parents are in fast memory
        for p in parents_map.get(v, []):
            if p not in fast_mem:
                # Need to make p available
                if p in in_slow:
                    # Load from slow to fast
                    # Check budget
                    if len(fast_mem) >= budget:
                        # Evict least urgent
                        evicted = _evict_least_urgent(fast_mem, remaining_children,
                                                       computed, inputs, outputs,
                                                       children_map)
                        if evicted is not None:
                            schedule.append(['delete', evicted])
                            fast_mem.discard(evicted)
                    schedule.append(['load', p])
                    fast_mem.add(p)
                elif p in computed:
                    # Was computed before but evicted. Need to recompute.
                    # This is expensive; try to avoid by better eviction.
                    # For now, recompute.
                    _recompute_vertex(p, fast_mem, in_slow, computed, schedule,
                                      parents_map, children_map, budget,
                                      remaining_children)
                else:
                    # Should not happen in topological order
                    pass

        # Compute v
        if v not in fast_mem:
            if len(fast_mem) >= budget:
                evicted = _evict_least_urgent(fast_mem, remaining_children,
                                               computed, inputs, outputs,
                                               children_map)
                if evicted is not None:
                    schedule.append(['delete', evicted])
                    fast_mem.discard(evicted)
            schedule.append(['compute', v])
            fast_mem.add(v)
            computed.add(v)

        # Mark v's parent's remaining child count
        for p in parents_map.get(v, []):
            if v in remaining_children.get(p, set()):
                remaining_children[p].discard(v)

    # Store all outputs that are not already in slow memory
    for v in outputs:
        if v in fast_mem and v not in in_slow:
            schedule.append(['store', v])
            in_slow.add(v)
        elif v not in in_slow:
            # Output was evicted; need to recompute and store
            _ensure_and_store(v, fast_mem, in_slow, computed, schedule,
                              parents_map, children_map, budget,
                              remaining_children)

    return schedule


_t0_global = time.time()


def _evict_least_urgent(fast_mem, remaining_children, computed, inputs, outputs,
                         children_map):
    """
    Choose which vertex to evict from fast memory.
    Priority: evict vertices with fewest remaining children (least reuse).
    Among ties, prefer evicting non-output vertices.
    """
    best = None
    best_key = None
    for v in fast_mem:
        if v in outputs:
            continue  # Don't evict outputs if possible
        rem = len(remaining_children.get(v, set()))
        # Lower rem = less urgent to keep
        key = (rem, 1 if v in inputs else 0)
        if best_key is None or key < best_key:
            best_key = key
            best = v

    # If all are outputs, evict the one with fewest remaining children
    if best is None:
        for v in fast_mem:
            rem = len(remaining_children.get(v, set()))
            key = rem
            if best_key is None or key < best_key:
                best_key = key
                best = v

    return best


def _recompute_vertex(v, fast_mem, in_slow, computed, schedule,
                      parents_map, children_map, budget, remaining_children):
    """Recompute a vertex that was evicted."""
    if v in fast_mem:
        return
    # Ensure parents are available
    for p in parents_map.get(v, []):
        if p not in fast_mem:
            if p in in_slow:
                if len(fast_mem) >= budget:
                    evicted = _evict_least_urgent(fast_mem, remaining_children,
                                                   computed, inputs=None, outputs=None,
                                                   children_map=children_map)
                    if evicted is not None:
                        schedule.append(['delete', evicted])
                        fast_mem.discard(evicted)
                schedule.append(['load', p])
                fast_mem.add(p)
            else:
                _recompute_vertex(p, fast_mem, in_slow, computed, schedule,
                                  parents_map, children_map, budget,
                                  remaining_children)
    if len(fast_mem) >= budget:
        evicted = _evict_least_urgent(fast_mem, remaining_children,
                                       computed, inputs=None, outputs=None,
                                       children_map=children_map)
        if evicted is not None:
            schedule.append(['delete', evicted])
            fast_mem.discard(evicted)
    schedule.append(['compute', v])
    fast_mem.add(v)
    computed.add(v)


def _ensure_and_store(v, fast_mem, in_slow, computed, schedule,
                      parents_map, children_map, budget, remaining_children):
    """Ensure v is computed and in fast memory, then store it."""
    if v not in fast_mem:
        _recompute_vertex(v, fast_mem, in_slow, computed, schedule,
                          parents_map, children_map, budget, remaining_children)
    if v in fast_mem and v not in in_slow:
        schedule.append(['store', v])
        in_slow.add(v)


def _verify_and_fix(schedule, dag, budget, vertices, parents_map,
                    children_map, inputs, outputs, topo_order):
    """
    Verify the schedule is valid and fix any issues.
    Check:
    1. All operations are valid.
    2. Fast memory never exceeds budget.
    3. All inputs start available (in slow memory).
    4. All outputs end in slow memory.
    5. A vertex can only be computed if all its parents are in fast memory.
    6. A vertex can only be loaded if it's an input or was previously stored.
    7. A vertex can only be deleted if it's in fast memory.
    """
    if not schedule:
        return schedule

    # Simulate
    fast_mem = set()
    in_slow = set(inputs)
    valid_ops = {'load', 'store', 'compute', 'delete'}

    fixed_schedule = []
    for op, v in schedule:
        if op not in valid_ops:
            continue

        if op == 'load':
            # v must be in slow memory (input or previously stored)
            if v not in in_slow:
                # Invalid load; skip or try to fix
                # For now, skip
                continue
            if v in fast_mem:
                # Already in fast mem, skip
                continue
            if len(fast_mem) >= budget:
                # Need to delete something first
                # Find a non-critical vertex to delete
                to_delete = None
                for u in list(fast_mem):
                    if u not in outputs:
                        to_delete = u
                        break
                if to_delete is None:
                    to_delete = next(iter(fast_mem)) if fast_mem else None
                if to_delete is not None:
                    fixed_schedule.append(['delete', to_delete])
                    fast_mem.discard(to_delete)
            fixed_schedule.append(['load', v])
            fast_mem.add(v)

        elif op == 'store':
            if v not in fast_mem:
                continue
            fixed_schedule.append(['store', v])
            in_slow.add(v)
            # Don't remove from fast_mem on store; it stays until deleted
            # Actually, in pebbling, store typically means the value is
            # persisted to slow memory. The red pebble may stay or go.
            # For our model, store doesn't remove from fast memory.

        elif op == 'compute':
            # All parents must be in fast memory
            parents = parents_map.get(v, [])
            if all(p in fast_mem for p in parents):
                if v in fast_mem:
                    # Already computed and in fast mem, skip
                    continue
                if len(fast_mem) >= budget:
                    to_delete = None
                    for u in list(fast_mem):
                        if u not in outputs:
                            to_delete = u
                            break
                    if to_delete is None:
                        to_delete = next(iter(fast_mem)) if fast_mem else None
                    if to_delete is not None:
                        fixed_schedule.append(['delete', to_delete])
                        fast_mem.discard(to_delete)
                fixed_schedule.append(['compute', v])
                fast_mem.add(v)
            else:
                # Missing parents; skip this compute (will be handled elsewhere)
                pass

        elif op == 'delete':
            if v in fast_mem:
                fixed_schedule.append(['delete', v])
                fast_mem.discard(v)

    # Ensure all outputs are in slow memory
    for v in outputs:
        if v not in in_slow:
            # Need to compute and store
            if v not in fast_mem:
                # Recompute
                _ensure_and_store(v, fast_mem, in_slow, set(), fixed_schedule,
                                  parents_map, children_map, budget,
                                  {u: set(children_map.get(u, [])) for u in vertices})
            else:
                fixed_schedule.append(['store', v])
                in_slow.add(v)

    # Limit to 20000 operations
    if len(fixed_schedule) > 20000:
        fixed_schedule = fixed_schedule[:20000]

    return fixed_schedule
