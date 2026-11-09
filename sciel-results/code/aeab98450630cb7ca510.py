"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque
from functools import lru_cache

def _parse_dag(dag):
    """Parse various possible DAG formats into a unified structure."""
    if isinstance(dag, dict):
        parents = dag.get('parents', None)
        outputs = dag.get('outputs', None)
        inputs = dag.get('inputs', None)
        vertices = dag.get('vertices', None)
        n = dag.get('n', None)
        
        if parents is None:
            if isinstance(dag, list):
                parents = [list(p) for p in dag]
                n = len(dag)
                if outputs is None:
                    outputs = [n - 1]
                if inputs is None:
                    inputs = [i for i in range(n) if not parents[i]]
            else:
                return None, None, None, None, None
        else:
            if isinstance(parents, list):
                parents = [list(p) for p in parents]
                n = len(parents)
            elif isinstance(parents, dict):
                n = max(parents.keys()) + 1 if parents else 0
                parents_list = [list(parents.get(i, [])) for i in range(n)]
                parents = parents_list
            
            if outputs is None:
                children = [[] for _ in range(n)]
                for v in range(n):
                    for p in parents[v]:
                        children[p].append(v)
                outputs = [v for v in range(n) if not children[v]]
            
            if inputs is None:
                inputs = [v for v in range(n) if not parents[v]]
        
        return parents, outputs, inputs, vertices, n
    
    elif isinstance(dag, list):
        parents = [list(p) if isinstance(p, (list, tuple)) else [] for p in dag]
        n = len(dag)
        outputs = [n - 1] if n > 0 else []
        inputs = [i for i in range(n) if not parents[i]]
        return parents, outputs, inputs, None, n
    
    return None, None, None, None, None


def _build_children(parents, n):
    """Build children list."""
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    return children


def _compute_depth(parents, n):
    """Compute depth (longest path from any input) for each vertex."""
    depth = [0] * n
    # Topological order is 0..n-1
    for v in range(n):
        if parents[v]:
            depth[v] = max(depth[p] for p in parents[v]) + 1
    return depth


def _dijkstra_pebbling(parents, outputs, fast_memory_budget, deadline):
    """
    Exact Dijkstra search over red-pebble configurations.
    State: frozenset of vertices currently in fast memory (red-pebbled).
    Goal: all outputs stored (blue).
    Transitions: load, store, compute, delete.
    Cost: 1 for load/store (transfers), 0 for compute/delete.
    """
    n = len(parents)
    if n == 0:
        return []

    children = _build_children(parents, n)
    is_output = set(outputs)
    is_input = [len(parents[v]) == 0 for v in range(n)]
    
    inputs_set = frozenset(v for v in range(n) if is_input[v])
    initial_state = (frozenset(), inputs_set)
    
    # Precompute parent sets as frozensets for faster membership checks
    parent_sets = [frozenset(parents[v]) for v in range(n)]
    
    def heuristic(fast, slow):
        """Lower bound on additional transfers needed."""
        # Each output not in slow needs at least 1 store
        count = 0
        for o in outputs:
            if o not in slow:
                count += 1
        # Additionally, if an output is in fast, it needs to be stored (but that's already counted if not in slow)
        # Actually outputs in fast are not in slow, so they're counted above
        return count
    
    # Dijkstra with A*
    best_g = {}
    best_g[initial_state] = 0
    parent_map = {}  # state -> (prev_state, op, vertex)
    
    pq = []
    h0 = heuristic(initial_state[0], initial_state[1])
    heapq.heappush(pq, (h0, 0, initial_state))
    
    max_ops = 20000
    max_states = 500000  # Limit state space
    
    while pq:
        if time.time() > deadline:
            return None
        if len(best_g) > max_states:
            return None
        
        f, g, state = heapq.heappop(pq)
        
        # If this state was already processed with a better g, skip
        if state in best_g and g > best_g[state]:
            continue
        
        fast, slow = state
        
        # Check if goal reached
        if all(o in slow for o in outputs):
            # Reconstruct path
            ops = []
            cur = state
            while cur in parent_map:
                prev, op, v = parent_map[cur]
                ops.append([op, v])
                cur = prev
            ops.reverse()
            # Check operation count
            if len(ops) <= max_ops:
                return ops
            else:
                best_g[state] = float('inf')
                continue
        
        # Generate neighbors
        # 1. Load: v in slow, v not in fast, |fast| < budget
        if len(fast) < fast_memory_budget:
            for v in slow:
                if v not in fast:
                    new_fast = fast | frozenset([v])
                    new_slow = slow - frozenset([v])
                    new_state = (new_fast, new_slow)
                    new_g = g + 1
                    if new_state not in best_g or new_g < best_g[new_state]:
                        best_g[new_state] = new_g
                        parent_map[new_state] = (state, 'load', v)
                        h = heuristic(new_fast, new_slow)
                        heapq.heappush(pq, (new_g + h, new_g, new_state))
        
        # 2. Store: v in fast
        for v in fast:
            new_fast = fast - frozenset([v])
            new_slow = slow | frozenset([v])
            new_state = (new_fast, new_slow)
            new_g = g + 1
            if new_state not in best_g or new_g < best_g[new_state]:
                best_g[new_state] = new_g
                parent_map[new_state] = (state, 'store', v)
                h = heuristic(new_fast, new_slow)
                heapq.heappush(pq, (new_g + h, new_g, new_state))
        
        # 3. Compute: all parents of v in fast, v not in fast, |fast| < budget
        if len(fast) < fast_memory_budget:
            for v in range(n):
                if v not in fast and parents[v]:
                    if parent_sets[v] <= fast:  # All parents are in fast
                        new_fast = fast | frozenset([v])
                        new_state = (new_fast, slow)
                        new_g = g  # compute costs 0
                        if new_state not in best_g or new_g < best_g[new_state]:
                            best_g[new_state] = new_g
                            parent_map[new_state] = (state, 'compute', v)
                            h = heuristic(new_fast, slow)
                            heapq.heappush(pq, (new_g + h, new_g, new_state))
        
        # 4. Delete: v in fast
        for v in fast:
            new_fast = fast - frozenset([v])
            new_state = (new_fast, slow)
            new_g = g  # delete costs 0
            if new_state not in best_g or new_g < best_g[new_state]:
                best_g[new_state] = new_g
                parent_map[new_state] = (state, 'delete', v)
                h = heuristic(new_fast, slow)
                heapq.heappush(pq, (new_g + h, new_g, new_state))
    
    return None


def _layered_pebbling(parents, outputs, fast_memory_budget, deadline):
    """
    Layer-by-layer pebbling with memory management.
    For layered DAGs, process one layer at a time:
    - Load previous layer values as needed
    - Compute current layer
    - Store current layer to slow memory
    - Delete previous layer from fast memory when no longer needed
    """
    n = len(parents)
    if n == 0:
        return []
    
    children = _build_children(parents, n)
    depth = _compute_depth(parents, n)
    is_output = set(outputs)
    is_input = [len(parents[v]) == 0 for v in range(n)]
    
    # Group vertices by depth (layer)
    max_depth = max(depth) if depth else 0
    layers = [[] for _ in range(max_depth + 1)]
    for v in range(n):
        layers[depth[v]].append(v)
    
    # Precompute: for each vertex, which layers' children need it
    # last_child_layer[v] = max depth of any child of v
    last_child_layer = [-1] * n
    for v in range(n):
        if children[v]:
            last_child_layer[v] = max(depth[c] for c in children[v])
    
    ops = []
    in_fast = set()
    current_layer = 0
    
    def make_space():
        """Make space in fast memory by deleting vertices that are no longer needed."""
        if len(in_fast) < fast_memory_budget:
            return True
        # Find vertices to delete: those whose last child layer is < current_layer
        for u in sorted(in_fast, key=lambda x: (last_child_layer[x], x)):
            if u in is_output:
                continue
            if last_child_layer[u] < current_layer:
                ops.append(['delete', u])
                in_fast.discard(u)
                if len(in_fast) < fast_memory_budget:
                    return True
        # If still no space, delete any non-output vertex with smallest last_child_layer
        for u in sorted(in_fast, key=lambda x: (last_child_layer[x], x)):
            if u not in is_output:
                ops.append(['delete', u])
                in_fast.discard(u)
                if len(in_fast) < fast_memory_budget:
                    return True
        return len(in_fast) < fast_memory_budget
    
    # Process layer by layer
    for d in range(max_depth + 1):
        if time.time() > deadline:
            return None
        
        current_layer = d
        layer = layers[d]
        if not layer:
            continue
        
        if d == 0:
            # Layer 0: inputs - load them
            for v in layer:
                make_space()
                if v not in in_fast and len(in_fast) < fast_memory_budget:
                    ops.append(['load', v])
                    in_fast.add(v)
        else:
            # Collect all unique parents of vertices in this layer
            needed_parents = set()
            for v in layer:
                for p in parents[v]:
                    needed_parents.add(p)
            
            # Load those parents that are not already in fast memory
            for p in sorted(needed_parents):
                if p not in in_fast:
                    make_space()
                    if len(in_fast) < fast_memory_budget:
                        ops.append(['load', p])
                        in_fast.add(p)
            
            # Compute vertices in this layer
            for v in layer:
                make_space()
                if v not in in_fast and len(in_fast) < fast_memory_budget:
                    ops.append(['compute', v])
                    in_fast.add(v)
            
            # Store vertices in this layer
            for v in layer:
                if v in in_fast:
                    ops.append(['store', v])
                    in_fast.discard(v)
        
        # Delete vertices from previous layers that are no longer needed
        for prev_d in range(d):
            for v in layers[prev_d]:
                if v in in_fast and v not in is_output:
                    # Check if v is still needed by any unprocessed child
                    if last_child_layer[v] < d:
                        ops.append(['delete', v])
                        in_fast.discard(v)
    
    # Ensure all outputs are stored
    for o in outputs:
        if o in in_fast:
            ops.append(['store', o])
            in_fast.discard(o)
    
    return ops


def _greedy_pebbling(parents, outputs, fast_memory_budget, deadline):
    """
    Greedy pebbling with last-use analysis.
    Processes vertices in topological order, evicting values that are no longer needed.
    Uses a priority-based eviction strategy.
    """
    n = len(parents)
    if n == 0:
        return []
    
    children = _build_children(parents, n)
    is_output = set(outputs)
    
    # Precompute last use for each vertex (the highest-indexed child)
    last_use = [-1] * n
    for v in range(n):
        if children[v]:
            last_use[v] = max(children[v])
    
    ops = []
    in_fast = set()
    
    def try_evict():
        """Try to evict a vertex from fast memory. Returns True if successful."""
        if not in_fast:
            return False
        # Find vertex with smallest last_use that is not an output
        # Prefer vertices that are no longer needed (last_use < current v)
        best = None
        best_key = (float('inf'), float('inf'))
        for u in in_fast:
            if u not in is_output:
                lu = last_use[u]
                # Key: (last_use, vertex_index) - prefer smaller last_use
                key = (lu, u)
                if key < best_key:
                    best_key = key
                    best = u
        if best is not None:
            ops.append(['delete', best])
            in_fast.discard(best)
            return True
        # If all are outputs, evict any
        if in_fast:
            u = next(iter(in_fast))
            ops.append(['delete', u])
            in_fast.discard(u)
            return True
        return False
    
    for v in range(n):
        if time.time() > deadline:
            return None
        
        # Ensure all parents are in fast memory
        for p in parents[v]:
            if p not in in_fast:
                while len(in_fast) >= fast_memory_budget:
                    if not try_evict():
                        break
                if len(in_fast) < fast_memory_budget:
                    ops.append(['load', p])
                    in_fast.add(p)
        
        # Compute v if it has parents
        if parents[v]:
            if v not in in_fast:
                while len(in_fast) >= fast_memory_budget:
                    if not try_evict():
                        break
                if len(in_fast) < fast_memory_budget:
                    ops.append(['compute', v])
                    in_fast.add(v)
        
        # Handle input-output vertices
        if not parents[v] and v in is_output:
            if v not in in_fast:
                while len(in_fast) >= fast_memory_budget:
                    if not try_evict():
                        break
                if len(in_fast) < fast_memory_budget:
                    ops.append(['load', v])
                    in_fast.add(v)
            if v in in_fast:
                ops.append(['store', v])
                in_fast.discard(v)
        
        # Store outputs that were computed
        if v in is_output and parents[v] and v in in_fast:
            ops.append(['store', v])
            in_fast.discard(v)
        
        # Clean up parents that are no longer needed
        for p in parents[v]:
            if p in in_fast and p not in is_output and last_use[p] == v:
                ops.append(['delete', p])
                in_fast.discard(p)
    
    # Final cleanup: store any remaining outputs
    for o in outputs:
        if o in in_fast:
            ops.append(['store', o])
            in_fast.discard(o)
    
    return ops


def _validate_ops(ops, parents, outputs, fast_memory_budget):
    """Validate that operations are legal and complete."""
    n = len(parents)
    in_fast = set()
    stored = set()
    
    for op, v in ops:
        if op == 'load':
            if v in in_fast:
                continue
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.discard(v)
            stored.add(v)
        elif op == 'compute':
            for p in parents[v]:
                if p not in in_fast:
                    return False
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'delete':
            if v not in in_fast:
                return False
            in_fast.discard(v)
        else:
            return False
    
    for o in outputs:
        if o not in stored:
            return False
    
    return True


def _count_transfers(ops):
    """Count the number of transfers (load + store operations)."""
    return sum(1 for op, v in ops if op in ('load', 'store'))


def pebble(dag, fast_memory_budget):
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin before 90s kill
    
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        result = pebble_baseline(dag, fast_memory_budget)
        if result is not None:
            return result
    except Exception:
        pass
    
    parents, outputs, inputs, vertices, n = _parse_dag(dag)
    
    if parents is None or n is None:
        return []
    
    if outputs is None:
        outputs = [n - 1]
    
    if fast_memory_budget is None:
        fast_memory_budget = n
    if fast_memory_budget <= 0:
        fast_memory_budget = 1
    
    best_ops = None
    best_transfers = float('inf')
    
    # Try layered pebbling first (typically best for layered DAGs)
    try:
        ops = _layered_pebbling(parents, outputs, fast_memory_budget, deadline)
        if ops is not None and _validate_ops(ops, parents, outputs, fast_memory_budget):
            transfers = _count_transfers(ops)
            if transfers < best_transfers:
                best_transfers = transfers
                best_ops = ops
    except Exception:
        pass
    
    # Try greedy pebbling
    try:
        ops = _greedy_pebbling(parents, outputs, fast_memory_budget, deadline)
        if ops is not None and _validate_ops(ops, parents, outputs, fast_memory_budget):
            transfers = _count_transfers(ops)
            if transfers < best_transfers:
                best_transfers = transfers
                best_ops = ops
    except Exception:
        pass
    
    # Try Dijkstra/A* exact search for small instances
    # Only attempt if n is small enough to be tractable
    if n <= 15:
        try:
            ops = _dijkstra_pebbling(parents, outputs, fast_memory_budget, deadline)
            if ops is not None and _validate_ops(ops, parents, outputs, fast_memory_budget):
                transfers = _count_transfers(ops)
                if transfers < best_transfers:
                    best_transfers = transfers
                    best_ops = ops
        except Exception:
            pass
    
    if best_ops is not None:
        return best_ops
    
    # Last resort: very simple approach
    try:
        ops = []
        in_fast = set()
        children = _build_children(parents, n)
        is_output = set(outputs)
        
        for v in range(n):
            if time.time() > deadline:
                break
            
            for p in parents[v]:
                if p not in in_fast:
                    while len(in_fast) >= fast_memory_budget:
                        evicted = False
                        for u in list(in_fast):
                            if u not in is_output:
                                ops.append(['delete', u])
                                in_fast.discard(u)
                                evicted = True
                                break
                        if not evicted:
                            break
                    if len(in_fast) < fast_memory_budget:
                        ops.append(['load', p])
                        in_fast.add(p)
            
            if parents[v]:
                while len(in_fast) >= fast_memory_budget:
                    evicted = False
                    for u in list(in_fast):
                        if u not in is_output:
                            ops.append(['delete', u])
                            in_fast.discard(u)
                            evicted = True
                            break
                    if not evicted:
                        break
                if len(in_fast) < fast_memory_budget:
                    ops.append(['compute', v])
                    in_fast.add(v)
            
            if v in is_output:
                if v not in in_fast and not parents[v]:
                    while len(in_fast) >= fast_memory_budget:
                        evicted = False
                        for u in list(in_fast):
                            if u not in is_output:
                                ops.append(['delete', u])
                                in_fast.discard(u)
                                evicted = True
                                break
                        if not evicted:
                            break
                    if len(in_fast) < fast_memory_budget:
                        ops.append(['load', v])
                        in_fast.add(v)
                
                if v in in_fast:
                    ops.append(['store', v])
                    in_fast.discard(v)
            
            for p in parents[v]:
                if p in in_fast and p not in is_output:
                    # Check if p is still needed
                    still_needed = False
                    for c in children[p]:
                        if c > v:
                            still_needed = True
                            break
                    if not still_needed:
                        ops.append(['delete', p])
                        in_fast.discard(p)
        
        if _validate_ops(ops, parents, outputs, fast_memory_budget):
            return ops
    except Exception:
        pass
    
    return []

# EVOLVE-BLOCK-END
