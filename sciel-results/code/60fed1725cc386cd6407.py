"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

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

    # Precompute which vertices can be computed given a set of fast vertices
    # A vertex v is computable if all its parents are in fast memory and v is not already in fast
    # A vertex v can be stored if v is in fast memory
    # A vertex v can be deleted if v is in fast memory
    # A vertex v can be loaded if v is not in fast memory and (v is input or v was previously computed/stored)
    # Actually, for the state-space search, we need to track what's "available" to load.
    # In standard pebbling, once a value is computed, it can be loaded back from slow memory.
    # So any vertex that has been computed at least once can be loaded.
    # But in the state-space, we need to know which vertices have been computed.
    # This makes the state space larger: (fast_set, computed_set).
    
    # Alternative: Since we can recompute, any vertex whose parents are all computable
    # (recursively) can eventually be computed. For the purpose of loading,
    # a vertex can be loaded if it has been computed at some point.
    # To keep the state space manageable, let's track (fast_set, computed_set).
    
    # Actually, let's think about this differently. The key insight is:
    # - A vertex can be loaded if it has been computed before (stored to slow memory)
    # - A vertex can be computed if all its parents are currently in fast memory
    # - A vertex can be stored if it's currently in fast memory
    # - A vertex can be deleted if it's currently in fast memory
    
    # State: (frozenset(fast), frozenset(computed))
    # where computed includes all vertices that have ever been computed or are inputs.
    # Wait, inputs start blue, so they're "available" from the start.
    # Let's define: available = set of vertices that can be loaded (inputs + computed vertices)
    # State: (frozenset(fast), frozenset(available))
    # Initially: fast = set(), available = set(inputs)
    # Goal: all outputs in available (since they've been stored) AND all outputs not in fast (they're blue)
    # Actually, the goal is that all outputs are blue, meaning they've been stored.
    # When we store a vertex, it moves from fast to available (slow memory).
    
    # Let's redefine:
    # fast: set of vertices currently in fast memory (red)
    # slow: set of vertices currently in slow memory (blue, available for loading)
    # Initially: fast = set(), slow = set(inputs)
    # Goal: all outputs in slow (they've been stored and are blue)
    # Operations:
    # - load v: v must be in slow, v not in fast, |fast| < budget. fast += {v}, slow -= {v}. Cost: 1
    # - store v: v must be in fast. fast -= {v}, slow += {v}. Cost: 1
    # - compute v: all parents of v must be in fast, v not in fast, |fast| < budget. fast += {v}. Cost: 0
    # - delete v: v must be in fast. fast -= {v}. Cost: 0
    
    # State: (frozenset(fast), frozenset(slow))
    # Initial: (frozenset(), frozenset(inputs))
    # Goal: all outputs in slow
    
    inputs_set = frozenset(v for v in range(n) if is_input[v])
    initial_state = (frozenset(), inputs_set)
    
    # For Dijkstra, we want minimum transfers (load + store count)
    # Use Dijkstra with cost = number of transfers
    
    # State space can be huge. Let's bound it.
    # Max vertices in fast: fast_memory_budget
    # Max vertices in slow: n (all vertices)
    # Total states: sum over k=0..budget of C(n,k) * 2^(n-k) which is way too large.
    
    # Let's limit the search: only consider states where |fast| <= budget
    # and use A* or Dijkstra with a good heuristic.
    
    # Heuristic: minimum additional transfers needed = 
    #   number of outputs not in slow (need to be computed and stored)
    #   + number of outputs in fast but not yet stored (need to be stored)
    # Actually, simpler: each output must be stored at least once.
    # Lower bound: number of outputs not yet in slow.
    
    def heuristic(state):
        fast, slow = state
        # Outputs not in slow need to be computed (and their parents loaded/computed) and stored
        # Minimum: each such output needs at least 1 store
        return len([o for o in outputs if o not in slow])
    
    # Dijkstra with A*
    # Priority queue: (f, g, state) where f = g + h, g = cost so far
    # We track g (actual cost) for each state
    
    best_g = {}
    best_g[initial_state] = 0
    parent = {}  # state -> (prev_state, op, vertex)
    
    pq = []
    h0 = heuristic(initial_state)
    heapq.heappush(pq, (h0, 0, initial_state))
    
    op_count = 0
    max_ops = 20000
    
    while pq:
        if time.time() > deadline:
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
            while cur in parent:
                prev, op, v = parent[cur]
                ops.append([op, v])
                cur = prev
            ops.reverse()
            # Check operation count
            if len(ops) <= max_ops:
                return ops
            else:
                # Too many operations, this path is invalid
                # Mark this state as not leading to valid solution
                best_g[state] = float('inf')
                continue
        
        # Generate neighbors
        # 1. Load: v in slow, v not in fast, |fast| < budget
        if len(fast) < fast_memory_budget:
            for v in slow:
                if v not in fast:
                    new_fast = frozenset(fast | {v})
                    new_slow = frozenset(slow - {v})
                    new_state = (new_fast, new_slow)
                    new_g = g + 1
                    if new_state not in best_g or new_g < best_g[new_state]:
                        best_g[new_state] = new_g
                        parent[new_state] = (state, 'load', v)
                        h = heuristic(new_state)
                        heapq.heappush(pq, (new_g + h, new_g, new_state))
        
        # 2. Store: v in fast
        for v in fast:
            new_fast = frozenset(fast - {v})
            new_slow = frozenset(slow | {v})
            new_state = (new_fast, new_slow)
            new_g = g + 1
            if new_state not in best_g or new_g < best_g[new_state]:
                best_g[new_state] = new_g
                parent[new_state] = (state, 'store', v)
                h = heuristic(new_state)
                heapq.heappush(pq, (new_g + h, new_g, new_state))
        
        # 3. Compute: all parents of v in fast, v not in fast, |fast| < budget
        if len(fast) < fast_memory_budget:
            for v in range(n):
                if v not in fast and parents[v]:
                    if all(p in fast for p in parents[v]):
                        new_fast = frozenset(fast | {v})
                        new_state = (new_fast, slow)
                        new_g = g  # compute costs 0
                        if new_state not in best_g or new_g < best_g[new_state]:
                            best_g[new_state] = new_g
                            parent[new_state] = (state, 'compute', v)
                            h = heuristic(new_state)
                            heapq.heappush(pq, (new_g + h, new_g, new_state))
        
        # 4. Delete: v in fast
        for v in fast:
            new_fast = frozenset(fast - {v})
            new_state = (new_fast, slow)
            new_g = g  # delete costs 0
            if new_state not in best_g or new_g < best_g[new_state]:
                best_g[new_state] = new_g
                parent[new_state] = (state, 'delete', v)
                h = heuristic(new_state)
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
    
    ops = []
    in_fast = set()
    
    def make_space():
        """Make space in fast memory by deleting vertices that are no longer needed."""
        if len(in_fast) < fast_memory_budget:
            return True
        # Find vertices to delete: those whose children have all been processed
        for u in list(in_fast):
            if u in is_output:
                continue
            # Check if all children of u are at depth <= current processing depth
            if children[u]:
                max_child_depth = max(depth[c] for c in children[u])
                # If the max child depth is less than current layer we're processing,
                # u is no longer needed
                if max_child_depth < current_layer:
                    ops.append(['delete', u])
                    in_fast.discard(u)
                    if len(in_fast) < fast_memory_budget:
                        return True
        # If still no space, delete any non-output vertex
        for u in list(in_fast):
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
                    still_needed = False
                    for c in children[v]:
                        if depth[c] >= d:
                            still_needed = True
                            break
                    if not still_needed:
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
        best = None
        best_lu = float('inf')
        for u in in_fast:
            if u not in is_output:
                lu = last_use[u]
                if lu < best_lu:
                    best_lu = lu
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
    if n <= 12:
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
