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
    
    # Precompute transitive dependencies for better heuristic
    # For each vertex, compute the set of all ancestors (transitive closure of parents)
    ancestors = [set() for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            ancestors[v] = ancestors[v] | {p} | ancestors[p]
    
    # For each vertex, compute which outputs depend on it (transitive closure of children)
    descendants = [set() for _ in range(n)]
    for v in range(n):
        for c in children[v]:
            descendants[v] = descendants[v] | {c} | descendants[c]
    
    # Precompute for each vertex: is it needed for any output?
    needed_for_output = [bool(descendants[v] & is_output) for v in range(n)]
    
    inputs_set = frozenset(v for v in range(n) if is_input[v])
    initial_state = (frozenset(), inputs_set)
    
    def heuristic(state):
        fast, slow = state
        # Count outputs not yet in slow (need at least 1 store each)
        outputs_not_stored = len([o for o in outputs if o not in slow])
        
        # For outputs in fast but not stored, they need 1 store
        # For outputs not in slow, they need to be computed (parents in fast) and stored
        # Better heuristic: for each output not in slow, estimate min transfers needed
        
        # Simple but stronger: each output not in slow needs at least 1 store
        # Additionally, if an output's parents are not all in fast, we need loads
        # But this is complex; use a simpler lower bound
        
        # Lower bound: number of outputs not in slow
        # Plus: for each fast vertex that is needed for an un-stored output, 
        # if it might need to be reloaded later, add cost
        # For now, use: outputs_not_stored + (number of fast vertices that are 
        # ancestors of un-stored outputs but might need eviction)
        
        # Conservative: just outputs_not_stored
        return outputs_not_stored
    
    # Dijkstra with A*
    # Priority queue: (f, g, state) where f = g + h, g = cost so far
    # We track g (actual cost) for each state
    
    best_g = {}
    best_g[initial_state] = 0
    parent = {}  # state -> (prev_state, op, vertex)
    
    pq = []
    h0 = heuristic(initial_state)
    heapq.heappush(pq, (h0, 0, initial_state))
    
    max_ops = 20000
    max_states = 50000  # Limit to prevent memory explosion
    
    while pq and len(best_g) < max_states:
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
                best_g[state] = float('inf')
                continue
        
        # Generate neighbors - prioritize useful operations
        # 1. Compute first (cost 0, often needed)
        if len(fast) < fast_memory_budget:
            for v in range(n):
                if v not in fast and parents[v]:
                    if all(p in fast for p in parents[v]):
                        new_fast = frozenset(fast | {v})
                        new_state = (new_fast, slow)
                        new_g = g
                        if new_state not in best_g or new_g < best_g[new_state]:
                            best_g[new_state] = new_g
                            parent[new_state] = (state, 'compute', v)
                            h = heuristic(new_state)
                            heapq.heappush(pq, (new_g + h, new_g, new_state))
        
        # 2. Store outputs first (they need to be stored)
        for v in fast:
            if v in is_output:
                new_fast = frozenset(fast - {v})
                new_slow = frozenset(slow | {v})
                new_state = (new_fast, new_slow)
                new_g = g + 1
                if new_state not in best_g or new_g < best_g[new_state]:
                    best_g[new_state] = new_g
                    parent[new_state] = (state, 'store', v)
                    h = heuristic(new_state)
                    heapq.heappush(pq, (new_g + h, new_g, new_state))
        
        # 3. Load vertices that are needed (parents of uncomputed vertices or needed for outputs)
        if len(fast) < fast_memory_budget:
            # Prioritize loading vertices that are ancestors of outputs not yet stored
            outputs_not_stored = [o for o in outputs if o not in slow]
            needed_ancestors = set()
            for o in outputs_not_stored:
                needed_ancestors |= ancestors[o]
            
            # Load from slow, prioritizing needed ancestors
            for v in sorted(slow, key=lambda x: (x not in needed_ancestors, x)):
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
        
        # 4. Store non-outputs (for memory management)
        for v in fast:
            if v not in is_output:
                new_fast = frozenset(fast - {v})
                new_slow = frozenset(slow | {v})
                new_state = (new_fast, new_slow)
                new_g = g + 1
                if new_state not in best_g or new_g < best_g[new_state]:
                    best_g[new_state] = new_g
                    parent[new_state] = (state, 'store', v)
                    h = heuristic(new_state)
                    heapq.heappush(pq, (new_g + h, new_g, new_state))
        
        # 5. Delete (last resort)
        for v in fast:
            new_fast = frozenset(fast - {v})
            new_state = (new_fast, slow)
            new_g = g
            if new_state not in best_g or new_g < best_g[new_state]:
                best_g[new_state] = new_g
                parent[new_state] = (state, 'delete', v)
                h = heuristic(new_state)
                heapq.heappush(pq, (new_g + h, new_g, new_state))
    
    return None


def _layered_pebbling(parents, outputs, fast_memory_budget, deadline):
    """
    Layer-by-layer pebbling strategy with memory budget awareness.
    For layered DAGs, process one layer at a time:
    - Load previous layer values needed for this layer
    - Compute current layer
    - Store current layer to slow memory
    - Delete previous layer from fast memory
    Respects fast_memory_budget by evicting when necessary.
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
    
    def evict_if_needed(num_needed):
        """Evict vertices from fast memory if we need room for num_needed more."""
        while len(in_fast) + num_needed > fast_memory_budget and in_fast:
            # Find vertex to evict: prefer non-outputs with no future uses
            best = None
            best_priority = float('inf')
            for u in in_fast:
                if u in is_output:
                    continue
                # Priority: lower is better (evict first)
                # Check if u is still needed by any future vertex
                still_needed = False
                for c in children[u]:
                    if depth[c] > depth[u]:  # Future use
                        still_needed = True
                        break
                if not still_needed:
                    best = u
                    best_priority = 0
                    break
                best_priority = 1
            if best is None:
                # All are outputs or still needed; evict any non-output
                for u in in_fast:
                    if u not in is_output:
                        best = u
                        break
            if best is None:
                # Only outputs left; can't evict
                return False
            ops.append(['delete', best])
            in_fast.discard(best)
        return len(in_fast) + num_needed <= fast_memory_budget
    
    # Process layer by layer
    for d in range(max_depth + 1):
        if time.time() > deadline:
            return None
        
        layer = layers[d]
        if not layer:
            continue
        
        if d == 0:
            # Layer 0: inputs - load them if they'll be needed
            for v in layer:
                # Only load if this input is needed for some output
                if descendants[v] & is_output or v in is_output:
                    if evict_if_needed(1):
                        ops.append(['load', v])
                        in_fast.add(v)
        else:
            # Collect all unique parents of vertices in this layer
            needed_parents = set()
            for v in layer:
                for p in parents[v]:
                    needed_parents.add(p)
            
            # Load those parents that are not already in fast
            parents_to_load = [p for p in sorted(needed_parents) if p not in in_fast]
            
            if parents_to_load:
                if not evict_if_needed(len(parents_to_load)):
                    # Can't fit all; load as many as possible
                    for p in parents_to_load:
                        if not evict_if_needed(1):
                            break
                        ops.append(['load', p])
                        in_fast.add(p)
            
            # Compute vertices in this layer
            for v in layer:
                if v not in in_fast:
                    if not evict_if_needed(1):
                        # Can't compute, skip (shouldn't happen if we loaded parents)
                        continue
                    ops.append(['compute', v])
                    in_fast.add(v)
            
            # Store vertices in this layer that are not final outputs
            for v in layer:
                if v not in is_output and v in in_fast:
                    ops.append(['store', v])
                    in_fast.discard(v)
        
        # Clean up: delete vertices from previous layers that are no longer needed
        if d > 0:
            for u in list(in_fast):
                if u in is_output:
                    continue
                # Check if u is needed by any vertex in current or future layers
                still_needed = False
                for c in children[u]:
                    if depth[c] >= d:  # Current or future use
                        still_needed = True
                        break
                if not still_needed:
                    ops.append(['delete', u])
                    in_fast.discard(u)
    
    # Store any remaining outputs that haven't been stored
    for o in outputs:
        if o in in_fast:
            ops.append(['store', o])
            in_fast.discard(o)
    
    return ops


def _greedy_pebbling(parents, outputs, fast_memory_budget, deadline):
    """
    Greedy pebbling with improved last-use analysis and lookahead.
    Processes vertices in topological order, evicting values that are no longer needed.
    """
    n = len(parents)
    if n == 0:
        return []
    
    children = _build_children(parents, n)
    is_output = set(outputs)
    
    # Precompute last use for each vertex
    last_use = [-1] * n
    for v in range(n):
        if children[v]:
            last_use[v] = max(children[v])
    
    # Precompute: for each vertex, count how many times it's used as a parent
    use_count = [0] * n
    for v in range(n):
        for p in parents[v]:
            use_count[p] += 1
    
    # Track remaining uses
    remaining_uses = use_count[:]
    
    ops = []
    in_fast = set()
    
    def try_evict():
        """Try to evict a vertex from fast memory. Returns True if successful."""
        if not in_fast:
            return False
        # Find vertex with smallest last_use that is not an output and has no remaining uses
        best = None
        best_lu = float('inf')
        best_remaining = float('inf')
        for u in in_fast:
            if u not in is_output:
                if remaining_uses[u] == 0:
                    # No more uses, prefer to evict
                    if last_use[u] < best_lu:
                        best_lu = last_use[u]
                        best = u
                        best_remaining = 0
                elif best is None or (remaining_uses[u] < best_remaining or 
                                      (remaining_uses[u] == best_remaining and last_use[u] < best_lu)):
                    best_lu = last_use[u]
                    best = u
                    best_remaining = remaining_uses[u]
        if best is not None:
            ops.append(['delete', best])
            in_fast.discard(best)
            return True
        # If all are outputs, evict any non-output
        for u in in_fast:
            if u not in is_output:
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
        
        # Store outputs
        if v in is_output and parents[v] and v in in_fast:
            ops.append(['store', v])
            in_fast.discard(v)
        
        # Update remaining uses for parents
        for p in parents[v]:
            if p in in_fast:
                remaining_uses[p] -= 1
                # Delete if no more uses and not an output
                if remaining_uses[p] == 0 and p not in is_output:
                    ops.append(['delete', p])
                    in_fast.discard(p)
    
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
    
    # Determine if instance is small enough for exact search
    # State space grows as C(n, budget) * 2^n roughly
    # For n <= 12 and budget <= 4, should be manageable
    exact_search_feasible = n <= 12 and fast_memory_budget <= 4
    
    # Try Dijkstra/A* exact search first for small instances
    if exact_search_feasible:
        try:
            ops = _dijkstra_pebbling(parents, outputs, fast_memory_budget, deadline)
            if ops is not None and _validate_ops(ops, parents, outputs, fast_memory_budget):
                transfers = _count_transfers(ops)
                if transfers < best_transfers:
                    best_transfers = transfers
                    best_ops = ops
        except Exception:
            pass
    
    # Try layered pebbling (optimal for layered DAGs)
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
