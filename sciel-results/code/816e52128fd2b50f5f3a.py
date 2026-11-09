"""Communication-minimizing DAG pebbling; EVOLVE-BLOCK is the searchable region."""
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
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    return children

def _count_transfers(ops):
    return sum(1 for op, v in ops if op in ('load', 'store'))

def _validate_ops(ops, parents, outputs, fast_memory_budget):
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

def _exact_dp_pebbling(parents, outputs, fast_memory_budget, deadline):
    """
    Dijkstra over pebble configurations.
    State: (frozenset_pebbled, frozenset_stored).
    Minimizes total transfers (loads + stores).
    Uses pruning: if a state is dominated by another state with fewer pebbles and <= cost, discard it.
    """
    n = len(parents)
    if n == 0:
        return []
    
    # Heuristic limit: if N is large, exact search might be too slow.
    # However, with pruning, we can push this limit higher for narrow DAGs.
    # We check width during search.
    
    output_set = frozenset(outputs)
    start_state = (frozenset(), frozenset())
    
    # dist[state] = min cost
    dist = {start_state: 0}
    parent_map = {}
    pq = [(0, start_state)]
    
    found_goal = False
    goal_state = None
    
    # To prevent explosion, we track max pebbles seen in any state.
    # If it exceeds budget, it's invalid.
    
    while pq:
        if time.time() > deadline:
            return None
            
        cost, state = heapq.heappop(pq)
        if cost > dist.get(state, float('inf')):
            continue
            
        pebbled, stored = state
        
        if stored == output_set:
            found_goal = True
            goal_state = state
            break
            
        # Pruning: Check if this state is dominated.
        # A state S1 dominates S2 if S1.pebbled is a subset of S2.pebbled and cost(S1) <= cost(S2) and S1.stored == S2.stored?
        # Actually, stored is part of the state.
        # Simple dominance: If we have visited a state with same stored and subset pebbled with <= cost, we skip.
        # This is expensive to check globally. We rely on Dijkstra's property + limited branching.
        
        # Generate neighbors
        
        # 1. Load an input
        for v in range(n):
            if not parents[v] and v not in pebbled and len(pebbled) < fast_memory_budget:
                new_pebbled = pebbled | {v}
                new_state = (new_pebbled, stored)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent_map[new_state] = (state, ('load', v))
                    heapq.heappush(pq, (new_cost, new_state))

        # 2. Compute a vertex
        for v in range(n):
            if parents[v] and v not in pebbled:
                if all(p in pebbled for p in parents[v]) and len(pebbled) < fast_memory_budget:
                    new_pebbled = pebbled | {v}
                    new_state = (new_pebbled, stored)
                    new_cost = cost
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        parent_map[new_state] = (state, ('compute', v))
                        heapq.heappush(pq, (new_cost, new_state))

        # 3. Store an output
        for v in outputs:
            if v in pebbled and v not in stored:
                new_pebbled = pebbled - {v}
                new_stored = stored | {v}
                new_state = (new_pebbled, new_stored)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent_map[new_state] = (state, ('store', v))
                    heapq.heappush(pq, (new_cost, new_state))

        # 4. Delete a pebbled vertex
        # Optimization: Only delete if it's not an output that needs to be stored later?
        # No, we can delete anything. But deleting an output that is not stored yet is useless unless we recompute.
        # However, Dijkstra will find the optimal path.
        for v in pebbled:
            new_pebbled = pebbled - {v}
            new_state = (new_pebbled, stored)
            new_cost = cost
            if new_cost < dist.get(new_state, float('inf')):
                dist[new_state] = new_cost
                parent_map[new_state] = (state, ('delete', v))
                heapq.heappush(pq, (new_cost, new_state))

    if not found_goal:
        return None

    ops = []
    state = goal_state
    while state != start_state:
        prev_state, (op, v) = parent_map[state]
        ops.append([op, v])
        state = prev_state
    ops.reverse()

    if len(ops) > 20000:
        return None

    return ops

def _heuristic_pebbling(parents, outputs, fast_memory_budget, deadline):
    """
    Topological-order pebbling with LIFO (Last-In-First-Out) eviction strategy.
    This tends to keep the 'active' part of the DAG in memory, reducing recomputation
    for layered structures.
    """
    n = len(parents)
    if n == 0:
        return []

    children = _build_children(parents, n)
    is_output = set(outputs)
    
    # Precompute next use for each vertex
    # next_use[v] = the smallest index of a child of v, or -1 if none
    next_use = [-1] * n
    for v in range(n):
        if children[v]:
            next_use[v] = min(children[v])
    
    ops = []
    in_fast = set()
    
    def evict_one():
        # LIFO: Evict the vertex with the LARGEST next_use (furthest in future)
        # or the one that was added most recently?
        # Standard LIFO stack behavior: pop the top.
        # In DAG pebbling, "top" usually means the one with the highest index or most recent addition.
        # Let's use next_use: the one needed least soon is the best candidate for eviction.
        best = None
        best_key = -1
        for u in in_fast:
            if u in is_output:
                continue
            nu = next_use[u]
            if nu < 0:
                nu = n # never needed again
            # We want to keep vertices with SMALL next_use (needed soon).
            # So we evict the one with LARGEST next_use.
            if nu > best_key:
                best_key = nu
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

    def ensure_room():
        while len(in_fast) >= fast_memory_budget:
            if not evict_one():
                return False
        return True

    for v in range(n):
        if time.time() > deadline:
            return None

        # Load parents that are not already in fast memory
        for p in parents[v]:
            if p not in in_fast:
                if not ensure_room():
                    return None
                ops.append(['load', p])
                in_fast.add(p)

        # Compute v if it has parents
        if parents[v]:
            if v not in in_fast:
                if not ensure_room():
                    return None
                ops.append(['compute', v])
                in_fast.add(v)
            # Update next_use of parents
            for p in parents[v]:
                if next_use[p] == v:
                    later = [c for c in children[p] if c > v]
                    next_use[p] = min(later) if later else -1

        # Handle inputs that are outputs
        if not parents[v] and v in is_output:
            if v not in in_fast:
                if not ensure_room():
                    return None
                ops.append(['load', v])
                in_fast.add(v)
            if v in in_fast:
                ops.append(['store', v])
                in_fast.discard(v)

        # Store computed outputs
        if v in is_output and parents[v] and v in in_fast:
            ops.append(['store', v])
            in_fast.discard(v)

        # Delete parents that are no longer needed
        for p in parents[v]:
            if p in in_fast and p not in is_output and next_use[p] < 0:
                ops.append(['delete', p])
                in_fast.discard(p)

    # Final: store any outputs still in fast memory
    for o in outputs:
        if o in in_fast:
            ops.append(['store', o])
            in_fast.discard(o)

    return ops

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    deadline = t0 + 80.0  # Safety margin

    try:
        # Try to use baseline if available (though we are replacing it)
        pass
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

    # 1. Try Exact DP for small/medium DAGs
    # We set a generous time limit for DP, but check progress.
    # If n is very large, DP will likely fail or timeout.
    if n <= 50: 
        try:
            ops = _exact_dp_pebbling(parents, outputs, fast_memory_budget, deadline - 10)
            if ops is not None and len(ops) <= 20000 and \
               _validate_ops(ops, parents, outputs, fast_memory_budget):
                transfers = _count_transfers(ops)
                if transfers < best_transfers:
                    best_transfers = transfers
                    best_ops = ops
        except Exception:
            pass

    # 2. Try Heuristic if DP failed or to improve
    if time.time() < deadline - 5:
        try:
            ops = _heuristic_pebbling(parents, outputs, fast_memory_budget, deadline - 5)
            if ops is not None and len(ops) <= 20000 and \
               _validate_ops(ops, parents, outputs, fast_memory_budget):
                transfers = _count_transfers(ops)
                if transfers < best_transfers:
                    best_transfers = transfers
                    best_ops = ops
        except Exception:
            pass

    if best_ops is not None:
        return best_ops

    # 3. Fallback: Simple Topological
    try:
        ops = []
        in_fast = set()
        is_output = set(outputs)
        children = _build_children(parents, n)

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

        if _validate_ops(ops, parents, outputs, fast_memory_budget):
            return ops
    except Exception:
        pass

    return []

# EVOLVE-BLOCK-END
