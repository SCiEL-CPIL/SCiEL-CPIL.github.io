"""Pebbling solver using Branch and Bound with bitmasks and heuristic pruning."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque

_DEADLINE = 75.0  # Safety margin under 90s wall-clock

def _parse_dag(dag):
    """Extract vertices, parents, outputs, inputs from the dag structure."""
    vertices = None
    for k in ('vertices', 'nodes', 'vertex', 'node'):
        if k in dag and dag[k] is not None:
            vertices = list(dag[k])
            break
    if vertices is None:
        all_v = set()
        parents = dag.get('parents', {})
        for v, ps in parents.items():
            all_v.add(v)
            for p in ps:
                all_v.add(p)
        vertices = sorted(all_v)

    if 'parents' in dag and dag['parents'] is not None:
        parents = {v: list(ps) for v, ps in dag['parents'].items()}
    elif 'edges' in dag and dag['edges'] is not None:
        parents = {}
        for e in dag['edges']:
            if isinstance(e, (list, tuple)) and len(e) == 2:
                p, c = e
                parents.setdefault(c, []).append(p)
            elif isinstance(e, dict):
                parents[e['child']] = [e['parent']]
        for v in vertices:
            parents.setdefault(v, [])
    else:
        parents = {v: [] for v in vertices}

    if 'outputs' in dag and dag['outputs'] is not None:
        outputs = set(dag['outputs'])
    elif 'output' in dag and dag['output'] is not None:
        outputs = set(dag['output']) if isinstance(dag['output'], (list, set, tuple)) else {dag['output']}
    else:
        children = {}
        for v, ps in parents.items():
            for p in ps:
                children.setdefault(p, []).append(v)
        outputs = {v for v in vertices if v not in children}

    if 'inputs' in dag and dag['inputs'] is not None:
        inputs = set(dag['inputs'])
    elif 'blue' in dag and dag['blue'] is not None:
        inputs = set(dag['blue'])
    elif 'blue_vertices' in dag and dag['blue_vertices'] is not None:
        inputs = set(dag['blue_vertices'])
    else:
        inputs = {v for v in vertices if not parents.get(v)}

    return vertices, parents, outputs, inputs

def _count_transfers(ops):
    return sum(1 for op, v in ops if op in ('load', 'store'))

def _validate_ops(ops, dag, budget):
    """Validate operation sequence. Returns True if legal and outputs end blue."""
    vertices, parents, outputs, inputs = _parse_dag(dag)
    blue = set(inputs)
    fast = set()
    available = set(inputs)
    for op, v in ops:
        if op == 'load':
            if v not in inputs or v in fast or len(fast) >= budget:
                return False
            fast.add(v)
            available.add(v)
        elif op == 'store':
            if v not in fast and v not in blue:
                return False
            if v in fast:
                fast.remove(v)
            blue.add(v)
        elif op == 'compute':
            for p in parents.get(v, []):
                if p not in available:
                    return False
            if len(fast) >= budget:
                return False
            fast.add(v)
            available.add(v)
        elif op == 'delete':
            if v not in fast:
                return False
            fast.remove(v)
            if v not in blue:
                available.discard(v)
        else:
            return False
    return outputs.issubset(blue)

def _toposort(vertices, parents):
    indeg = {v: len(parents.get(v, [])) for v in vertices}
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for w in children[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    return topo

def _solve_small_dag(vertices, parents, outputs, inputs, budget, time_limit=30.0):
    """Branch and Bound with bitmasks for N <= 60."""
    n = len(vertices)
    if n > 60:
        return None
    
    idx = {v: i for i, v in enumerate(vertices)}
    # Bitmasks
    parent_mask = [0] * n
    for v in vertices:
        i = idx[v]
        for p in parents.get(v, []):
            parent_mask[i] |= (1 << idx[p])
    
    init_blue_mask = 0
    for v in inputs:
        init_blue_mask |= (1 << idx[v])
    
    output_mask = 0
    for v in outputs:
        output_mask |= (1 << idx[v])
        
    # Precompute ancestors of outputs for heuristic
    # anc_mask[i] = bitmask of ancestors of i (including i)
    anc_mask = [0] * n
    topo = _toposort(vertices, parents)
    for v in topo:
        i = idx[v]
        anc_mask[i] = (1 << i)
        for p in parents.get(v, []):
            anc_mask[i] |= anc_mask[idx[p]]
            
    # Heuristic: min transfers to finish
    # 1. Each output not in blue needs at least 1 store.
    # 2. If fast is full and we need to compute, we need to free space.
    def heuristic(blue_mask, fast_mask):
        missing_outputs = output_mask & ~blue_mask
        h = bin(missing_outputs).count('1')
        # If we have vertices to compute that are not in fast and not in blue,
        # and fast is full, we might need extra stores/deletes.
        # Simple admissible bound: just the missing outputs.
        return h

    best_cost = float('inf')
    best_path = None
    start_time = time.time()
    
    # State: (blue_mask, fast_mask, cost, path)
    # Using Dijkstra/A*
    # Priority: cost + heuristic
    init_state = (init_blue_mask, 0)
    if output_mask & ~init_blue_mask == 0:
        return []
        
    # We use a priority queue.
    # To avoid memory explosion, we limit the number of states explored.
    max_states = 100000
    states_explored = 0
    best_known = {init_state: 0}
    pq = [(heuristic(*init_state), 0, init_state, [])]
    
    while pq:
        if time.time() - start_time > time_limit:
            break
        if states_explored > max_states:
            break
            
        h, g, state, path = heapq.heappop(pq)
        if g + h >= best_cost:
            continue
            
        blue_mask, fast_mask = state
        states_explored += 1
        
        if (blue_mask & output_mask) == output_mask:
            best_cost = g
            best_path = path
            break
            
        # Generate successors
        available_mask = blue_mask | fast_mask
        fast_count = bin(fast_mask).count('1')
        
        # 1. Load
        # Can load any input not in fast
        for v in inputs:
            i = idx[v]
            bit = 1 << i
            if not (fast_mask & bit) and fast_count < budget:
                new_fast = fast_mask | bit
                new_state = (blue_mask, new_fast)
                ng = g + 1
                if ng < best_known.get(new_state, float('inf')):
                    best_known[new_state] = ng
                    heapq.heappush(pq, (ng + heuristic(*new_state), ng, new_state, path + [('load', v)]))
                    
        # 2. Store
        # Can store any vertex in fast
        for v in vertices:
            i = idx[v]
            bit = 1 << i
            if fast_mask & bit:
                new_blue = blue_mask | bit
                new_fast = fast_mask & ~bit
                new_state = (new_blue, new_fast)
                ng = g + 1
                if ng < best_known.get(new_state, float('inf')):
                    best_known[new_state] = ng
                    heapq.heappush(pq, (ng + heuristic(*new_state), ng, new_state, path + [('store', v)]))
                    
        # 3. Compute
        # Can compute any vertex not in fast, not in blue (if blue, it's already done? No, blue means stored. If in blue, we don't need to compute again unless we deleted it? No, delete only removes from fast. If in blue, it's available. So we only compute if NOT in available? No, if in blue, it is available. If we compute it again, it goes to fast. Is it useful? Maybe if we need it in fast for a child. But if it's in blue, it's available. So computing it again is only useful if we want it in fast.
        # However, if v is in blue, it is available. If we compute it, it enters fast.
        # But if v is in blue, we don't *need* to compute it to make it available.
        # So we only compute if v is NOT in available_mask?
        # Actually, if v is in blue, it is available. If we compute it, we pay 0 cost but occupy fast.
        # It might be useful to have it in fast to avoid loading? No, inputs are loaded.
        # It might be useful to have it in fast to compute children without checking blue? No, available = blue | fast.
        # So computing a vertex already in blue is only useful if we want to move it to fast.
        # But we can just "load" it? No, load is for inputs only.
        # So if v is in blue and not in fast, and we want it in fast, we must compute it.
        # Cost is 0.
        
        for v in vertices:
            i = idx[v]
            bit = 1 << i
            if fast_mask & bit:
                continue
            # Check if parents are available
            if (parent_mask[i] & ~available_mask) == 0:
                # Parents are available
                # Can we compute?
                # If fast is full, we can't.
                if fast_count < budget:
                    new_fast = fast_mask | bit
                    # Blue stays same
                    new_state = (blue_mask, new_fast)
                    ng = g # Compute cost is 0
                    if ng < best_known.get(new_state, float('inf')):
                        best_known[new_state] = ng
                        heapq.heappush(pq, (ng + heuristic(*new_state), ng, new_state, path + [('compute', v)]))
                        
        # 4. Delete
        for v in vertices:
            i = idx[v]
            bit = 1 << i
            if fast_mask & bit:
                new_fast = fast_mask & ~bit
                # Blue stays same
                new_state = (blue_mask, new_fast)
                ng = g # Delete cost is 0
                if ng < best_known.get(new_state, float('inf')):
                    best_known[new_state] = ng
                    heapq.heappush(pq, (ng + heuristic(*new_state), ng, new_state, path + [('delete', v)]))
                    
    return best_path

def _naive_schedule(vertices, parents, outputs, inputs, budget):
    """Naive topological schedule."""
    ops = []
    for v in vertices:
        ps = parents.get(v, [])
        if not ps:
            ops.append(('load', v))
        else:
            ops.append(('compute', v))
        if v in outputs:
            ops.append(('store', v))
        else:
            ops.append(('delete', v))
    return ops

def _greedy_schedule(vertices, parents, outputs, inputs, budget):
    """Greedy: compute in topo order, store if output or needed later, delete if not."""
    topo = _toposort(vertices, parents)
    # Count how many times each vertex is needed as a parent
    usage_count = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            usage_count[p] += 1
            
    blue = set(inputs)
    fast = set()
    ops = []
    
    for v in topo:
        ps = parents.get(v, [])
        # Ensure parents are available
        for p in ps:
            if p not in fast and p not in blue:
                # This shouldn't happen if we manage fast correctly
                pass
            
        if not ps:
            # Input
            if v not in fast and len(fast) < budget:
                ops.append(('load', v))
                fast.add(v)
        else:
            # Compute
            # If fast is full, we might need to delete something not needed
            if len(fast) >= budget:
                # Find a vertex in fast that is not an output and not needed later
                # "Needed later" means it has children in topo order that haven't been processed
                # Simple heuristic: delete any non-output
                for u in list(fast):
                    if u not in outputs:
                        ops.append(('delete', u))
                        fast.remove(u)
                        break
                if len(fast) >= budget:
                    # Still full, delete any
                    u = next(iter(fast))
                    ops.append(('delete', u))
                    fast.remove(u)
                    
            if v not in fast:
                ops.append(('compute', v))
                fast.add(v)
                
        # Decide what to do with v
        # If v is output, store it
        if v in outputs:
            ops.append(('store', v))
            fast.remove(v)
            blue.add(v)
        else:
            # If v is needed by future vertices, keep in fast
            # Check if any child of v is in topo[v+1:]
            # We can precompute children
            children = {v: [] for v in vertices}
            for u in vertices:
                for p in parents.get(u, []):
                    children[p].append(u)
            
            needed = False
            for c in children[v]:
                if c in topo[topo.index(v)+1:]:
                    needed = True
                    break
            
            if not needed:
                ops.append(('delete', v))
                fast.remove(v)
                
    return ops

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    try:
        budget = int(fast_memory_budget)
        if budget < 1:
            budget = 1
        vertices, parents, outputs, inputs = _parse_dag(dag)

        if not vertices:
            return []

        if outputs.issubset(inputs):
            ops = []
            for v in outputs:
                ops.append(('load', v))
                ops.append(('store', v))
                ops.append(('delete', v))
            return ops

        best_ops = None
        best_cost = float('inf')

        # 1. Try Exact B&B for small DAGs
        if len(vertices) <= 40:
            try:
                ops = _solve_small_dag(vertices, parents, outputs, inputs, budget, time_limit=20.0)
                if ops is not None:
                    if _validate_ops(ops, dag, budget):
                        c = _count_transfers(ops)
                        if c < best_cost:
                            best_cost = c
                            best_ops = ops
            except Exception:
                pass

        # 2. Try Greedy
        if time.time() - t0 < _DEADLINE - 5:
            try:
                ops = _greedy_schedule(vertices, parents, outputs, inputs, budget)
                if _validate_ops(ops, dag, budget):
                    c = _count_transfers(ops)
                    if c < best_cost:
                        best_cost = c
                        best_ops = ops
            except Exception:
                pass

        # 3. Fallback Naive
        if best_ops is None:
            ops = _naive_schedule(vertices, parents, outputs, inputs, budget)
            if _validate_ops(ops, dag, budget):
                best_ops = ops
            else:
                # Last resort
                best_ops = ops

        return best_ops if best_ops is not None else []

    except Exception:
        return []
# EVOLVE-BLOCK-END
