import heapq
import time

def pebble(dag, fast_memory_budget):
    # Parse DAG
    if isinstance(dag, list):
        n = len(dag)
        parents_list = [0] * n
        is_output = [False] * n
        for i, v in enumerate(dag):
            if isinstance(v, dict):
                if 'parents' in v:
                    parents_list[i] = v['parents']
                if 'output' in v:
                    is_output[i] = v['output']
                elif 'is_output' in v:
                    is_output[i] = v['is_output']
    elif isinstance(dag, dict):
        n = len(dag)
        ids = sorted(dag.keys())
        id_to_idx = {id: i for i, id in enumerate(ids)}
        parents_list = [[] for _ in range(n)]
        is_output = [False] * n
        for id, v in dag.items():
            idx = id_to_idx[id]
            if isinstance(v, dict):
                if 'parents' in v:
                    parents_list[idx] = [id_to_idx[p] for p in v['parents']]
                if 'output' in v:
                    is_output[idx] = v['output']
                elif 'is_output' in v:
                    is_output[idx] = v['is_output']
            elif isinstance(v, list):
                parents_list[idx] = [id_to_idx[p] for p in v]
    else:
        return []

    if not any(is_output):
        is_output = [True] * n

    # Filter to ancestors of outputs
    # Build reverse graph
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents_list[i]:
            children[p].append(i)
            
    relevant = [False] * n
    stack = [i for i in range(n) if is_output[i]]
    while stack:
        u = stack.pop()
        if relevant[u]:
            continue
        relevant[u] = True
        for p in parents_list[u]:
            if not relevant[p]:
                stack.append(p)
                
    # If no relevant vertices, return empty
    if not any(relevant):
        return []

    # Remap indices to only relevant vertices?
    # Or just skip irrelevant vertices in transitions.
    # Skipping is easier.

    inputs = [i for i in range(n) if len(parents_list[i]) == 0 and relevant[i]]
    
    start_mask_C = 0
    for i in inputs:
        start_mask_C |= (1 << i)
    start_mask_R = 0
    start_state = (start_mask_C, start_mask_R)
    
    goal_mask_C = 0
    for i in range(n):
        if is_output[i] and relevant[i]:
            goal_mask_C |= (1 << i)

    parent_masks = [0] * n
    for i in range(n):
        m = 0
        for p in parents_list[i]:
            if relevant[p]:
                m |= (1 << p)
        parent_masks[i] = m

    pq = [(0, start_mask_C, start_mask_R)]
    best_cost = {start_state: 0}
    parent_state = {start_state: None}
    
    start_time = time.time()
    deadline = start_time + 50.0
    
    while pq:
        if time.time() > deadline:
            break
            
        cost, mask_C, mask_R = heapq.heappop(pq)
        state = (mask_C, mask_R)
        
        if best_cost.get(state, float('inf')) < cost:
            continue
            
        if (mask_C & goal_mask_C) == goal_mask_C:
            # Reconstruct
            operations = []
            curr = state
            while parent_state[curr] is not None:
                prev, op = parent_state[curr]
                operations.append(op)
                curr = prev
            operations.reverse()
            result = [[op, v] for op, v in operations]
            return result
            
        available_mask = mask_C | mask_R
        
        # 1. Compute v
        # Only for relevant v
        for v in range(n):
            if not relevant[v]:
                continue
            if (mask_C & (1 << v)) or (mask_R & (1 << v)):
                continue
            if (parent_masks[v] & ~available_mask) == 0:
                new_mask_R = mask_R | (1 << v)
                if bin(new_mask_R).count('1') > fast_memory_budget:
                    continue
                new_state = (mask_C, new_mask_R)
                new_cost = cost
                if new_cost < best_cost.get(new_state, float('inf')):
                    best_cost[new_state] = new_cost
                    parent_state[new_state] = (state, ('compute', v))
                    heapq.heappush(pq, (new_cost, mask_C, new_mask_R))
                    
        # 2. Store v
        r_bits = mask_R
        while r_bits:
            lsb = r_bits & -r_bits
            v = lsb.bit_length() - 1
            r_bits -= lsb
            if not relevant[v]:
                continue
            new_mask_C = mask_C | (1 << v)
            new_mask_R = mask_R & ~(1 << v)
            new_state = (new_mask_C, new_mask_R)
            new_cost = cost + 1
            if new_cost < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = new_cost
                parent_state[new_state] = (state, ('store', v))
                heapq.heappush(pq, (new_cost, new_mask_C, new_mask_R))
                
        # 3. Load v
        # Only load if v is a parent of some relevant u not in C
        # Precompute "needed" mask?
        # v is needed if there exists u such that v in parents(u), relevant[u], and u not in C
        c_bits = mask_C
        while c_bits:
            lsb = c_bits & -c_bits
            v = lsb.bit_length() - 1
            c_bits -= lsb
            if not relevant[v]:
                continue
            if (mask_R & (1 << v)):
                continue
            # Check if v is needed
            needed = False
            for u in range(n):
                if not relevant[u]:
                    continue
                if (mask_C & (1 << u)):
                    continue
                if v in parents_list[u]:
                    needed = True
                    break
            if not needed:
                continue
                
            new_mask_R = mask_R | (1 << v)
            if bin(new_mask_R).count('1') > fast_memory_budget:
                continue
            new_state = (mask_C, new_mask_R)
            new_cost = cost + 1
            if new_cost < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = new_cost
                parent_state[new_state] = (state, ('load', v))
                heapq.heappush(pq, (new_cost, mask_C, new_mask_R))
                
        # 4. Delete v
        r_bits = mask_R
        while r_bits:
            lsb = r_bits & -r_bits
            v = lsb.bit_length() - 1
            r_bits -= lsb
            if not relevant[v]:
                continue
            new_mask_R = mask_R & ~(1 << v)
            new_state = (mask_C, new_mask_R)
            new_cost = cost
            if new_cost < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = new_cost
                parent_state[new_state] = (state, ('delete', v))
                heapq.heappush(pq, (new_cost, mask_C, new_mask_R))

    # If not found, return empty
    return []
