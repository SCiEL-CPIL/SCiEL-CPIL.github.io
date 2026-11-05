import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    # Parse DAG structure
    # Case 1: dag is a list of lists (parents)
    # Case 2: dag is a dict {'edges': ..., 'outputs': ...}
    
    if isinstance(dag, dict):
        edges = dag.get('edges', [])
        outputs = set(dag.get('outputs', []))
        # Infer N
        N = 0
        for p_list in edges:
            for p in p_list:
                if p + 1 > N:
                    N = p + 1
        # Also check outputs
        for o in outputs:
            if o >= N:
                N = o + 1
        # Ensure edges list is long enough
        while len(edges) < N:
            edges.append([])
    else:
        edges = dag
        N = len(edges)
        # Infer outputs as sinks
        children = [[] for _ in range(N)]
        for i in range(N):
            for p in edges[i]:
                children[p].append(i)
        outputs = set([i for i in range(N) if len(children[i]) == 0])
        # Re-define children for the dict case if needed, but we need children for both
        if 'children' not in locals():
             children = [[] for _ in range(N)]
             for i in range(N):
                 for p in edges[i]:
                     children[p].append(i)
             # If outputs were not provided in dict, use sinks
             if not outputs:
                 outputs = set([i for i in range(N) if len(children[i]) == 0])

    if N == 0:
        return []

    # Build children list if not already done
    if 'children' not in locals():
        children = [[] for _ in range(N)]
        for i in range(N):
            for p in edges[i]:
                children[p].append(i)

    is_input = [len(edges[i]) == 0 for i in range(N)]
    is_output = [i in outputs for i in range(N)]

    in_fast = set()
    ops = []
    needed_count = [len(children[i]) for i in range(N)]
    next_ptr = [0] * N

    for i in range(N):
        if time.time() > deadline:
            break

        # 1. Ensure parents are in fast memory
        if is_input[i]:
            if i not in in_fast:
                ops.append(["load", i])
                in_fast.add(i)
        else:
            for p in edges[i]:
                if p not in in_fast:
                    ops.append(["load", p])
                    in_fast.add(p)
        
        # 2. Compute
        ops.append(["compute", i])
        in_fast.add(i)
        
        # 3. If output, store and delete
        if is_output[i]:
            ops.append(["store", i])
            ops.append(["delete", i])
            in_fast.remove(i)
        
        # 4. Update needed counts for parents and delete if done
        for p in edges[i]:
            needed_count[p] -= 1
            # Advance pointer for p to next unprocessed child
            while next_ptr[p] < len(children[p]) and children[p][next_ptr[p]] <= i:
                next_ptr[p] += 1
            
            if needed_count[p] == 0:
                if p in in_fast:
                    ops.append(["delete", p])
                    in_fast.remove(p)
        
        # 5. Memory Management
        if len(in_fast) > fast_memory_budget:
            victim = None
            max_dist = -1
            for v in in_fast:
                if next_ptr[v] < len(children[v]):
                    dist = children[v][next_ptr[v]]
                else:
                    dist = N
                
                if dist > max_dist:
                    max_dist = dist
                    victim = v
            
            if victim is not None:
                ops.append(["delete", victim])
                in_fast.remove(victim)

    return ops
