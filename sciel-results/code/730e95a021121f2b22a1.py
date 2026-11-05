import time
import sys

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin for 60s limit

    # Parse DAG structure
    # Assume dag is either:
    # 1. List of lists: dag[i] = parents of i
    # 2. Tuple (parents, outputs)
    # 3. Dict with keys 'parents', 'outputs'
    
    if isinstance(dag, dict):
        parents = dag.get('parents', [])
        outputs = dag.get('outputs', [])
    elif isinstance(dag, (tuple, list)) and len(dag) == 2 and isinstance(dag[0], list) and isinstance(dag[1], list):
        # Heuristic: if it looks like (list, list)
        # Check if dag[0] is list of lists and dag[1] is list of ints
        if all(isinstance(x, list) for x in dag[0]) and all(isinstance(x, int) for x in dag[1]):
            parents = dag[0]
            outputs = dag[1]
        else:
            parents = dag[0]
            outputs = dag[1]
    else:
        parents = dag
        outputs = []

    N = len(parents)
    M = fast_memory_budget
    
    if N == 0:
        return []

    # Identify outputs if not provided
    if not outputs:
        # Outputs are nodes with no children (sinks)
        has_child = [False] * N
        for i in range(N):
            for p in parents[i]:
                has_child[p] = True
        outputs = [i for i in range(N) if not has_child[i]]
    
    outputs_set = set(outputs)

    # Build use_lists: for each node u, list of indices i where u is a parent of i
    use_lists = [[] for _ in range(N)]
    for i in range(N):
        for p in parents[i]:
            use_lists[p].append(i)
    
    # use_lists are already sorted because i increases
    ptr = [0] * N
    
    # Initial memory state: Inputs (sources) start Blue
    # Sources are nodes with no parents
    sources = [i for i in range(N) if len(parents[i]) == 0]
    
    mem = set()
    ops = []
    
    # Add sources to memory
    # If sources > M, we might need to evict? 
    # But we can't compute anything without parents.
    # We will add them and let the eviction logic handle overflow if it occurs.
    # However, eviction requires "next use".
    # For sources, next use is their first child index.
    
    for s in sources:
        mem.add(s)
        
    # If initial mem > M, we have a problem. 
    # We will just proceed and evict as needed when we try to add more.
    # But we should probably evict now if len(mem) > M?
    # If we don't, and we try to load a parent, we will evict then.
    # Let's stick to evicting only when we need space for a NEW load or after compute.
    
    def get_next_use(u):
        if ptr[u] < len(use_lists[u]):
            return use_lists[u][ptr[u]]
        else:
            if u in outputs_set:
                return N  # Virtual use at the end
            else:
                return float('inf')

    def find_victim(mem_set):
        # Find node in mem_set with max next use
        best_victim = None
        best_use = -1
        for u in mem_set:
            nu = get_next_use(u)
            if nu > best_use:
                best_use = nu
                best_victim = u
        return best_victim

    # Main loop
    for i in range(N):
        if time.time() > deadline:
            break
            
        # Process parents of i
        for p in parents[i]:
            if p not in mem:
                # Need to load p
                if len(mem) >= M:
                    # Evict someone
                    victim = find_victim(mem)
                    if victim is not None:
                        ops.append(['store', victim])
                        mem.remove(victim)
                
                ops.append(['load', p])
                mem.add(p)
            
            # Advance pointer for p
            # Since p is a parent of i, i is in use_lists[p]
            # We assume ptr[p] points to i (or an earlier index if we missed it? No, order is strict)
            ptr[p] += 1

        # Compute i
        ops.append(['compute', i])
        
        # Add i to memory
        mem.add(i)
        
        # If memory overflow, evict
        if len(mem) > M:
            victim = find_victim(mem)
            if victim is not None:
                ops.append(['store', victim])
                mem.remove(victim)

    # Finalize outputs
    for o in outputs:
        if time.time() > deadline:
            break
            
        if o not in mem:
            if len(mem) >= M:
                victim = find_victim(mem)
                if victim is not None:
                    ops.append(['store', victim])
                    mem.remove(victim)
            
            ops.append(['load', o])
            mem.add(o)

    return ops
