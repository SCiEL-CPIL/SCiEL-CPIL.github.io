import time

def construct():
    P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    n = P['n']
    edges = P['edges']
    
    adj = [[False] * n for _ in range(n)]
    neighbors = [set() for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        neighbors[u].add(v)
        neighbors[v].add(u)
        
    start_time = time.time()
    deadline = start_time + 80.0 # 80s limit to be safe

    def is_clique(S, v):
        for u in S:
            if not adj[u][v]:
                return False
        return True

    def is_independent(S, v):
        for u in S:
            if adj[u][v]:
                return False
        return True

    # To speed up, we can sort vertices by degree descending?
    # But the problem asks for partition of 0..13.
    # The order of assignment doesn't change the final partition structure, just the search path.
    # However, the output must be a partition of 0..13.
    # The function returns the lists. The order of lists doesn't matter.
    # The order of vertices in lists doesn't matter.
    # So we can permute the vertex assignment order.
    # Let's assign vertices in order of decreasing degree to prune earlier.
    deg = [len(neighbors[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    
    # We need to map back to original indices?
    # No, the parts contain original vertex indices.
    # We just pick the next vertex to place from `order`.
    
    # State:
    # parts: list of lists
    # types: list of 'C', 'I', None
    # pos: index in `order`
    # max_part: highest part index used
    
    def solve(pos, parts, types, max_part):
        if time.time() > deadline:
            return False # Timeout, return failure to fallback or raise? 
                         # Better to return a valid suboptimal result if we have one?
                         # The problem says "Minimize number of parts".
                         # If we timeout, we should return the best found so far.
                         # But `solve` returns True/False.
                         # We can use a global variable to store the best solution.
                         
        if pos == n:
            return True
            
        v = order[pos]
        
        # Try existing parts
        for i in range(max_part + 1):
            S = parts[i]
            t = types[i]
            
            if t is None:
                # Try C
                parts[i].append(v)
                types[i] = 'C'
                if solve(pos + 1, parts, types, max_part):
                    return True
                parts[i].pop()
                types[i] = None
                
                # Try I
                parts[i].append(v)
                types[i] = 'I'
                if solve(pos + 1, parts, types, max_part):
                    return True
                parts[i].pop()
                types[i] = None
                continue
                
            elif t == 'C':
                if is_clique(S, v):
                    parts[i].append(v)
                    if solve(pos + 1, parts, types, max_part):
                        return True
                    parts[i].pop()
            else: # I
                if is_independent(S, v):
                    parts[i].append(v)
                    if solve(pos + 1, parts, types, max_part):
                        return True
                    parts[i].pop()

        # Try new part
        if max_part + 1 < len(parts):
            i = max_part + 1
            parts[i] = [v]
            types[i] = 'C'
            if solve(pos + 1, parts, types, max_part + 1):
                return True
            
            types[i] = 'I'
            if solve(pos + 1, parts, types, max_part + 1):
                return True
            
            parts[i] = []
            types[i] = None

        return False

    # We need to handle the timeout properly.
    # If solve returns False due to timeout, we might not have a solution.
    # We can run the search for K=1, 2, ...
    # If we find a solution for K, we return it.
    # If we timeout during K, we break and return the best K-1? 
    # No, if we timeout during K, we haven't found a solution for K.
    # We should return the best valid partition found so far.
    # But we only return when we find a valid partition for a specific K.
    # If we don't find any, we return a greedy solution?
    # The greedy solution is always valid.
    
    # Let's store the best solution found in a non-local variable.
    best_solution = None
    
    # Reset deadline check inside solve?
    # The `time.time()` call has overhead.
    # For N=14, it should be fast.
    
    # To ensure we always return *something* valid:
    # We can run the exact solver.
    # If it times out, we return a greedy partition.
    
    # Greedy partition fallback
    def greedy():
        parts = []
        for v in range(n):
            placed = False
            for p in parts:
                # Check if v can be added to p
                # Determine if p is clique or independent set
                # If p is size 1, it's both.
                # We need to track type?
                # Greedy: 
                # Try to add to clique part.
                # Try to add to independent part.
                
                # Let's just check if adding v keeps it a clique OR keeps it an independent set.
                # If p is empty, add.
                # If p is not empty:
                #   Check if all pairs in p + {v} are edges -> Clique
                #   Check if no pairs in p + {v} are edges -> Independent
                #   If either is true, we can add v.
                #   But we need to commit to a type for future additions.
                #   If p is currently a clique, we can only add if it remains a clique.
                #   If p is currently an independent set, we can only add if it remains an independent set.
                #   If p is size 1, it's both. We can choose.
                #   To simplify greedy:
                #   If p is size 1, we can add v and mark it as Clique if connected, Independent if not.
                #   Actually, if p=[u], and we add v:
                #      If u-v is edge, p+{v} is a clique. It is NOT an independent set.
                #      If u-v is not edge, p+{v} is an independent set. It is NOT a clique.
                #      So the type is forced!
                #   Therefore, for size > 1, the type is determined by the content.
                #   For size 1, the type is determined by the next addition.
                #   So we can just check:
                #   Can v be added to p such that p+{v} is a clique?
                #   OR Can v be added to p such that p+{v} is an independent set?
                #   If yes, add v.
                
                trial = p + [v]
                is_c = True
                is_i = True
                for i in range(len(trial)):
                    for j in range(i+1, len(trial)):
                        u, w = trial[i], trial[j]
                        if adj[u][w]:
                            is_i = False
                        else:
                            is_c = False
                if is_c or is_i:
                    p.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts

    # Try exact search
    for K in range(1, n + 1):
        parts = [[] for _ in range(K)]
        types = [None] * K
        # Symmetry breaking: First vertex in order goes to Part 0
        # But `solve` handles placement.
        # We can pre-place the first vertex to reduce branches.
        # Let's just call solve.
        
        # To handle timeout:
        # We'll wrap the call.
        # If it returns True, we have a solution.
        # If it returns False, we check if it was a timeout.
        # We can use a flag.
        
        timeout_flag = [False]
        original_solve = solve
        def solve_with_timeout(pos, parts, types, max_part):
            if time.time() > deadline:
                timeout_flag[0] = True
                return False
            return original_solve(pos, parts, types, max_part)
            
        # Recursion limit? N=14, depth 14. Fine.
        
        if solve_with_timeout(0, parts, types, -1):
            if not timeout_flag[0]:
                final_parts = [p for p in parts if p]
                return final_parts
            else:
                # Timeout
                break
        else:
            if timeout_flag[0]:
                break

    return greedy()
