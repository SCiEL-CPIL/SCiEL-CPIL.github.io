"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math
    from itertools import combinations
    import time
    import sys
    sys.setrecursionlimit(10000)
    
    n = 128
    
    # Precompute LCM table
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    lcm_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            lcm_table[i][j] = lcm(i+1, j+1)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all forbidden triples where lcm(a,b) == lcm(a,c) == lcm(b,c)
    forbidden_triples = set()
    for a in range(n):
        for b in range(a+1, n):
            lab = lcm_table[a][b]
            for c in range(b+1, n):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    forbidden_triples.add((a, b, c))
    
    # Build conflict structure: for each vertex v, list of pairs (a,b) such that {a,b,v} is forbidden
    v_to_pairs = [[] for _ in range(n)]
    for a, b, c in forbidden_triples:
        for v in [a, b, c]:
            others = [x for x in [a,b,c] if x != v]
            pair = (min(others), max(others))
            v_to_pairs[v].append(pair)
    
    # Vertex degree (number of forbidden triples containing it)
    vertex_degree = [len(v_to_pairs[i]) for i in range(n)]
    
    # Order vertices by degree descending for backtracking
    bt_order = sorted(range(n), key=lambda x: -vertex_degree[x])
    
    # Precompute greedy solution for initial upper bound
    def greedy_solve(order):
        selected = []
        selected_set = set()
        # For fast conflict checking
        for v in order:
            can_add = True
            for a, b in v_to_pairs[v]:
                if a in selected_set and b in selected_set:
                    can_add = False
                    break
            if can_add:
                selected.append(v)
                selected_set.add(v)
        return selected
    
    # Get initial solution via greedy
    best_solution = greedy_solve(bt_order)
    best_size = len(best_solution)
    
    # Backtracking with branch and bound
    start_time = time.time()
    timeout = 50  # seconds
    
    def backtrack(selected, index, selected_set):
        nonlocal best_solution, best_size, start_time, timeout
        
        if time.time() - start_time > timeout:
            return
        
        # Pruning: upper bound
        if len(selected) + (n - index) <= best_size:
            return
        
        if index == n:
            if len(selected) > best_size:
                best_size = len(selected)
                best_solution = list(selected)
            return
        
        v = bt_order[index]
        
        # Check if v can be added
        can_add = True
        for a, b in v_to_pairs[v]:
            if a in selected_set and b in selected_set:
                can_add = False
                break
        
        # Branch 1: include v (if possible)
        if can_add:
            selected.append(v)
            selected_set.add(v)
            backtrack(selected, index + 1, selected_set)
            selected.pop()
            selected_set.remove(v)
        
        # Branch 2: exclude v
        backtrack(selected, index + 1, selected_set)
    
    # Run backtracking
    backtrack([], 0, set())
    
    # Convert back to values (vertex i = value i+1)
    result = sorted([v + 1 for v in best_solution])
    
    return result
# EVOLVE-BLOCK-END
