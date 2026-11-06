def construct():
    import math
    import random
    
    n = 256
    max_sum = 2 * n
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
        
    # Precompute conflicts for all 1..256
    # conflicts[a] contains b > a such that a+b is square
    # We need symmetric access for set intersection
    conflicts = {a: set() for a in range(1, n + 1)}
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a + b) in squares:
                conflicts[a].add(b)
                conflicts[b].add(a)
        
    # Diagonal check: 2a is square?
    # If 2a is square, a cannot be in the set.
    valid_nodes = [a for a in range(1, n + 1) if 2 * a not in squares]
    
    def can_add(S, v):
        return len(conflicts[v] & S) == 0

    def get_initial_solution():
        S = set()
        nodes = valid_nodes[:]
        random.shuffle(nodes)
        for v in nodes:
            if can_add(S, v):
                S.add(v)
        return S

    def local_search(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            # Try adding any valid node
            for v in valid_nodes:
                if v not in S and can_add(S, v):
                    S.add(v)
                    improved = True
        return S

    def simulated_annealing(initial_S, max_iters=3000, T_start=1.0, T_end=0.01, alpha=0.9995):
        S = set(initial_S)
        current_score = len(S)
        best_S = set(S)
        best_score = current_score
        
        T = T_start
        for i in range(max_iters):
            r = random.random()
            S_new = set(S)
            delta = 0
            valid_move = False
            
            if r < 0.3:
                # Add
                candidates = [v for v in valid_nodes if v not in S]
                if candidates:
                    v = random.choice(candidates)
                    if can_add(S, v):
                        S_new.add(v)
                        delta = 1
                        valid_move = True
            elif r < 0.6:
                # Remove
                if S:
                    u = random.choice(list(S))
                    S_new.discard(u)
                    delta = -1
                    valid_move = True
            else:
                # Swap
                if S:
                    u = random.choice(list(S))
                    S_temp = S - {u}
                    candidates = [v for v in valid_nodes if v not in S_temp]
                    if candidates:
                        v = random.choice(candidates)
                        if can_add(S_temp, v):
                            S_new = S_temp | {v}
                            delta = 0
                            valid_move = True
                        else:
                            # Fallback: just remove u
                            S_new = S_temp
                            delta = -1
                            valid_move = True
                    else:
                        S_new = S_temp
                        delta = -1
                        valid_move = True

            if not valid_move:
                continue
            
            if delta > 0:
                S = S_new
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
            else:
                if delta == 0:
                    S = S_new
                else:
                    p = math.exp(delta / T)
                    if random.random() < p:
                        S = S_new
                        current_score += delta
            
            T = T * alpha
            if T < T_end:
                break
        
        return best_S

    random.seed(42)
    best_global = set()
    best_global_score = 0
    
    for restart in range(50):
        init = get_initial_solution()
        init = local_search(init)
        
        res = simulated_annealing(init, max_iters=2000)
        
        if len(res) > best_global_score:
            best_global_score = len(res)
            best_global = res
            
        res2 = local_search(res)
        if len(res2) > best_global_score:
            best_global_score = len(res2)
            best_global = res2
            
    result = sorted(best_global)
    
    for i, a in enumerate(result):
        for b in result[i:]:
            s = a + b
            root = math.isqrt(s)
            if root * root == s:
                raise ValueError(f"Conflict: {a} + {b} = {s}")
                
    return result
