import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 12
    m = 40
    total_sets = 1 << n
    
    # Helper function to check comparability
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b
    
    def calc_score(S):
        score = 0
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if is_comparable(S[i], S[j]):
                    score += 1
        return score

    def greedy_init(randomize_ties=False):
        S = []
        in_S = [False] * total_sets
        
        # Start with the empty set
        S.append(0)
        in_S[0] = True
        
        # G[y] = number of comparable sets in S
        G = [0] * total_sets
        for y in range(total_sets):
            if y != 0:
                G[y] = 1 # 0 is comparable to all
        
        for _ in range(m - 1):
            best_y = -1
            best_g = -1
            candidates = []
            # Find y with max G[y]
            for y in range(total_sets):
                if not in_S[y]:
                    if G[y] > best_g:
                        best_g = G[y]
                        best_y = y
                        candidates = [y]
                    elif G[y] == best_g:
                        candidates.append(y)
            
            if best_y == -1:
                break
            
            if randomize_ties and len(candidates) > 1:
                best_y = random.choice(candidates)
            
            S.append(best_y)
            in_S[best_y] = True
            
            # Update G for all other candidates
            for u in range(total_sets):
                if not in_S[u]:
                    if is_comparable(u, best_y):
                        G[u] += 1
        return S

    def local_search(S):
        # Prepare C and G
        C = [0] * total_sets
        in_S = [False] * total_sets
        for x in S:
            in_S[x] = True
        
        # Calculate C for x in S
        for i, x in enumerate(S):
            count = 0
            for j, y in enumerate(S):
                if i != j and is_comparable(x, y):
                    count += 1
            C[x] = count
            
        # Calculate G for y not in S
        G = [0] * total_sets
        for y in range(total_sets):
            if not in_S[y]:
                count = 0
                for x in S:
                    if is_comparable(x, y):
                        count += 1
                G[y] = count
                
        current_score = calc_score(S)
        best_score = current_score
        best_S = S.copy()
        
        def update_C_G(x, y):
            # Remove x, Add y
            # Update C for remaining nodes in S (excluding x)
            for u in S:
                if u == x: continue
                if is_comparable(u, y):
                    C[u] += 1
                if is_comparable(u, x):
                    C[u] -= 1
                    
            # Update G for nodes not in S
            # Note: x is now not in S, y is now in S.
            for v in range(total_sets):
                if in_S[v]: continue # v is in S
                if is_comparable(v, y):
                    G[v] += 1
                if is_comparable(v, x):
                    G[v] -= 1
                
            # Set C[y] and G[x]
            c_y = 0
            for u in S:
                if u == x: continue
                if is_comparable(u, y):
                    c_y += 1
            C[y] = c_y
            
            g_x = 0
            for u in S:
                if u == y: continue
                if is_comparable(u, x):
                    g_x += 1
            G[x] = g_x
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_x = -1
            best_y = -1
            
            # Find best swap
            for x in S:
                cx = C[x]
                for y in range(total_sets):
                    if in_S[y]:
                        continue
                    gy = G[y]
                    comp_xy = 1 if is_comparable(x, y) else 0
                    delta = gy - comp_xy - cx
                    if delta > best_delta:
                        best_delta = delta
                        best_x = x
                        best_y = y
            
            if best_delta > 0:
                # Apply swap
                S.remove(best_x)
                in_S[best_x] = False
                S.append(best_y)
                in_S[best_y] = True
                
                current_score += best_delta
                update_C_G(best_x, best_y)
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_S = S.copy()
            else:
                improved = False
                
        return best_S, best_score

    # 1. Generate multiple diverse initializations
    best_overall_S = []
    best_overall_score = 0
    
    # Try 5 randomized greedy starts
    for i in range(5):
        if time.time() > deadline - 10: break
        S_init = greedy_init(randomize_ties=True)
        S_ls, score_ls = local_search(S_init)
        if score_ls > best_overall_score:
            best_overall_score = score_ls
            best_overall_S = S_ls
    
    # 2. Run ILS on the best solution found
    S = best_overall_S
    current_score = best_overall_score
    best_score = current_score
    best_S = S.copy()
    
    # Pre-compute in_S for the current best
    in_S = [False] * total_sets
    for x in S:
        in_S[x] = True
        
    # We need to re-calc C and G for the ILS loop if we want to use the fast update, 
    # but the local_search function above re-calcs them from scratch.
    # To avoid duplication, let's implement a tighter ILS loop that re-uses the logic.
    
    # Actually, let's just call local_search again in a loop with perturbations?
    # No, local_search re-calculates C and G from scratch every time it's called, which is O(40*40 + 4096*40) ~ 160k ops.
    # This is fast enough.
    
    while time.time() < deadline - 2:
        # Perturb
        num_swaps = random.randint(1, 3)
        S_perturbed = S.copy()
        in_S_perturbed = [False] * total_sets
        for x in S_perturbed:
            in_S_perturbed[x] = True
            
        for _ in range(num_swaps):
            x = random.choice(S_perturbed)
            # Find random y not in S
            while True:
                y = random.randint(0, total_sets - 1)
                if not in_S_perturbed[y]:
                    break
            S_perturbed.remove(x)
            in_S_perturbed[x] = False
            S_perturbed.append(y)
            in_S_perturbed[y] = True
            
        # Local Search on perturbed
        S_ls, score_ls = local_search(S_perturbed)
        
        if score_ls > best_score:
            best_score = score_ls
            best_S = S_ls
            S = S_ls # Update current best for next perturbation base? 
            # Or should we perturb from the *original* best?
            # Usually ILS perturbs from the current local optimum.
            # If we update S, we are doing "Iterated Local Search" properly.
            
    # Convert best_S to lists
    result = []
    for mask in best_S:
        s = []
        for i in range(n):
            if mask & (1 << i):
                s.append(i)
        result.append(s)
        
    return result
