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
    
    # 1. Greedy Initialization
    S = []
    in_S = [False] * total_sets
    
    # Start with the empty set (0) as it's comparable to everything
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
        # Find y with max G[y]
        for y in range(total_sets):
            if not in_S[y] and G[y] > best_g:
                best_g = G[y]
                best_y = y
        
        if best_y == -1:
            break
            
        S.append(best_y)
        in_S[best_y] = True
        
        # Update G for all other candidates
        # G[u] += 1 if u is comparable to best_y
        for u in range(total_sets):
            if not in_S[u]:
                if is_comparable(u, best_y):
                    G[u] += 1
    
    # 2. Prepare for Local Search
    # C[x] for x in S: number of comparable sets in S (excluding self)
    C = [0] * total_sets
    for i, x in enumerate(S):
        count = 0
        for j, y in enumerate(S):
            if i != j and is_comparable(x, y):
                count += 1
        C[x] = count
        
    # G[y] for y not in S is already maintained from greedy phase? 
    # Let's recalc to be safe and consistent with C.
    for y in range(total_sets):
        if not in_S[y]:
            count = 0
            for x in S:
                if is_comparable(x, y):
                    count += 1
            G[y] = count
            
    def calc_score():
        score = 0
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if is_comparable(S[i], S[j]):
                    score += 1
        return score
    
    current_score = calc_score()
    best_score = current_score
    best_S = S.copy()
    
    def update_C_G(x, y):
        # Remove x, Add y
        # 1. Update C for remaining nodes in S (excluding x)
        # C[u] changes by +1 if u comp y, -1 if u comp x
        for u in S:
            if u == x: continue
            if is_comparable(u, y):
                C[u] += 1
            if is_comparable(u, x):
                C[u] -= 1
        
        # 2. Update G for nodes not in S
        # G[v] changes by +1 if v comp y, -1 if v comp x
        # Note: x is now not in S, y is now in S.
        for v in range(total_sets):
            if in_S[v]: continue # v is in S, we don't track G for them in this way
            if is_comparable(v, y):
                G[v] += 1
            if is_comparable(v, x):
                G[v] -= 1
                
        # 3. Set C[y] and G[x]
        # C[y] is number of neighbors of y in new S (old S - {x} + {y})
        # Neighbors in old S - {x} are those u in S, u!=x, comp(y,u)
        # Plus 0 for self? No, C counts neighbors.
        # We already updated C[u] for u in S. 
        # C[y] needs to be calculated.
        c_y = 0
        for u in S:
            if u == x: continue
            if is_comparable(u, y):
                c_y += 1
        C[y] = c_y
        
        # G[x] is number of neighbors of x in new S (old S - {x} + {y})
        g_x = 0
        for u in S:
            if u == y: continue
            if is_comparable(u, x):
                g_x += 1
        G[x] = g_x
        
    def local_search():
        nonlocal current_score, best_score, best_S
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_x = -1
            best_y = -1
            
            # Find best swap
            # x in S, y not in S
            # Delta = G[y] - C[x] - (1 if comp(x,y) else 0)
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

    def simulated_annealing():
        nonlocal current_score, best_score, best_S
        temp = 10.0
        cooling = 0.999
        min_temp = 0.01
        last_improvement = time.time()
        
        while time.time() < deadline and temp > min_temp:
            # Pick random x in S
            x = random.choice(S)
            cx = C[x]
            
            # Pick random y not in S
            y = random.randint(0, total_sets - 1)
            while in_S[y]:
                y = random.randint(0, total_sets - 1)
            gy = G[y]
            
            comp_xy = 1 if is_comparable(x, y) else 0
            delta = gy - comp_xy - cx
            
            if delta > 0:
                # Accept
                S.remove(x)
                in_S[x] = False
                S.append(y)
                in_S[y] = True
                
                current_score += delta
                update_C_G(x, y)
                last_improvement = time.time()
                
                if current_score > best_score:
                    best_score = current_score
                    best_S = S.copy()
            else:
                # Probabilistic acceptance
                prob = pow(2.718281828, delta / temp)
                if random.random() < prob:
                    S.remove(x)
                    in_S[x] = False
                    S.append(y)
                    in_S[y] = True
                    
                    current_score += delta
                    update_C_G(x, y)
            
            temp *= cooling
            if time.time() - last_improvement > 10.0:
                # Restart with higher temp if stuck
                temp = 10.0
                last_improvement = time.time()

    # Initial Local Search
    local_search()
    
    # ILS Loop with Simulated Annealing Perturbation
    while time.time() < deadline:
        # Perturb using SA for a short duration
        sa_start = time.time()
        simulated_annealing()
        
        # Refine with Local Search
        local_search()

    # Convert best_S to lists
    result = []
    for mask in best_S:
        s = []
        for i in range(n):
            if mask & (1 << i):
                s.append(i)
        result.append(s)
        
    return result
