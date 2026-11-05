def construct():
    import random
    import time
    from math import gcd

    N = 128
    start_time = time.time()
    time_limit = 55

    # 1. Precompute conflicts
    # conflicts_with[x] is a list of pairs (a, b) such that (a, b, x) is a forbidden triple
    conflicts_with = [set() for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            for k in range(j + 1, N + 1):
                g_ij = gcd(i, j)
                g_ik = gcd(i, k)
                g_jk = gcd(j, k)
                if g_ij == g_ik == g_jk:
                    conflicts_with[i].add((j, k))
                    conflicts_with[j].add((i, k))
                    conflicts_with[k].add((i, j))

    # 2. Helper function to check validity
    def is_valid_addition(x, S):
        for a, b in conflicts_with[x]:
            if a in S and b in S:
                return False
        return True

    # 3. Initial Set: Powers of 2
    S = set()
    x = 1
    while x <= N:
        S.add(x)
        x *= 2
        
    best_S = S.copy()
    best_size = len(S)
    current_S = S.copy()

    # 4. Greedy Fill
    candidates = [x for x in range(1, N + 1) if x not in current_S]
    random.shuffle(candidates)
    for x in candidates:
        if is_valid_addition(x, current_S):
            current_S.add(x)
            
    if len(current_S) > best_size:
        best_size = len(current_S)
        best_S = current_S.copy()

    # 5. Local Search
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() - start_time < time_limit:
        if no_improve_count > max_no_improve:
            break
            
        improved = False
        
        # Try Add
        if len(current_S) < N:
            candidates = [x for x in range(1, N + 1) if x not in current_S]
            if candidates:
                x = random.choice(candidates)
                if is_valid_addition(x, current_S):
                    current_S.add(x)
                    improved = True
                    if len(current_S) > best_size:
                        best_size = len(current_S)
                        best_S = current_S.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    continue

        # Try Swap
        if current_S:
            y = random.choice(list(current_S))
            S_minus_y = current_S - {y}
            
            candidates = [x for x in range(1, N + 1) if x not in current_S]
            if candidates:
                random.shuffle(candidates)
                for x in candidates[:20]:
                    if is_valid_addition(x, S_minus_y):
                        current_S = S_minus_y
                        current_S.add(x)
                        improved = True
                        if len(current_S) > best_size:
                            best_size = len(current_S)
                            best_S = current_S.copy()
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                        break
                if not improved:
                    no_improve_count += 1
        else:
            no_improve_count += 1

    return list(best_S)
