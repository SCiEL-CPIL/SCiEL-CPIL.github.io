def construct():
    import time
    import random
    from math import gcd
    
    deadline = time.time() + 85
    
    n = 128
    gcd_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, n+1):
            gcd_table[i][j] = gcd(i, j)
    
    def can_add(x, S):
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                a, b = S[i], S[j]
                if gcd_table[a][b] == gcd_table[a][x] == gcd_table[b][x]:
                    return False
        return True
    
    def hill_climb(S):
        S = list(S)
        S_set = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, 129):
                if x not in S_set and can_add(x, S):
                    S.append(x)
                    S_set.add(x)
                    improved = True
            if improved:
                S.sort()
        return S
    
    all_elements = list(range(1, 129))
    orders = [
        all_elements,
        list(reversed(all_elements)),
        [x for x in all_elements if x % 2 == 0],
        [x for x in all_elements if x % 2 == 1],
        sorted(all_elements, key=lambda x: -x.bit_count()),
        sorted(all_elements, key=lambda x: x.bit_count()),
        random.sample(all_elements, 128),
        random.sample(all_elements, 128),
    ]
    
    best_S = []
    for order in orders:
        if time.time() >= deadline:
            break
        S = hill_climb([x for x in order if can_add(x, [])])  # just greedy
        # Actually let me do proper greedy
        S = []
        S_set = set()
        for x in order:
            if x not in S_set and can_add(x, S):
                S.append(x)
                S_set.add(x)
        S = hill_climb(S)
        if len(S) > len(best_S):
            best_S = S
    
    rng = random.Random(42)
    for _ in range(200):
        if time.time() >= deadline:
            break
        if best_S:
            kick_size = min(len(best_S), rng.randint(2, 8))
            to_remove = rng.sample(best_S, kick_size)
            S = [x for x in best_S if x not in to_remove]
            S = hill_climb(S)
            if len(S) > len(best_S):
                best_S = S
        else:
            order = random.sample(all_elements, 128)
            S = []
            S_set = set()
            for x in order:
                if x not in S_set and can_add(x, S):
                    S.append(x)
                    S_set.add(x)
            S = hill_climb(S)
            if len(S) > len(best_S):
                best_S = S
    
    return best_S
