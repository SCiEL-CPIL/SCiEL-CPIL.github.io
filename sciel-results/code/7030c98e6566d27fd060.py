def construct():
    scale = 5354228880
    weights = {i: scale // i for i in range(1, 25)}
    
    def try_greedy(order):
        S = []
        SS = {0}
        for k in order:
            w_k = weights[k]
            if w_k in SS:
                continue
            
            valid = True
            for j in S:
                if j < k:
                    diff = weights[j] - w_k
                    if diff in SS:
                        valid = False
                        break
                # if j > k, diff is negative, not in SS (SS contains non-negative)
                # if j == k, not in S
            
            if valid:
                S.append(k)
                new_sums = {s + w_k for s in SS}
                SS = SS | new_sums
        
        return sorted(S)
    
    orders = [
        list(range(24, 0, -1)),
        list(range(1, 25)),
        list(range(24, 0, -1, 2)) + list(range(23, 0, -1, 2)), # Even then Odd descending
        list(range(1, 25, 2)) + list(range(24, 0, -1, 2)), # Odd asc then Even desc
    ]
    
    import random
    rnd = random.Random(42)
    for _ in range(10):
        o = list(range(1, 25))
        rnd.shuffle(o)
        orders.append(o)
        
    best = []
    for o in orders:
        sol = try_greedy(o)
        if len(sol) > len(best):
            best = sol
            
    return best
