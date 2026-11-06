import random

def construct():
    n = 192
    best_set = []
    
    def can_add(x, S, nS):
        # Check new x: no 3 in S sum to multiple of x
        if nS >= 3:
            for i in range(nS):
                si = S[i]
                for j in range(i + 1, nS):
                    sij = si + S[j]
                    for k in range(j + 1, nS):
                        if (sij + S[k]) % x == 0:
                            return False
        
        # Check existing a: no pair b,c in S (b!=a, c!=a) such that x+b+c is multiple of a
        for idx_a in range(nS):
            a = S[idx_a]
            for i in range(nS):
                if i == idx_a:
                    continue
                si = S[i]
                for j in range(i + 1, nS):
                    if j == idx_a:
                        continue
                    if (x + si + S[j]) % a == 0:
                        return False
        return True

    # Orders to try
    orders = []
    orders.append(list(range(n, 0, -1)))
    orders.append(list(range(1, n + 1)))
    orders.append(list(range(n // 2, n + 1)))
    orders.append(list(range(n // 3, n + 1)))
    
    # Random permutations
    random.seed(42) # For reproducibility in case of flakiness? No, just use default
    for _ in range(15):
        perm = list(range(1, n + 1))
        random.shuffle(perm)
        orders.append(perm)

    for order in orders:
        S = []
        nS = 0
        for x in order:
            if can_add(x, S, nS):
                S.append(x)
                nS += 1
                if nS > len(best_set):
                    best_set = S[:]
                    if nS == 48:
                        return best_set
    return best_set
