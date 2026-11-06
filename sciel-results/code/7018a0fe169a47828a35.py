def construct():
    import numpy as np
    from math import gcd
    from scipy.optimize import milp, LinearConstraint, Bounds
    
    N = 128
    nums = list(range(1, N+1))
    
    # Precompute GCD matrix
    G = np.zeros((N+1, N+1), dtype=int)
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            G[i, j] = g
            G[j, i] = g

    # Find all bad triples
    # A triple (a, b, c) with a < b < c is bad if G[a,b] == G[a,c] == G[b,c]
    bad_triples = []
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            gab = G[a, b]
            for c in range(b+1, N+1):
                if G[a, c] == gab and G[b, c] == gab:
                    bad_triples.append((a, b, c))
    
    # If no bad triples, we can take all numbers
    if not bad_triples:
        return nums

    # Setup ILP
    # Variables x_1 ... x_128
    # Minimize -sum(x_i)
    c_obj = -np.ones(N)
    
    # Constraints: x_a + x_b + x_c <= 2 for each bad triple
    # We need to build the constraint matrix A_ub such that A_ub @ x <= b_ub
    # Number of constraints = len(bad_triples)
    # Number of variables = N
    m = len(bad_triples)
    A_ub = np.zeros((m, N))
    b_ub = np.full(m, 2)
    
    for i, (a, b, c) in enumerate(bad_triples):
        A_ub[i, a-1] = 1
        A_ub[i, b-1] = 1
        A_ub[i, c-1] = 1
        
    constraints = LinearConstraint(A_ub, ub=b_ub)
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(N)
    
    res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.status != 0:
        # Fallback: return a simple valid set, e.g. primes
        def is_prime(n):
            if n < 2: return False
            for i in range(2, int(np.sqrt(n))+1):
                if n % i == 0: return False
            return True
        return [x for x in nums if is_prime(x)]
        
    # Extract solution
    # res.x is array of 1s and 0s
    selected = [i+1 for i in range(N) if res.x[i] > 0.5]
    
    # Safety check
    def is_valid(S_list):
        n = len(S_list)
        S_set = set(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i+1, n):
                b = S_list[j]
                g = G[a, b]
                for k in range(j+1, n):
                    c = S_list[k]
                    if G[a, c] == g and G[b, c] == g:
                        return False
        return True
        
    if not is_valid(selected):
        # Should not happen, but if it does, repair or fallback
        # Simple fallback: greedy removal
        S = list(selected)
        changed = True
        while changed:
            changed = False
            n = len(S)
            for i in range(n):
                a = S[i]
                for j in range(i+1, n):
                    b = S[j]
                    g = G[a, b]
                    for k in range(j+1, n):
                        c = S[k]
                        if G[a, c] == g and G[b, c] == g:
                            S.remove(c)
                            changed = True
                            break
                    if changed: break
                if changed: break
        if not S:
             S = [2]
        return S
        
    return selected
