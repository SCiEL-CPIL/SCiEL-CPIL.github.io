    import time
    import sys
    from math import gcd
    
    def construct():
        N = 512
        start_time = time.time()
        deadline = start_time + 55 # 5s margin
        
        # 1. Initial solution: Divisors of 360
        def get_divisors(n):
            divs = []
            for i in range(1, int(n**0.5) + 1):
                if n % i == 0:
                    divs.append(i)
                    if i != n // i:
                        divs.append(n // i)
            return divs
        
        init_clique = get_divisors(360)
        best_clique_bits = 0
        for v in init_clique:
            best_clique_bits |= (1 << v)
        best_size = len(init_clique)
        
        # 2. Build Graph
        adj = [0] * (N + 1)
        for u in range(1, N + 1):
            for v in range(u + 1, N + 1):
                g = gcd(u, v)
                if u * v <= N * g:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # 3. Max Clique Solver
        # Using non-recursive or iterative? Recursion is easier.
        # We need to update best_clique_bits globally.
        
        # To avoid global variable issues in nested function, use a list or nonlocal
        global_best = [best_clique_bits, best_size]
        
        def bron_kerbosch(R, P, X):
            nonlocal global_best
            
            # Time check
            if time.time() > deadline:
                return
                
            # Pruning
            if len(R) + P.bit_count() <= global_best[1]:
                return
                
            if P == 0:
                if len(R) > global_best[1]:
                    global_best[0] = R
                    global_best[1] = len(R)
                return
            
            # Pivot
            # Choose u from P U X
            candidates = P | X
            if candidates == 0:
                return
                
            max_deg = -1
            pivot = -1
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (P & adj[v]).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pivot = v
                temp -= lsb
            
            if pivot == -1:
                return

            # Expand on P \ adj[pivot]
            candidates_to_expand = P & ~adj[pivot]
            
            temp = candidates_to_expand
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp -= lsb
                
                # Check time
                if time.time() > deadline:
                    return

                # Recurse
                new_R = R | (1 << v)
                new_P = P & adj[v]
                new_X = X & adj[v]
                
                bron_kerbosch(new_R, new_P, new_X)
                
                # Move v
                P &= ~(1 << v)
                X |= (1 << v)
                
                # Prune
                if len(R) + P.bit_count() <= global_best[1]:
                    break

        # Initialize P
        P_init = 0
        for i in range(1, N + 1):
            P_init |= (1 << i)
            
        bron_kerbosch(0, P_init, 0)
        
        # Extract result
        result_bits = global_best[0]
        result = []
        temp = result_bits
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            result.append(v)
            temp -= lsb
            
        if not result:
            return init_clique
            
        return result
    