    def construct():
        import random
        import math
        
        N = math.factorial(12)
        primes = [2]*10 + [3]*5 + [5]*2 + [7] + [11]
        
        best_max_global = float('inf')
        best_solution_global = None
        
        for k in range(1, 13):
            if k == 1:
                if N < best_max_global:
                    best_max_global = N
                    best_solution_global = [N]
                continue
            
            # State: list of lists of primes
            # We'll use a helper to initialize
            def init_state(k):
                bins = [[] for _ in range(k)]
                # Greedy: assign largest primes to emptiest bins
                sorted_primes = sorted(primes, reverse=True)
                prods = [1]*k
                for p in sorted_primes:
                    min_idx = 0
                    min_val = prods[0]
                    for i in range(1, k):
                        if prods[i] < min_val:
                            min_val = prods[i]
                            min_idx = i
                    bins[min_idx].append(p)
                    prods[min_idx] *= p
                return bins, prods

            def calc_prods(bins):
                prods = []
                for b in bins:
                    p = 1
                    for x in b:
                        p *= x
                    prods.append(p)
                return prods

            def is_valid(prods):
                if any(p <= 12 for p in prods):
                    return False
                if len(set(prods)) != len(prods):
                    return False
                return True

            def get_max(prods):
                if not is_valid(prods):
                    return float('inf')
                return max(prods)

            best_k_max = float('inf')
            best_k_prods = None
            
            for restart in range(200):
                bins, prods = init_state(k)
                current_max = get_max(prods)
                
                # SA
                T = 50.0
                cooling = 0.995
                min_T = 0.5
                iterations_per_T = 50
                
                while T > min_T:
                    for _ in range(iterations_per_T):
                        # Pick random move
                        i = random.randint(0, k-1)
                        if not bins[i]:
                            continue
                        j = random.randint(0, k-1)
                        if i == j:
                            continue
                        p_idx = random.randint(0, len(bins[i])-1)
                        p_val = bins[i][p_idx]
                        
                        # Move p_val from i to j
                        bins[i].pop(p_idx)
                        bins[j].append(p_val)
                        
                        old_prods = prods
                        # Update prods
                        prods[i] //= p_val
                        prods[j] *= p_val
                        
                        new_max = get_max(prods)
                        
                        # Accept?
                        delta = new_max - current_max
                        if delta < 0 or random.random() < math.exp(-delta / T):
                            current_max = new_max
                            # Update best
                            if current_max < best_k_max:
                                best_k_max = current_max
                                best_k_prods = list(prods)
                        else:
                            # Revert
                            bins[j].pop()
                            bins[i].insert(p_idx, p_val)
                            prods[i] *= p_val
                            prods[j] //= p_val
                            
                    T *= cooling
                
            if best_k_max < best_max_global:
                best_max_global = best_k_max
                best_solution_global = best_k_prods
        
        if best_solution_global is None:
            return [N]
            
        return best_solution_global
    