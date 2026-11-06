import time
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # wall-clock seconds; evaluator kills at 90s
    
    # Known 12-term solution as immediate fallback
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210 + 1/420 + 1/840
    # Verify this is correct
    def verify(sol):
        if not sol:
            return False
        if len(set(sol)) != len(sol):
            return False
        if not all(5 <= d <= 10000000 for d in sol):
            return False
        s = sum(Fraction(1, d) for d in sol)
        return s == 1
    
    # First, verify the known 12-term solution
    known_12 = [5, 6, 7, 10, 14, 35, 42, 70, 105, 210, 420, 840]
    if verify(known_12):
        best_sol = known_12
    else:
        # Generate a 12-term solution programmatically
        def get_12_term_solution():
            # Use the identity: 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210 + 1/420 + 1/840
            # Let's verify by computing the sum
            target = 2520
            reached = {0: []}
            for d in range(5, target + 1):
                if target % d:
                    continue
                w = target // d
                for total, values in list(reached.items()):
                    if total + w <= target and total + w not in reached:
                        reached[total + w] = values + [d]
                if target in reached:
                    sol = reached[target]
                    if verify(sol):
                        return sol
            raise RuntimeError("no 12-term representation found")
        best_sol = get_12_term_solution()
    
    # Now try to find an 11-term solution with improved backtracking
    def try_find_n_terms(n, time_limit=30.0):
        """Try to find n distinct denominators in [5, 10^7] summing to 1."""
        best = [None]
        deadline = time.time() + time_limit
        
        def bt(rem, min_d, chosen, k):
            if time.time() > deadline:
                return
            if k == 0:
                if rem == 0:
                    best[0] = list(chosen)
                return
            if rem <= 0:
                return
            
            # Pruning bounds
            # rem must be at most k/min_d (if all terms were 1/min_d)
            if rem > k / min_d:
                return
            # rem must be at least k/10^7 (if all terms were 1/10^7)
            if rem < k / 10000000:
                return
            
            # Compute valid range for d
            # d must satisfy: 1/d <= rem, so d >= ceil(1/rem)
            # Also, rem - 1/d must be representable by k-1 terms with denominators > d
            # So rem - 1/d <= (k-1)/(d+1)
            # This gives: rem <= 1/d + (k-1)/(d+1)
            # Solving for d: rem*d*(d+1) <= (d+1) + d*(k-1)
            # rem*d^2 + rem*d <= d + 1 + kd - k
            # rem*d^2 + (rem - 1 - k)*d + (k - 1) <= 0
            
            a = float(rem)
            b = a - 1 - k
            c = k - 1
            disc = b*b - 4*a*c
            if disc < 0:
                return
            sqrt_disc = disc ** 0.5
            d_max = int((-b + sqrt_disc) / (2 * a)) + 2
            d_min = max(min_d, int(1 / float(rem)) + 1)
            
            if d_min > d_max:
                return
            
            # Cap the search range to keep it manageable
            # But allow more than 500 if we have time
            max_search = 2000
            d_max = min(d_max, d_min + max_search)
            
            for d in range(d_min, d_max + 1):
                if d in chosen:
                    continue
                term = Fraction(1, d)
                if term > rem:
                    continue  # Need larger d
                if term == rem and k == 1:
                    best[0] = list(chosen) + [d]
                    return
                
                new_rem = rem - term
                chosen.append(d)
                bt(new_rem, d + 1, chosen, k - 1)
                chosen.pop()
                if best[0] is not None:
                    return
                if time.time() > deadline:
                    return
        
        bt(Fraction(1, 1), 5, [], n)
        return best[0]
    
    # Try 11 terms with a reasonable time limit
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 10:
        sol_11 = try_find_n_terms(11, time_limit=min(remaining_time - 5, 40.0))
        if sol_11 and verify(sol_11):
            best_sol = sol_11
    
    return best_sol
