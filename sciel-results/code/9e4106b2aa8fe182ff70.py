"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Generate all 5-smooth numbers <= 128
    smooth = []
    for i in range(1, 129):
        x = i
        while x % 2 == 0:
            x //= 2
        while x % 3 == 0:
            x //= 3
        while x % 5 == 0:
            x //= 5
        if x == 1:
            smooth.append(i)
    
    smooth_set = set(smooth)
    n_smooth = len(smooth)
    
    # Precompute valid 2-term antichains: for each sum, store a valid pair
    two_term = {}
    for i in range(n_smooth):
        a = smooth[i]
        for j in range(i + 1, n_smooth):
            b = smooth[j]
            s = a + b
            if s > 128:
                break
            # Check antichain: since a < b, only need to check if a divides b
            if b % a != 0:
                if s not in two_term:
                    two_term[s] = [a, b]
    
    # Precompute valid 3-term antichains
    three_term = {}
    for i in range(n_smooth):
        a = smooth[i]
        for j in range(i + 1, n_smooth):
            b = smooth[j]
            if a + b > 128:
                break
            # Check a and b are antichain
            if b % a == 0:
                continue
            for k in range(j + 1, n_smooth):
                c = smooth[k]
                s = a + b + c
                if s > 128:
                    break
                # Check all pairs: a|b already checked, need a|c, b|c
                if c % a == 0 or c % b == 0:
                    continue
                if s not in three_term:
                    three_term[s] = [a, b, c]
    
    answers = []
    
    for n in range(1, 129):
        # Check 1 term
        if n in smooth_set:
            answers.append([n])
            continue
            
        # Check 2 terms (precomputed)
        if n in two_term:
            answers.append(two_term[n])
            continue
            
        # Check 3 terms (precomputed)
        if n in three_term:
            answers.append(three_term[n])
            continue
            
        # Check 4+ terms with backtracking
        def find_antichain(n, k, start_idx, current_terms, current_sum):
            if len(current_terms) == k:
                if current_sum == n:
                    return current_terms[:]
                return None
            if current_sum > n:
                return None
            
            rem = n - current_sum
            terms_left = k - len(current_terms)
            if rem < terms_left:
                return None
            # Upper bound: largest possible sum with remaining terms
            if rem > sum(smooth[-terms_left:]):
                return None
            
            for idx in range(start_idx, n_smooth):
                term = smooth[idx]
                if term > rem:
                    break
                # Check compatibility
                compatible = True
                for t in current_terms:
                    if term % t == 0 or t % term == 0:
                        compatible = False
                        break
                if compatible:
                    res = find_antichain(n, k, idx + 1, current_terms + [term], current_sum + term)
                    if res:
                        return res
            return None

        for k in range(4, 11):
            res = find_antichain(n, k, 0, [], 0)
            if res:
                answers.append(res)
                break

    return answers

# EVOLVE-BLOCK-END
