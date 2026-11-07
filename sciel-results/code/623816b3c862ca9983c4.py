import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin
    
    # 1. Identify primes in [2, 40]
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [p for p in range(2, 41) if is_prime(p)]
    
    result = []
    
    for p in primes:
        # Get available powers p^e <= 40
        powers = []
        e = 1
        val = p
        while val <= 40:
            powers.append((e, val))
            e += 1
            val *= p
            
        if not powers:
            continue
            
        # We want to select a subset of these powers such that
        # the exponents have distinct subset sums.
        # Max size of powers is small (<=5).
        # Try all subsets of indices, largest first.
        
        n_powers = len(powers)
        best_subset = []
        
        # Generate all subsets of range(n_powers)
        # Since n_powers <= 5, 2^5 = 32 subsets.
        for mask in range(1, 1 << n_powers):
            # Check time
            if time.time() > deadline:
                break
                
            subset_indices = [i for i in range(n_powers) if (mask >> i) & 1]
            
            # If this subset is smaller than current best, skip?
            # We want max size.
            if len(subset_indices) <= len(best_subset):
                continue
                
            # Check if exponents in subset_indices have distinct subset sums
            exponents = [powers[i][0] for i in subset_indices]
            
            # Generate all subset sums of exponents
            sums = set()
            valid = True
            # Iterate all subsets of exponents
            for sub_mask in range(1 << len(exponents)):
                s = 0
                for j in range(len(exponents)):
                    if (sub_mask >> j) & 1:
                        s += exponents[j]
                if s in sums:
                    valid = False
                    break
                sums.add(s)
                
            if valid:
                best_subset = subset_indices
                # Since we iterate masks, we don't guarantee largest first.
                # But we can just keep the best found.
        
        # Add the numbers corresponding to best_subset
        for idx in best_subset:
            result.append(powers[idx][1])
            
    return result
