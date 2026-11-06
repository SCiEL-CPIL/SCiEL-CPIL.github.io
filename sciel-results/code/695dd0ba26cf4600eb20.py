import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0
    
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i:limit+1:i] = [False] * len(sieve[i*i:limit+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    MAX_L = 100000
    
    initial_counts = np.zeros(MAX_L + 1, dtype=np.int8)
    initial_used = {}
    initial_budget = 0.0
    
    beam = [(0, initial_used, initial_counts, initial_budget)]
    
    best_L = 0
    best_used = {}
    
    BEAM_WIDTH = 10 # Reduced to allow more steps
    
    while time.time() - start_time < deadline:
        next_beam = []
        
        for L, used, counts, budget in beam:
            # Find current L if not provided or if we need to verify
            # We store L in beam, but it might be stale if counts changed? No, we update L in moves.
            
            if L > best_L:
                best_L = L
                best_used = used
                
            if L >= MAX_L:
                continue
                
            # Generate moves
            # 1. Add new prime
            remaining_budget = 1.0 - budget
            for p in primes:
                if p in used:
                    continue
                cost = 1.0 / p
                if cost > remaining_budget + 1e-9:
                    continue
                
                r = L % p
                new_counts = counts.copy()
                new_counts[r::p] += 1
                
                # Check if L improved
                if new_counts[L] == 0:
                    # L didn't even cover L? Impossible if r = L % p.
                    # Wait, if L=0, r=0. new_counts[0] += 1.
                    # We care about 1..MAX_L.
                    # If L=0, we want to cover 1.
                    # My L definition: "largest L such that 1..L covered".
                    # So if 1 is not covered, L=0.
                    # If 1 is covered, L>=1.
                    # In my loop, L is the current coverage length.
                    # If L=0, we want to cover 1.
                    # r = 0 % p = 0.
                    # new_counts[0::p] += 1.
                    # This covers 0, p, 2p...
                    # It does NOT cover 1 (unless p=1, which is not prime).
                    # So this move is invalid for increasing L from 0 to 1?
                    # No, we need to cover 1.
                    # So r must be 1 % p = 1.
                    
                    # Correction:
                    # The "target" to cover is L+1.
                    # Let target = L + 1.
                    # r = target % p.
                    
                    pass # Logic fix below

            # Logic Fix:
            # Current state covers 1..L.
            # We want to cover L+1.
            # So target = L + 1.
            # r = target % p.
            
            # Let's restart the move generation logic with target = L+1.
            
            target = L + 1
            if target > MAX_L:
                continue
                
            # Add moves
            for p in primes:
                if p in used:
                    continue
                cost = 1.0 / p
                if cost > remaining_budget + 1e-9:
                    continue
                
                r = target % p
                new_counts = counts.copy()
                new_counts[r::p] += 1
                
                # Find new L
                # We know 1..L are covered.
                # We added coverage to r, r+p, ...
                # target is covered.
                # Check if L+1, L+2, ... are covered.
                new_L = L
                # Scan forward
                # We can use numpy to find first zero from target onwards
                # But we only care if it's contiguous from 1.
                # Since 1..L were covered, and we only added coverage, 1..L are still covered.
                # So we just need to find the first zero >= target.
                unc = np.nonzero(new_counts[target:] == 0)[0]
                if len(unc) == 0:
                    new_L = MAX_L
                else:
                    new_L = target + unc[0] - 1
                    
                if new_L > L:
                    next_beam.append((new_L, {**used, p: r}, new_counts, budget + cost))
                elif new_L == L:
                    # Neutral move, might be useful?
                    # Usually we want improvement.
                    pass

            # Change moves
            for p, old_r in used.items():
                r = target % p
                if r == old_r:
                    continue
                new_counts = counts.copy()
                new_counts[r::p] += 1
                new_counts[old_r::p] -= 1
                
                # Check if 1..L are still covered
                # We removed old_r::p. If any n in 1..L was ONLY covered by p at old_r, it breaks.
                # We can check min(new_counts[1:L+1])
                if np.min(new_counts[1:L+1]) == 0:
                    continue # Breaks coverage
                
                # Find new L
                unc = np.nonzero(new_counts[target:] == 0)[0]
                if len(unc) == 0:
                    new_L = MAX_L
                else:
                    new_L = target + unc[0] - 1
                    
                if new_L >= L:
                    new_used = {**used}
                    new_used[p] = r
                    next_beam.append((new_L, new_used, new_counts, budget))
                    
        if not next_beam:
            break
            
        next_beam.sort(key=lambda x: x[0], reverse=True)
        
        seen = set()
        unique_beam = []
        for state in next_beam:
            L, used, counts, budget = state
            key = tuple(sorted(used.items()))
            if key in seen:
                continue
            seen.add(key)
            unique_beam.append(state)
            
        beam = unique_beam[:BEAM_WIDTH]
        
    return [[p, r] for p, r in best_used.items()]
