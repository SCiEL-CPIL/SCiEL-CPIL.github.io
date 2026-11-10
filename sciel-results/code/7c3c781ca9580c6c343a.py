import time
import random
import sys

def construct():
    def get_cost(seq):
        """
        Returns the number of duplicate consecutive block sums.
        seq is a strictly increasing list of integers.
        """
        n = len(seq)
        if n == 0:
            return 0
        sums = set()
        duplicates = 0
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    duplicates += 1
                else:
                    sums.add(s)
        return duplicates

    def simulated_annealing(target_len, time_limit):
        """
        Attempts to find a strictly increasing sequence of length target_len
        with distinct consecutive block sums.
        """
        if target_len > 256:
            return None
            
        # Initialize with a random subset
        # We pick target_len random distinct integers from 1..256
        current_seq = sorted(random.sample(range(1, 257), target_len))
        current_cost = get_cost(current_seq)
        
        if current_cost == 0:
            return current_seq

        best_seq = current_seq
        best_cost = current_cost
        
        temp = 10.0
        cooling = 0.999
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if current_cost == 0:
                return current_seq
                
            # Pick a random index to remove
            idx = random.randint(0, target_len - 1)
            old_val = current_seq[idx]
            
            # Pick a random new value not in current_seq
            # To be efficient, we can just pick a random number from 1..256 
            # and retry if it's already in the set, or use a set for membership.
            # For small target_len, a set is good.
            current_set = set(current_seq)
            
            new_val = None
            attempts = 0
            while new_val is None or new_val in current_set:
                new_val = random.randint(1, 256)
                attempts += 1
                if attempts > 100: # Fallback, though unlikely for small L
                    break
            
            if new_val in current_set:
                continue

            # Create new sequence
            # Remove old_val, add new_val, sort
            new_seq = current_seq[:idx] + current_seq[idx+1:] + [new_val]
            new_seq.sort()
            
            new_cost = get_cost(new_seq)
            
            # Acceptance criterion
            delta = new_cost - current_cost
            if delta < 0 or random.random() < pow(2.718281828, -delta / temp):
                current_seq = new_seq
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_seq = current_seq
                    if best_cost == 0:
                        return best_seq
            
            temp *= cooling
            if temp < 0.01:
                temp = 0.01 # Floor temperature to allow some moves
                
        return None

    # 1. Get a baseline greedy sequence to know the lower bound
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
            
    greedy_len = len(values)
    
    # 2. Try to improve using Simulated Annealing
    # We try lengths from greedy_len + 1 up to a limit, or just start high?
    # It's safer to try greedy_len + 1, then + 2, etc.
    # But SA is stochastic. If it fails for L, it might succeed for L-1.
    # However, if L is too large, it's hard.
    # Let's estimate max length.
    # With 256 values, and sums up to 256*256/2 ~ 32k.
    # Number of sums is L(L+1)/2.
    # Distinct sums required.
    # Max L is roughly sqrt(2 * 32000) ~ 250? No, sums are bounded by 256*L.
    # Actually, the sums can be large.
    # Let's just try from greedy_len + 1 up to greedy_len + 5 or 10.
    
    best_found = values
    best_len = greedy_len
    
    # We have ~50 seconds.
    # Let's allocate time per length attempt.
    # Try L = greedy_len + 1
    # If found, try L = greedy_len + 2
    # ...
    
    time_per_attempt = 5.0 # seconds
    
    for L in range(greedy_len + 1, greedy_len + 10):
        if time.time() > time.time() + 50: # Global limit check
            break
            
        # Allocate remaining time or fixed slice?
        # Fixed slice is safer to ensure we try multiple lengths.
        # But if we find a solution for L, we can try L+1.
        
        start_t = time.time()
        # Limit this specific SA run to time_per_attempt
        res = simulated_annealing(L, time_per_attempt)
        
        if res:
            best_found = res
            best_len = L
            # Continue to try for longer sequences
        else:
            # If we failed for L, it's likely we failed for L+1 as well?
            # Not necessarily, but usually larger L is harder.
            # We can break if we fail for L, assuming L-1 was the max.
            # But maybe L was just hard.
            # Let's break to be safe and return the best found so far.
            # Or try one more time?
            # Given the stochastic nature, if it fails for L, L+1 is very unlikely.
            break

    return best_found
