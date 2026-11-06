import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    N = 1000
    best_set = []
    best_size = 0
    
    # Helper to check if adding x to set S (with counts) is valid
    # Returns (valid, new_counts) or (False, None)
    # We modify counts in place? No, better to check first.
    # For speed, we might just check validity without copying counts if we revert.
    
    def is_valid_addition(x, chosen, counts):
        # Check x+x
        if counts.get(2*x, 0) >= 2:
            return False
        # Check x+y
        for y in chosen:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        return True

    def get_tightness(x, chosen, counts):
        """Calculate how many sums would reach count 2 if we add x."""
        tight = 0
        if counts.get(2*x, 0) == 1:
            tight += 1
        for y in chosen:
            s = x + y
            if counts.get(s, 0) == 1:
                tight += 1
        return tight

    def run_greedy(noise=0.0, time_limit=None):
        """
        Greedy construction.
        Picks the element that causes the least "tightness" (sums reaching count 2).
        """
        chosen = []
        counts = defaultdict(int)
        candidates = list(range(1, N + 1))
        cand_set = set(candidates)
        
        if time_limit is None:
            time_limit = deadline
        
        while candidates:
            if time.time() > time_limit:
                break
            
            best_x = None
            best_score = float('inf')
            
            # Sample candidates or iterate all?
            # Use a subset for speed when many candidates remain
            if len(candidates) > 200:
                sample_size = min(len(candidates), 200)
                sample = random.sample(candidates, sample_size)
            else:
                sample = candidates
            
            for x in sample:
                # Fast validity check
                if counts.get(2*x, 0) >= 2:
                    continue
                    
                tight = 0
                valid = True
                
                for y in chosen:
                    s = x + y
                    c = counts.get(s, 0)
                    if c >= 2:
                        valid = False
                        break
                    if c == 1:
                        tight += 1
                        
                if not valid:
                    continue
                
                if counts.get(2*x, 0) == 1:
                    tight += 1
                
                score = tight + (random.random() * noise if noise > 0 else 0)
                if score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
                
            # Add best_x
            counts[2*best_x] = counts.get(2*best_x, 0) + 1
            for y in chosen:
                s = best_x + y
                counts[s] = counts.get(s, 0) + 1
            chosen.append(best_x)
            
            # Remove from candidates efficiently
            candidates.remove(best_x)
            cand_set.discard(best_x)
            
        return chosen

    def local_search(initial_set, time_limit):
        """
        Tries to grow the set by adding elements.
        If stuck, tries swaps to find a better configuration that allows growth.
        """
        chosen = list(initial_set)
        counts = defaultdict(int)
        
        # Rebuild counts
        for x in chosen:
            counts[2*x] = counts.get(2*x, 0) + 1
        for i in range(len(chosen)):
            for j in range(i+1, len(chosen)):
                s = chosen[i] + chosen[j]
                counts[s] = counts.get(s, 0) + 1
                
        candidates = set(range(1, N + 1))
        for x in chosen:
            candidates.discard(x)
            
        if not candidates:
            return chosen
            
        consecutive_failures = 0
        
        while time.time() < time_limit:
            # Try to add
            added_any = False
            cand_list = list(candidates)
            random.shuffle(cand_list)
            
            for x in cand_list:
                if time.time() > time_limit:
                    break
                if is_valid_addition(x, chosen, counts):
                    # Add it
                    counts[2*x] = counts.get(2*x, 0) + 1
                    for y in chosen:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                    chosen.append(x)
                    candidates.discard(x)
                    added_any = True
                    consecutive_failures = 0
                    break # Restart loop to pick next best candidate
            
            if added_any:
                continue
                
            # If we couldn't add, try a swap to escape local max
            if len(candidates) == 0:
                break
                
            consecutive_failures += 1
            if consecutive_failures > 50:
                break
                
            x_to_try = random.choice(list(candidates))
            
            # Try to find a y to swap with - limit attempts for speed
            chosen_list = list(chosen)
            random.shuffle(chosen_list)
            max_swap_attempts = min(len(chosen_list), 20)
            
            swap_success = False
            for y in chosen_list[:max_swap_attempts]:
                # Check if removing y and adding x is valid
                # We need to check sums involving x and (chosen - {y})
                
                valid = True
                # Check x+x
                count_2x = counts.get(2*x, 0)
                if y * 2 == 2 * x:
                    count_2x -= 1
                if count_2x >= 2:
                    continue
                
                # Check x+z for all z in chosen except y
                for z in chosen:
                    if z == y:
                        continue
                    s = x + z
                    c = counts.get(s, 0)
                    if z + y == s:
                        c -= 1
                    if c >= 2:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Swap is valid - execute it
                # Remove y
                counts[2*y] -= 1
                if counts[2*y] == 0: del counts[2*y]
                for z in chosen:
                    if z != y:
                        s = y + z
                        counts[s] -= 1
                        if counts[s] == 0: del counts[s]
                        
                # Add x
                counts[2*x] = counts.get(2*x, 0) + 1
                for z in chosen:
                    if z != y:
                        s = x + z
                        counts[s] = counts.get(s, 0) + 1
                            
                # Update sets
                chosen.remove(y)
                chosen.append(x)
                candidates.discard(x)
                candidates.add(y)
                
                swap_success = True
                break
                
            if not swap_success:
                # If we can't even swap, break and restart
                break
                
        return chosen

    # Main loop
    while time.time() < deadline:
        # 1. Generate a good baseline with Greedy
        # Use varying noise to explore different greedy paths
        noise = random.uniform(0, 1.5)
        greedy_time = min(8.0, (deadline - time.time()) * 0.3)
        if greedy_time < 0.5:
            break
        baseline = run_greedy(noise=noise, time_limit=time.time() + greedy_time)
        
        if len(baseline) > best_size:
            best_size = len(baseline)
            best_set = baseline
            
        # 2. Refine with Local Search
        # Allocate remaining time for this round
        remaining = deadline - time.time()
        if remaining < 2:
            break
            
        # Use more time for local search as we get closer to deadline
        ls_time = min(remaining * 0.7, 15.0)
        if ls_time < 1:
            break
            
        refined = local_search(best_set if len(best_set) > len(baseline) else baseline, time.time() + ls_time)
        
        if len(refined) > best_size:
            best_size = len(refined)
            best_set = refined
            
    return best_set
