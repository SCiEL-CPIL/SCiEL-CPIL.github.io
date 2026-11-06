import time
import random
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    N = 1000
    best_set = []
    best_size = 0
    
    # Precompute FFT size for convolution
    # Max sum is 1000+1000 = 2000. We need array size at least 2001.
    # Next power of 2 >= 2001 is 4096.
    FFT_SIZE = 4096
    FFT_SIZE_LOG2 = 12
    
    def get_rep_counts_fft(chosen):
        """
        Compute representation counts for all sums using FFT.
        Returns a numpy array where rep_counts[s] is the number of pairs (a,b) with a<=b such that a+b=s.
        """
        if not chosen:
            return np.zeros(FFT_SIZE, dtype=np.float64)
        
        # Create boolean vector
        vec = np.zeros(FFT_SIZE, dtype=np.float64)
        for x in chosen:
            vec[x] = 1.0
            
        # FFT
        f = np.fft.rfft(vec, n=FFT_SIZE)
        f_sq = f * f
        
        # Inverse FFT
        conv = np.fft.irfft(f_sq, n=FFT_SIZE)
        
        # conv[s] counts pairs (a,b) such that a+b=s.
        # This includes (a,b) and (b,a) for a!=b, and (a,a) once.
        # We want a <= b.
        # For s, let k = number of pairs (a,b) with a<b and a+b=s.
        # Let d = 1 if s/2 is integer and s/2 in chosen, else 0.
        # Total unordered pairs = k + d.
        # conv[s] = 2*k + d.
        # So k + d = (conv[s] + d) / 2.
        
        rep_counts = np.zeros(FFT_SIZE, dtype=np.int32)
        
        # We only care about s up to 2000
        for s in range(2, min(FFT_SIZE, 2001)):
            conv_val = conv[s]
            # conv_val should be close to an integer
            # d is 1 if s is even and s//2 is in chosen
            if s % 2 == 0:
                half = s // 2
                if vec[half] > 0.5:
                    d = 1
                else:
                    d = 0
            else:
                d = 0
            
            # k = (conv_val - d) / 2
            # count = k + d = (conv_val + d) / 2
            count = int(round((conv_val + d) / 2.0))
            rep_counts[s] = count
            
        return rep_counts

    def is_valid_addition(x, chosen, rep_counts):
        """
        Check if x can be added to chosen.
        rep_counts is the array of current counts.
        """
        # New sums involving x:
        # 1. x + x = 2x
        # 2. x + y for y in chosen
        
        # Check 2x
        if 2*x < len(rep_counts) and rep_counts[2*x] >= 2:
            return False
        # Note: if rep_counts[2x] == 1, it's valid (will become 2)
        
        # Check x + y
        for y in chosen:
            s = x + y
            if s < len(rep_counts) and rep_counts[s] >= 2:
                return False
                
        return True

    def get_tight_count(x, chosen, rep_counts):
        """
        Return tightness score for adding x, or None if invalid.
        Tightness = number of sums that will reach count 2.
        """
        if not is_valid_addition(x, chosen, rep_counts):
            return None
            
        tight = 0
        
        # 2x
        if 2*x < len(rep_counts) and rep_counts[2*x] == 1:
            tight += 1
            
        # x + y
        for y in chosen:
            s = x + y
            if s < len(rep_counts) and rep_counts[s] == 1:
                tight += 1
                
        return tight

    def run_greedy(noise=0.0, time_limit=None):
        """
        Greedy construction using FFT for global count checks.
        """
        if time_limit is None:
            time_limit = deadline
            
        chosen = []
        # We will maintain rep_counts manually for speed during the loop,
        # but we can also use FFT if needed. 
        # Actually, maintaining counts manually is O(N) per step.
        # FFT is O(N log N) per step but gives global view.
        # For N=1000, manual is likely faster per step.
        # However, the "spectral" idea suggests using FFT.
        # Let's stick to manual counts for the inner loop for speed, 
        # but use FFT to *validate* or *score* if we want.
        # The prompt suggests "maximizing spectral entropy or minimizing peak energy".
        # Let's try a heuristic based on the distribution of counts.
        
        # Actually, let's just use the manual counts for validity and tightness,
        # as it's the same logic but potentially faster than calling FFT every step.
        # The "structural change" is the *representation*. 
        # If I use FFT, I can compute *all* potential additions' scores at once?
        # No, validity depends on current counts.
        
        # Let's use a hybrid: Manual counts for speed, but if we get stuck, 
        # use FFT to analyze the "landscape".
        
        # For now, implement standard greedy with manual counts, 
        # but add a "spectral kick" later.
        
        chosen = []
        counts = defaultdict(int)
        candidates = set(range(1, N + 1))
        
        while candidates:
            if time.time() > time_limit:
                break
                
            best_x = None
            best_score = float('inf')
            
            for x in candidates:
                score = get_tight_count(x, chosen, counts)
                # get_tight_count expects rep_counts array, but here we pass dict.
                # Let's adapt get_tight_count to handle dict or array.
                # To keep it simple, let's write a specific check for dict.
                
                # Re-implement tight check for dict
                if counts.get(2*x, 0) >= 2:
                    continue
                tight = 0
                if counts.get(2*x, 0) == 1:
                    tight += 1
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
                    
                total = tight + (random.random() * noise if noise > 0 else 0)
                if total < best_score:
                    best_score = total
                    best_x = x
                    
            if best_x is None:
                break
                
            # Add best_x
            counts[2*best_x] = counts.get(2*best_x, 0) + 1
            for y in chosen:
                s = best_x + y
                counts[s] = counts.get(s, 0) + 1
            chosen.append(best_x)
            candidates.remove(best_x)
            
        return chosen, counts

    def spectral_kick(chosen, counts, time_limit):
        """
        Use FFT to analyze the set and find a "good" element to remove
        or a "good" element to add, based on spectral properties.
        """
        # Compute rep_counts via FFT
        rep_arr = get_rep_counts_fft(chosen)
        
        # Metric: Minimize max rep count (should be <= 2). 
        # Or maximize "slack": number of sums with count < 2.
        # Let's look at the "pressure" on each element.
        # Removing element x reduces counts for 2x and x+y.
        # We want to remove an element that "frees up" the most constrained sums.
        
        # For each x in chosen, calculate how many sums involving x are at count 2.
        # If we remove x, those sums drop to count 1 (or 0), potentially allowing new additions.
        
        # We also want to find a candidate z not in chosen that has low tightness.
        
        # Let's pick a random x to remove, but biased towards those with high "constraint".
        # Constraint of x = number of sums s involving x such that rep_counts[s] == 2.
        
        constraints = {}
        for x in chosen:
            c = 0
            # 2x
            if 2*x < len(rep_arr) and rep_arr[2*x] == 2:
                c += 1
            # x + y
            for y in chosen:
                if y == x: continue
                s = x + y
                if s < len(rep_arr) and rep_arr[s] == 2:
                    c += 1
            constraints[x] = c
            
        # Pick x with highest constraint (or random among top few)
        if not constraints:
            return chosen, counts
            
        # Sort by constraint
        sorted_items = sorted(constraints.items(), key=lambda item: item[1], reverse=True)
        # Pick from top 10%
        k = max(1, len(sorted_items) // 10)
        candidates_to_remove = [item[0] for item in sorted_items[:k]]
        x_to_remove = random.choice(candidates_to_remove)
        
        # Remove x_to_remove
        new_chosen = [x for x in chosen if x != x_to_remove]
        
        # Rebuild counts
        new_counts = defaultdict(int)
        for x in new_chosen:
            new_counts[2*x] = new_counts.get(2*x, 0) + 1
        for i in range(len(new_chosen)):
            for j in range(i+1, len(new_chosen)):
                s = new_chosen[i] + new_chosen[j]
                new_counts[s] = new_counts.get(s, 0) + 1
                
        # Now greedily refill
        candidates = set(range(1, N + 1))
        for x in new_chosen:
            candidates.discard(x)
            
        while candidates:
            if time.time() > time_limit:
                break
                
            best_x = None
            best_score = float('inf')
            
            for x in candidates:
                # Check validity
                if new_counts.get(2*x, 0) >= 2:
                    continue
                tight = 0
                if new_counts.get(2*x, 0) == 1:
                    tight += 1
                valid = True
                for y in new_chosen:
                    s = x + y
                    c = new_counts.get(s, 0)
                    if c >= 2:
                        valid = False
                        break
                    if c == 1:
                        tight += 1
                if not valid:
                    continue
                    
                total = tight + random.random() * 0.5
                if total < best_score:
                    best_score = total
                    best_x = x
                    
            if best_x is None:
                break
                
            new_counts[2*best_x] = new_counts.get(2*best_x, 0) + 1
            for y in new_chosen:
                s = best_x + y
                new_counts[s] = new_counts.get(s, 0) + 1
            new_chosen.append(best_x)
            candidates.remove(best_x)
            
        return new_chosen, new_counts

    def local_search(initial_set, time_limit):
        chosen = list(initial_set)
        
        # Build counts
        counts = defaultdict(int)
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
            return chosen, counts
            
        while time.time() < time_limit:
            # Try to add
            added_any = False
            cand_list = list(candidates)
            random.shuffle(cand_list)
            
            for x in cand_list:
                if time.time() > time_limit:
                    break
                # Check validity
                if counts.get(2*x, 0) >= 2:
                    continue
                valid = True
                for y in chosen:
                    s = x + y
                    if counts.get(s, 0) >= 2:
                        valid = False
                        break
                if valid:
                    # Add
                    counts[2*x] = counts.get(2*x, 0) + 1
                    for y in chosen:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                    chosen.append(x)
                    candidates.discard(x)
                    added_any = True
                    break
                    
            if added_any:
                continue
                
            # If stuck, try spectral kick
            if time.time() > time_limit:
                break
                
            # Use spectral kick to escape
            new_chosen, new_counts = spectral_kick(chosen, counts, time_limit)
            
            if len(new_chosen) > len(chosen):
                chosen = new_chosen
                counts = new_counts
                # Update candidates
                candidates = set(range(1, N + 1))
                for x in chosen:
                    candidates.discard(x)
            else:
                # If kick didn't help, try a simple random swap
                # or break to restart
                # Let's try a simple swap
                swap_success = False
                for _ in range(5):
                    if time.time() > time_limit:
                        break
                    if not candidates:
                        break
                    x_to_try = random.choice(list(candidates))
                    y_to_try = random.choice(chosen)
                    
                    # Check if swap is valid
                    # Remove y, add x
                    old_counts = dict(counts)
                    
                    # Remove y
                    counts[2*y_to_try] -= 1
                    if counts[2*y_to_try] == 0: del counts[2*y_to_try]
                    for z in chosen:
                        if z != y_to_try:
                            s = y_to_try + z
                            counts[s] -= 1
                            if counts[s] == 0: del counts[s]
                            
                    # Check x
                    if counts.get(2*x_to_try, 0) < 2:
                        valid = True
                        for z in chosen:
                            if z != y_to_try:
                                s = x_to_try + z
                                if counts.get(s, 0) >= 2:
                                    valid = False
                                    break
                        if valid:
                            # Add x
                            counts[2*x_to_try] = counts.get(2*x_to_try, 0) + 1
                            for z in chosen:
                                if z != y_to_try:
                                    s = x_to_try + z
                                    counts[s] = counts.get(s, 0) + 1
                            
                            chosen.remove(y_to_try)
                            chosen.append(x_to_try)
                            candidates.discard(x_to_try)
                            candidates.add(y_to_try)
                            swap_success = True
                            break
                            
                    # Revert
                    counts.clear()
                    counts.update(old_counts)
                    
                if not swap_success:
                    # Give up on this local search instance
                    break
                    
        return chosen, counts

    # Main loop
    while time.time() < deadline:
        # 1. Generate baseline
        noise = random.uniform(0, 1.0)
        baseline, _ = run_greedy(noise=noise, time_limit=min(deadline, time.time() + 2))
        
        if len(baseline) > best_size:
            best_size = len(baseline)
            best_set = baseline
            
        # 2. Refine
        remaining = deadline - time.time()
        if remaining < 1:
            break
            
        ls_time = min(remaining * 0.5, 5.0)
        refined, _ = local_search(best_set if len(best_set) > len(baseline) else baseline, time.time() + ls_time)
        
        if len(refined) > best_size:
            best_size = len(refined)
            best_set = refined
            
    return best_set
