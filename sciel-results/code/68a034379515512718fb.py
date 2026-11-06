import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 150s limit
    
    N = 1000
    MAX_SUM = 2 * N  # 2000
    
    def run_greedy_fast(noise_scale=0.0, seed=42, tie_break='min_x'):
        """
        Fast greedy using numpy array for sum counts.
        For each candidate x, we need to check if any (x+y) or 2x has count >= 2.
        We maintain counts as a numpy array of length MAX_SUM+1.
        """
        random.seed(seed)
        chosen = []
        chosen_set = set()
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        
        while True:
            best_x = None
            best_score = float('inf')
            best_tight = float('inf')
            best_tie_break_val = None
            
            if time.time() > deadline:
                break
            
            # For each candidate x not in chosen_set:
            #   valid: no sum s in {x+y for y in chosen} U {2x} has counts[s] >= 2
            #   tight_count: number of such sums with counts[s] == 1
            #
            # Key insight: for a fixed x, the sums are x+1, x+2, ..., x+N (for y in 1..N)
            # but only those y that are in chosen_set matter.
            # However, we can't easily do this in O(1) without precomputation.
            #
            # Alternative: maintain a boolean array 'in_set' and use it to compute
            # the relevant sums quickly.
            
            # Let's try a different approach: maintain the set of chosen elements
            # and for each candidate x, compute the sums that would be affected.
            # The sums affected by adding x are: {x + y : y in chosen} U {2x}
            
            # For speed, let's iterate over candidates and use numpy where possible.
            
            # Pre-compute: for each possible x, which y values are in chosen?
            # This is expensive to recompute each iteration.
            
            # Let's just do a smart loop with early termination.
            # The key optimization: instead of checking all y in chosen for each x,
            # we can note that if x is small, most x+y will be small too.
            
            # Actually, let's try a completely different approach:
            # Maintain the counts array. For candidate x, the new sums are:
            #   S_x = {x + y : y in chosen} ∪ {2x}
            # We need: max(counts[s] for s in S_x) < 2 for validity
            # And: tight = sum(1 for s in S_x if counts[s] == 1)
            
            # To compute S_x efficiently, we need to know which y are in chosen.
            # Let's maintain a numpy array 'in_chosen' of length N+1.
            
            in_chosen = np.zeros(N + 1, dtype=bool)
            
            best_x = None
            best_score = float('inf')
            best_tight = float('inf')
            best_tie_break_val = None
            
            # Iterate through candidates
            for x in range(1, N + 1):
                if in_chosen[x]:
                    continue
                
                # Check 2x
                s2 = 2 * x
                if counts[s2] >= 2:
                    continue
                
                tight = 0
                if counts[s2] == 1:
                    tight += 1
                
                # Check x + y for y in chosen
                # We need to iterate over y in chosen_set
                # But we can break early if we find a sum with count >= 2
                
                valid = True
                for y in chosen:
                    s = x + y
                    c = counts[s]
                    if c >= 2:
                        valid = False
                        break
                    if c == 1:
                        tight += 1
                
                if not valid:
                    continue
                
                score = tight + random.random() * noise_scale
                
                if score < best_score:
                    best_score = score
                    best_x = x
                    best_tight = tight
                    if tie_break == 'min_x':
                        best_tie_break_val = x
                    elif tie_break == 'max_x':
                        best_tie_break_val = -x
                    else:
                        best_tie_break_val = random.random()
                elif score == best_score:
                    if tie_break == 'min_x':
                        if x < best_tie_break_val:
                            best_x = x
                            best_tight = tight
                            best_tie_break_val = x
                    elif tie_break == 'max_x':
                        if x > -best_tie_break_val:
                            best_x = x
                            best_tight = tight
                            best_tie_break_val = -x
                    else:
                        if random.random() < 0.5:
                            best_x = x
                            best_tight = tight
                            best_tie_break_val = random.random()
            
            if best_x is None:
                break
            
            # Add best_x
            chosen.append(best_x)
            in_chosen[best_x] = True
            s2 = 2 * best_x
            counts[s2] += 1
            for y in chosen[:-1]:  # all previously chosen
                s = best_x + y
                counts[s] += 1
        
        return chosen

    def run_greedy_numpy(noise_scale=0.0, seed=42, tie_break='min_x'):
        """
        Greedy with numpy-based sum checking.
        Key idea: maintain a bitset of chosen elements. For candidate x,
        the sums x+y for y in chosen can be computed via convolution-like operation.
        But since we just need max count and tight count, let's try a different angle.
        
        Actually, let's maintain the counts array and for each x, compute the affected sums
        using numpy vectorized operations where possible.
        """
        random.seed(seed)
        chosen = []
        in_chosen = np.zeros(N + 1, dtype=bool)
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        
        while True:
            if time.time() > deadline:
                break
            
            best_x = None
            best_score = float('inf')
            best_tight = float('inf')
            best_tie_break_val = None
            
            # Get list of chosen elements
            chosen_list = [i for i in range(1, N + 1) if in_chosen[i]]
            
            for x in range(1, N + 1):
                if in_chosen[x]:
                    continue
                
                # Check 2x
                s2 = 2 * x
                if counts[s2] >= 2:
                    continue
                
                tight = 1 if counts[s2] == 1 else 0
                
                # Check x + y for y in chosen_list
                # Vectorized: sums = x + chosen_list, check counts[sums]
                if len(chosen_list) > 0:
                    sums = x + np.array(chosen_list)
                    sum_counts = counts[sums]
                    if np.any(sum_counts >= 2):
                        continue
                    tight += np.sum(sum_counts == 1)
                
                score = tight + random.random() * noise_scale
                
                if score < best_score:
                    best_score = score
                    best_x = x
                    best_tight = tight
                    if tie_break == 'min_x':
                        best_tie_break_val = x
                    elif tie_break == 'max_x':
                        best_tie_break_val = -x
                    else:
                        best_tie_break_val = random.random()
                elif score == best_score:
                    if tie_break == 'min_x':
                        if x < best_tie_break_val:
                            best_x = x
                            best_tight = tight
                            best_tie_break_val = x
                    elif tie_break == 'max_x':
                        if x > -best_tie_break_val:
                            best_x = x
                            best_tight = tight
                            best_tie_break_val = -x
                    else:
                        if random.random() < 0.5:
                            best_x = x
                            best_tight = tight
                            best_tie_break_val = random.random()
            
            if best_x is None:
                break
            
            # Add best_x
            chosen.append(best_x)
            in_chosen[best_x] = True
            s2 = 2 * best_x
            counts[s2] += 1
            if len(chosen) > 1:
                prev_chosen = np.array(chosen[:-1])
                new_sums = best_x + prev_chosen
                counts[new_sums] += 1
        
        return chosen

    best_set = []
    best_size = 0
    
    # Phase 1: Deterministic Greedy with various tie-breakers
    for seed in range(30):
        if time.time() > deadline:
            break
        tie_break = ['min_x', 'max_x', 'random'][seed % 3]
        s = run_greedy_numpy(noise_scale=0.0, seed=seed, tie_break=tie_break)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # Phase 2: Randomized Greedy Restarts
    while time.time() < deadline:
        seed = random.randint(0, 1000000)
        noise = random.uniform(0.0, 3.0)
        tie_break = random.choice(['min_x', 'max_x', 'random'])
        s = run_greedy_numpy(noise_scale=noise, seed=seed, tie_break=tie_break)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    return best_set
