import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit (kill at 150s, but let's be safe)
    
    N = 1000
    
    def run_greedy(noise_scale=0.0, seed=42, tie_break='min_x'):
        random.seed(seed)
        chosen = []
        chosen_set = set()
        counts = defaultdict(int)
        
        # Pre-compute candidates
        candidates = list(range(1, N + 1))
        
        while True:
            best_x = None
            best_score = float('inf')
            best_tight = float('inf')
            best_tie_break_val = None
            best_new_sums = None
            
            if time.time() > deadline:
                break
                
            # Iterate through all candidates not in chosen
            for x in candidates:
                if x in chosen_set:
                    continue
                
                valid = True
                tight_count = 0
                new_sums = []
                
                # Check x + y for y in chosen
                for y in chosen:
                    s = x + y
                    c = counts.get(s, 0)
                    if c >= 2:
                        valid = False
                        break
                    if c == 1:
                        tight_count += 1
                    new_sums.append((s, c))
                
                if not valid:
                    continue
                
                # Check 2*x
                s_double = 2 * x
                c_double = counts.get(s_double, 0)
                if c_double >= 2:
                    valid = False
                    continue
                if c_double == 1:
                    tight_count += 1
                new_sums.append((s_double, c_double))
                
                if not valid:
                    continue
                
                # Score: prefer lower tight_count
                score = tight_count + random.random() * noise_scale
                
                if score < best_score:
                    best_score = score
                    best_x = x
                    best_tight = tight_count
                    best_new_sums = new_sums
                    if tie_break == 'min_x':
                        best_tie_break_val = x
                    elif tie_break == 'max_x':
                        best_tie_break_val = -x
                    else:
                        best_tie_break_val = random.random()
                elif score == best_score:
                    # Tie breaking
                    if tie_break == 'min_x':
                        if x < best_tie_break_val:
                            best_x = x
                            best_new_sums = new_sums
                            best_tie_break_val = x
                    elif tie_break == 'max_x':
                        if x > -best_tie_break_val: # Note: best_tie_break_val is -x
                            best_x = x
                            best_new_sums = new_sums
                            best_tie_break_val = -x
                    else:
                        # Random tie break
                        if random.random() < 0.5:
                            best_x = x
                            best_new_sums = new_sums
                            best_tie_break_val = random.random()
            
            if best_x is None:
                break
                
            # Add best_x
            chosen.append(best_x)
            chosen_set.add(best_x)
            for s, c in best_new_sums:
                counts[s] = c + 1
        
        return chosen

    best_set = []
    best_size = 0
    
    # Phase 1: Deterministic Greedy with various tie-breakers
    # Try a few fixed seeds with low noise and different tie-breaks
    for seed in range(20):
        if time.time() > deadline:
            break
        tie_break = ['min_x', 'max_x', 'random'][seed % 3]
        s = run_greedy(noise_scale=0.0, seed=seed, tie_break=tie_break)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # Phase 2: Randomized Greedy Restarts
    # Higher noise to explore different structures
    while time.time() < deadline:
        seed = random.randint(0, 100000)
        noise = random.uniform(0.0, 2.0)
        tie_break = random.choice(['min_x', 'max_x', 'random'])
        s = run_greedy(noise_scale=noise, seed=seed, tie_break=tie_break)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    return best_set
