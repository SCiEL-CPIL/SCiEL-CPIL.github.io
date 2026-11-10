import math
import random
import time

def construct():
    # Weights: higher a -> higher weight
    weights = {a: 1.0 / (512 - a) for a in range(1, 512)}
    
    # Precompute GCDs or use math.gcd
    # For speed, we can precompute a coprime matrix or just use math.gcd which is fast in Python 3
    
    def get_score(S):
        return sum(weights[a] for a in S)
        
    def is_valid_swap(S, out_idx, in_val):
        # Check if in_val is coprime to all in S except S[out_idx]
        for i, val in enumerate(S):
            if i == out_idx:
                continue
            if math.gcd(in_val, val) != 1:
                return False
        return True

    # Initial Solution: Greedy (same as current, serves as a good start)
    def greedy_init():
        candidates = list(range(511, 0, -1))
        chosen = []
        for a in candidates:
            if all(math.gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
        return chosen

    # Simulated Annealing
    def simulate_annealing(start_state, start_temp, end_temp, steps):
        current_state = start_state[:]
        current_score = get_score(current_state)
        best_state = current_state[:]
        best_score = current_score
        
        # Precompute list of all candidates for random selection
        all_candidates = list(range(1, 512))
        
        for i in range(steps):
            # Cooling schedule
            T = start_temp * (end_temp / start_temp) ** (i / steps)
            
            # Generate neighbor
            # 1. Pick an index to remove
            out_idx = random.randint(0, len(current_state) - 1)
            out_val = current_state[out_idx]
            
            # 2. Pick a candidate to add
            # We should bias towards high weights or just pick random from all not in state
            # To be efficient, let's pick a random candidate from the universe
            in_val = random.choice(all_candidates)
            
            # If in_val is already in state (and not the one being removed), skip or handle
            if in_val in current_state:
                if in_val == out_val:
                    continue
                else:
                    # It's in the state at a different index, so swapping is just a permutation, score doesn't change.
                    # We can skip this move as it yields no gain and wastes time.
                    continue

            # Check validity
            if not is_valid_swap(current_state, out_idx, in_val):
                continue
                
            new_score = current_score - weights[out_val] + weights[in_val]
            delta = new_score - current_score
            
            # Acceptance criterion
            if delta > 0 or random.random() < math.exp(delta / T):
                current_state[out_idx] = in_val
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_state = current_state[:]
                    
        return best_state, best_score

    # Multi-Start Simulated Annealing
    # Time limit 120s. Let's aim for ~50s of computation to be safe.
    start_time = time.time()
    time_limit = 100.0 # seconds
    
    best_overall_state = greedy_init()
    best_overall_score = get_score(best_overall_state)
    
    # Run multiple SA instances
    # Each SA run: ~10000 steps? 
    # Let's estimate time per step. is_valid_swap is O(N) where N is size of set (~10-20).
    # 10000 steps * 20 gcds = 200k gcds. Very fast.
    # We can do more steps.
    
    num_runs = 10
    steps_per_run = 50000
    start_temp = 1.0
    end_temp = 0.001
    
    for run in range(num_runs):
        # Start from greedy or a perturbed greedy
        # Perturbation: take greedy, swap a few random elements
        start_state = best_overall_state[:]
        # Perturb
        for _ in range(5):
            if start_state:
                idx = random.randint(0, len(start_state) - 1)
                new_val = random.randint(1, 511)
                if new_val not in start_state:
                    if is_valid_swap(start_state, idx, new_val):
                        start_state[idx] = new_val
        
        state, score = simulate_annealing(start_state, start_temp, end_temp, steps_per_run)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_state = state
            
        # Check time
        if time.time() - start_time > time_limit:
            break
            
    return best_overall_state
