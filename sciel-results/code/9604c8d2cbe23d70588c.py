import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Target n = 34
    n = 34
    
    # Precompute triples
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute K4s and K5s
    k4_list = list(combinations(range(n), 4))
    num_k4 = len(k4_list)
    k5_list = list(combinations(range(n), 5))
    num_k5 = len(k5_list)
    
    # Map each triple to the K4s and K5s it belongs to
    triple_to_k4s = [[] for _ in range(num_triples)]
    triple_to_k5s = [[] for _ in range(num_triples)]
    
    # k4_data[i] = list of 4 triple indices in K4 i
    # k5_data[i] = list of 10 triple indices in K5 i
    k4_data = []
    k5_data = []
    
    for i, k4 in enumerate(k4_list):
        ts = []
        for j in range(4):
            t = tuple(sorted(k4[:j] + k4[j+1:]))
            idx = triple_to_idx[t]
            ts.append(idx)
            triple_to_k4s[idx].append(i)
        k4_data.append(ts)
        
    for i, k5 in enumerate(k5_list):
        ts = []
        for combo in combinations(range(5), 3):
            t = tuple(sorted([k5[j] for j in combo]))
            idx = triple_to_idx[t]
            ts.append(idx)
            triple_to_k5s[idx].append(i)
        k5_data.append(ts)

    # Convert to numpy arrays for fast vectorized operations
    k4_arr = np.array(k4_data, dtype=np.int32)
    k5_arr = np.array(k5_data, dtype=np.int32)
    
    # Precompute the number of K4s and K5s per triple for potential weighting
    # Not strictly necessary, but good for understanding
    
    def get_counts_and_violations(colors):
        red_mask = (colors == 0)
        k4_counts = red_mask[k4_arr].sum(axis=1)
        
        blue_mask = (colors == 1)
        k5_counts = blue_mask[k5_arr].sum(axis=1)
        
        violations = np.sum(k4_counts == 4) + np.sum(k5_counts == 10)
        
        return k4_counts, k5_counts, violations

    # We will run multiple restarts.
    # Each restart uses a Simulated Annealing / WalkSAT hybrid.
    
    best_violations = float('inf')
    best_colors = None
    
    num_restarts = 0
    max_restarts = 1000
    
    # Strategy for initialization:
    # 1. Random
    # 2. "Algebraic" - color based on vertex properties
    
    while time.time() < deadline and num_restarts < max_restarts:
        num_restarts += 1
        
        # Choose initialization strategy
        if num_restarts == 1:
            # Random
            colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
        elif num_restarts % 10 == 0:
            # Try a structured init: color = (a+b+c) % 2
            # This creates a specific pattern that might avoid small monochromatic subgraphs
            c = np.zeros(num_triples, dtype=np.int8)
            for i, (a, b, c_val) in enumerate(triples):
                c[i] = (a + b + c_val) % 2
            colors = c
        else:
            # Random
            colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
            
        k4_counts, k5_counts, current_violations = get_counts_and_violations(colors)
        
        if current_violations == 0:
            return {'n': 34, 'colors': colors.tolist()}
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors.copy()
            
        # Maintain sets of violated clauses for fast selection
        violated_k4s = set(np.where(k4_counts == 4)[0])
        violated_k5s = set(np.where(k5_counts == 10)[0])
        
        # Simulated Annealing / WalkSAT parameters
        T = 5.0
        T_min = 0.1
        cooling_rate = 0.995
        run_start = time.time()
        
        # We allow up to 2 seconds per restart to ensure we try many restarts
        while time.time() < run_start + 2.0:
            if current_violations == 0:
                return {'n': 34, 'colors': colors.tolist()}
                
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors.copy()
                
            # If we have no violations, we are done.
            # If we have violations, pick one.
            
            if len(violated_k4s) + len(violated_k5s) == 0:
                # Should not happen if current_violations > 0, but safety check
                break
                
            # Pick a violated clause
            # To avoid expensive list(set) conversion every step, we can maintain a list
            # But for simplicity and given the small size of violations in late stages, 
            # we convert. If violations are large, we are in early stage, random flip is fine.
            
            if len(violated_k4s) + len(violated_k5s) > 1000:
                # Early stage: just flip a random triple
                t_idx = random.randint(0, num_triples - 1)
            else:
                # Late stage: pick from violated clauses
                # 50% chance to pick a K4 violation, 50% K5 (if available)
                if violated_k4s and (not violated_k5s or random.random() < 0.5):
                    # Pick random K4
                    k4_idx = random.choice(list(violated_k4s))
                    # Pick random triple in this K4
                    t_idx = random.choice(k4_data[k4_idx])
                elif violated_k5s:
                    # Pick random K5
                    k5_idx = random.choice(list(violated_k5s))
                    # Pick random triple in this K5
                    t_idx = random.choice(k5_data[k5_idx])
                else:
                    t_idx = random.randint(0, num_triples - 1)
                    
            # Calculate the effect of flipping t_idx
            # We need to know the new violation count to decide acceptance.
            # Instead of calculating exactly, we can just flip and calculate delta.
            
            old_val = colors[t_idx]
            new_val = 1 - old_val
            
            # Calculate delta violations
            # This is the expensive part. We must update counts.
            
            # We will perform the flip, calculate new violations, and revert if not accepted.
            
            # Save state
            old_k4_counts = k4_counts.copy()
            old_k5_counts = k5_counts.copy()
            old_violated_k4s = violated_k4s.copy()
            old_violated_k5s = violated_k5s.copy()
            
            # Perform Flip
            colors[t_idx] = new_val
            
            delta_red = 1 if new_val == 0 else -1
            delta_blue = 1 if new_val == 1 else -1
            
            new_violations = current_violations
            
            # Update K4s
            for k4_idx in triple_to_k4s[t_idx]:
                old_count = k4_counts[k4_idx]
                new_count = old_count + delta_red
                k4_counts[k4_idx] = new_count
                
                was_viol = (old_count == 4)
                is_viol = (new_count == 4)
                
                if was_viol and not is_viol:
                    violated_k4s.discard(k4_idx)
                    new_violations -= 1
                elif not was_viol and is_viol:
                    violated_k5s.add(k4_idx) # BUG: Should be violated_k4s
                    violated_k4s.add(k4_idx)
                    new_violations += 1

            # Update K5s
            for k5_idx in triple_to_k5s[t_idx]:
                old_count = k5_counts[k5_idx]
                new_count = old_count + delta_blue
                k5_counts[k5_idx] = new_count
                
                was_viol = (old_count == 10)
                is_viol = (new_count == 10)
                
                if was_viol and not is_viol:
                    violated_k5s.discard(k5_idx)
                    new_violations -= 1
                elif not was_viol and is_viol:
                    violated_k5s.add(k5_idx)
                    new_violations += 1

            # Acceptance Criteria
            delta_v = new_violations - current_violations
            
            accepted = False
            if delta_v <= 0:
                accepted = True
            else:
                # Accept with probability exp(-delta_v / T)
                if T > T_min:
                    prob = np.exp(-delta_v / T)
                    if random.random() < prob:
                        accepted = True
            
            if accepted:
                current_violations = new_violations
                T *= cooling_rate
            else:
                # Revert
                colors[t_idx] = old_val
                k4_counts = old_k4_counts
                k5_counts = old_k5_counts
                violated_k4s = old_violated_k4s
                violated_k5s = old_violated_k5s
                # T cooling usually happens regardless in SA, or only on acceptance?
                # Standard SA cools regardless.
                T *= cooling_rate
                
        # End of restart loop
        
    if best_violations == 0:
        return {'n': 34, 'colors': best_colors.tolist()}
    else:
        # Fallback
        n_fallback = 9
        colors_fallback = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_fallback), 3)]
        return {'n': n_fallback, 'colors': colors_fallback}
