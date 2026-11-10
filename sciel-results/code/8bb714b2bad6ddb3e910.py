import time
import random
import math

def calc_score(A):
    """
    Compute the score for a set A: max(|A+A|, |A*A|).
    """
    if not A:
        return 0
    A = list(A)
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    return max(len(sumset), len(prodset))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 20
    max_val = 10000
    
    # Helper to generate smooth numbers (2, 3, 5-smooth)
    def generate_smooth_numbers(limit):
        smooths = set()
        # 2^a * 3^b * 5^c
        for a in range(0, 15):
            p2 = 2 ** a
            if p2 > limit: break
            for b in range(0, 10):
                p3 = 3 ** b
                if p2 * p3 > limit: break
                for c in range(0, 8):
                    p5 = 5 ** c
                    val = p2 * p3 * p5
                    if val > limit: break
                    smooths.add(val)
        return sorted(smooths)

    # Pre-compute a rich pool of "structurally interesting" numbers
    # Small numbers tend to have more additive/multiplicative overlaps
    # Smooth numbers have many divisors and multiples, leading to overlaps in A*A
    smooths_1000 = generate_smooth_numbers(1000)
    smooths_10000 = generate_smooth_numbers(10000)
    
    # Pool 1: Small integers (1-100)
    small_ints = list(range(1, 101))
    
    # Pool 2: Smooth numbers up to 1000
    # Pool 3: Smooth numbers up to 10000 (subset)
    
    # Combine for a general pool
    general_pool = sorted(set(small_ints + smooths_1000 + [random.randint(1, max_val) for _ in range(50)]))
    
    best_global_set = None
    best_global_score = float('inf')
    
    # Multi-Start Strategy
    # We will run several independent ILS instances with different random seeds
    # Each instance runs for a fraction of the total time, or we run one long instance
    # with periodic restarts if stuck.
    
    # Let's do 5 restarts, each with its own SA/ILS phase.
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart_idx in range(num_restarts):
        restart_start = time.time()
        restart_deadline = restart_start + time_per_restart
        
        # Initialize based on restart index
        if restart_idx == 0:
            # Start with small integers
            current_set = list(range(1, 21))
        elif restart_idx == 1:
            # Start with small smooths
            current_set = smooths_1000[:20]
        elif restart_idx == 2:
            # Start with a mix of small and medium smooths
            random.seed(100 + restart_idx)
            candidates = smooths_1000 + small_ints
            current_set = sorted(random.sample(candidates, 20))
        else:
            # Random restarts from the general pool
            random.seed(200 + restart_idx)
            current_set = sorted(random.sample(general_pool, 20))
        
        current_score = calc_score(current_set)
        
        if current_score < best_global_score:
            best_global_score = current_score
            best_global_set = list(current_set)
            
        # Local Search / ILS Loop for this restart
        T = 500.0
        T_min = 1.0
        alpha = 0.99
        
        # Adaptive Pool Logic:
        # We want to propose moves that are likely to create overlaps.
        # Overlaps in A+A happen if a+b = c+d.
        # Overlaps in A*A happen if a*b = c*d.
        # Numbers with many factors (smooth) or small numbers help.
        
        # We will maintain a "hot" list of candidates.
        # Initially, use general_pool.
        # If stuck, bias towards numbers that share factors with current_set.
        
        step_count = 0
        no_improve_count = 0
        last_improve_score = current_score
        
        while time.time() < restart_deadline:
            # Generate move
            # Pick an index to replace
            idx = random.randint(0, n - 1)
            old_val = current_set[idx]
            current_set.pop(idx)
            
            # Determine candidate pool for this move
            # Strategy: 
            # 1. 50% chance: pick from general_pool (exploration)
            # 2. 30% chance: pick a smooth number that is a multiple of some element in current_set (exploitation of product structure)
            # 3. 20% chance: pick a small number that is a sum of two elements in current_set (exploitation of sum structure)
            
            r = random.random()
            candidates = []
            
            if r < 0.5:
                candidates = general_pool
            elif r < 0.8:
                # Multiples of existing elements
                # Find small multiples
                for elem in current_set:
                    for mult in range(2, 10):
                        val = elem * mult
                        if val <= max_val and val not in current_set:
                            candidates.append(val)
                if not candidates:
                    candidates = general_pool
            else:
                # Sums of existing elements
                for i in range(len(current_set)):
                    for j in range(i, len(current_set)):
                        val = current_set[i] + current_set[j]
                        if val <= max_val and val not in current_set:
                            candidates.append(val)
                if not candidates:
                    candidates = general_pool
            
            # Pick a random candidate from the chosen subset
            # Ensure it's not already in current_set
            new_val = None
            # Filter candidates not in current_set for efficiency if list is small, else just try
            valid_candidates = [c for c in candidates if c not in current_set]
            if not valid_candidates:
                current_set.insert(idx, old_val)
                continue
                
            new_val = random.choice(valid_candidates)
            
            # Perform swap
            current_set.insert(idx, new_val)
            
            # Calculate new score
            new_score = calc_score(current_set)
            
            # Acceptance
            delta = new_score - current_score
            if delta < 0:
                current_score = new_score
                no_improve_count = 0
                if new_score < best_global_score:
                    best_global_score = new_score
                    best_global_set = list(current_set)
            else:
                # Deterioration
                if delta == 0:
                    # Accept neutral moves to move around landscape
                    current_score = new_score
                    no_improve_count += 1
                else:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_score = new_score
                        no_improve_count += 1
                    else:
                        # Reject: revert
                        current_set.pop(idx)
                        current_set.insert(idx, old_val)
                        no_improve_count += 1
            
            # Cooling
            T *= alpha
            if T < T_min:
                T = T_min
                
            # Perturbation / Restart within ILS if stuck
            if no_improve_count > 500:
                # Perturb: change 3 random elements
                for _ in range(3):
                    p_idx = random.randint(0, n - 1)
                    p_old = current_set[p_idx]
                    current_set.pop(p_idx)
                    p_new = random.choice(general_pool)
                    # Ensure distinct
                    while p_new in current_set:
                        p_new = random.choice(general_pool)
                    current_set.insert(p_idx, p_new)
                current_score = calc_score(current_set)
                no_improve_count = 0
                T = 500.0 # Reheat

            step_count += 1
            
    # Final verification
    if best_global_set is None or len(best_global_set) != 20:
        return list(range(1, 21))
        
    final_set = sorted(list(set(best_global_set)))
    if len(final_set) < 20:
        return list(range(1, 21))
    if any(x > 10000 for x in final_set):
        return list(range(1, 21))
        
    return final_set

if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Set: {res}")
    print(f"Score: {calc_score(res)}")
