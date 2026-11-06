import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety
    
    base = 51975
    # Generate odd divisors > 1
    moduli = [d for d in range(3, base + 1, 2) if base % d == 0]
    # Sort moduli, maybe by size or randomly
    # Using a fixed order for reproducibility in local search, but randomizing in SA
    
    best_uncovered = float('inf')
    best_residues = None
    
    # Number of restarts
    # We have ~145 seconds.
    # One full coordinate descent pass: 47 moduli * O(N) each.
    # O(N) is roughly 50k ops. 47 * 50k = 2.35M ops.
    # NumPy is fast, maybe 1ms per modulus? 47ms per pass.
    # We can do ~1000 passes in 145s.
    # But we also need SA moves.
    
    # Let's try a few restarts with different initializations.
    for seed in range(5): # 5 restarts
        if time.time() > deadline - 10:
            break
            
        random.seed(seed)
        np.random.seed(seed)
        
        residues = {}
        # Initialize randomly
        for m in moduli:
            residues[m] = random.randint(0, m - 1)
        
        # Compute initial coverage
        count_cover = np.zeros(base, dtype=np.int8)
        for m in moduli:
            r = residues[m]
            count_cover[r::m] += 1
            
        current_uncovered = np.sum(count_cover == 0)
        if current_uncovered < best_uncovered:
            best_uncovered = current_uncovered
            best_residues = residues.copy()
            
        # Local Search: Coordinate Descent with SA
        temperature = 10.0
        cooling = 0.99
        min_temperature = 0.01
        
        # Precompute indices for each modulus to speed up updates?
        # Or just use slicing.
        
        while temperature > min_temperature and time.time() < deadline - 5:
            improved = False
            # Shuffle moduli to avoid bias
            current_moduli = moduli.copy()
            random.shuffle(current_moduli)
            
            for m in current_moduli:
                if time.time() > deadline - 5:
                    break
                
                r_old = residues[m]
                
                # Remove m from coverage
                count_cover[r_old::m] -= 1
                
                # Calculate U[r] = sum of 1 where count_cover[i]==0 and i%m == r
                # This is a bit slow if done naively in Python.
                # We can use NumPy:
                # uncovered_mask = (count_cover == 0)
                # We need sum of uncovered_mask[i] for i % m == r.
                # This is equivalent to summing uncovered_mask[r::m]
                
                # To find best r, we can iterate r and sum.
                # Cost O(N) per r, O(N*m) total? No.
                # sum(uncovered_mask[r::m]) is O(N/m).
                # Total cost sum over r is O(N).
                
                best_r = r_old
                best_score = np.sum(count_cover[r_old::m] == 0)
                
                # Optimization: Only check if m is small? No, check all.
                # But for large m, N/m is small.
                # For m=3, N/3 ~ 17k. For m=51975, N/m = 1.
                # So checking all r is cheap for large m, expensive for small m.
                # Total cost per modulus is O(N).
                
                # Let's just loop r.
                # For speed, we can break early if we find 0? No, we want min.
                # If best_score becomes 0, we can break.
                
                if best_score > 0:
                    for r in range(m):
                        if r == r_old:
                            continue
                        s = np.sum(count_cover[r::m] == 0)
                        if s < best_score:
                            best_score = s
                            best_r = r
                            if s == 0:
                                break
                
                # Decide whether to accept
                old_score = np.sum(count_cover[r_old::m] == 0)
                # Note: old_score was calculated before removal? No, we removed it.
                # We need the score of the old configuration relative to others.
                # Let's re-calculate or store it.
                # Actually, best_score is the score of best_r.
                # The score of r_old is np.sum(count_cover[r_old::m] == 0).
                
                delta = best_score - old_score
                
                if delta <= 0 or (delta > 0 and random.random() < np.exp(-delta / temperature)):
                    # Accept
                    if best_r != r_old:
                        count_cover[r_old::m] += 1 # Restore old
                        count_cover[best_r::m] -= 1 # Apply new? No.
                        # Current state: m is removed.
                        # We want to set m to best_r.
                        # So we just add best_r back.
                        count_cover[best_r::m] += 1
                        residues[m] = best_r
                        improved = True
                    else:
                        count_cover[r_old::m] += 1
                else:
                    # Reject, restore old
                    count_cover[r_old::m] += 1
                
            temperature *= cooling
            
            # If no improvement for a while, restart or cool faster?
            # For now, just let it run.
            
            # Update best
            current_uncovered = np.sum(count_cover == 0)
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_residues = residues.copy()
        
        # After SA, do a deterministic polish
        for _ in range(10): # 10 passes of pure hill climbing
            if time.time() > deadline - 2:
                break
            improved = False
            for m in moduli:
                r_old = residues[m]
                count_cover[r_old::m] -= 1
                
                best_r = r_old
                best_score = np.sum(count_cover[r_old::m] == 0)
                
                for r in range(m):
                    if r == r_old:
                        continue
                    s = np.sum(count_cover[r::m] == 0)
                    if s < best_score:
                        best_score = s
                        best_r = r
                        if s == 0:
                            break
                
                if best_r != r_old:
                    count_cover[r_old::m] += 1
                    count_cover[best_r::m] += 1
                    residues[m] = best_r
                    improved = True
                else:
                    count_cover[r_old::m] += 1
                    
            if not improved:
                break
                
        final_uncovered = np.sum(count_cover == 0)
        if final_uncovered < best_uncovered:
            best_uncovered = final_uncovered
            best_residues = residues.copy()

    # Construct return value from best solution
    if best_residues is None:
        # Fallback
        best_residues = {m: 0 for m in moduli}
        
    classes = [[m, best_residues[m]] for m in moduli]
    return classes
