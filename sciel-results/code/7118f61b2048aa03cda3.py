import numpy as np
import time
import random

def construct():
    N = 51975
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Generate all divisors of 51975 greater than 1
    divisors = []
    for d in range(2, N + 1):
        if N % d == 0:
            divisors.append(d)
    divisors.sort()
    
    # cover_count[x] is the number of classes covering x
    cover_count = np.zeros(N, dtype=np.int32)
    selected_classes = [] # list of [m, r]
    used_moduli = set()
    
    # Helper to apply a class to cover_count
    def apply_class(m, r, sign=1):
        indices = np.arange(r, N, m)
        cover_count[indices] += sign
    
    # Helper to get current score
    def get_score():
        return np.sum(cover_count == 0)
    
    best_score = get_score()
    best_classes = []
    
    # Phase 1: Greedy Construction
    # We will iteratively pick the (m, r) that gives the highest gain.
    # To avoid getting stuck in a poor local optimum, we can randomize the selection among top candidates.
    
    while time.time() < deadline - 20.0:
        best_gain = -1
        best_m = None
        best_r = None
        candidates = []
        
        for m in divisors:
            if m in used_moduli:
                continue
            
            # For each residue r, calculate gain
            # Gain is the number of indices i in [r, r+m, ...] where cover_count[i] == 0
            # We can vectorize this check for all r? No, r changes the starting point.
            # But we can iterate r.
            # Optimization: If m is large, there are fewer indices.
            for r in range(m):
                indices = np.arange(r, N, m)
                # Count how many of these are currently uncovered
                gain = np.sum(cover_count[indices] == 0)
                
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
                    candidates = [(m, r, gain)]
                elif gain == best_gain and gain > 0:
                    candidates.append((m, r, gain))
        
        if best_m is None or best_gain <= 0:
            break
            
        # Randomly select from top candidates to diversify
        if len(candidates) > 1:
            # Use a softmax-like selection or just random choice from top few
            # For simplicity, pick random from top 10 if available
            top_k = min(10, len(candidates))
            # Sort candidates by gain descending (already mostly sorted, but let's be sure)
            candidates.sort(key=lambda x: x[2], reverse=True)
            chosen = random.choice(candidates[:top_k])
            best_m, best_r = chosen[0], chosen[1]
            
        apply_class(best_m, best_r, 1)
        selected_classes.append([best_m, best_r])
        used_moduli.add(best_m)
        
        current_score = get_score()
        if current_score < best_score:
            best_score = current_score
            best_classes = [c[:] for c in selected_classes]
            
        if best_score == 0:
            break
            
    # Phase 2: Local Search
    # We try to improve the current best solution (or the current greedy solution if it's better)
    # We will work on the current state (cover_count, selected_classes, used_moduli)
    # and try to improve it.
    
    # If the greedy solution was better, use that as the starting point for local search?
    # Actually, we should always keep the best state.
    # Let's restore the best state if it's different from current.
    if best_classes != selected_classes:
        cover_count = np.zeros(N, dtype=np.int32)
        selected_classes = [c[:] for c in best_classes]
        used_moduli = set(m for m, r in selected_classes)
        for m, r in selected_classes:
            apply_class(m, r, 1)
        
    current_score = get_score()
    improved = True
    no_improve_count = 0
    
    while time.time() < deadline - 5.0 and improved:
        improved = False
        no_improve_count += 1
        
        # Try all possible moves
        # Move Type 1: Change residue of an existing modulus
        for i, (m, r) in enumerate(selected_classes):
            # Try all other residues for this modulus
            for r_new in range(m):
                if r_new == r:
                    continue
                
                # Calculate gain of changing r to r_new
                # Remove old, add new
                indices_old = np.arange(r, N, m)
                indices_new = np.arange(r_new, N, m)
                
                # We need to calculate the change in score
                # Score = sum(cover_count == 0)
                # Change = (new_score - old_score)
                # We can simulate the change on a copy or calculate directly.
                # Direct calculation:
                # Positions in indices_old that become uncovered: cover_count[i] == 1
                # Positions in indices_new that become covered: cover_count[i] == 0
                # But indices_old and indices_new might overlap.
                # If they overlap, the net change at those positions is 0.
                
                # Let's use a temporary array for efficiency?
                # Or just calculate:
                # gain_old = sum(cover_count[indices_old] == 1)  # these will become 0 if not covered by others? No, they decrease by 1.
                # If cover_count[i] == 1, it becomes 0. So +1 to score (worse).
                # If cover_count[i] == 0, it stays 0.
                
                # loss_old = sum(cover_count[indices_old] == 1)
                
                # gain_new = sum(cover_count[indices_new] == 0)
                # If cover_count[i] == 0, it becomes 1. So -1 to score (better).
                
                # But we must be careful with overlaps.
                # Overlap set O = indices_old intersect indices_new
                # For i in O:
                #   cover_count[i] decreases by 1 then increases by 1 -> net 0.
                #   So we can ignore overlaps.
                
                # Let's create a mask for non-overlapping parts.
                # This is getting complex. 
                # Simpler: Apply, check score, revert.
                # Since N is small, this might be fast enough with NumPy.
                
                # Apply old removal
                cover_count[indices_old] -= 1
                # Apply new addition
                cover_count[indices_new] += 1
                
                new_score = get_score()
                
                # Revert
                cover_count[indices_new] -= 1
                cover_count[indices_old] += 1
                
                if new_score < current_score:
                    # Accept move
                    selected_classes[i] = [m, r_new]
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in selected_classes]
                    improved = True
                    no_improve_count = 0
                    break # Restart local search loop to re-evaluate all moves
            if improved:
                break
        
        if improved:
            continue
            
        # Move Type 2: Swap modulus
        # Try swapping an existing modulus m_old with an unused modulus m_new
        if no_improve_count > 10:
             # Random perturbation to escape local minima
             # Pick a random class and change its residue randomly
             if selected_classes:
                 idx = random.randint(0, len(selected_classes) - 1)
                 m, r = selected_classes[idx]
                 r_new = random.randint(0, m - 1)
                 if r_new != r:
                     indices_old = np.arange(r, N, m)
                     indices_new = np.arange(r_new, N, m)
                     cover_count[indices_old] -= 1
                     cover_count[indices_new] += 1
                     selected_classes[idx] = [m, r_new]
                     current_score = get_score()
                     if current_score < best_score:
                         best_score = current_score
                         best_classes = [c[:] for c in selected_classes]
                     improved = True
                     no_improve_count = 0
                     continue

        if no_improve_count > 50:
             break

    return best_classes if best_classes else []
