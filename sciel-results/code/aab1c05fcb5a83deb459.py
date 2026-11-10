import random
from math import gcd
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin

    # Candidate moduli: Even numbers in [20, 200]
    # This ensures gcd(m_i, m_j) >= 2 for all pairs, making disjointness possible.
    # Using only evens is a strong heuristic because it maximizes the chance of finding
    # compatible residues. Odd moduli often have gcd=1 with many others, forcing
    # stricter constraints.
    moduli = [m for m in range(20, 201) if m % 2 == 0]
    
    n = len(moduli)
    
    # Precompute gcds
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
            
    best_score = 0.0
    best_set = []
    
    # Helper to check if a residue r for modulus at index i is valid given selected indices and residues
    def is_valid(i, r, sel_indices, sel_residues):
        for k in range(len(sel_indices)):
            idx = sel_indices[k]
            r2 = sel_residues[k]
            g = gcds[i][idx]
            if (r - r2) % g == 0:
                return False
        return True

    # Helper to find a valid residue for index i, or None if none exists
    def find_valid_residue(i, sel_indices, sel_residues):
        m = moduli[i]
        # Try 0 first, then random? Or just iterate.
        # Iterating is deterministic and fast enough for m<=200.
        # To add diversity in restarts, we can shuffle the order of checking residues.
        # But for now, let's stick to a simple check.
        for r in range(m):
            if is_valid(i, r, sel_indices, sel_residues):
                return r
        return None

    # Run multiple random restarts
    # We use randomization to escape local minima in the greedy assignment
    for restart in range(5000):
        if time.time() > deadline:
            break
            
        selected_indices = []
        selected_residues = []
        
        # Order of processing:
        # 1. Smallest moduli first (greedy for max sum)
        # 2. Randomized order
        # 3. Randomized order with small bias towards small moduli
        
        order = list(range(n))
        
        # Strategy 1: Pure greedy (small m first)
        # Strategy 2: Random
        # Strategy 3: Random but prioritize small m with some probability
        
        if restart % 3 == 0:
            pass # Already sorted ascending
        elif restart % 3 == 1:
            random.shuffle(order)
        else:
            # Weighted shuffle: smaller moduli have higher chance of being earlier
            # Simple approach: sort by random key + small penalty for large m
            # Or just shuffle and then sort by (random_key, m) with some noise?
            # Let's just shuffle, but ensure the first few are small moduli.
            # Pick 5 small moduli randomly from the first 20, put them first, shuffle rest.
            small_pool = list(range(min(20, n)))
            random.shuffle(small_pool)
            large_pool = list(range(min(20, n), n))
            random.shuffle(large_pool)
            order = small_pool + large_pool
            random.shuffle(order) # This ruins the priority. 
            # Better:
            # order = small_pool[:3] + [x for x in order if x not in small_pool[:3]]
            # Let's stick to simple shuffle for now, the greedy base is strong.
            random.shuffle(order)

        for i in order:
            m = moduli[i]
            r = find_valid_residue(i, selected_indices, selected_residues)
            if r is not None:
                selected_indices.append(i)
                selected_residues.append(r)
                
        # Calculate score
        score = 0.0
        for idx in selected_indices:
            score += 1.0 / moduli[idx]
            
        if score > best_score:
            best_score = score
            best_set = [[moduli[idx], selected_residues[k]] for k, idx in enumerate(selected_indices)]
            
    # Local Search / Refinement on the best set found
    if best_set and time.time() < deadline:
        # Convert best_set to indices and residues for easier manipulation
        # best_set is [[m, r], ...]
        # We need to map back to indices in 'moduli'
        # Since moduli are distinct, we can find index by value
        
        # Reconstruct selected_indices and selected_residues from best_set
        curr_sel_indices = []
        curr_sel_residues = []
        for m, r in best_set:
            # Find index
            idx = moduli.index(m)
            curr_sel_indices.append(idx)
            curr_sel_residues.append(r)
            
        curr_score = best_score
        
        # Try to add missing moduli by swapping or perturbing
        selected_moduli_set = set(moduli[i] for i in curr_sel_indices)
        missing_indices = [i for i in range(n) if moduli[i] not in selected_moduli_set]
        
        # Sort missing by value (1/m) descending -> m ascending
        missing_indices.sort(key=lambda i: moduli[i])
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add each missing modulus
            # We can try to add it directly if possible (shouldn't be possible if greedy failed, 
            # but maybe due to order? No, if it's missing, greedy couldn't place it.
            # However, if we change residues of existing ones, it might become possible.)
            
            for target_idx in missing_indices[:10]: # Focus on best missing
                if time.time() > deadline:
                    break
                
                # Try to find a valid residue for target_idx given current set
                # This will likely fail.
                # So, we try to perturb an existing residue to make room.
                
                # Pick a random existing index to perturb
                if not curr_sel_indices:
                    break
                
                victim_pos = random.randint(0, len(curr_sel_indices) - 1)
                victim_idx = curr_sel_indices[victim_pos]
                old_victim_r = curr_sel_residues[victim_pos]
                
                # Try a few random new residues for the victim
                found = False
                for _ in range(20):
                    new_r = random.randint(0, moduli[victim_idx] - 1)
                    if new_r == old_victim_r:
                        continue
                    
                    # Temporarily set victim to new_r
                    curr_sel_residues[victim_pos] = new_r
                    
                    # Check if target can be added
                    target_r = find_valid_residue(target_idx, curr_sel_indices, curr_sel_residues)
                    if target_r is not None:
                        # We can add target!
                        # Update state
                        curr_sel_indices.append(target_idx)
                        curr_sel_residues.append(target_r)
                        
                        # Update score
                        curr_score += 1.0 / moduli[target_idx]
                        
                        # Update missing list
                        if target_idx in missing_indices:
                            missing_indices.remove(target_idx)
                        
                        improved = True
                        found = True
                        break
                    
                    # Revert
                    curr_sel_residues[victim_pos] = old_victim_r
                
                if found:
                    # Restart the inner loop to try adding more
                    break

        # Final update if improved
        if curr_score > best_score:
            best_score = curr_score
            best_set = [[moduli[idx], r] for idx, r in zip(curr_sel_indices, curr_sel_residues)]

    return best_set
