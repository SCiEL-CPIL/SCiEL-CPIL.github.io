import time
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute comparability matrix
    masks = np.arange(total_masks)
    i_exp = masks[:, None]
    j_exp = masks[None, :]
    
    # i subset j: (i & j) == i
    i_sub_j = (i_exp & j_exp) == i_exp
    j_sub_i = (i_exp & j_exp) == j_exp
    comp_matrix = (i_sub_j | j_sub_i).astype(np.uint8)
    
    # comp_matrix[i][j] is 1 if i and j are comparable (including i=j)
    
    def count_score(family):
        # family is numpy array of ints
        sub = comp_matrix[np.ix_(family, family)]
        return int((np.sum(sub) - len(family)) // 2)

    best_score = 0
    best_family = np.array([], dtype=int)
    
    deadline = time.time() + 85
    
    # Helper to generate a random chain
    def random_chain():
        chain = [0]
        cur = 0
        bits = list(range(n))
        np.random.shuffle(bits)
        for b in bits:
            cur |= (1 << b)
            chain.append(cur)
        return chain

    # Multiple restarts
    for restart in range(500): # Upper bound, will break on time
        if time.time() > deadline: break
        
        # Initialization
        # Start with a random chain (length 13)
        chain = random_chain()
        # Fill the rest with random sets
        # Ensure no duplicates
        family_list = chain[:]
        used = set(family_list)
        
        # Greedily add sets that maximize comparable pairs? 
        # Or just random? Random is faster to start.
        # Let's try random first, then local search will fix it.
        while len(family_list) < m:
            s = np.random.randint(total_masks)
            if s not in used:
                family_list.append(s)
                used.add(s)
        
        family = np.array(family_list, dtype=int)
        current_score = count_score(family)
        
        if current_score > best_score:
            best_score = current_score
            best_family = family.copy()
            
        # Local Search with Simulated Annealing
        # Maintain scores_all incrementally
        # scores_all[s] = sum(comp_matrix[s][family])
        
        # Initial computation of scores_all
        scores_all = comp_matrix[:, family].sum(axis=1).astype(np.int32)
        
        temp = 5.0
        cooling = 0.99
        min_temp = 0.1
        
        # We need to be careful with the "mask" of used elements.
        # Instead of creating a boolean mask array every time, 
        # we can just check if the best_s is in family?
        # No, we need to exclude family from candidates.
        # We can set scores_all[family] = -inf? 
        # But scores_all is updated incrementally.
        # Better: Keep a set of family elements? 
        # Or just mask the vals vector.
        
        iterations = 0
        while temp > min_temp:
            if time.time() > deadline: break
            iterations += 1
            
            # Pick random index to replace
            i = np.random.randint(m)
            old_mask = family[i]
            
            # contrib_old = sum(comp[old_mask][family[j]] for j != i)
            # scores_all[old_mask] = sum(comp[old_mask][family])
            # Since comp[old_mask][old_mask] = 1, 
            # contrib_old = scores_all[old_mask] - 1
            
            contrib_old = scores_all[old_mask] - 1
            
            # We want to find s not in family that maximizes:
            # new_contrib = sum(comp[s][family[j]] for j != i)
            # = scores_all[s] - comp[s][old_mask]
            
            # vals[s] = scores_all[s] - comp_matrix[s, old_mask]
            # We need s not in family.
            
            # Compute vals for all s
            # To avoid allocating a new array every time, we can reuse a buffer?
            # Or just compute it. 4096 is small.
            vals = scores_all - comp_matrix[:, old_mask]
            
            # Mask out family members
            # Create a temporary array to mask
            # We can do: vals[family] = -np.inf
            # But this modifies vals.
            
            vals[family] = -np.inf
            
            best_s = np.argmax(vals)
            best_val = vals[best_s]
            
            if best_val == -np.inf:
                break # No valid move
            
            delta = best_val - contrib_old
            
            # Acceptance
            if delta > 0:
                accept = True
            else:
                if np.random.rand() < np.exp(delta / temp):
                    accept = True
                else:
                    accept = False
            
            if accept:
                # Update family
                family[i] = best_s
                
                # Update scores_all
                # scores_all[k] = sum(comp[k][family])
                # New family has old_mask removed, best_s added.
                # scores_all_new[k] = scores_all[k] - comp[k, old_mask] + comp[k, best_s]
                
                scores_all -= comp_matrix[:, old_mask]
                scores_all += comp_matrix[:, best_s]
                
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_family = family.copy()
            
            temp *= cooling

    # Convert best_family to list of lists
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
        
    return result
