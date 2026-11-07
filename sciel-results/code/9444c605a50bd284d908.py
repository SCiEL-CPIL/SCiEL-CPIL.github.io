import time
import random
import math
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 50  # Leave 10s margin for safety

    # Precompute strict comparability matrix using numpy
    comp = np.zeros((total_masks, total_masks), dtype=np.uint8)
    
    for a in range(total_masks):
        for b in range(a + 1, total_masks):
            if (a & ~b) == 0 and a != b:
                comp[a, b] = 1
                comp[b, a] = 1
            elif (b & ~a) == 0 and b != a:
                comp[a, b] = 1
                comp[b, a] = 1
    
    # Multi-chain initialization
    family = []
    fam_set = set()
    
    # Start with empty set
    family.append(0)
    fam_set.add(0)
    
    # scores[s] = number of sets in family strictly comparable to s
    scores = np.zeros(total_masks, dtype=np.int32)
    
    def update_scores_for_addition(mask, delta):
        scores += delta * comp[:, mask]
    
    update_scores_for_addition(0, 1)
    
    # Greedily add sets
    while len(family) < m and time.time() < deadline:
        best_score = -1
        best_mask = -1
        
        for s in range(total_masks):
            if s in fam_set:
                continue
            if scores[s] > best_score:
                best_score = scores[s]
                best_mask = s
                
        if best_mask == -1:
            break
            
        family.append(best_mask)
        fam_set.add(best_mask)
        update_scores_for_addition(best_mask, 1)
    
    # Local Search with simulated annealing
    def get_total_score(fam_list):
        if len(fam_list) < 2:
            return 0
        fam_arr = np.array(fam_list, dtype=np.int32)
        sub_comp = comp[np.ix_(fam_arr, fam_arr)]
        return int(np.triu(sub_comp, k=1).sum())
    
    current_score = get_total_score(family)
    best_family = family[:]
    best_score = current_score
    
    # Simulated annealing
    T = 10.0
    cooling = 0.95
    
    while time.time() < deadline and T > 0.01:
        improved = False
        
        for i in range(len(family)):
            if time.time() > deadline:
                break
            A = family[i]
            fam_arr = np.array(family, dtype=np.int32)
            # Score of A with rest of family (comp[A, A] = 0, so this is correct)
            score_A_others = int(np.sum(comp[A, fam_arr]))
            
            best_gain = 0
            best_B = -1
            
            for B in range(total_masks):
                if B in fam_set:
                    continue
                # Score of B with family excluding position i (where A was)
                # = sum(comp[B, fam_arr]) - comp[B, A]
                score_B_with_family = int(np.sum(comp[B, fam_arr])) - int(comp[B, A])
                gain = score_B_with_family - score_A_others
                
                if gain > best_gain:
                    best_gain = gain
                    best_B = B
                    
            if best_B != -1:
                accept = False
                if best_gain > 0:
                    accept = True
                else:
                    if random.random() < math.exp(best_gain / T):
                        accept = True
                
                if accept:
                    fam_set.remove(A)
                    fam_set.add(best_B)
                    family[i] = best_B
                    
                    update_scores_for_addition(A, -1)
                    update_scores_for_addition(best_B, 1)
                    
                    new_score = get_total_score(family)
                    if new_score > best_score:
                        best_score = new_score
                        best_family = family[:]
                    
                    improved = True
        
        T *= cooling
    
    # Restore best family
    family = best_family
    
    return _convert(family)

def _convert(family_masks):
    result = []
    for mask in family_masks:
        s = []
        for b in range(12):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    return result
