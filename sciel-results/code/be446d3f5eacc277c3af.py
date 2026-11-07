import time
import random
import math
import numpy as np

P = {'n': 12, 'm': 40}

def construct():
    deadline = time.time() + 85
    n = P['n']
    m = P['m']
    N = 1 << n
    
    # Precompute comparability matrix as NumPy array
    # comp_mat_np[a][b] = True if a and b are comparable
    comp_mat_np = np.zeros((N, N), dtype=bool)
    for a in range(N):
        for b in range(N):
            if (a & b) == a or (a & b) == b:
                comp_mat_np[a][b] = True
    
    def count_comparable(family):
        count = 0
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                if comp_mat_np[family[i]][family[j]]:
                    count += 1
        return count

    def greedy_init(start_mask=None, seed=42):
        random.seed(seed)
        F = []
        fam_set = set()
        comp_in_F = np.zeros(N, dtype=int)
        
        if start_mask is None:
            start_mask = random.randint(0, N-1)
        
        F.append(start_mask)
        fam_set.add(start_mask)
        comp_in_F[comp_mat_np[:, start_mask]] += 1
        
        while len(F) < m:
            scores = comp_in_F.copy()
            # Mask out current family
            for y in fam_set:
                scores[y] = -1000000
            
            best_y = np.argmax(scores)
            best_val = scores[best_y]
            
            if best_val < 0:
                break
                
            F.append(best_y)
            fam_set.add(best_y)
            comp_in_F[comp_mat_np[:, best_y]] += 1
            
        return F

    def local_search(family, max_iters, deadline):
        # family is a list of masks
        comp_in_F = np.zeros(N, dtype=int)
        
        for z in family:
            comp_in_F[comp_mat_np[:, z]] += 1
        
        current_score = count_comparable(family)
        best_score = current_score
        best_family = family[:]
        
        fam_set = set(family)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            improved = False
            for i in range(m):
                x = family[i]
                k_x = comp_in_F[x] - 1
                
                scores = comp_in_F.copy()
                for y in fam_set:
                    scores[y] = -1000000
                
                # Subtract 1 for comparable to x
                scores[comp_mat_np[:, x]] -= 1
                
                best_y = np.argmax(scores)
                best_val = scores[best_y]
                
                if best_val > k_x:
                    # Update comp_in_F
                    comp_in_F[comp_mat_np[:, best_y]] += 1
                    comp_in_F[comp_mat_np[:, x]] -= 1
                    
                    fam_set.remove(x)
                    fam_set.add(best_y)
                    family[i] = best_y
                    current_score += best_val - k_x
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_family = family[:]
                    
                    improved = True
            
            if not improved:
                break
                
        return best_family, best_score

    best_overall = None
    best_score_overall = 0
    
    # Run multiple restarts with Greedy Init + Local Search
    for restart in range(100):
        if time.time() > deadline - 5:
            break
        
        start = random.randint(0, N-1)
        family = greedy_init(start, seed=100+restart)
        
        # Run local search with moderate iterations
        fam, score = local_search(family, 500, deadline)
        
        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam[:]
            
    if best_overall is None:
        best_overall = list(range(m))
        
    result = []
    for mask in best_overall:
        result.append([b for b in range(n) if (mask >> b) & 1])
        
    return result
