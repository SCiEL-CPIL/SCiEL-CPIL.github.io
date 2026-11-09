import time
import math
from fractions import Fraction

def construct():
    t0 = time.time()
    DEADLINE = 115.0
    MIN_M, MAX_M = 20, 200
    moduli = list(range(MIN_M, MAX_M + 1))
    n = len(moduli)
    
    # Precompute gcds
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            g = math.gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Precompute class masks
    # class_masks[m_j][g][k] = mask of residues in [0, m_j) congruent to k mod g
    # We store as dict of dict of list
    class_masks = {}
    for m_j in range(MIN_M, MAX_M + 1):
        class_masks[m_j] = {}
        for g in range(1, m_j + 1):
            masks_for_g = []
            for k in range(g):
                mask = 0
                for x in range(m_j):
                    if x % g == k:
                        mask |= (1 << x)
                masks_for_g.append(mask)
            class_masks[m_j][g] = masks_for_g

    # Initial masks
    masks = [0] * n
    for i, m in enumerate(moduli):
        masks[i] = (1 << m) - 1
        
    # Weights and suffix sums (using float for speed in pruning)
    weights = [1.0 / m for m in moduli]
    suffix_sum = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
        
    best_score = 0.0
    best_solution = []
    
    def search(idx, current_score, masks, solution):
        nonlocal best_score, best_solution
        
        if time.time() > DEADLINE:
            return
            
        if idx == n:
            if current_score > best_score:
                best_score = current_score
                best_solution = solution[:]
            return
            
        # Upper bound
        ub = current_score + suffix_sum[idx]
        if ub <= best_score:
            return
            
        m = moduli[idx]
        
        # Option 2: Include m (try first to find high score early)
        if masks[idx] != 0:
            # Try residues
            # We can optimize by checking which residues are allowed
            allowed_residues = []
            mm = masks[idx]
            while mm:
                lsb = mm & -mm
                r = lsb.bit_length() - 1
                allowed_residues.append(r)
                mm ^= lsb
                
            # Heuristic: Sort residues by impact (optional, might be slow)
            # For now, just iterate
            for r in allowed_residues:
                new_masks = masks[:]
                valid = True
                
                for j in range(idx + 1, n):
                    m_j = moduli[j]
                    g = gcds[idx][j]
                    if g == 1:
                        continue
                    k = r % g
                    mask_to_clear = class_masks[m_j][g][k]
                    
                    if new_masks[j] & mask_to_clear:
                        new_masks[j] &= ~mask_to_clear
                        if new_masks[j] == 0:
                            valid = False
                            break
                
                if valid:
                    solution.append((idx, r))
                    search(idx + 1, current_score + weights[idx], new_masks, solution)
                    solution.pop()
                    
        # Option 1: Skip m
        search(idx + 1, current_score, masks, solution)
        
    search(0, 0.0, masks, [])
    
    result = []
    for idx, r in best_solution:
        m = moduli[idx]
        result.append([m, r])
        
    return result
