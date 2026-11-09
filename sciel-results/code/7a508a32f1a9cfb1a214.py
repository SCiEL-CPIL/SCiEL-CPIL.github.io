# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    # Moduli 20..200
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    
    # Precompute gcd table
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute forbidden masks: for each m and each possible (m2, r2), 
    # store the bitmask of forbidden residues in m
    # forbidden_mask[m][m2][r2] = bitmask of r in [0,m) where r ≡ r2 (mod gcd(m,m2))
    # Only compute for gcd > 1
    forbidden_masks = {}
    for m in moduli:
        forbidden_masks[m] = {}
        for m2 in moduli:
            if m2 == m:
                continue
            g = gcd(m, m2)
            if g == 1:
                # All residues forbidden
                forbidden_masks[m][m2] = None  # None means all forbidden
                continue
            # Precompute masks for each r2
            masks = []
            for r2 in range(m2):
                mask = 0
                r0 = r2 % g
                for r in range(r0, m, g):
                    mask |= (1 << r)
                masks.append(mask)
            forbidden_masks[m][m2] = masks
    
    def compute_valid_mask_fast(m, existing_classes):
        """Fast bitmask of valid residues for modulus m given existing classes."""
        m_val = m
        all_mask = (1 << m_val) - 1
        valid = all_mask
        
        for (m2, r2) in existing_classes:
            if m2 not in forbidden_masks[m]:
                return 0, 0
            masks = forbidden_masks[m][m2]
            if masks is None:
                return 0, 0
            valid &= ~masks[r2]
            if valid == 0:
                return 0, 0
        
        count = bin(valid).count('1')
        return valid, count
    
    def get_valid_residues(valid_mask, m):
        """Get list of valid residues from bitmask."""
        residues = []
        for r in range(m):
            if (valid_mask >> r) & 1:
                residues.append(r)
        return residues
    
    def pick_best_residue(valid_mask, m, existing_classes):
        """Pick a residue from valid_mask that maximizes flexibility for future additions."""
        if valid_mask == 0:
            return None
        
        valid_residues = get_valid_residues(valid_mask, m)
        
        if len(valid_residues) == 1:
            return valid_residues[0]
        
        # For each candidate, estimate flexibility: count total valid residues available
        # for all unused moduli if we pick this residue
        used_moduli = set(m2 for m2, r2 in existing_classes)
        
        # Limit evaluation to a subset if too many candidates
        if len(valid_residues) > 30:
            step = max(1, len(valid_residues) // 30)
            candidates = valid_residues[::step]
        else:
            candidates = valid_residues
        
        best_r = candidates[0]
        best_score = -1
        
        for r_cand in candidates:
            # Estimate: sum over unused moduli of (number of valid residues they'd have)
            flex = 0
            for m2 in moduli:
                if m2 in used_moduli or m2 == m:
                    continue
                masks = forbidden_masks[m2].get(m)
                if masks is None:
                    continue
                mask = masks[r_cand]
                n_forbidden = bin(mask).count('1')
                flex += (m2 - n_forbidden)
            
            if flex > best_score:
                best_score = flex
                best_r = r_cand
        
        return best_r
    
    def score_float(classes):
        """Fast float score."""
        total = 0.0
        for (m, r) in classes:
            total += 1.0 / m
        return total
    
    def score_exact(classes):
        """Exact rational score."""
        total = Fraction(0, 1)
        for (m, r) in classes:
            total += Fraction(1, m)
        return total
    
    # Greedy construction with priority on small moduli
    def greedy_construct(order=None):
        if order is None:
            order = sorted(moduli)  # smallest first
        classes = []
        for m in order:
            valid_mask, count = compute_valid_mask_fast(m, classes)
            if count > 0:
                r = pick_best_residue(valid_mask, m, classes)
                classes.append([m, r])
        return classes
    
    def local_search(classes, max_iter=2000, time_limit=None):
        """Local search: try to improve by adding missing small moduli, or swapping."""
        if time_limit is None:
            time_limit = deadline
        best = [list(c) for c in classes]
        best_score = score_float(best)
        used_moduli = set(m for m, r in best)
        
        for it in range(max_iter):
            if time.time() > time_limit:
                break
            
            improved = False
            
            # Try to add any unused modulus (sorted by 1/m descending = m ascending)
            for m in sorted(moduli):
                if m in used_moduli:
                    continue
                valid_mask, count = compute_valid_mask_fast(m, best)
                if count > 0:
                    r = pick_best_residue(valid_mask, m, best)
                    new_classes = best + [[m, r]]
                    new_score = score_float(new_classes)
                    if new_score > best_score:
                        best = new_classes
                        best_score = new_score
                        used_moduli.add(m)
                        improved = True
                        break
            
            if not improved:
                # Try 1-swap: remove one class, see if we can add two better ones
                for i in range(len(best)):
                    if time.time() > time_limit:
                        break
                    m_rem, r_rem = best[i]
                    temp = [best[j] for j in range(len(best)) if j != i]
                    temp_used = set(m for m, r in temp)
                    added = 0
                    # Try to add up to 2 moduli
                    for m in sorted(moduli):
                        if m in temp_used or m == m_rem:
                            continue
                        valid_mask, count = compute_valid_mask_fast(m, temp)
                        if count > 0:
                            r = pick_best_residue(valid_mask, m, temp)
                            temp = temp + [[m, r]]
                            temp_used.add(m)
                            added += 1
                            if added == 2:
                                break
                    if added > 0:
                        new_score = score_float(temp)
                        if new_score > best_score:
                            best = temp
                            best_score = new_score
                            used_moduli = set(m for m, r in best)
                            improved = True
                            break
            
            if not improved:
                break
        
        return best
    
    # Multi-start search
    best_classes = None
    best_score = Fraction(0, 1)
    
    # Start 1: pure greedy, smallest moduli first
    classes = greedy_construct()
    classes = local_search(classes, max_iter=500)
    s = score_exact(classes)
    if s > best_score:
        best_score = s
        best_classes = [list(c) for c in classes]
    
    # Multiple randomized starts
    random.seed(42)
    num_starts = 30
    for start in range(num_starts):
        if time.time() > deadline:
            break
        
        # Create weighted random ordering: strongly favor small moduli
        alpha = 1.0 + 0.5 * random.random()
        weights = [1.0 / (m ** alpha) for m in moduli]
        
        # Weighted sampling without replacement using cumulative sums
        order = []
        remaining = list(zip(moduli, weights))
        while remaining:
            total_w = sum(w for _, w in remaining)
            if total_w <= 0:
                order.append(remaining[0][0])
                remaining.pop(0)
                continue
            r = random.random() * total_w
            cum = 0
            chosen_idx = 0
            for i, (m, w) in enumerate(remaining):
                cum += w
                if r <= cum:
                    chosen_idx = i
                    break
            order.append(remaining[chosen_idx][0])
            remaining.pop(chosen_idx)
        
        classes = greedy_construct(order)
        classes = local_search(classes, max_iter=200)
        s = score_exact(classes)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in classes]
    
    # Final intensive local search on best
    if best_classes is not None:
        best_classes = local_search(best_classes, max_iter=2000)
    
    return best_classes if best_classes is not None else []
# EVOLVE-BLOCK-END
