"""Model-and-optimize approach for pairwise disjoint residue classes."""
# EVOLVE-BLOCK-START
import time
import math
import random
from math import gcd
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Moduli range
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
    
    def can_add(classes, m, r):
        """Check if class (m, r) is disjoint from all classes in the set."""
        mi = mod_idx[m]
        for (m2, r2) in classes:
            mj = mod_idx[m2]
            g = gcd_table[mi][mj]
            if (r - r2) % g == 0:
                return False
        return True
    
    def find_best_residue(classes, m, prefer_max_slack=True):
        """Find the best residue for modulus m given existing classes."""
        mi = mod_idx[m]
        valid_residues = []
        for r in range(m):
            if can_add(classes, m, r):
                valid_residues.append(r)
        
        if not valid_residues:
            return None, 0
        
        if len(valid_residues) == 1:
            return valid_residues[0], m
        
        # Score each valid residue by slack (distance to nearest conflicting residue)
        best_r = valid_residues[0]
        best_slack = -1
        
        for r in valid_residues:
            min_dist = m  # large initial value
            for (m2, r2) in classes:
                mj = mod_idx[m2]
                g = gcd_table[mi][mj]
                if g > 1:
                    diff = (r - r2) % g
                    dist = min(diff, g - diff)
                    if dist < min_dist:
                        min_dist = dist
                        if min_dist <= best_slack:
                            break
            if min_dist > best_slack:
                best_slack = min_dist
                best_r = r
        
        return best_r, best_slack
    
    def score(classes):
        """Compute sum of 1/m for the set of classes."""
        total = 0.0
        for (m, r) in classes:
            total += 1.0 / m
        return total
    
    def greedy_construct(order=None):
        """Greedy construction: add classes in the given order if possible."""
        if order is None:
            order = sorted(moduli)  # smallest moduli first
        classes = []
        used_moduli = set()
        for m in order:
            if m in used_moduli:
                continue
            r, slack = find_best_residue(classes, m)
            if r is not None:
                classes.append([m, r])
                used_moduli.add(m)
        return classes
    
    def try_add_all(classes):
        """Try to add all remaining moduli, starting with smallest."""
        used_moduli = set(m for m, r in classes)
        remaining = [m for m in sorted(moduli) if m not in used_moduli]
        
        for m in remaining:
            if time.time() - start_time > deadline - 2:
                break
            r, slack = find_best_residue(classes, m)
            if r is not None:
                classes.append([m, r])
                used_moduli.add(m)
        
        return classes
    
    def local_search(classes, max_iter=1000):
        """Improve by trying to add missing moduli, swap residues, or swap moduli."""
        best_classes = [list(c) for c in classes]
        best_score = score(best_classes)
        used_moduli = set(m for m, r in best_classes)
        
        for iteration in range(max_iter):
            if time.time() - start_time > deadline - 3:
                break
            
            improved = False
            
            # 1. Try adding a new modulus (smallest first for max gain)
            for m in sorted(moduli):
                if m in used_moduli:
                    continue
                r, slack = find_best_residue(best_classes, m)
                if r is not None:
                    new_classes = best_classes + [[m, r]]
                    new_score = best_score + 1.0/m
                    if new_score > best_score + 1e-12:
                        best_classes = new_classes
                        best_score = new_score
                        used_moduli.add(m)
                        improved = True
                        break
            
            # 2. Try 1-for-1 swap: remove m_old, add m_new if 1/m_new > 1/m_old
            if not improved:
                missing = [m for m in sorted(moduli) if m not in used_moduli]
                
                for m_new in missing:
                    if time.time() - start_time > deadline - 3:
                        break
                    
                    found_swap = False
                    for i in range(len(best_classes)):
                        m_old, r_old = best_classes[i]
                        if m_old <= m_new:
                            continue
                        
                        temp_classes = [list(c) for c in best_classes if c[0] != m_old]
                        r_new, _ = find_best_residue(temp_classes, m_new)
                        if r_new is not None:
                            new_classes = temp_classes + [[m_new, r_new]]
                            new_score = score(new_classes)
                            if new_score > best_score + 1e-12:
                                best_classes = new_classes
                                best_score = new_score
                                used_moduli = set(m for m, r in best_classes)
                                improved = True
                                found_swap = True
                                break
                    
                    if found_swap:
                        break
            
            # 3. Try 2-for-1 swap: remove two moduli, add one if net gain positive
            if not improved:
                missing = [m for m in sorted(moduli) if m not in used_moduli]
                n = len(best_classes)
                
                for m_new in missing:
                    if time.time() - start_time > deadline - 3:
                        break
                    if 1.0/m_new < 1e-5:  # very small gain, skip
                        continue
                    
                    found_2swap = False
                    for i in range(n):
                        for j in range(i+1, n):
                            if time.time() - start_time > deadline - 3:
                                break
                            m_old1 = best_classes[i][0]
                            m_old2 = best_classes[j][0]
                            # Only consider if net gain is positive
                            if 1.0/m_new <= 1.0/m_old1 + 1.0/m_old2 + 1e-12:
                                continue
                            
                            temp_classes = [list(c) for k, c in enumerate(best_classes) if k != i and k != j]
                            r_new, _ = find_best_residue(temp_classes, m_new)
                            if r_new is not None:
                                new_classes = temp_classes + [[m_new, r_new]]
                                new_score = score(new_classes)
                                if new_score > best_score + 1e-12:
                                    best_classes = new_classes
                                    best_score = new_score
                                    used_moduli = set(m for m, r in best_classes)
                                    improved = True
                                    found_2swap = True
                                    break
                        if found_2swap:
                            break
                    if found_2swap:
                        break
            
            # 4. Residue re-optimization: try to improve slack for each class
            if not improved:
                for i in range(len(best_classes)):
                    m, r = best_classes[i]
                    other_classes = [c for j, c in enumerate(best_classes) if j != i]
                    new_r, _ = find_best_residue(other_classes, m)
                    if new_r is not None and new_r != r:
                        # Accept if it improves the minimum slack across all pairs
                        # This is a neutral move in score but may help future additions
                        # Try it and see if it enables an addition
                        temp_classes = [list(c) for j, c in enumerate(best_classes) if j != i] + [[m, new_r]]
                        # Check if any missing modulus can now be added
                        used_temp = set(m2 for m2, r2 in temp_classes)
                        for m_add in sorted(moduli):
                            if m_add in used_temp:
                                continue
                            r_add, _ = find_best_residue(temp_classes, m_add)
                            if r_add is not None:
                                best_classes = temp_classes + [[m_add, r_add]]
                                best_score = score(best_classes)
                                used_moduli = set(m2 for m2, r2 in best_classes)
                                improved = True
                                break
                        if improved:
                            break
                        else:
                            # Still accept the residue change if it improves local slack
                            # (helps future searches)
                            best_classes[i] = [m, new_r]
            
            # 5. New: Try removing one modulus and re-adding multiple missing ones
            if not improved:
                n = len(best_classes)
                for i in range(n):
                    if time.time() - start_time > deadline - 3:
                        break
                    m_remove = best_classes[i][0]
                    temp_classes = [list(c) for j, c in enumerate(best_classes) if j != i]
                    used_temp = set(m2 for m2, r2 in temp_classes)
                    
                    # Try adding back multiple missing moduli
                    added_any = False
                    for m_add in sorted(moduli):
                        if m_add in used_temp:
                            continue
                        r_add, _ = find_best_residue(temp_classes, m_add)
                        if r_add is not None:
                            temp_classes.append([m_add, r_add])
                            used_temp.add(m_add)
                            added_any = True
                    
                    if added_any:
                        new_score = score(temp_classes)
                        if new_score > best_score + 1e-12:
                            best_classes = temp_classes
                            best_score = new_score
                            used_moduli = set(m for m, r in best_classes)
                            improved = True
                            break
            
            if not improved:
                break
        
        return best_classes
    
    # Multi-start approach
    best_classes = None
    best_score = 0.0
    
    # Start 1: Greedy with smallest moduli first
    classes = greedy_construct()
    classes = try_add_all(classes)
    classes = local_search(classes, max_iter=2000)
    s = score(classes)
    if s > best_score:
        best_score = s
        best_classes = [list(c) for c in classes]
    
    # Additional starts with different orderings
    random.seed(42)
    num_starts = 80
    
    for start_idx in range(num_starts):
        if time.time() - start_time > deadline - 25:
            break
        
        # Create a random ordering biased toward small moduli
        remaining = list(moduli)
        order = []
        while remaining:
            alpha = 1.0 + start_idx * 0.08
            weights = [1.0 / (m ** alpha) for m in remaining]
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0
            chosen_idx = 0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen_idx = i
                    break
            order.append(remaining[chosen_idx])
            remaining.pop(chosen_idx)
        
        classes = greedy_construct(order)
        classes = try_add_all(classes)
        
        if time.time() - start_time > deadline - 15:
            s = score(classes)
            if s > best_score:
                best_score = s
                best_classes = [list(c) for c in classes]
            continue
        
        classes = local_search(classes, max_iter=800)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in classes]
    
    # Final polish with more iterations
    if best_classes is not None:
        for round_num in range(8):
            if time.time() - start_time > deadline - 2:
                break
            max_iter = 1000 + round_num * 500
            improved_classes = local_search(best_classes, max_iter=max_iter)
            improved_score = score(improved_classes)
            if improved_score > best_score + 1e-12:
                best_classes = improved_classes
                best_score = improved_score
            else:
                break
    
    # Final high-precision polish: exhaustive residue search for small sets
    if best_classes is not None and time.time() - start_time < deadline - 5:
        n_classes = len(best_classes)
        if n_classes <= 30:
            # Try all possible residues for each class in the current set
            # and see if we can improve the score by enabling more additions
            current_score = score(best_classes)
            for _ in range(3):
                if time.time() - start_time > deadline - 2:
                    break
                improved = False
                for i in range(n_classes):
                    m_i, r_i = best_classes[i]
                    other_classes = [c for j, c in enumerate(best_classes) if j != i]
                    # Try all valid residues for m_i
                    for r_try in range(m_i):
                        if not can_add(other_classes, m_i, r_try):
                            continue
                        temp_classes = [list(c) for j, c in enumerate(best_classes) if j != i] + [[m_i, r_try]]
                        # Check if this enables adding more moduli
                        used_temp = set(m2 for m2, r2 in temp_classes)
                        for m_add in sorted(moduli):
                            if m_add in used_temp:
                                continue
                            r_add, _ = find_best_residue(temp_classes, m_add)
                            if r_add is not None:
                                new_classes = temp_classes + [[m_add, r_add]]
                                new_score = score(new_classes)
                                if new_score > current_score + 1e-12:
                                    best_classes = new_classes
                                    current_score = new_score
                                    n_classes = len(best_classes)
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if not improved:
                    break
    
    return best_classes if best_classes is not None else []
# EVOLVE-BLOCK-END
