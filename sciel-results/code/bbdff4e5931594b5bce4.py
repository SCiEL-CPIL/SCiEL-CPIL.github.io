"""Model-and-optimize approach for pairwise disjoint residue classes."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Precompute gcd table for moduli 20..200
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    
    # gcd_table[i][j] = gcd(moduli[i], moduli[j])
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Precompute sorted moduli for quick access
    sorted_moduli = sorted(moduli)
    
    def can_add(classes, m, r):
        """Check if class (m, r) is disjoint from all classes in the set."""
        mi = mod_idx[m]
        for (m2, r2) in classes:
            mj = mod_idx[m2]
            g = gcd_table[mi][mj]
            if (r - r2) % g == 0:
                return False
        return True
        
    def find_any_valid_residue(classes, m):
        """Find the first valid residue for modulus m given existing classes."""
        mi = mod_idx[m]
        for r in range(m):
            if can_add(classes, m, r):
                return r
        return None
        
    def find_random_valid_residue(classes, m, random_obj):
        """Find a random valid residue for modulus m given existing classes."""
        mi = mod_idx[m]
        valid_residues = []
        for r in range(m):
            if can_add(classes, m, r):
                valid_residues.append(r)
        if not valid_residues:
            return None
        return random_obj.choice(valid_residues)

    def score(classes):
        """Compute sum of 1/m for the set of classes."""
        total = 0.0
        for (m, r) in classes:
            total += 1.0 / m
        return total

    def greedy_construct(order=None, random_obj=None, randomize=False):
        """Greedy construction: add classes in the given order if possible."""
        if order is None:
            order = sorted_moduli
        classes = []
        used_moduli = set()
        for m in order:
            if m in used_moduli:
                continue
            if randomize and random_obj:
                r = find_random_valid_residue(classes, m, random_obj)
            else:
                r = find_any_valid_residue(classes, m)
            if r is not None:
                classes.append([m, r])
                used_moduli.add(m)
        return classes

    def local_search(classes, max_iter=200, random_obj=None):
        """Improve by trying to add missing moduli, swap residues, or swap moduli."""
        best_classes = [list(c) for c in classes]
        best_score = score(best_classes)
        used_moduli = set(m for m, r in best_classes)
        
        for iteration in range(max_iter):
            if time.time() - start_time > deadline - 3:
                break
            
            improved = False
            n_classes = len(best_classes)
            
            # 1. Try adding a new modulus (smallest first for best 1/m)
            for m in sorted_moduli:
                if m in used_moduli:
                    continue
                r = find_any_valid_residue(best_classes, m)
                if r is not None:
                    new_classes = best_classes + [[m, r]]
                    new_score = best_score + 1.0/m
                    if new_score > best_score + 1e-12:
                        best_classes = new_classes
                        best_score = new_score
                        used_moduli.add(m)
                        improved = True
                        break
            
            # 2. Try 1-for-2 swap: Remove m_old, Add m_new1, m_new2
            # This is often the most powerful move for increasing sum of reciprocals
            if not improved and n_classes > 0:
                # Sort classes by modulus descending to try removing larger ones first
                # Actually, we want to remove m_old where 1/m_old is small (m_old large)
                # and replace with smaller moduli.
                classes_sorted_by_m_desc = sorted(best_classes, key=lambda x: -x[0])
                
                found_1for2 = False
                for i in range(len(classes_sorted_by_m_desc)):
                    if time.time() - start_time > deadline - 3:
                        break
                    m_old, r_old = classes_sorted_by_m_desc[i]
                    temp_classes = [list(c) for j, c in enumerate(best_classes) if c[0] != m_old]
                    
                    # Try adding two new moduli
                    # Limit search to small moduli to keep it fast
                    candidate_mods = [m for m in sorted_moduli if m not in used_moduli and m != m_old]
                    
                    for m_new1 in candidate_mods[:30]: # Limit to top 30 smallest
                        if 1.0/m_new1 <= 1.0/m_old + 1e-12:
                             # Even if 1/m_new1 > 1/m_old, we need m_new2 to be valid too
                             # But if m_new1 is very large, m_new2 must be very small.
                             # We just check feasibility first.
                             pass
                        
                        r_new1 = find_any_valid_residue(temp_classes, m_new1)
                        if r_new1 is None:
                            continue
                        
                        temp_classes_with_1 = temp_classes + [[m_new1, r_new1]]
                        
                        for m_new2 in candidate_mods[:30]:
                            if m_new2 == m_new1:
                                continue
                            
                            # Check score improvement
                            gain = (1.0/m_new1 + 1.0/m_new2) - 1.0/m_old
                            if gain <= 1e-12:
                                continue
                            
                            r_new2 = find_any_valid_residue(temp_classes_with_1, m_new2)
                            if r_new2 is None:
                                continue
                            
                            new_classes = temp_classes_with_1 + [[m_new2, r_new2]]
                            new_score = best_score + gain
                            if new_score > best_score + 1e-12:
                                best_classes = new_classes
                                best_score = new_score
                                used_moduli = set(m for m, r in best_classes)
                                improved = True
                                found_1for2 = True
                                break
                        if found_1for2:
                            break
                    if found_1for2:
                        break

            # 3. Try 2-for-1 swap: Remove two large moduli, add one small modulus
            if not improved:
                missing = [m for m in sorted_moduli if m not in used_moduli]
                n_classes = len(best_classes)
                
                for m_new in missing[:10]:
                    if time.time() - start_time > deadline - 3:
                        break
                    
                    if m_new >= 200:
                        continue
                    
                    if 1.0/m_new < 2.0/200:
                        continue
                    
                    found_2for1 = False
                    for i in range(n_classes):
                        if time.time() - start_time > deadline - 3:
                            break
                        m_old1 = best_classes[i][0]
                        if m_old1 <= m_new:
                            continue
                        for j in range(i+1, n_classes):
                            m_old2 = best_classes[j][0]
                            if m_old2 <= m_new:
                                continue
                            if 1.0/m_new <= 1.0/m_old1 + 1.0/m_old2 + 1e-12:
                                continue
                            
                            temp_classes = [list(c) for k, c in enumerate(best_classes) if k != i and k != j]
                            r_new = find_any_valid_residue(temp_classes, m_new)
                            if r_new is not None:
                                new_classes = temp_classes + [[m_new, r_new]]
                                new_score = score(new_classes)
                                if new_score > best_score + 1e-12:
                                    best_classes = new_classes
                                    best_score = new_score
                                    used_moduli = set(m for m, r in best_classes)
                                    improved = True
                                    found_2for1 = True
                                    break
                        if found_2for1:
                            break
                    if found_2for1:
                        break
                        
            # 4. Try residue reassignment: change a residue to enable adding a missing modulus
            if not improved:
                missing = [m for m in sorted_moduli if m not in used_moduli]
                for m_new in missing[:10]:
                    if time.time() - start_time > deadline - 3:
                        break
                    
                    # Try changing one existing class's residue to make room for m_new
                    for i in range(len(best_classes)):
                        m_i, r_i = best_classes[i]
                        mi_idx = mod_idx[m_i]
                        mn_idx = mod_idx[m_new]
                        g_mn = gcd_table[mi_idx][mn_idx]
                        
                        if g_mn <= 1:
                            continue  # No constraint between them
                        
                        # Try all other residues for m_i
                        for r_try in range(m_i):
                            if r_try == r_i:
                                continue
                            
                            # Check if r_try is valid for m_i with other classes
                            valid = True
                            for j in range(len(best_classes)):
                                if j == i:
                                    continue
                                m_j, r_j = best_classes[j]
                                mj_idx = mod_idx[m_j]
                                g = gcd_table[mi_idx][mj_idx]
                                if (r_try - r_j) % g == 0:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            
                            # Check if m_new can be added with m_i having residue r_try
                            temp_classes = [list(c) for k, c in enumerate(best_classes) if k != i]
                            temp_classes.append([m_i, r_try])
                            r_new = find_any_valid_residue(temp_classes, m_new)
                            if r_new is not None:
                                new_classes = temp_classes + [[m_new, r_new]]
                                new_score = score(new_classes)
                                if new_score > best_score + 1e-12:
                                    best_classes = new_classes
                                    best_score = new_score
                                    used_moduli = set(m for m, r in best_classes)
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_classes

    # Multi-start approach
    best_classes = None
    best_score = 0.0
    
    # Start 1: Greedy with smallest moduli first (Deterministic)
    classes = greedy_construct()
    classes = local_search(classes, max_iter=300)
    s = score(classes)
    if s > best_score:
        best_score = s
        best_classes = [list(c) for c in classes]
    
    # Start 2+: Randomized Greedy with Local Search
    random.seed(42)
    num_starts = 20
    
    for start_idx in range(num_starts):
        if time.time() - start_time > deadline - 15:
            break
        
        # Create a slightly perturbed ordering
        order = list(sorted_moduli)
        # Randomly swap some adjacent elements to create diversity
        for _ in range(10):
            i = random.randint(0, len(order)-2)
            if random.random() < 0.5:
                order[i], order[i+1] = order[i+1], order[i]
        
        classes = greedy_construct(order=order, random_obj=random, randomize=True)
        
        if time.time() - start_time > deadline - 8:
            s = score(classes)
            if s > best_score:
                best_score = s
                best_classes = [list(c) for c in classes]
            continue
        
        classes = local_search(classes, max_iter=200, random_obj=random)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in classes]
            
    # Final polish: intensive local search on best solution
    if best_classes is not None:
        for round_num in range(5):
            if time.time() - start_time > deadline - 2:
                break
            max_iter = 400 + round_num * 100
            improved_classes = local_search(best_classes, max_iter=max_iter)
            improved_score = score(improved_classes)
            if improved_score > best_score + 1e-12:
                best_classes = improved_classes
                best_score = improved_score
            else:
                break
    
    return best_classes if best_classes is not None else []
# EVOLVE-BLOCK-END
