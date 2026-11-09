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
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
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
    
    # For each pair of moduli, precompute which residues are compatible
    # compatible[i][j] is a set of residues r_i in [0, moduli[i]) that are compatible
    # with each residue r_j in [0, moduli[j])
    # Actually, let's think differently.
    
    # The constraint: for classes (m1, r1) and (m2, r2), we need r1 != r2 mod gcd(m1, m2)
    # So for a given modulus m and residue r, it conflicts with (m2, r2) if r ≡ r2 (mod gcd(m, m2))
    
    # Strategy: This is essentially a constraint satisfaction / optimization problem.
    # We want to select a subset of moduli and assign residues to maximize sum of 1/m.
    
    # Key insight: If two moduli are coprime (gcd=1), then ANY residue is valid for both
    # because r1 != r2 mod 1 is always true (since there's only 1 residue class mod 1).
    # So coprime moduli are always compatible regardless of residues.
    
    # For moduli with gcd > 1, we need to avoid r1 ≡ r2 (mod g).
    
    # Model: We can think of this as a graph problem where nodes are (m, r) pairs,
    # but that's too large. Instead, let's use a different approach.
    
    # Key observation: For a given set of moduli, we can check if there exist valid residues
    # using a backtracking or constraint propagation approach.
    
    # Better approach: Since we want to maximize sum of 1/m, we should prefer smaller moduli.
    # The optimal solution likely includes many small moduli.
    
    # Let's use a greedy + local search approach with a smarter residue selection.
    
    def can_add(classes, m, r):
        """Check if class (m, r) is disjoint from all classes in the set."""
        mi = mod_idx[m]
        for (m2, r2) in classes:
            mj = mod_idx[m2]
            g = gcd_table[mi][mj]
            if (r - r2) % g == 0:
                return False
        return True
    
    def find_best_residue(classes, m):
        """Find the best residue for modulus m given existing classes.
        Uses a forward-looking heuristic: minimize conflicts with potential future moduli."""
        mi = mod_idx[m]
        valid_residues = []
        for r in range(m):
            if can_add(classes, m, r):
                valid_residues.append(r)
        
        if not valid_residues:
            return None, 0
        
        if len(valid_residues) == 1:
            return valid_residues[0], 0
        
        # For each candidate residue, estimate how many other moduli it would block
        # A residue r for modulus m blocks modulus m2 if for all residues r2,
        # there exists some class (m3, r3) such that... actually simpler:
        # r blocks m2 if the set {r mod gcd(m,m2)} is the only residue class mod gcd(m,m2)
        # that's compatible with all existing classes.
        # Simpler proxy: count how many other moduli m2 have gcd(m,m2) > 1 and
        # where r mod gcd(m,m2) is a "rare" compatible residue.
        
        # Even simpler: for each candidate r, count the number of moduli m2 (not in classes)
        # where gcd(m, m2) > 1 and the residue class r mod gcd(m,m2) conflicts with
        # at least one existing class's residue mod gcd(m2, existing_m).
        # This is complex. Let's use a lighter heuristic:
        # Score = sum over existing classes of (1/gcd(m, m2)) * indicator(r conflicts with "many" future moduli)
        
        # Lightest useful heuristic: minimize the sum of 1/gcd(m,m2) over existing classes
        # where the residue r is "close" to conflicting (small distance in the gcd space)
        # This is similar to slack but weighted by inverse gcd (more weight for larger gcds)
        
        best_r = valid_residues[0]
        best_score = float('inf')
        
        for r in valid_residues:
            s = 0.0
            for (m2, r2) in classes:
                mj = mod_idx[m2]
                g = gcd_table[mi][mj]
                if g > 1:
                    diff = (r - r2) % g
                    dist = min(diff, g - diff)
                    # Weight by 1/g: larger gcd means more constraint, so small distance is worse
                    s += (1.0 / g) * (1.0 / max(dist, 1))
            if s < best_score:
                best_score = s
                best_r = r
        
        return best_r, best_score
    
    def score(classes):
        """Compute sum of 1/m for the set of classes."""
        total = 0.0
        for (m, r) in classes:
            total += 1.0 / m
        return total
    
    def score_exact(classes):
        """Compute exact score using Fraction."""
        total = Fraction(0, 1)
        for (m, r) in classes:
            total += Fraction(1, m)
        return total
    
    def greedy_construct(order=None):
        """Greedy construction: add classes in the given order if possible."""
        if order is None:
            order = sorted(moduli, key=lambda m: 1.0/m)  # smallest moduli first (highest 1/m)
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
    
    def try_add_all(classes, max_time=5.0):
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
    
    def local_search(classes, max_iter=200):
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
            
            # 2. Try Swap-Add: Remove m_old, Add m_new if 1/m_new > 1/m_old
            if not improved:
                missing = [m for m in sorted(moduli) if m not in used_moduli]
                
                for m_new in missing:
                    if time.time() - start_time > deadline - 3:
                        break
                    
                    found_swap = False
                    # Only consider removing m_old where m_old > m_new (so 1/m_new > 1/m_old)
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

            # 3. Try 2-for-1 swap: Remove two large moduli, add one small modulus
            # This can escape local minima where no single swap works
            if not improved:
                missing = [m for m in sorted(moduli) if m not in used_moduli]
                n_classes = len(best_classes)
                
                for m_new in missing[:10]:  # Only try top 10 missing moduli
                    if time.time() - start_time > deadline - 3:
                        break
                    
                    if m_new >= 200:
                        continue
                    
                    # Find pairs of existing moduli whose combined removal allows adding m_new
                    # Only consider pairs where both m_old > m_new (so 2/m_new > 1/m_old1 + 1/m_old2 is possible)
                    # Actually we need 1/m_new > 1/m_old1 + 1/m_old2 for improvement
                    # So m_new must be quite small.
                    if 1.0/m_new < 2.0/200:  # Even removing two smallest possible won't help
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
                            # Check if 1/m_new > 1/m_old1 + 1/m_old2
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
                                    found_2for1 = True
                                    break
                        if found_2for1:
                            break
                    if found_2for1:
                        break
            
            # 4. Try residue reassignment: change a residue to enable adding a missing modulus
            if not improved:
                missing = [m for m in sorted(moduli) if m not in used_moduli]
                for m_new in missing[:5]:
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
                    
                        # Current residue r_i blocks m_new if there's no valid r for m_new
                        # Try all other residues for m_i and see if any allows m_new
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
                            r_new, _ = find_best_residue(temp_classes, m_new)
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
    
    # Start 1: Greedy with smallest moduli first
    classes = greedy_construct()
    classes = try_add_all(classes)
    classes = local_search(classes, max_iter=500)
    s = score(classes)
    if s > best_score:
        best_score = s
        best_classes = [list(c) for c in classes]
    
    # Start 2: Greedy with random ordering biased toward small moduli
    random.seed(123)
    num_starts = 60
    
    for start_idx in range(num_starts):
        if time.time() - start_time > deadline - 15:
            break
        
        # Create a random ordering
        remaining = list(moduli)
        order = []
        while remaining:
            # Weight by 1/m^alpha to bias toward small moduli
            alpha = 1.0 + start_idx * 0.05
            weights = [1.0 / (m ** alpha) for m in remaining]
            total_w = sum(weights)
            if total_w <= 0:
                order.append(remaining[0])
                remaining.pop(0)
            else:
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
        
        if time.time() - start_time > deadline - 8:
            s = score(classes)
            if s > best_score:
                best_score = s
                best_classes = [list(c) for c in classes]
            continue
        
        classes = local_search(classes, max_iter=150)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in classes]
    
    # Final polish: intensive local search on best solution
    if best_classes is not None:
        # Run multiple rounds of local search with increasing iteration counts
        for round_num in range(4):
            if time.time() - start_time > deadline - 2:
                break
            max_iter = 300 + round_num * 200
            improved_classes = local_search(best_classes, max_iter=max_iter)
            improved_score = score(improved_classes)
            if improved_score > best_score + 1e-12:
                best_classes = improved_classes
                best_score = improved_score
            else:
                break
    
    return best_classes if best_classes is not None else []
# EVOLVE-BLOCK-END
