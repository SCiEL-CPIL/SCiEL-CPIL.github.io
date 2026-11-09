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
        Returns (residue, num_conflicts_avoided) or (None, 0) if no valid residue."""
        mi = mod_idx[m]
        valid_residues = []
        for r in range(m):
            if can_add(classes, m, r):
                valid_residues.append(r)
        
        if not valid_residues:
            return None, 0
        
        # Score each valid residue by how much "slack" it provides
        # A residue r has slack if for each existing class (m2, r2),
        # the constraint (r - r2) % g != 0 is "far from being violated"
        # We can measure this as the minimum over all (m2,r2) of 
        # min(|r - r2 mod g|, g - |r - r2 mod g|)
        # Larger minimum distance means more robust.
        
        best_r = valid_residues[0]
        best_slack = -1
        
        for r in valid_residues:
            min_dist = m
            for (m2, r2) in classes:
                mj = mod_idx[m2]
                g = gcd_table[mi][mj]
                if g > 1:
                    diff = (r - r2) % g
                    dist = min(diff, g - diff)
                    min_dist = min(min_dist, dist)
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
                # Iterate over missing moduli (candidates for m_new)
                # Sort by 1/m descending (small m first) to find best swaps early
                missing = [m for m in sorted(moduli) if m not in used_moduli]
                
                for m_new in missing:
                    if time.time() - start_time > deadline - 3:
                        break
                    
                    # Check if m_new can be added if we remove some m_old
                    # We only need to consider m_old that conflict with m_new
                    # Or just try all m_old? That's O(N^2).
                    # Let's try m_old that are "large" (low value) first, as we want to replace them.
                    # Sort existing classes by 1/m ascending (largest m first)
                    existing_sorted = sorted(best_classes, key=lambda c: c[0]) # Largest m first
                    
                    for m_old, r_old in existing_sorted:
                        # If 1/m_new <= 1/m_old, no point swapping (value doesn't improve)
                        if 1.0/m_new <= 1.0/m_old + 1e-12:
                            break # Since sorted by m asc? No, sorted by m asc means 1/m desc.
                            # Wait, sorted(best_classes, key=lambda c: c[0]) is m ascending.
                            # So 1/m is descending.
                            # If 1/m_new <= 1/m_old, then m_new >= m_old.
                            # Since we iterate m_new in ascending order (small to large),
                            # and m_old in ascending order (small to large).
                            # We want to replace large m_old with small m_new.
                            # So we should iterate m_old in *descending* order of value (ascending m)
                            # and stop when 1/m_old >= 1/m_new? No.
                            # We want 1/m_new > 1/m_old => m_new < m_old.
                            # So we only care about m_old > m_new.
                            
                            # Let's re-evaluate sorting.
                            # existing_sorted is m ascending.
                            # We want to find m_old such that m_old > m_new.
                            # If m_old <= m_new, swap is not beneficial (or neutral).
                            # So we can break if m_old > m_new is false?
                            # If m_old <= m_new, then 1/m_old >= 1/m_new. Swap doesn't improve score.
                            # So we only consider m_old where m_old > m_new.
                            # Since existing_sorted is ascending, all m_old <= m_new are at the beginning.
                            # We can skip them.
                            pass
                            
                    # Refined Swap-Add Logic:
                    # For each m_new (small to large), find if removing any m_old (large) allows adding m_new.
                    # We only care if m_old > m_new.
                    
                    found_swap = False
                    # Find indices of m_old > m_new
                    # To optimize, we can just iterate all and check condition.
                    for i in range(len(best_classes)):
                        m_old, r_old = best_classes[i]
                        if m_old <= m_new:
                            continue # Swap not beneficial for score
                        
                        # Check if removing best_classes[i] allows adding m_new
                        temp_classes = [list(c) for c in best_classes if c[0] != m_old]
                        r_new, _ = find_best_residue(temp_classes, m_new)
                        if r_new is not None:
                            # Valid swap
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
                        break # Restart loop with new state
            
            # 3. Try Residue Swap for Slack (if no add or swap-add found)
            if not improved:
                # Keep the original residue swap logic but maybe make it lighter
                # Or skip it if we have enough time?
                # Let's keep a simplified version: try to improve slack of a random class
                # This might help future additions.
                pass # Skipping complex residue swap for now to save time for Swap-Add
            
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
    num_starts = 80
    
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
        for round_num in range(3):
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
