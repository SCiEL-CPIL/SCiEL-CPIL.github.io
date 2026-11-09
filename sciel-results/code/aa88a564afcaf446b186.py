"""Pairwise disjoint residue classes - greedy + add-remove-swap local search."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

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

    def find_valid_residue(classes, m):
        """Find a valid residue for modulus m given existing classes.
        Returns the residue with maximum slack, or None."""
        mi = mod_idx[m]
        
        # Fast validity check using NumPy
        residues = np.arange(m)
        valid = np.ones(m, dtype=bool)
        
        for m2, r2 in classes:
            mj = mod_idx[m2]
            g = gcd_table[mi][mj]
            if g == 1:
                continue
            target_mod = r2 % g
            invalid_mask = (residues % g) == target_mod
            valid &= ~invalid_mask
            
        if not np.any(valid):
            return None

        # Calculate slack for valid residues
        valid_indices = np.where(valid)[0]
        best_r = None
        best_slack = -1
        
        # To speed up, if there are many valid residues, we might not need to check all
        # But m is small (<=200), so checking all is fine.
        for r in valid_indices:
            min_dist = m
            for (m2, r2) in classes:
                mj = mod_idx[m2]
                g = gcd_table[mi][mj]
                if g > 1:
                    diff = (r - r2) % g
                    dist = min(diff, g - diff)
                    if dist < min_dist:
                        min_dist = dist
            if min_dist > best_slack:
                best_slack = min_dist
                best_r = r
                
        return best_r

    def score(classes):
        """Compute sum of 1/m for the set of classes."""
        total = 0.0
        for (m, r) in classes:
            total += 1.0 / m
        return total

    def try_add_all(classes):
        """Try to add all remaining moduli, starting with smallest."""
        used_moduli = set(m for m, r in classes)
        remaining = [m for m in sorted(moduli) if m not in used_moduli]

        for m in remaining:
            if time.time() - start_time > deadline - 3:
                break
            r = find_valid_residue(classes, m)
            if r is not None:
                classes.append([m, r])
                used_moduli.add(m)

        return classes

    def add_remove_swap(classes, max_iter=300):
        """Local search with add, remove, and swap moves."""
        best_classes = [list(c) for c in classes]
        best_score = score(best_classes)
        used_moduli = set(m for m, r in best_classes)

        for iteration in range(max_iter):
            if time.time() - start_time > deadline - 3:
                break

            improved = False
            n_classes = len(best_classes)

            # Move 1: Add a new modulus
            for m in sorted(moduli):
                if m in used_moduli:
                    continue
                r = find_valid_residue(best_classes, m)
                if r is not None:
                    new_classes = best_classes + [[m, r]]
                    new_score = best_score + 1.0 / m
                    if new_score > best_score + 1e-12:
                        best_classes = new_classes
                        best_score = new_score
                        used_moduli.add(m)
                        improved = True
                        break

            # Move 2: 1-for-2 Swap (remove 1 large, add 2 small)
            if not improved:
                # Try removing each modulus and see if we can add two smaller ones
                sorted_indices = sorted(range(n_classes), key=lambda i: -best_classes[i][0])
                for i in sorted_indices:
                    if time.time() - start_time > deadline - 3:
                        break
                    m_remove = best_classes[i][0]
                    temp_classes = [list(c) for j, c in enumerate(best_classes) if j != i]
                    temp_used = set(m for m, r in temp_classes)
                    temp_score = best_score - 1.0 / m_remove
                    
                    # Try to add two smaller moduli
                    added_count = 0
                    for m_add in sorted(moduli):
                        if m_add in temp_used:
                            continue
                        if m_add >= m_remove:
                            break
                        r = find_valid_residue(temp_classes, m_add)
                        if r is not None:
                            temp_classes.append([m_add, r])
                            temp_used.add(m_add)
                            temp_score += 1.0 / m_add
                            added_count += 1
                            if added_count == 2:
                                break
                    
                    if added_count == 2 and temp_score > best_score + 1e-12:
                        best_classes = temp_classes
                        best_score = temp_score
                        used_moduli = set(m for m, r in best_classes)
                        improved = True
                        break

            # Move 3: 2-for-1 Swap (remove 2 large, add 1 small)
            if not improved:
                # Try removing pairs of largest moduli
                # Limit pairs to avoid O(N^2) explosion, just check top 10
                top_indices = sorted(range(n_classes), key=lambda i: -best_classes[i][0])[:10]
                for i in range(len(top_indices)):
                    for j in range(i+1, len(top_indices)):
                        if time.time() - start_time > deadline - 3:
                            break
                        idx1 = top_indices[i]
                        idx2 = top_indices[j]
                        m1 = best_classes[idx1][0]
                        m2 = best_classes[idx2][0]
                        
                        temp_classes = [list(c) for k, c in enumerate(best_classes) if k != idx1 and k != idx2]
                        temp_used = set(m for m, r in temp_classes)
                        temp_score = best_score - 1.0 / m1 - 1.0 / m2
                        
                        # Try to add one smaller modulus
                        added = False
                        for m_add in sorted(moduli):
                            if m_add in temp_used:
                                continue
                            if m_add >= min(m1, m2):
                                break
                            r = find_valid_residue(temp_classes, m_add)
                            if r is not None:
                                temp_classes.append([m_add, r])
                                temp_used.add(m_add)
                                temp_score += 1.0 / m_add
                                added = True
                                break
                        
                        if added and temp_score > best_score + 1e-12:
                            best_classes = temp_classes
                            best_score = temp_score
                            used_moduli = set(m for m, r in best_classes)
                            improved = True
                            break
                    if improved:
                        break

            # Move 4: Residue reassignment
            if not improved:
                for i in range(n_classes):
                    if time.time() - start_time > deadline - 3:
                        break
                    m, r = best_classes[i]
                    mi = mod_idx[m]
                    # Try all other residues for this modulus
                    best_r = r
                    best_slack = -1
                    for r2 in range(m):
                        if r2 == r:
                            continue
                        valid = True
                        min_dist = m
                        for j in range(n_classes):
                            if i == j:
                                continue
                            m2, r2_other = best_classes[j]
                            mj = mod_idx[m2]
                            g = gcd_table[mi][mj]
                            if (r2 - r2_other) % g == 0:
                                valid = False
                                break
                            if g > 1:
                                diff = (r2 - r2_other) % g
                                dist = min(diff, g - diff)
                                if dist < min_dist:
                                    min_dist = dist
                        if valid and min_dist > best_slack:
                            best_slack = min_dist
                            best_r = r2

                    if best_r != r:
                        temp_classes = [list(c) for c in best_classes]
                        temp_classes[i][1] = best_r
                        temp_classes = try_add_all(temp_classes)
                        if score(temp_classes) > best_score + 1e-12:
                            best_classes = temp_classes
                            best_score = score(best_classes)
                            used_moduli = set(m for m, r in best_classes)
                            improved = True
                            break

            if not improved:
                break

        return best_classes

    def greedy_construct(order=None):
        """Greedy construction: add classes in the given order if possible."""
        if order is None:
            order = sorted(moduli, key=lambda m: 1.0 / m)
        classes = []
        used_moduli = set()
        for m in order:
            if m in used_moduli:
                continue
            r = find_valid_residue(classes, m)
            if r is not None:
                classes.append([m, r])
                used_moduli.add(m)
        return classes

    # Multi-start approach
    best_classes = None
    best_score = 0.0

    # Start 1: Greedy with smallest moduli first
    classes = greedy_construct()
    classes = try_add_all(classes)
    classes = add_remove_swap(classes, max_iter=500)
    s = score(classes)
    if s > best_score:
        best_score = s
        best_classes = [list(c) for c in classes]

    # Start 2: Multiple random starts
    random.seed(42)
    num_starts = 60

    for start_idx in range(num_starts):
        if time.time() - start_time > deadline - 15:
            break

        # Create a random ordering biased toward small moduli
        remaining = list(moduli)
        order = []
        while remaining:
            alpha = 1.0 + start_idx * 0.1
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

        classes = add_remove_swap(classes, max_iter=200)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in classes]

    # Final polish
    if best_classes is not None:
        for round_num in range(4):
            if time.time() - start_time > deadline - 2:
                break
            max_iter = 400 + round_num * 200
            improved_classes = add_remove_swap(best_classes, max_iter=max_iter)
            improved_score = score(improved_classes)
            if improved_score > best_score + 1e-12:
                best_classes = improved_classes
                best_score = improved_score
            else:
                break

    return best_classes if best_classes is not None else []
# EVOLVE-BLOCK-END
