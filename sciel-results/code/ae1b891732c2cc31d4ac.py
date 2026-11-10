"""
Optimized search for pairwise disjoint residue classes with distinct moduli in [20, 200].
Maximizes sum of 1/m.
"""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Precompute gcd table
    gcd_table = {}
    for m1 in range(20, 201):
        for m2 in range(m1 + 1, 201):
            gcd_table[(m1, m2)] = gcd(m1, m2)
    # Symmetric access
    def get_gcd(m1, m2):
        if m1 == m2:
            return m1
        if m1 < m2:
            return gcd_table.get((m1, m2), gcd(m1, m2))
        else:
            return gcd_table.get((m2, m1), gcd(m2, m1))

    # All moduli in range
    all_moduli = list(range(20, 201))
    
    # Precompute 1/m as float for speed
    inv_mod = {m: 1.0 / m for m in all_moduli}

    def check_conflict_fast(r_new, m_new, selected_moduli, selected_residues):
        """Check if (m_new, r_new) conflicts with any in selected."""
        for i, m2 in enumerate(selected_moduli):
            g = get_gcd(m_new, m2)
            if r_new % g == selected_residues[i] % g:
                return True
        return False

    def find_valid_residue(m_new, selected_moduli, selected_residues):
        """Find a valid residue for m_new given selected classes."""
        # Try residues in order of 0, 1, 2, ... 
        # But we can be smarter: check which residues are "free"
        # For each selected m2, the forbidden residues are those r where r % gcd(m_new, m2) == r2 % gcd(m_new, m2)
        # Let's just iterate. It's fast enough for small n.
        for r in range(m_new):
            if not check_conflict_fast(r, m_new, selected_moduli, selected_residues):
                return r
        return None

    def compute_score(selected_moduli):
        """Compute sum of 1/m."""
        return sum(inv_mod[m] for m in selected_moduli)

    def greedy_fill(selected_moduli, selected_residues, order):
        """Greedy add moduli in given order."""
        sel_m = list(selected_moduli)
        sel_r = list(selected_residues)
        existing = set(sel_m)
        for m in order:
            if m in existing:
                continue
            r = find_valid_residue(m, sel_m, sel_r)
            if r is not None:
                sel_m.append(m)
                sel_r.append(r)
                existing.add(m)
        return sel_m, sel_r

    def try_add_greedy(selected_moduli, selected_residues, m_add, fill_order):
        """Try adding m_add then greedy fill."""
        if m_add in set(selected_moduli):
            return None, None
        sel_m = list(selected_moduli)
        sel_r = list(selected_residues)
        r = find_valid_residue(m_add, sel_m, sel_r)
        if r is None:
            return None, None
        sel_m.append(m_add)
        sel_r.append(r)
        existing = set(sel_m)
        for m in fill_order:
            if m in existing:
                continue
            r2 = find_valid_residue(m, sel_m, sel_r)
            if r2 is not None:
                sel_m.append(m)
                sel_r.append(r2)
                existing.add(m)
        return sel_m, sel_r

    def try_swap(selected_moduli, selected_residues, m_out, m_in, fill_order):
        """Remove m_out, add m_in, then greedy fill."""
        # Find index of m_out
        idx_out = None
        for i, m in enumerate(selected_moduli):
            if m == m_out:
                idx_out = i
                break
        if idx_out is None:
            return None, None
        
        sel_m = [m for i, m in enumerate(selected_moduli) if i != idx_out]
        sel_r = [r for i, r in enumerate(selected_residues) if i != idx_out]
        
        r = find_valid_residue(m_in, sel_m, sel_r)
        if r is None:
            return None, None
        sel_m.append(m_in)
        sel_r.append(r)
        
        existing = set(sel_m)
        for m in fill_order:
            if m in existing:
                continue
            r2 = find_valid_residue(m, sel_m, sel_r)
            if r2 is not None:
                sel_m.append(m)
                sel_r.append(r2)
                existing.add(m)
        return sel_m, sel_r

    # Phase 1: Multiple greedy starts
    best_m = []
    best_r = []
    best_score = 0.0

    random.seed(42)

    # Try increasing order (small moduli first is generally good for 1/m sum)
    order = list(range(20, 201))
    sel_m, sel_r = greedy_fill([], [], order)
    best_m = sel_m
    best_r = sel_r
    best_score = compute_score(sel_m)

    # Try a few randomized orders
    for trial in range(30):
        if time.time() > deadline - 30:
            break
        order = list(range(20, 201))
        random.shuffle(order)
        sel_m, sel_r = greedy_fill([], [], order)
        score = compute_score(sel_m)
        if score > best_score:
            best_score = score
            best_m = sel_m
            best_r = sel_r

    if time.time() > deadline - 10:
        return [[m, r] for m, r in zip(best_m, best_r)]

    # Phase 2: Local search
    current_m = best_m[:]
    current_r = best_r[:]
    current_score = best_score
    no_improve = 0
    max_no_improve = 20
    fill_order = list(range(20, 201))

    while time.time() < deadline - 5:
        improved = False

        # Try adding any missing modulus
        existing = set(current_m)
        candidates = sorted([m for m in range(20, 201) if m not in existing])
        
        for m_add in candidates:
            temp_m, temp_r = try_add_greedy(current_m, current_r, m_add, fill_order)
            if temp_m is not None:
                new_score = compute_score(temp_m)
                if new_score > current_score:
                    current_m = temp_m
                    current_r = temp_r
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_m = current_m[:]
                        best_r = current_r[:]
                    improved = True
                    no_improve = 0
                    break

        if improved:
            continue

        # Try single swaps
        existing = set(current_m)
        candidates_in = sorted([m for m in range(20, 201) if m not in existing])
        candidates_out = sorted(current_m, reverse=True)
        
        swap_done = False
        for m_out in candidates_out:
            if swap_done:
                break
            for m_in in candidates_in:
                if m_in >= m_out:
                    break
                temp_m, temp_r = try_swap(current_m, current_r, m_out, m_in, fill_order)
                if temp_m is not None:
                    new_score = compute_score(temp_m)
                    if new_score > current_score:
                        current_m = temp_m
                        current_r = temp_r
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best_m = current_m[:]
                            best_r = current_r[:]
                        swap_done = True
                        no_improve = 0
                        break

        if swap_done:
            continue

        # Try perturbation: remove random large moduli and re-greedy
        if len(current_m) > 10:
            n_remove = random.randint(2, min(5, len(current_m) - 3))
            indices = list(range(len(current_m)))
            # Bias removal toward larger moduli (smaller 1/m)
            weights = [1.0 / current_m[i] for i in indices]
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            to_remove = set(random.choices(indices, k=n_remove, weights=weights))
            
            temp_m = [current_m[i] for i in indices if i not in to_remove]
            temp_r = [current_r[i] for i in indices if i not in to_remove]
            temp_m, temp_r = greedy_fill(temp_m, temp_r, fill_order)
            new_score = compute_score(temp_m)
            
            if new_score > current_score:
                current_m = temp_m
                current_r = temp_r
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_m = current_m[:]
                    best_r = current_r[:]
                no_improve = 0
                continue

        no_improve += 1
        if no_improve >= max_no_improve:
            # Reset and try aggressive perturbation
            current_m = best_m[:]
            current_r = best_r[:]
            current_score = best_score
            no_improve = 0
            
            n_remove = random.randint(3, min(8, len(current_m) - 2))
            indices = list(range(len(current_m)))
            random.shuffle(indices)
            to_remove = set(indices[:n_remove])
            
            temp_m = [current_m[i] for i in indices if i not in to_remove]
            temp_r = [current_r[i] for i in indices if i not in to_remove]
            temp_m, temp_r = greedy_fill(temp_m, temp_r, fill_order)
            new_score = compute_score(temp_m)
            if new_score > current_score:
                current_m = temp_m
                current_r = temp_r
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_m = current_m[:]
                    best_r = current_r[:]

    # Final check: try to add any remaining moduli one by one to best_m
    # This is a final polish step
    existing = set(best_m)
    for m in sorted([m for m in range(20, 201) if m not in existing]):
        if time.time() > deadline - 1:
            break
        r = find_valid_residue(m, best_m, best_r)
        if r is not None:
            # Check if adding it allows us to keep the score or improve
            # Actually, adding always improves score since 1/m > 0
            best_m.append(m)
            best_r.append(r)
            existing.add(m)

    return [[m, r] for m, r in zip(best_m, best_r)]
# EVOLVE-BLOCK-END
