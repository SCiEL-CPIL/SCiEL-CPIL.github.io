"""Improved search for pairwise disjoint residue classes with distinct moduli in [20, 200]."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Precompute gcd table
    gcd_table = {}
    for m1 in range(20, 201):
        for m2 in range(20, 201):
            if m1 != m2:
                gcd_table[(m1, m2)] = gcd(m1, m2)

    def is_conflict(r_new, m_new, selected):
        """Check if (m_new, r_new) conflicts with any in selected."""
        for m2, r2 in selected:
            g = gcd_table.get((m_new, m2), gcd(m_new, m2))
            if r_new % g == r2 % g:
                return True
        return False

    def find_valid_residue(m_new, selected):
        """Find a valid residue for m_new given selected classes."""
        for r in range(m_new):
            if not is_conflict(r, m_new, selected):
                return r
        return None

    def compute_score(selected):
        """Compute sum of 1/m."""
        return float(sum(Fraction(1, m) for m, r in selected))

    def greedy_fill(selected, order):
        """Greedy add moduli in given order."""
        existing = {m for m, r in selected}
        sel_list = list(selected)
        for m in order:
            if m in existing:
                continue
            r = find_valid_residue(m, sel_list)
            if r is not None:
                sel_list.append([m, r])
                existing.add(m)
        return sel_list

    def try_add_greedy(selected, m_add, fill_order):
        """Try adding m_add then greedy fill."""
        if m_add in {m for m, r in selected}:
            return None
        sel_list = [x[:] for x in selected]
        r = find_valid_residue(m_add, sel_list)
        if r is None:
            return None
        sel_list.append([m_add, r])
        existing = {m for m, r in sel_list}
        for m in fill_order:
            if m in existing:
                continue
            r2 = find_valid_residue(m, sel_list)
            if r2 is not None:
                sel_list.append([m, r2])
                existing.add(m)
        return sel_list

    def try_swap(selected, m_out, m_in, fill_order):
        """Remove m_out, add m_in, then greedy fill."""
        sel_list = [x[:] for x in selected if x[0] != m_out]
        r = find_valid_residue(m_in, sel_list)
        if r is None:
            return None
        sel_list.append([m_in, r])
        existing = {m for m, r in sel_list}
        for m in fill_order:
            if m in existing:
                continue
            r2 = find_valid_residue(m, sel_list)
            if r2 is not None:
                sel_list.append([m, r2])
                existing.add(m)
        return sel_list

    # Phase 1: Multiple greedy starts
    best_selected = []
    best_score = 0.0

    random.seed(42)

    # Try increasing order
    order = list(range(20, 201))
    selected = greedy_fill([], order)
    best_selected = [x[:] for x in selected]
    best_score = compute_score(selected)

    # Try a few randomized orders
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        order = list(range(20, 201))
        random.shuffle(order)
        selected = greedy_fill([], order)
        score = compute_score(selected)
        if score > best_score:
            best_score = score
            best_selected = [x[:] for x in selected]

    if time.time() > deadline - 10:
        return best_selected

    # Phase 2: Local search
    current_selected = [x[:] for x in best_selected]
    current_score = best_score
    no_improve = 0
    max_no_improve = 30
    fill_order = list(range(20, 201))

    while time.time() < deadline - 5:
        improved = False

        # Try adding any missing modulus
        existing = {m for m, r in current_selected}
        candidates = sorted([m for m in range(20, 201) if m not in existing])
        
        for m_add in candidates:
            temp = try_add_greedy(current_selected, m_add, fill_order)
            if temp is not None:
                new_score = compute_score(temp)
                if new_score > current_score:
                    current_selected = temp
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_selected = [x[:] for x in current_selected]
                    improved = True
                    no_improve = 0
                    break

        if improved:
            continue

        # Try single swaps
        existing = {m for m, r in current_selected}
        candidates_in = sorted([m for m in range(20, 201) if m not in existing])
        candidates_out = sorted([x[0] for x in current_selected], reverse=True)
        
        swap_done = False
        for m_out in candidates_out:
            if swap_done:
                break
            for m_in in candidates_in:
                if m_in >= m_out:
                    break
                temp = try_swap(current_selected, m_out, m_in, fill_order)
                if temp is not None:
                    new_score = compute_score(temp)
                    if new_score > current_score:
                        current_selected = temp
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best_selected = [x[:] for x in current_selected]
                        swap_done = True
                        no_improve = 0
                        break

        if swap_done:
            continue

        # Try perturbation: remove random large moduli and re-greedy
        if len(current_selected) > 10:
            n_remove = random.randint(2, min(5, len(current_selected) - 3))
            indices = list(range(len(current_selected)))
            # Bias removal toward larger moduli
            weights = [1.0 / current_selected[i][0] for i in indices]
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            to_remove = set(random.sample(indices, n_remove, counts=None)) if False else set(random.choices(indices, k=n_remove))
            
            temp_selected = [current_selected[i][:] for i in indices if i not in to_remove]
            temp_selected = greedy_fill(temp_selected, fill_order)
            new_score = compute_score(temp_selected)
            
            if new_score > current_score:
                current_selected = temp_selected
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_selected = [x[:] for x in current_selected]
                no_improve = 0
                continue

        no_improve += 1
        if no_improve >= max_no_improve:
            # Reset and try aggressive perturbation
            current_selected = [x[:] for x in best_selected]
            current_score = best_score
            no_improve = 0
            
            n_remove = random.randint(3, min(8, len(current_selected) - 2))
            indices = list(range(len(current_selected)))
            random.shuffle(indices)
            to_remove = set(indices[:n_remove])
            
            temp_selected = [current_selected[i][:] for i in indices if i not in to_remove]
            temp_selected = greedy_fill(temp_selected, fill_order)
            new_score = compute_score(temp_selected)
            if new_score > current_score:
                current_selected = temp_selected
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_selected = [x[:] for x in current_selected]

    return best_selected
# EVOLVE-BLOCK-END
