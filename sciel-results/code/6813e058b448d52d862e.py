"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit

    def check_conflict(res, m, selected):
        """Check if residue res with modulus m conflicts with any selected class."""
        for m2, r2 in selected:
            g = gcd(m, m2)
            if g > 1 and res % g == r2 % g:
                return True
        return False

    def find_valid_residues(m, selected):
        """Find all valid residues for modulus m."""
        valid = []
        for r in range(m):
            if not check_conflict(r, m, selected):
                valid.append(r)
        return valid

    def score(selected):
        return sum(1.0 / m for m, r in selected)

    def greedy_fill(selected, start_m=20):
        """Fill with smallest possible moduli that fit, in increasing order."""
        existing_moduli = {m for m, r in selected}
        for m in range(start_m, 201):
            if m in existing_moduli:
                continue
            residues = find_valid_residues(m, selected)
            if residues:
                # Pick residue that minimizes future conflicts
                # Heuristic: pick residue that's "spread out" from others
                best_r = residues[0]
                best_conflict_count = float('inf')
                for r in residues:
                    # Count how many future moduli would be blocked
                    conflict_count = 0
                    for m2 in range(m + 1, 201):
                        g = gcd(m, m2)
                        if g > 1:
                            # This residue blocks r % g for modulus m2
                            # Count how many residues of m2 are blocked
                            blocked = r % g
                            # Number of residues in [0, m2) that are ≡ blocked (mod g)
                            # This is m2 // g
                            conflict_count += m2 // g
                    if conflict_count < best_conflict_count:
                        best_conflict_count = conflict_count
                        best_r = r
                selected.append([m, best_r])
                existing_moduli.add(m)
        return selected

    # Phase 1: Initial greedy construction with smart residue selection
    selected = []
    selected = greedy_fill(selected)
    best_selected = [x[:] for x in selected]
    best_score = score(selected)

    if time.time() > deadline:
        return best_selected

    # Phase 2: Backtracking search for better residue assignments
    # Try to improve by reassigning residues for a subset of moduli
    
    def backtrack_improve(selected, moduli_list, idx, current_score):
        """Try to improve by reassigning residues starting from index idx."""
        if time.time() > deadline:
            return None, current_score
        
        if idx == len(moduli_list):
            return None, current_score
        
        m = moduli_list[idx]
        # Find which selected class has this modulus
        existing_idx = None
        for i, (m2, r2) in enumerate(selected):
            if m2 == m:
                existing_idx = i
                break
        
        if existing_idx is None:
            return backtrack_improve(selected, moduli_list, idx + 1, current_score)
        
        old_r = selected[existing_idx][1]
        
        # Try different residues
        valid_residues = find_valid_residues(m, [x for i, x in enumerate(selected) if i != existing_idx])
        
        best_new_selected = None
        best_new_score = current_score
        
        for r in valid_residues:
            if r == old_r:
                continue
            new_selected = [x[:] for x in selected]
            new_selected[existing_idx][1] = r
            new_score = score(new_selected)
            
            if new_score > best_new_score:
                best_new_score = new_score
                best_new_selected = new_selected
        
        if best_new_selected is not None:
            # Recurse with the improved selection
            result, new_score = backtrack_improve(best_new_selected, moduli_list, idx + 1, best_new_score)
            if result is not None:
                return result, new_score
            return best_new_selected, best_new_score
        
        return backtrack_improve(selected, moduli_list, idx + 1, current_score)

    # Phase 3: Local search with swaps
    while time.time() < deadline:
        improved = False

        # Step A: Try to add any missing modulus
        existing_moduli = {m for m, r in selected}
        candidates_to_add = [m for m in range(20, 201) if m not in existing_moduli]

        for m_in in candidates_to_add:
            residues = find_valid_residues(m_in, selected)
            if residues:
                # Pick best residue
                best_r = residues[0]
                best_conflict_count = float('inf')
                for r in residues:
                    conflict_count = 0
                    for m2 in range(m_in + 1, 201):
                        g = gcd(m_in, m2)
                        if g > 1:
                            conflict_count += m2 // g
                    if conflict_count < best_conflict_count:
                        best_conflict_count = conflict_count
                        best_r = r
                selected.append([m_in, best_r])
                # Re-fill with greedy
                existing_moduli = {m for m, r in selected}
                for m in range(20, 201):
                    if m in existing_moduli:
                        continue
                    residues = find_valid_residues(m, selected)
                    if residues:
                        best_r = residues[0]
                        best_conflict_count = float('inf')
                        for r in residues:
                            conflict_count = 0
                            for m2 in range(m + 1, 201):
                                g = gcd(m, m2)
                                if g > 1:
                                    conflict_count += m2 // g
                            if conflict_count < best_conflict_count:
                                best_conflict_count = conflict_count
                                best_r = r
                        selected.append([m, best_r])
                        existing_moduli.add(m)
                new_score = score(selected)
                if new_score > best_score:
                    best_score = new_score
                    best_selected = [x[:] for x in selected]
                improved = True
                break

        if improved:
            continue

        # Step B: Try swaps - remove large m, add small m
        selected_sorted = sorted(selected, key=lambda x: x[0], reverse=True)
        existing_moduli = {m for m, r in selected}
        candidates_to_swap_in = [m for m in range(20, 201) if m not in existing_moduli]

        swap_performed = False
        for m_out_val, r_out in selected_sorted:
            for m_in in candidates_to_swap_in:
                if m_in >= m_out_val:
                    break

                temp_selected = [x[:] for x in selected if x[0] != m_out_val]
                residues = find_valid_residues(m_in, temp_selected)
                if residues:
                    best_r = residues[0]
                    best_conflict_count = float('inf')
                    for r in residues:
                        conflict_count = 0
                        for m2 in range(m_in + 1, 201):
                            g = gcd(m_in, m2)
                            if g > 1:
                                conflict_count += m2 // g
                        if conflict_count < best_conflict_count:
                            best_conflict_count = conflict_count
                            best_r = r
                    temp_selected.append([m_in, best_r])
                    # Re-fill
                    existing_moduli = {m for m, r in temp_selected}
                    for m in range(20, 201):
                        if m in existing_moduli:
                            continue
                        residues = find_valid_residues(m, temp_selected)
                        if residues:
                            best_r = residues[0]
                            best_conflict_count = float('inf')
                            for r in residues:
                                conflict_count = 0
                                for m2 in range(m + 1, 201):
                                    g = gcd(m, m2)
                                    if g > 1:
                                        conflict_count += m2 // g
                                if conflict_count < best_conflict_count:
                                    best_conflict_count = conflict_count
                                    best_r = r
                            temp_selected.append([m, best_r])
                            existing_moduli.add(m)
                    new_score = score(temp_selected)
                    if new_score > best_score:
                        selected = temp_selected
                        best_score = new_score
                        best_selected = [x[:] for x in selected]
                        swap_performed = True
                        break

            if swap_performed:
                break

        if swap_performed:
            continue

        # Step C: Perturbation - remove some moduli and re-greedy
        if len(selected) > 5:
            n_remove = min(3, len(selected) - 1)
            sel_sorted = sorted(range(len(selected)), key=lambda i: -selected[i][0])
            top_half = sel_sorted[:max(1, len(sel_sorted) // 2)]
            to_remove = set(random.sample(top_half, min(n_remove, len(top_half))))

            temp_selected = []
            for i, item in enumerate(selected):
                if i not in to_remove:
                    temp_selected.append(item[:])

            existing_moduli = {m for m, r in temp_selected}
            for m in range(20, 201):
                if m in existing_moduli:
                    continue
                residues = find_valid_residues(m, temp_selected)
                if residues:
                    best_r = residues[0]
                    best_conflict_count = float('inf')
                    for r in residues:
                        conflict_count = 0
                        for m2 in range(m + 1, 201):
                            g = gcd(m, m2)
                            if g > 1:
                                conflict_count += m2 // g
                            if conflict_count < best_conflict_count:
                                best_conflict_count = conflict_count
                                best_r = r
                    temp_selected.append([m, best_r])
                    existing_moduli.add(m)

            new_score = score(temp_selected)
            if new_score > best_score:
                selected = temp_selected
                best_score = new_score
                best_selected = [x[:] for x in selected]

    return best_selected
# EVOLVE-BLOCK-END
