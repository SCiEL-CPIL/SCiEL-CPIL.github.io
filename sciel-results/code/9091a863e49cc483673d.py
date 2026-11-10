"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    def check_conflict(res, m, selected):
        """Check if residue res with modulus m conflicts with any selected class."""
        for m2, r2 in selected:
            g = gcd(m, m2)
            if res % g == r2 % g:
                return True
        return False

    def find_valid_residue(m, selected):
        """Find a valid residue for modulus m. Returns None if impossible."""
        for r in range(m):
            if not check_conflict(r, m, selected):
                return r
        return None

    def greedy_fill(selected):
        """Fill with smallest possible moduli that fit, in increasing order."""
        existing_moduli = {m for m, r in selected}
        for m in range(20, 201):
            if m in existing_moduli:
                continue
            r = find_valid_residue(m, selected)
            if r is not None:
                selected.append([m, r])
                existing_moduli.add(m)
        return selected

    def score(selected):
        return sum(1.0 / m for m, r in selected)

    # Phase 1: Greedy construction - select moduli in increasing order
    selected = []
    selected = greedy_fill(selected)
    best_selected = [x[:] for x in selected]
    best_score = score(selected)

    if time.time() > deadline:
        return best_selected

    # Phase 2: Local search with add, swap, and multi-swap operations
    while time.time() < deadline:
        improved = False

        # Step A: Try to add any missing modulus (smallest first for max 1/m)
        existing_moduli = {m for m, r in selected}
        candidates_to_add = [m for m in range(20, 201) if m not in existing_moduli]

        for m_in in candidates_to_add:
            r = find_valid_residue(m_in, selected)
            if r is not None:
                selected.append([m_in, r])
                selected = greedy_fill(selected)
                new_score = score(selected)
                if new_score > best_score:
                    best_score = new_score
                    best_selected = [x[:] for x in selected]
                improved = True
                break

        if improved:
            continue

        # Step B: Try single swaps - remove large m, add small m
        selected_sorted = sorted(selected, key=lambda x: x[0], reverse=True)
        existing_moduli = {m for m, r in selected}
        candidates_to_swap_in = [m for m in range(20, 201) if m not in existing_moduli]

        swap_performed = False
        for m_out_val, r_out in selected_sorted:
            for m_in in candidates_to_swap_in:
                if m_in >= m_out_val:
                    break  # No point swapping for equal or larger modulus

                temp_selected = [x[:] for x in selected if x[0] != m_out_val]
                r = find_valid_residue(m_in, temp_selected)
                if r is not None:
                    temp_selected.append([m_in, r])
                    temp_selected = greedy_fill(temp_selected)
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

        # Step C: Try double swaps - remove two large, add two small
        selected_sorted_desc = sorted(selected, key=lambda x: x[0], reverse=True)
        existing_moduli = {m for m, r in selected}
        candidates_small = [m for m in range(20, 201) if m not in existing_moduli]

        double_swap_performed = False
        # Try removing the two largest and adding the two smallest
        if len(selected) >= 2 and len(candidates_small) >= 2:
            # Try all pairs of small candidates to add
            for i in range(min(5, len(candidates_small))):
                for j in range(i + 1, min(5, len(candidates_small))):
                    m_in1, m_in2 = candidates_small[i], candidates_small[j]
                    # Try removing pairs of largest moduli
                    for k in range(len(selected_sorted_desc)):
                        for l in range(k + 1, len(selected_sorted_desc)):
                            m_out1 = selected_sorted_desc[k][0]
                            m_out2 = selected_sorted_desc[l][0]
                            # Only worth it if we're swapping for smaller
                            if m_in1 >= m_out1 or m_in2 >= m_out2:
                                continue

                            temp_selected = [x[:] for x in selected
                                             if x[0] != m_out1 and x[0] != m_out2]
                            r1 = find_valid_residue(m_in1, temp_selected)
                            if r1 is None:
                                continue
                            temp_selected.append([m_in1, r1])
                            r2 = find_valid_residue(m_in2, temp_selected)
                            if r2 is None:
                                continue
                            temp_selected.append([m_in2, r2])
                            temp_selected = greedy_fill(temp_selected)
                            new_score = score(temp_selected)
                            if new_score > best_score:
                                selected = temp_selected
                                best_score = new_score
                                best_selected = [x[:] for x in selected]
                                double_swap_performed = True
                                break
                        if double_swap_performed:
                            break
                    if double_swap_performed:
                        break

        if double_swap_performed:
            continue

        # Step D: Perturbation - remove some moduli and re-greedy
        if len(selected) > 5:
            # Remove 2-3 random moduli, preferring larger ones
            n_remove = min(3, len(selected) - 1)
            # Sort by modulus descending, remove from top randomly
            sel_sorted = sorted(range(len(selected)), key=lambda i: -selected[i][0])
            # Pick n_remove from the top half
            top_half = sel_sorted[:max(1, len(sel_sorted) // 2)]
            import random
            to_remove = set(random.sample(top_half, min(n_remove, len(top_half))))

            temp_selected = []
            for i, item in enumerate(selected):
                if i not in to_remove:
                    temp_selected.append(item[:])

            temp_selected = greedy_fill(temp_selected)
            new_score = score(temp_selected)
            if new_score > best_score:
                selected = temp_selected
                best_score = new_score
                best_selected = [x[:] for x in selected]

    return best_selected
# EVOLVE-BLOCK-END
