import math
import time
import random
from fractions import Fraction

def construct():
    deadline = time.time() + 110
    best_score = 0.0
    best_solution = []
    
    def get_valid_residues(m, selected):
        """
        Returns a list of valid residues r for modulus m,
        given a list of selected (m2, r2) pairs.
        """
        valid = []
        constraints = []
        for m2, r2 in selected:
            g = math.gcd(m, m2)
            constraints.append((g, r2 % g))
        
        for r in range(m):
            is_valid = True
            for g, r2_mod in constraints:
                if r % g == r2_mod:
                    is_valid = False
                    break
            if is_valid:
                valid.append(r)
        return valid

    def can_add(m_new, selected):
        """
        Checks if m_new can be added to selected.
        Returns a valid residue if yes, None if no.
        """
        valids = get_valid_residues(m_new, selected)
        if valids:
            return random.choice(valids)
        return None

    def count_conflicts(m, r, all_moduli, selected_mods):
        """Count how many unselected moduli would be blocked by choosing (m, r)."""
        conflicts = 0
        for m2 in all_moduli:
            if m2 in selected_mods or m2 == m:
                continue
            g = math.gcd(m, m2)
            if r % g == 0:  # Check if this residue conflicts with any residue of m2
                # Actually, we need to check if ALL residues of m2 are blocked
                # A residue r of m conflicts with m2 if r % gcd(m,m2) == r2 % gcd(m,m2) for all r2
                # No, the constraint is: for a specific r2, r % g != r2 % g
                # So choosing r blocks residue r2 of m2 if r % g == r2 % g
                # But we want to know if choosing r makes it IMPOSSIBLE to add m2
                # That happens if for every r2 in [0, m2), r % g == r2 % g
                # Which means g == m2 (i.e., m2 divides m) and r % g == r2 % g for all r2
                # That's impossible unless m2=1
                # Actually, m2 can still be added with some other residue r2' where r2' % g != r % g
                # So m2 is only blocked if ALL its residues satisfy r2' % g == r % g
                # That requires g >= m2, i.e., g = m2, meaning m2 | m
                # In that case, all residues r2' have r2' % m2 in [0, m2-1], and r % m2 is fixed
                # So only one residue class is blocked, not all
                # Thus m2 is never fully blocked by a single residue choice
                # The real issue is: which residues of m2 are available?
                pass
        return conflicts

    def score_residue_choice(m, r, selected, all_moduli):
        """
        Score a residue choice by how many future moduli it leaves available.
        Higher is better (more future moduli remain addable).
        """
        selected_with_new = selected + [(m, r)]
        selected_mods = set(m2 for m2, _ in selected_with_new)
        available_count = 0
        for m2 in all_moduli:
            if m2 in selected_mods:
                continue
            valids = get_valid_residues(m2, selected_with_new)
            if valids:
                available_count += 1
        return available_count

    def greedy_smart():
        """
        Greedy that picks the residue minimizing future conflicts.
        """
        selected = []
        all_moduli = list(range(20, 201))
        
        for m in all_moduli:
            if time.time() > deadline:
                break
            valids = get_valid_residues(m, selected)
            if not valids:
                continue
            
            if len(valids) == 1:
                r = valids[0]
            else:
                # Pick the residue that leaves the most future moduli available
                best_r = valids[0]
                best_score_r = -1
                for r in valids:
                    s = score_residue_choice(m, r, selected, all_moduli)
                    if s > best_score_r:
                        best_score_r = s
                        best_r = r
                r = best_r
            
            selected.append((m, r))
        return selected

    def local_search(selected):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            selected_mods = set(m for m, r in selected)
            skipped = [m for m in range(20, 201) if m not in selected_mods]
            skipped.sort(key=lambda x: -1.0/x)
            
            for m_k in skipped:
                if time.time() > deadline:
                    return selected
                
                r_k = can_add(m_k, selected)
                if r_k is not None:
                    selected.append((m_k, r_k))
                    improved = True
                    break
                
                for i, (m_i, r_i) in enumerate(selected):
                    if time.time() > deadline:
                        return selected
                    
                    others = [(m, r) for j, (m, r) in enumerate(selected) if j != i]
                    valids_i = get_valid_residues(m_i, others)
                    
                    for r_i_new in valids_i:
                        if r_i_new == r_i:
                            continue
                        
                        selected[i] = (m_i, r_i_new)
                        r_k_new = can_add(m_k, selected)
                        if r_k_new is not None:
                            selected.append((m_k, r_k_new))
                            improved = True
                            break
                        else:
                            selected[i] = (m_i, r_i)
                    
                    if improved:
                        break
                
                if improved:
                    break
        
        return selected

    def branch_and_bound(selected):
        """
        Branch and bound on the conflict graph to find a better independent set.
        Uses fractional knapsack upper bound.
        """
        selected_mods = set(m for m, r in selected)
        candidates = [m for m in range(20, 201) if m not in selected_mods]
        candidates.sort(key=lambda x: -1.0/x)
        
        best_local = selected[:]
        best_local_score = sum(1.0/m for m, r in selected)
        
        def compute_conflict_matrix(cand_list):
            """For each pair of candidates, check if they can coexist."""
            n = len(cand_list)
            conflict = [[False]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    m1, m2 = cand_list[i], cand_list[j]
                    g = math.gcd(m1, m2)
                    # They conflict if for every r1 in [0,m1) and r2 in [0,m2), r1%g == r2%g
                    # No, they conflict if there do NOT EXIST r1, r2 with r1%g != r2%g
                    # That happens when g >= min(m1, m2), i.e., one divides the other
                    # Actually: r1%g and r2%g can be made different iff g > 1 and both have at least 2 residue classes mod g
                    # r1%g ranges over [0, g-1] as r1 ranges over [0, m1) iff m1 >= g
                    # Similarly for r2
                    # So we need m1 >= g and m2 >= g, which is always true since g <= min(m1,m2)
                    # And we need at least 2 distinct values in each range
                    # r1%g takes values [0, min(m1,g)-1]... no, it takes values 0..g-1 if m1 >= g
                    # Since g <= m1, r1%g takes all values 0..g-1
                    # So as long as g >= 2, we can pick r1%g != r2%g
                    # If g == 1, no constraint
                    # So two moduli ALWAYS can coexist? No wait...
                    # The constraint is r1 % g != r2 % g for the SPECIFIC chosen r1, r2
                    # We get to CHOOSE r1 and r2, so we can always pick them to satisfy the constraint
                    # UNLESS the only valid residues for r1 (given other selected) force r1%g to a specific value
                    # This is getting complex. For B&B, let's just try to add and check.
                    conflict[i][j] = False  # Assume they can coexist; we'll verify when adding
            return conflict
        
        # Simple B&B: try adding candidates in order of 1/m, with backtracking
        def bb(idx, current_selected, current_score):
            nonlocal best_local, best_local_score
            if time.time() > deadline - 5:
                return
            
            if current_score > best_local_score:
                best_local_score = current_score
                best_local = current_selected[:]
            
            if idx >= len(candidates):
                return
            
            m = candidates[idx]
            
            # Upper bound: current_score + sum of 1/m for all remaining candidates
            ub = current_score + sum(1.0/candidates[j] for j in range(idx+1, len(candidates)))
            if ub <= best_local_score:
                return
            
            # Try adding m
            valids = get_valid_residues(m, current_selected)
            if valids:
                r = valids[0]
                new_selected = current_selected + [(m, r)]
                bb(idx+1, new_selected, current_score + 1.0/m)
            
            # Try not adding m
            bb(idx+1, current_selected, current_score)
        
        bb(0, selected[:], best_local_score)
        return best_local

    # Run multiple greedy restarts
    num_restarts = 15
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        
        sol = greedy_smart()
        sol = local_search(sol)
        
        score = sum(1.0/m for m, r in sol)
        if score > best_score:
            best_score = score
            best_solution = sol[:]

    # If time remains, do B&B and more local search on the best solution
    if time.time() < deadline - 10:
        best_solution = branch_and_bound(best_solution)
        best_score = sum(1.0/m for m, r in best_solution)
    
    if time.time() < deadline - 5:
        best_solution = local_search(best_solution)
        best_score = sum(1.0/m for m, r in best_solution)

    return [[m, r] for m, r in best_solution]
