import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin

    # Precompute gcd table for all pairs in [20, 200]
    gcd_table = {}
    for m1 in range(20, 201):
        for m2 in range(20, 201):
            gcd_table[(m1, m2)] = math.gcd(m1, m2)

    def is_valid(selected, m, r):
        """Check if (m, r) is disjoint from all (m_j, r_j) in selected."""
        for m_j, r_j in selected:
            g = gcd_table[(m, m_j)]
            if r % g == r_j % g:
                return False
        return True

    def find_valid_residue(selected, m):
        """Find a valid residue for modulus m given selected pairs. Return None if none exists."""
        for r in range(m):
            if is_valid(selected, m, r):
                return r
        return None

    def find_valid_residue_fast(selected, m, blocked_mods):
        """Find a valid residue using precomputed blocked mods."""
        for r in range(m):
            valid = True
            for m_j in blocked_mods:
                g = gcd_table[(m, m_j)]
                if r % g == blocked_mods[m_j] % g:
                    valid = False
                    break
            if valid:
                return r
        return None

    def greedy_pass(order=None, random_seed=None):
        """Generates a solution using a greedy approach with given order."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()

        if order is None:
            order = list(range(20, 201))

        selected = []
        for m in order:
            if time.time() > deadline:
                break
            rs = list(range(m))
            if random_seed is not None:
                rng.shuffle(rs)
            for r in rs:
                if is_valid(selected, m, r):
                    selected.append((m, r))
                    break
        score = sum(1.0 / m for m, r in selected)
        return selected, score

    def local_search(selected, max_time=None):
        """Improves a given solution using local search."""
        if max_time is None:
            max_time = deadline - 5
        selected = [tuple(x) for x in selected]
        current_score = sum(1.0 / m for m, r in selected)
        best_score = current_score
        best_solution = list(selected)

        improved = True
        while improved and time.time() < max_time:
            improved = False

            # Move type 1: Add a new modulus
            m_set = {m for m, r in selected}
            for m_add in range(20, 201):
                if m_add in m_set:
                    continue
                r = find_valid_residue(selected, m_add)
                if r is not None:
                    new_selected = list(selected) + [(m_add, r)]
                    new_score = current_score + 1.0 / m_add
                    if new_score > best_score + 1e-12:
                        selected = new_selected
                        current_score = new_score
                        best_score = new_score
                        improved = True

            # Move type 2: Remove one modulus and try to add multiple
            for idx in range(len(selected)):
                if time.time() > max_time:
                    break
                m_to_remove, r_to_remove = selected[idx]
                others = selected[:idx] + selected[idx+1:]
                new_selected = list(others)
                new_score = current_score - 1.0 / m_to_remove

                m_set_new = {m for m, r in new_selected}
                for m_add in range(20, 201):
                    if m_add in m_set_new:
                        continue
                    r = find_valid_residue(new_selected, m_add)
                    if r is not None:
                        new_selected.append((m_add, r))
                        new_score += 1.0 / m_add
                        m_set_new.add(m_add)

                if new_score > best_score + 1e-12:
                    selected = new_selected
                    current_score = new_score
                    best_score = new_score
                    improved = True
                    break

        return best_solution, best_score

    def backtracking_csp(selected, max_time=None, depth_limit=50):
        """
        CSP-based backtracking to find a valid residue assignment for a subset of moduli.
        Uses dynamic variable ordering (MRV heuristic) to select next modulus.
        Tries to find residues for all moduli in the given set, or as many as possible.
        Returns (solution_list, score) where solution_list is list of (m, r).
        """
        if max_time is None:
            max_time = deadline - 5
        
        # Work with a copy
        current = [tuple(x) for x in selected]
        current_score = sum(1.0 / m for m, r in current)
        
        # All moduli we want to consider
        all_moduli = list(range(20, 201))
        selected_mods = {m for m, r in current}
        
        # We'll try to add missing moduli using backtracking
        # Start with current assignment and try to extend it
        
        best_score = current_score
        best_sol = list(current)
        
        def backtrack(assigned, remaining, depth=0):
            nonlocal best_score, best_sol
            
            if time.time() > max_time:
                return
            
            if depth > depth_limit:
                return
            
            # Calculate current score
            cur_score = sum(1.0 / m for m, r in assigned)
            if cur_score > best_score:
                best_score = cur_score
                best_sol = list(assigned)
            
            if not remaining:
                return
            
            # MRV heuristic: pick the modulus with smallest domain
            # Domain size = number of valid residues
            best_var = None
            best_domain_size = float('inf')
            best_domain = None
            
            for m in remaining:
                # Count valid residues
                domain = []
                for r in range(m):
                    valid = True
                    for m_j, r_j in assigned:
                        g = gcd_table[(m, m_j)]
                        if r % g == r_j % g:
                            valid = False
                            break
                    if valid:
                        domain.append(r)
                
                if len(domain) < best_domain_size:
                    best_domain_size = len(domain)
                    best_var = m
                    best_domain = domain
                    if best_domain_size == 0:
                        return  # Dead end
                    if best_domain_size == 1:
                        break  # Can't do better than 1
            
            if best_var is None:
                return
            
            # Try each value in the domain (ordered by some heuristic)
            # Sort by how many conflicts it causes (fewer is better)
            # For simplicity, just try in order
            
            remaining_new = [m for m in remaining if m != best_var]
            
            for r in best_domain:
                if time.time() > max_time:
                    return
                
                new_assigned = assigned + [(best_var, r)]
                backtrack(new_assigned, remaining_new, depth + 1)
            
            return
        
        # Start backtracking with current assignment and all missing moduli
        missing = [m for m in all_moduli if m not in selected_mods]
        
        # Use MRV to order the missing moduli initially
        # But backtrack will handle dynamic ordering
        
        backtrack(list(current), missing, 0)
        
        return best_sol, best_score

    def swap_search(selected, max_time=None):
        """Try swapping residues to enable adding more moduli."""
        if max_time is None:
            max_time = deadline - 5
        selected = [tuple(x) for x in selected]
        current_score = sum(1.0 / m for m, r in selected)
        best_score = current_score
        best_solution = list(selected)

        while time.time() < max_time:
            improved = False
            m_set = {m for m, r in selected}

            for m_add in range(20, 201):
                if m_add in m_set:
                    continue
                if time.time() > max_time:
                    break

                for idx in range(len(selected)):
                    m_j, r_j = selected[idx]
                    g = gcd_table[(m_add, m_j)]
                    if g == 1:
                        continue
                    for new_r_j in range(m_j):
                        if new_r_j == r_j:
                            continue
                        others = selected[:idx] + selected[idx+1:]
                        valid = True
                        for m_k, r_k in others:
                            g2 = gcd_table[(m_j, m_k)]
                            if new_r_j % g2 == r_k % g2:
                                valid = False
                                break
                        if not valid:
                            continue
                        test_selected = others + [(m_j, new_r_j)]
                        r_add = find_valid_residue(test_selected, m_add)
                        if r_add is not None:
                            new_selected = test_selected + [(m_add, r_add)]
                            new_score = current_score + 1.0 / m_add
                            if new_score > best_score + 1e-12:
                                selected = new_selected
                                current_score = new_score
                                best_score = new_score
                                improved = True
                                m_set.add(m_add)
                                break
                    if improved:
                        break
                if improved:
                    break

            if not improved:
                break

        return best_solution, best_score

    best_score = 0.0
    best_solution = []

    # 1. Run deterministic greedy (smallest moduli first)
    sol, score = greedy_pass(order=list(range(20, 201)), random_seed=None)
    if score > best_score:
        best_score = score
        best_solution = sol

    # 2. Run multiple randomized greedy passes with different orderings
    num_passes = 200
    for i in range(num_passes):
        if time.time() > deadline - 60:  # Reserve 60s for CSP and local search
            break
        seed = i
        if i % 3 == 0:
            order = list(range(20, 201))
            random.Random(seed).shuffle(order)
        elif i % 3 == 1:
            order = list(range(200, 19, -1))
            random.Random(seed).shuffle(order)
        else:
            order = list(range(20, 201))
            random.Random(seed).shuffle(order)

        sol, score = greedy_pass(order=order, random_seed=seed)
        if score > best_score:
            best_score = score
            best_solution = sol

    # 3. Apply CSP backtracking to the best solution found
    if time.time() < deadline - 20:
        improved_sol, improved_score = backtracking_csp(best_solution)
        if improved_score > best_score:
            best_score = improved_score
            best_solution = improved_sol

    # 4. Apply local search to the best solution found
    if time.time() < deadline - 10:
        improved_sol, improved_score = local_search(best_solution)
        if improved_score > best_score:
            best_score = improved_score
            best_solution = improved_sol

    # 5. Apply swap search if time permits
    if time.time() < deadline - 5:
        improved_sol, improved_score = swap_search(best_solution)
        if improved_score > best_score:
            best_score = improved_score
            best_solution = improved_sol

    # 6. Final polish: try to add any missing small moduli
    if time.time() < deadline - 2:
        m_set = {m for m, r in best_solution}
        for m_add in range(20, 201):
            if m_add in m_set:
                continue
            r = find_valid_residue(best_solution, m_add)
            if r is not None:
                best_solution.append((m_add, r))
                best_score += 1.0 / m_add
                m_set.add(m_add)

    return [[m, r] for m, r in best_solution]
