import math
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s margin for safety

    # Precompute gcd table for speed
    gcd_table = {}
    for m1 in range(20, 201):
        for m2 in range(20, 201):
            gcd_table[(m1, m2)] = math.gcd(m1, m2)

    def is_valid(selected, m, r):
        """
        Check if (m, r) is disjoint from all (m_j, r_j) in selected.
        """
        for m_j, r_j in selected:
            g = gcd_table[(m, m_j)]
            if r % g == r_j % g:
                return False
        return True

    def is_valid_fast(selected, m, r, m_to_r):
        """
        Faster validity check using dict lookup for selected moduli.
        m_to_r: dict mapping modulus -> residue for selected set
        """
        for m_j in m_to_r:
            g = gcd_table[(m, m_j)]
            if r % g == m_to_r[m_j] % g:
                return False
        return True

    def greedy_pass(random_seed=None):
        """
        Generates a solution using a greedy approach.
        random_seed: if provided, randomizes the choice of residue among valid ones.
        """
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        selected = []
        m_to_r = {}
        
        for m in range(20, 201):
            if time.time() > deadline - 10.0:
                break
                
            # Find all valid residues for m
            valid_r = None
            if random_seed is not None:
                rs_to_check = list(range(m))
                rng.shuffle(rs_to_check)
            else:
                rs_to_check = list(range(m))
                
            for r in rs_to_check:
                if is_valid_fast(selected, m, r, m_to_r):
                    valid_r = r
                    break
            
            if valid_r is not None:
                selected.append((m, valid_r))
                m_to_r[m] = valid_r
        
        score = sum(1.0 / m for m, r in selected)
        return selected, score

    def local_search_with_perturbation(selected, max_time=None):
        """
        Improves a given solution using:
        1. Residue perturbation: try changing residues of existing moduli
        2. 1-removal + re-add: remove one modulus, try to add others
        3. 2-opt swap: try swapping one modulus for another
        """
        if max_time is None:
            max_time = deadline - time.time() - 2.0
            
        selected = [tuple(x) for x in selected]
        current_score = sum(1.0 / m for m, r in selected)
        m_to_r = {m: r for m, r in selected}
        selected_moduli = set(m_to_r.keys())
        
        improved = True
        iteration = 0
        while improved and (time.time() - start_time) < (deadline - max_time):
            improved = False
            iteration += 1
            
            # Strategy 1: Residue perturbation - try changing residues of existing moduli
            # This can unlock conflicts and allow more moduli to be added
            for idx in range(len(selected)):
                if time.time() > deadline - max_time:
                    return selected, current_score
                    
                m_cur, r_cur = selected[idx]
                
                # Try different residues for this modulus
                # Only try a subset for speed
                candidates = list(range(m_cur))
                random.Random(idx + iteration * 1000).shuffle(candidates)
                candidates = candidates[:min(20, len(candidates))]  # Limit to 20 candidates
                
                for r_new in candidates:
                    if r_new == r_cur:
                        continue
                    
                    # Check if changing residue is valid with all others
                    others = [s for i, s in enumerate(selected) if i != idx]
                    valid = True
                    for m_j, r_j in others:
                        g = gcd_table[(m_cur, m_j)]
                        if r_new % g == r_j % g:
                            valid = False
                            break
                    
                    if valid:
                        # Try to add more moduli after this change
                        new_selected = list(selected)
                        new_selected[idx] = (m_cur, r_new)
                        new_m_to_r = dict(m_to_r)
                        new_m_to_r[m_cur] = r_new
                        new_score = current_score
                        
                        # Try to add any missing moduli
                        for m_add in range(20, 201):
                            if m_add in new_m_to_r:
                                continue
                            valid_r = None
                            for r in range(m_add):
                                if is_valid_fast(new_selected, m_add, r, new_m_to_r):
                                    valid_r = r
                                    break
                            if valid_r is not None:
                                new_selected.append((m_add, valid_r))
                                new_m_to_r[m_add] = valid_r
                                new_score += 1.0 / m_add
                        
                        if new_score > current_score + 1e-12:
                            selected = new_selected
                            m_to_r = new_m_to_r
                            selected_moduli = set(m_to_r.keys())
                            current_score = new_score
                            improved = True
                            break
                
                if improved:
                    break
            
            if improved:
                continue
            
            # Strategy 2: Remove one modulus and try to add others
            for idx in range(len(selected)):
                if time.time() > deadline - max_time:
                    return selected, current_score
                    
                m_to_remove = selected[idx][0]
                others = [s for i, s in enumerate(selected) if i != idx]
                
                new_selected = list(others)
                new_m_to_r = {m: r for m, r in others}
                new_score = current_score - (1.0 / m_to_remove)
                
                # Attempt to add any modulus from 20 to 200 not in new_selected
                for m_add in range(20, 201):
                    if m_add in new_m_to_r:
                        continue
                    
                    valid_r = None
                    for r in range(m_add):
                        if is_valid_fast(new_selected, m_add, r, new_m_to_r):
                            valid_r = r
                            break
                    
                    if valid_r is not None:
                        new_selected.append((m_add, valid_r))
                        new_m_to_r[m_add] = valid_r
                        new_score += 1.0 / m_add
                
                if new_score > current_score + 1e-12:
                    selected = new_selected
                    m_to_r = new_m_to_r
                    selected_moduli = set(m_to_r.keys())
                    current_score = new_score
                    improved = True
                    break
        
        return selected, current_score

    def polish_pass(selected):
        """
        Final high-precision polish: try all possible residue assignments for each modulus
        in a limited neighborhood to find better local optima.
        """
        selected = [tuple(x) for x in selected]
        current_score = sum(1.0 / m for m, r in selected)
        m_to_r = {m: r for m, r in selected}
        
        improved = True
        while improved:
            improved = False
            for idx in range(len(selected)):
                if time.time() > deadline - 1.0:
                    return selected, current_score
                    
                m_cur, r_cur = selected[idx]
                others = [s for i, s in enumerate(selected) if i != idx]
                
                # Try ALL residues for this modulus (up to m_cur values)
                best_r = r_cur
                best_score = current_score
                
                for r_new in range(m_cur):
                    valid = True
                    for m_j, r_j in others:
                        g = gcd_table[(m_cur, m_j)]
                        if r_new % g == r_j % g:
                            valid = False
                            break
                    
                    if valid:
                        # Score doesn't change with residue, but it might unlock additions
                        # For now, just check if it's valid
                        pass
                
                # Try adding missing moduli after each valid residue change
                for r_new in range(m_cur):
                    # Check validity
                    valid = True
                    for m_j, r_j in others:
                        g = gcd_table[(m_cur, m_j)]
                        if r_new % g == r_j % g:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    new_selected = list(selected)
                    new_selected[idx] = (m_cur, r_new)
                    new_m_to_r = dict(m_to_r)
                    new_m_to_r[m_cur] = r_new
                    new_score = current_score
                    
                    # Try to add any missing moduli
                    for m_add in range(20, 201):
                        if m_add in new_m_to_r:
                            continue
                        valid_r = None
                        for r in range(m_add):
                            if is_valid_fast(new_selected, m_add, r, new_m_to_r):
                                valid_r = r
                                break
                        if valid_r is not None:
                            new_selected.append((m_add, valid_r))
                            new_m_to_r[m_add] = valid_r
                            new_score += 1.0 / m_add
                    
                    if new_score > best_score + 1e-12:
                        best_score = new_score
                        best_r = r_new
                        best_selected = new_selected
                        best_m_to_r = new_m_to_r
                
                if best_score > current_score + 1e-12:
                    selected = best_selected
                    m_to_r = best_m_to_r
                    current_score = best_score
                    improved = True
        
        return selected, current_score

    best_score = 0.0
    best_solution = []
    
    # 1. Run Deterministic Greedy
    sol, score = greedy_pass(random_seed=None)
    if score > best_score:
        best_score = score
        best_solution = sol
        
    # 2. Run Randomized Greedy Passes with local search on promising ones
    num_passes = 200
    top_solutions = []
    
    for i in range(num_passes):
        if time.time() > deadline - 30.0:  # Reserve 30s for local search
            break
            
        seed = i
        sol, score = greedy_pass(random_seed=seed)
        
        if score > best_score:
            best_score = score
            best_solution = sol
        
        # Keep top solutions for later local search
        if len(top_solutions) < 10:
            top_solutions.append((score, sol))
        elif score > top_solutions[0][0]:
            top_solutions.pop(0)
            top_solutions.append((score, sol))
    
    # 3. Apply Local Search with perturbation to the best solution
    if time.time() < deadline - 5.0:
        improved_sol, improved_score = local_search_with_perturbation(best_solution)
        if improved_score > best_score:
            best_score = improved_score
            best_solution = improved_sol

    # 4. Apply Local Search to a few other top solutions if time allows
    for score, sol in top_solutions[:3]:
        if time.time() < deadline - 5.0:
            improved_sol, improved_score = local_search_with_perturbation(sol)
            if improved_score > best_score:
                best_score = improved_score
                best_solution = improved_sol

    # 5. Final polish pass
    if time.time() < deadline - 1.0:
        polished_sol, polished_score = polish_pass(best_solution)
        if polished_score > best_score:
            best_score = polished_score
            best_solution = polished_sol
    
    return [[m, r] for m, r in best_solution]
