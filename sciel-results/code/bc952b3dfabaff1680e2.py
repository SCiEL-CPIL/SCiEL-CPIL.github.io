import math
import time
import random

def construct():
    deadline = time.time() + 110
    best_score = 0
    best_solution = []
    
    def get_valid_residues(m, selected):
        """
        Returns a list of valid residues r for modulus m,
        given a list of selected (m2, r2) pairs.
        """
        valid = []
        # Precompute constraints
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

    def calculate_score(selected):
        return sum(1.0 / m for m, r in selected)

    def greedy_randomized():
        selected = []
        for m in range(20, 201):
            if time.time() > deadline:
                break
            valids = get_valid_residues(m, selected)
            if valids:
                r = random.choice(valids)
                selected.append((m, r))
        return selected

    def local_search(selected):
        # Returns a potentially improved selected list
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Identify skipped moduli
            selected_mods = set(m for m, r in selected)
            skipped = [m for m in range(20, 201) if m not in selected_mods]
            
            # Sort by 1/m descending (smallest m first) to prioritize high value
            skipped.sort(key=lambda x: -1.0/x)
            
            for m_k in skipped:
                if time.time() > deadline:
                    return selected
                
                # 1. Try to add m_k directly
                r_k = can_add(m_k, selected)
                if r_k is not None:
                    selected.append((m_k, r_k))
                    improved = True
                    break # Restart loop to re-evaluate other skips
                
                # 2. Try to swap a residue in selected to enable m_k
                # Iterate through selected moduli
                for i, (m_i, r_i) in enumerate(selected):
                    if time.time() > deadline:
                        return selected
                    
                    # Find valid residues for m_i if it were not in selected
                    # Actually, we need valid residues for m_i given (selected \ {m_i})
                    others = [(m, r) for j, (m, r) in enumerate(selected) if j != i]
                    valids_i = get_valid_residues(m_i, others)
                    
                    # Try each valid residue for m_i
                    for r_i_new in valids_i:
                        if r_i_new == r_i:
                            continue
                        
                        # Temporarily swap
                        selected[i] = (m_i, r_i_new)
                        
                        # Check if m_k can now be added
                        r_k_new = can_add(m_k, selected)
                        if r_k_new is not None:
                            # Success! Keep the swap and add m_k
                            selected.append((m_k, r_k_new))
                            improved = True
                            break
                        else:
                            # Revert
                            selected[i] = (m_i, r_i)
                    
                    if improved:
                        break
                
                if improved:
                    break
        
        return selected

    # Run multiple greedy restarts
    num_restarts = 20
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        
        sol = greedy_randomized()
        sol = local_search(sol)
        
        score = calculate_score(sol)
        if score > best_score:
            best_score = score
            best_solution = sol[:]

    # If time remains, do more local search on the best solution
    if time.time() < deadline - 5:
        # Deep local search
        best_solution = local_search(best_solution)
        best_score = calculate_score(best_solution)

    return [[m, r] for m, r in best_solution]
