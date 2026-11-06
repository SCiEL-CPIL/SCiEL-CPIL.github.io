import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    MAX_MOD = 128
    moduli = list(range(2, MAX_MOD + 1))
    
    # Precompute gcd table
    n_mod = len(moduli)
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    best_result = [[1, 2]]  # fallback: at least one valid class
    best_count = 1
    
    def try_add_residue(m_idx, selected):
        """Find a residue for modulus moduli[m_idx] disjoint from all selected."""
        m = moduli[m_idx]
        # For each candidate residue 0..m-1, check disjointness
        # To speed up, precompute constraints
        constraints = []
        for (r_i, m_i) in selected:
            m_i_idx = moduli.index(m_i)
            g = gcd_table[m_idx][m_i_idx]
            if g > 1:
                constraints.append((g, r_i))
        # Try residues in random order for diversity
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            valid = True
            for (g, r_i) in constraints:
                if (r - r_i) % g == 0:
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def greedy_randomized():
        """One greedy pass with random ordering."""
        selected = []  # list of [residue, modulus]
        used_moduli = set()
        order = list(range(n_mod))
        random.shuffle(order)
        
        for idx in order:
            if time.time() > deadline:
                break
            m = moduli[idx]
            if m in used_moduli:
                continue
            # Check gcd > 1 with all selected moduli
            compatible = True
            for (r_i, m_i) in selected:
                m_i_idx = moduli.index(m_i)
                if gcd_table[idx][m_i_idx] == 1:
                    compatible = False
                    break
            if not compatible:
                continue
            # Try to find a valid residue
            r = try_add_residue(idx, selected)
            if r is not None:
                selected.append([r, m])
                used_moduli.add(m)
        
        return selected
    
    # Run multiple randomized greedy passes
    num_restarts = 500
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        result = greedy_randomized()
        if len(result) > best_count:
            best_count = len(result)
            best_result = result
    
    # Try to improve best_result with local search
    if time.time() < deadline - 5.0:
        # Try adding more moduli to best_result
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            current = [[r, m] for r, m in best_result]
            used = set(m for _, m in current)
            for idx in range(n_mod):
                if time.time() > deadline - 2.0:
                    break
                m = moduli[idx]
                if m in used:
                    continue
                compatible = True
                for (r_i, m_i) in current:
                    m_i_idx = moduli.index(m_i)
                    if gcd_table[idx][m_i_idx] == 1:
                        compatible = False
                        break
                if not compatible:
                    continue
                r = try_add_residue(idx, current)
                if r is not None:
                    current.append([r, m])
                    used.add(m)
                    improved = True
            if len(current) > len(best_result):
                best_result = current
    
    return best_result
