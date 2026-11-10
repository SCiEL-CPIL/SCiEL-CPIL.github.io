import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    moduli_range = list(range(2, 129))
    n_moduli = len(moduli_range)
    
    # Precompute gcd matrix
    gcd_table = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(n_moduli):
            gcd_table[i][j] = gcd(moduli_range[i], moduli_range[j])
            
    def is_compatible(residues, moduli_idx_list, new_r, new_m_idx):
        """Check if new residue r with modulus at new_m_idx is compatible with all existing."""
        for i in moduli_idx_list:
            g = gcd_table[new_m_idx][i]
            if new_r % g == residues[i] % g:
                return False
        return True

    def solve_residues_for_clique(clique_indices, time_limit=5.0):
        """
        Attempt to find valid residues for a given set of moduli (clique).
        Uses backtracking with a time limit.
        Returns list of [r, m] pairs or None.
        """
        if not clique_indices:
            return []
        
        # Sort by modulus size ascending to constrain early?
        # Or by "constraint degree" within the clique?
        # Since it's a clique, all have same degree.
        # Let's sort by modulus.
        order = sorted(clique_indices, key=lambda i: moduli_range[i])
        
        residues = [0] * n_moduli
        selected_idx = []
        
        def backtrack(idx):
            if time.time() > start_time + time_limit:
                return False
            if idx == len(order):
                return True
            
            m_idx = order[idx]
            m = moduli_range[m_idx]
            
            # Try residues
            for r in range(m):
                if is_compatible(residues, selected_idx, r, m_idx):
                    residues[m_idx] = r
                    selected_idx.append(m_idx)
                    if backtrack(idx + 1):
                        return True
                    selected_idx.pop()
                    
            return False

        if backtrack(0):
            result = []
            for m_idx in order:
                result.append([residues[m_idx], moduli_range[m_idx]])
            return result
        return None

    # Candidate cliques: sets of moduli where every pair has gcd > 1
    # Largest cliques are likely multiples of small primes.
    
    candidates = []
    
    # Multiples of 2
    candidates.append([i for i in range(n_moduli) if moduli_range[i] % 2 == 0])
    # Multiples of 3
    candidates.append([i for i in range(n_moduli) if moduli_range[i] % 3 == 0])
    # Multiples of 4
    candidates.append([i for i in range(n_moduli) if moduli_range[i] % 4 == 0])
    # Multiples of 5
    candidates.append([i for i in range(n_moduli) if moduli_range[i] % 5 == 0])
    # Multiples of 6
    candidates.append([i for i in range(n_moduli) if moduli_range[i] % 6 == 0])
    
    best = []
    
    for cand in candidates:
        if time.time() > deadline - 2:
            break
        res = solve_residues_for_clique(cand)
        if res and len(res) > len(best):
            best = res
            
    # If we didn't find a good clique solution, fall back to greedy search
    if len(best) < 20:
        def greedy_search(order):
            residues = [0] * n_moduli
            selected = []
            selected_idx = []
            for m_idx in order:
                m = moduli_range[m_idx]
                for r in range(m):
                    if is_compatible(residues, selected_idx, r, m_idx):
                        residues[m_idx] = r
                        selected.append([r, m])
                        selected_idx.append(m_idx)
                        break
            return selected
            
        # Try a few greedy orders
        orders = [
            list(range(n_moduli)),
            list(reversed(range(n_moduli))),
            sorted(range(n_moduli), key=lambda i: moduli_range[i]),
            sorted(range(n_moduli), key=lambda i: -moduli_range[i])
        ]
        
        for order in orders:
            if time.time() > deadline - 5:
                break
            res = greedy_search(order)
            if len(res) > len(best):
                best = res
                
    if not best:
        best = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
        
    return best
