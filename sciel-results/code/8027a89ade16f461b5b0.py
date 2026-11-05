import time
import math
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Precompute gcd values
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    # For each modulus m, we can pick exactly one residue r in [0, m)
    # Two choices (m1, r1) and (m2, r2) are incompatible iff r1 % gcd(m1,m2) == r2 % gcd(m1,m2)
    
    # Strategy: Greedy + local search with efficient compatibility
    
    moduli = list(range(20, 201))
    n_mods = len(moduli)
    
    # Precompute gcd between all pairs of moduli
    gcd_table = {}
    for i, m1 in enumerate(moduli):
        for j, m2 in enumerate(moduli):
            if i != j:
                gcd_table[(i, j)] = gcd(m1, m2)
    
    def is_compatible_fast(selected_residues, m_idx, r_new):
        """Check if (m_idx, r_new) is compatible with all selected residues."""
        for j, r_old in selected_residues.items():
            g = gcd_table.get((m_idx, j), 1)
            if r_new % g == r_old % g:
                return False
        return True
    
    def find_compatible_residue(selected_residues, m_idx):
        """Find a compatible residue for modulus m_idx given selected residues."""
        m = moduli[m_idx]
        for r in range(m):
            if is_compatible_fast(selected_residues, m_idx, r):
                return r
        return None
    
    # Greedy: sort by 1/m descending, pick best compatible residue
    def greedy_init():
        selected = {}  # m_idx -> r
        # Try multiple orderings
        best_score = 0
        best_selected = None
        
        for attempt in range(50):
            order = list(range(n_mods))
            if attempt == 0:
                # Sort by 1/m descending
                order.sort(key=lambda i: -1.0/moduli[i])
            elif attempt == 1:
                # Sort by m ascending
                order.sort(key=lambda i: moduli[i])
            elif attempt == 2:
                # Random
                random.shuffle(order)
            else:
                # Random with bias toward small m
                order.sort(key=lambda i: random.random() * moduli[i])
            
            sel = {}
            for i in order:
                r = find_compatible_residue(sel, i)
                if r is not None:
                    sel[i] = r
            
            score = sum(1.0/moduli[i] for i in sel)
            if score > best_score:
                best_score = score
                best_selected = sel
        
        return best_selected, best_score
    
    selected, current_score = greedy_init()
    best_selected = dict(selected)
    best_score = current_score
    
    # Local search with simulated annealing
    random.seed(42)
    T = 0.01
    cooling = 0.9995
    min_T = 1e-8
    
    def score_of(sel):
        return sum(1.0/moduli[i] for i in sel)
    
    while T > min_T and time.time() < deadline:
        if time.time() > deadline - 5:
            break
            
        sol = dict(selected)
        move_type = random.choice(['swap_r', 'add_m', 'remove_m', 'replace_m'])
        
        if move_type == 'swap_r' and sol:
            idx = random.choice(list(sol.keys()))
            m = moduli[idx]
            new_r = random.randint(0, m-1)
            if new_r != sol[idx]:
                # Check compatibility
                compatible = True
                for j, r_old in sol.items():
                    if j == idx:
                        continue
                    g = gcd_table.get((idx, j), 1)
                    if new_r % g == r_old % g:
                        compatible = False
                        break
                if compatible:
                    sol[idx] = new_r
                else:
                    T *= cooling
                    continue
        elif move_type == 'add_m':
            current_idxs = set(sol.keys())
            candidates = [i for i in range(n_mods) if i not in current_idxs]
            if candidates:
                m_idx = random.choice(candidates)
                r = find_compatible_residue(sol, m_idx)
                if r is not None:
                    sol[m_idx] = r
                else:
                    T *= cooling
                    continue
        elif move_type == 'remove_m' and len(sol) > 1:
            idx = random.choice(list(sol.keys()))
            del sol[idx]
        elif move_type == 'replace_m' and sol:
            idx = random.choice(list(sol.keys()))
            current_idxs = set(sol.keys()) - {idx}
            candidates = [i for i in range(n_mods) if i not in current_idxs]
            if candidates:
                m_idx = random.choice(candidates)
                # Check if m_idx is compatible with all other selected
                compatible = True
                for j, r_old in sol.items():
                    if j == idx:
                        continue
                    g = gcd_table.get((m_idx, j), 1)
                    # Need to find an r for m_idx compatible with all others
                    # Just check if any r works
                    m = moduli[m_idx]
                    found_r = None
                    for r_new in range(m):
                        ok = True
                        for k, r_k in sol.items():
                            if k == idx:
                                continue
                            g2 = gcd_table.get((m_idx, k), 1)
                            if r_new % g2 == r_k % g2:
                                ok = False
                                break
                        if ok:
                            found_r = r_new
                            break
                    if found_r is not None:
                        sol[m_idx] = found_r
                        del sol[idx]
                    else:
                        T *= cooling
                        continue
        
        new_score = score_of(sol)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            selected = sol
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_selected = dict(selected)
        
        T *= cooling
    
    # Convert to required format
    result = []
    for i, r in best_selected.items():
        result.append([moduli[i], r])
    
    return result
