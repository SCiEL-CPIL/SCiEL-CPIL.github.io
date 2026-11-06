"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    moduli = list(range(2, 129))
    n_moduli = len(moduli)
    
    # Precompute gcd for all pairs
    gcd_table = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i, n_moduli):
            g = _gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute conflict_count
    conflict_count = [0] * n_moduli
    for i in range(n_moduli):
        for j in range(i + 1, n_moduli):
            if gcd_table[i][j] > 1:
                conflict_count[i] += 1
                conflict_count[j] += 1
    
    def is_valid(residues, included):
        """Check all pairs are disjoint."""
        for i in range(len(included)):
            mi = included[i]
            for j in range(i + 1, len(included)):
                mj = included[j]
                g = gcd_table[mi][mj]
                if g > 1:
                    if (residues[mi] - residues[mj]) % g == 0:
                        return False
        return True
    
    # Greedy initial solution
    def greedy():
        residues = [0] * n_moduli
        included = []
        order = sorted(range(n_moduli), key=lambda x: -conflict_count[x])
        for m_idx in order:
            m = moduli[m_idx]
            for r in range(m):
                ok = True
                for m_sel in included:
                    g = gcd_table[m_idx][m_sel]
                    if g > 1 and (r - residues[m_sel]) % g == 0:
                        ok = False
                        break
                if ok:
                    residues[m_idx] = r
                    included.append(m_idx)
                    break
        return included, residues
    
    included, residues = greedy()
    best_included = list(included)
    best_residues = residues[:]
    best_count = len(included)
    
    # Simulated annealing
    random.seed(42)
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    max_steps = 200000
    
    included_set = set(included)
    current_conflicts = 0  # Should be 0 from greedy
    
    step = 0
    while time.time() < deadline and step < max_steps:
        step += 1
        T = max(T_min, T * cooling)
        
        move_type = random.random()
        
        if move_type < 0.25 and len(included) < n_moduli:
            # Add modulus
            uninc = [i for i in range(n_moduli) if i not in included_set]
            if not uninc:
                continue
            m_idx = random.choice(uninc)
            m = moduli[m_idx]
            r = random.randint(0, m - 1)
            old_r = residues[m_idx]
            residues[m_idx] = r
            
            # Count new conflicts
            new_conflicts = 0
            for m_sel in included:
                g = gcd_table[m_idx][m_sel]
                if g > 1 and (r - residues[m_sel]) % g == 0:
                    new_conflicts += 1
            
            included.append(m_idx)
            included_set.add(m_idx)
            
            old_score = len(included) - 1 - 100 * current_conflicts
            new_score = len(included) - 100 * (current_conflicts + new_conflicts)
            delta = new_score - old_score
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_conflicts += new_conflicts
                if current_conflicts == 0 and len(included) > best_count:
                    best_count = len(included)
                    best_included = list(included)
                    best_residues = residues[:]
            else:
                included.pop()
                included_set.discard(m_idx)
                residues[m_idx] = old_r
                
        elif move_type < 0.45 and len(included) > 1:
            # Remove modulus
            idx = random.randint(0, len(included) - 1)
            m_idx = included[idx]
            
            removed_conflicts = 0
            for m_sel in included:
                if m_sel == m_idx:
                    continue
                g = gcd_table[m_idx][m_sel]
                if g > 1 and (residues[m_idx] - residues[m_sel]) % g == 0:
                    removed_conflicts += 1
            
            included.pop(idx)
            included_set.discard(m_idx)
            
            old_score = len(included) + 1 - 100 * current_conflicts
            new_score = len(included) - 100 * (current_conflicts - removed_conflicts)
            delta = new_score - old_score
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_conflicts -= removed_conflicts
                if current_conflicts == 0 and len(included) > best_count:
                    best_count = len(included)
                    best_included = list(included)
                    best_residues = residues[:]
            else:
                included.insert(idx, m_idx)
                included_set.add(m_idx)
                
        else:
            # Change residue
            if not included:
                continue
            m_idx = random.choice(included)
            m = moduli[m_idx]
            old_r = residues[m_idx]
            new_r = random.randint(0, m - 1)
            if new_r == old_r:
                continue
            
            old_conf = 0
            new_conf = 0
            for m_sel in included:
                if m_sel == m_idx:
                    continue
                g = gcd_table[m_idx][m_sel]
                if g > 1:
                    if (old_r - residues[m_sel]) % g == 0:
                        old_conf += 1
                    if (new_r - residues[m_sel]) % g == 0:
                        new_conf += 1
            
            delta_conf = new_conf - old_conf
            delta = -100 * delta_conf
            
            residues[m_idx] = new_r
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_conflicts += delta_conf
                if current_conflicts == 0 and len(included) > best_count:
                    best_count = len(included)
                    best_included = list(included)
                    best_residues = residues[:]
            else:
                residues[m_idx] = old_r
    
    result = [[best_residues[m_idx], moduli[m_idx]] for m_idx in best_included]
    
    if not result:
        return [[1, 2]]
    
    # Final validation
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            r1, m1 = result[i]
            r2, m2 = result[j]
            g = _gcd(m1, m2)
            if (r1 - r2) % g == 0:
                valid = False
                break
        if not valid:
            break
    
    if valid:
        return result
    return [[1, 2]]
# EVOLVE-BLOCK-END
