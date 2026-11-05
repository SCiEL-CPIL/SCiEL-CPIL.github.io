import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    
    # Precompute gcd table
    gcd_table = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i, n_moduli):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute 1/m values
    inv_m = [1.0 / m for m in moduli]
    
    def is_compatible(m_idx, r, sel_indices, sel_residues):
        """Check if (m_idx, r) is compatible with all selected."""
        for k in range(len(sel_indices)):
            g = gcd_table[m_idx][sel_indices[k]]
            if (r - sel_residues[k]) % g == 0:
                return False
        return True
    
    def compute_score(sel_indices):
        return sum(inv_m[i] for i in sel_indices)
    
    def greedy_randomized(seed_offset=0):
        """Fast randomized greedy construction."""
        rng = random.Random(seed_offset)
        sel_indices = []
        sel_residues = []
        score = 0.0
        for m_idx in range(n_moduli):
            m = moduli[m_idx]
            # Try residues in random order
            residues = list(range(m))
            rng.shuffle(residues)
            for r in residues:
                if is_compatible(m_idx, r, sel_indices, sel_residues):
                    sel_indices.append(m_idx)
                    sel_residues.append(r)
                    score += inv_m[m_idx]
                    break
        return sel_indices, sel_residues, score
    
    def simulated_annealing(init_indices, init_residues, time_budget):
        """SA with add, remove, and residue-change moves."""
        current_indices = list(init_indices)
        current_residues = list(init_residues)
        current_score = compute_score(current_indices)
        
        best_indices = list(current_indices)
        best_residues = list(current_residues)
        best_score = current_score
        
        T = 0.5
        cooling_rate = 0.998
        min_T = 0.005
        step = 0
        
        while T > min_T and time.time() < time_budget:
            step += 1
            new_indices = list(current_indices)
            new_residues = list(current_residues)
            
            move_type = random.random()
            
            if move_type < 0.35:
                # Add a random missing modulus
                used_set = set(new_indices)
                missing = [i for i in range(n_moduli) if i not in used_set]
                if missing:
                    m_idx = random.choice(missing)
                    m = moduli[m_idx]
                    residues = list(range(m))
                    random.shuffle(residues)
                    for r in residues:
                        if is_compatible(m_idx, r, new_indices, new_residues):
                            new_indices.append(m_idx)
                            new_residues.append(r)
                            break
                    else:
                        continue
            elif move_type < 0.55:
                # Remove a random modulus
                if not new_indices:
                    continue
                pos = random.randint(0, len(new_indices) - 1)
                new_indices.pop(pos)
                new_residues.pop(pos)
            elif move_type < 0.85:
                # Change residue of a random modulus
                if not new_indices:
                    continue
                pos = random.randint(0, len(new_indices) - 1)
                m_idx = new_indices[pos]
                m = moduli[m_idx]
                old_r = new_residues[pos]
                # Remove and reinsert with new residue
                new_indices.pop(pos)
                new_residues.pop(pos)
                residues = list(range(m))
                random.shuffle(residues)
                for r in residues:
                    if is_compatible(m_idx, r, new_indices, new_residues):
                        new_indices.append(m_idx)
                        new_residues.append(r)
                        break
                else:
                    # Revert
                    new_indices.insert(pos, m_idx)
                    new_residues.insert(pos, old_r)
                    continue
            else:
                # Swap: remove one, add another
                if not new_indices:
                    continue
                pos = random.randint(0, len(new_indices) - 1)
                removed_idx = new_indices.pop(pos)
                removed_res = new_residues.pop(pos)
                
                used_set = set(new_indices)
                missing = [i for i in range(n_moduli) if i not in used_set]
                if not missing:
                    new_indices.insert(pos, removed_idx)
                    new_residues.insert(pos, removed_res)
                    continue
                m_idx = random.choice(missing)
                m = moduli[m_idx]
                residues = list(range(m))
                random.shuffle(residues)
                added = False
                for r in residues:
                    if is_compatible(m_idx, r, new_indices, new_residues):
                        new_indices.append(m_idx)
                        new_residues.append(r)
                        added = True
                        break
                if not added:
                    new_indices.insert(pos, removed_idx)
                    new_residues.insert(pos, removed_res)
                    continue
            
            new_score = compute_score(new_indices)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / max(T, 1e-9)):
                current_indices = new_indices
                current_residues = new_residues
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_indices = list(current_indices)
                    best_residues = list(current_residues)
            
            T *= cooling_rate
        
        return best_indices, best_residues, best_score
    
    # Phase 1: Many fast randomized greedy trials
    best_indices = []
    best_residues = []
    best_score = 0.0
    
    trial = 0
    while time.time() < deadline - 40:
        trial += 1
        indices, residues, score = greedy_randomized(seed_offset=trial * 7919)
        if score > best_score:
            best_score = score
            best_indices = indices
            best_residues = residues
        
        # Early stop if we're very close to theoretical max
        # Max possible: sum of 1/m for all m in [20,200] ~ H(200)-H(19)
        # H(200) ~ ln(200)+0.5772 ~ 5.298+0.577 = 5.875
        # H(19) ~ ln(19)+0.5772 ~ 2.944+0.577 = 3.521
        # Max ~ 2.354
        if best_score > 2.30:
            break
    
    # Phase 2: Simulated annealing from best greedy solution
    remaining_time = deadline - time.time()
    if remaining_time > 5 and best_indices:
        sa_indices, sa_residues, sa_score = simulated_annealing(
            best_indices, best_residues, deadline - 5
        )
        if sa_score > best_score:
            best_score = sa_score
            best_indices = sa_indices
            best_residues = sa_residues
    
    # Phase 3: A few more SA runs from different greedy starts if time allows
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        for extra_trial in range(3):
            if time.time() > deadline - 10:
                break
            indices, residues, score = greedy_randomized(seed_offset=100000 + extra_trial * 331)
            if score > best_score:
                best_score = score
                best_indices = indices
                best_residues = residues
            
            # Quick SA from this solution
            sa_time = min(deadline - 10, time.time() + 8)
            sa_indices, sa_residues, sa_score = simulated_annealing(
                best_indices, best_residues, sa_time
            )
            if sa_score > best_score:
                best_score = sa_score
                best_indices = sa_indices
                best_residues = sa_residues
    
    return [[moduli[i], r] for i, r in zip(best_indices, best_residues)]
