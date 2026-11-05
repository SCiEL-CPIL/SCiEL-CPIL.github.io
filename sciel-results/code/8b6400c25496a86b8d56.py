import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    
    # Precompute gcds
    gcds = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i, n_moduli):
            g = math.gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
    
    # Helper to check if (m_idx, r) is compatible with a set
    def is_compatible(m_idx, r, selected_indices, selected_residues):
        for k in range(len(selected_indices)):
            m_old_idx = selected_indices[k]
            r_old = selected_residues[k]
            g = gcds[m_idx][m_old_idx]
            if (r - r_old) % g == 0:
                return False
        return True
    
    def get_score_float(indices):
        return sum(1.0 / moduli[idx] for idx in indices)
    
    def get_score_exact(indices):
        total = Fraction(0, 1)
        for idx in indices:
            total += Fraction(1, moduli[idx])
        return total
    
    def find_valid_residues(m_idx, selected_indices, selected_residues):
        """Return all valid residues for modulus m_idx given current selection."""
        m = moduli[m_idx]
        valid = []
        for r in range(m):
            compatible = True
            for k in range(len(selected_indices)):
                m_old_idx = selected_indices[k]
                r_old = selected_residues[k]
                g = gcds[m_idx][m_old_idx]
                if (r - r_old) % g == 0:
                    compatible = False
                    break
            if compatible:
                valid.append(r)
        return valid
    
    def greedy_construction():
        """Greedy: process moduli in increasing order, pick best residue."""
        selected_indices = []
        selected_residues = []
        
        for m_idx in range(n_moduli):
            m = moduli[m_idx]
            valid_residues = find_valid_residues(m_idx, selected_indices, selected_residues)
            if valid_residues:
                # Pick a random valid residue (for diversity)
                r = random.choice(valid_residues)
                selected_indices.append(m_idx)
                selected_residues.append(r)
        
        return selected_indices, selected_residues
    
    def greedy_randomized(alpha=0.5):
        """Greedy with randomization: with probability alpha, pick random valid;
        otherwise pick the residue that conflicts least with future moduli."""
        selected_indices = []
        selected_residues = []
        
        for m_idx in range(n_moduli):
            m = moduli[m_idx]
            valid_residues = find_valid_residues(m_idx, selected_indices, selected_residues)
            if valid_residues:
                if random.random() < alpha:
                    r = random.choice(valid_residues)
                else:
                    # Pick residue that minimizes future conflicts
                    # For each candidate r, count how many future moduli it conflicts with
                    best_r = None
                    best_conflicts = float('inf')
                    for r in valid_residues:
                        conflicts = 0
                        for future_idx in range(m_idx + 1, n_moduli):
                            g = gcds[m_idx][future_idx]
                            if (r - 0) % g == 0:  # Will conflict if future picks 0
                                # Actually we don't know what future will pick.
                                # Heuristic: residues closer to 0 mod g are "riskier"
                                # Just count how many distinct g values divide r
                                # Simpler: pick r that maximizes gcd(r, m) or something?
                                # Let's just pick random among top half to break ties
                                conflicts += 1
                        if conflicts < best_conflicts:
                            best_conflicts = conflicts
                            best_r = r
                    r = best_r
                selected_indices.append(m_idx)
                selected_residues.append(r)
        
        return selected_indices, selected_residues
    
    def local_search(current_indices, current_residues, time_budget):
        """Simulated annealing local search."""
        curr_idx = list(current_indices)
        curr_res = list(current_residues)
        curr_score = get_score_float(curr_idx)
        
        best_idx = list(curr_idx)
        best_res = list(curr_res)
        best_score = curr_score
        
        T = 0.1
        cooling_rate = 0.9995
        min_T = 0.001
        
        step = 0
        while T > min_T and time.time() < time_budget:
            step += 1
            move_type = random.random()
            
            new_idx = list(curr_idx)
            new_res = list(curr_res)
            
            if move_type < 0.5 and len(curr_idx) < n_moduli:
                # Add a modulus
                # Prefer smaller moduli
                candidates = [i for i in range(n_moduli) if i not in curr_idx]
                if candidates:
                    # Weight by 1/moduli
                    weights = [1.0 / moduli[c] for c in candidates]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cum = 0
                    chosen = candidates[0]
                    for c, w in zip(candidates, weights):
                        cum += w
                        if r <= cum:
                            chosen = c
                            break
                    new_idx.append(chosen)
                    valid = find_valid_residues(chosen, curr_idx, curr_res)
                    if valid:
                        new_res.append(random.choice(valid))
                    else:
                        continue
            elif move_type < 0.9 and curr_idx:
                # Change residue
                pos = random.randint(0, len(curr_idx) - 1)
                m_idx = curr_idx[pos]
                old_r = curr_res[pos]
                temp_idx = curr_idx[:pos] + curr_idx[pos+1:]
                temp_res = curr_res[:pos] + curr_res[pos+1:]
                valid = find_valid_residues(m_idx, temp_idx, temp_res)
                if valid:
                    new_r = random.choice(valid)
                    new_res = list(curr_res)
                    new_res[pos] = new_r
                else:
                    continue
            else:
                # Remove a modulus
                if not curr_idx:
                    continue
                pos = random.randint(0, len(curr_idx) - 1)
                new_idx = list(curr_idx)
                new_res = list(curr_res)
                new_idx.pop(pos)
                new_res.pop(pos)
            
            new_score = get_score_float(new_idx)
            delta = new_score - curr_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                curr_idx = new_idx
                curr_res = new_res
                curr_score = new_score
                
                if curr_score > best_score:
                    best_score = curr_score
                    best_idx = list(curr_idx)
                    best_res = list(curr_res)
            
            T *= cooling_rate
        
        return best_idx, best_res, best_score
    
    def hill_climb_2swap(current_indices, current_residues, time_budget):
        """Try to improve by swapping residues or adding/removing pairs."""
        curr_idx = list(current_indices)
        curr_res = list(current_residues)
        curr_score = get_score_float(curr_idx)
        
        best_idx = list(curr_idx)
        best_res = list(curr_res)
        best_score = curr_score
        
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            for i in range(len(curr_idx)):
                for j in range(len(curr_idx)):
                    if i == j:
                        continue
                    # Try swapping residues i and j
                    old_i = curr_res[i]
                    old_j = curr_res[j]
                    if old_i == old_j:
                        continue
                    
                    # Check if swap is valid
                    new_i = old_j
                    new_j = old_i
                    
                    # Check i with all others
                    valid = True
                    m_i = curr_idx[i]
                    for k in range(len(curr_idx)):
                        if k == i:
                            continue
                        m_k = curr_idx[k]
                        r_k = curr_res[k]
                        if k == j:
                            r_k = new_j
                        g = gcds[m_i][m_k]
                        if (new_i - r_k) % g == 0:
                            valid = False
                            break
                    
                    if valid:
                        m_j = curr_idx[j]
                        for k in range(len(curr_idx)):
                            if k == j:
                                continue
                            m_k = curr_idx[k]
                            r_k = curr_res[k]
                            if k == i:
                                r_k = new_i
                            g = gcds[m_j][m_k]
                            if (new_j - r_k) % g == 0:
                                valid = False
                                break
                    
                    if valid:
                        curr_res[i] = new_i
                        curr_res[j] = new_j
                        new_score = get_score_float(curr_idx)
                        if new_score > best_score:
                            best_score = new_score
                            best_idx = list(curr_idx)
                            best_res = list(curr_res)
                        else:
                            curr_res[i] = old_i
                            curr_res[j] = old_j
                        improved = True
            
            # Try adding one modulus
            candidates = [i for i in range(n_moduli) if i not in curr_idx]
            random.shuffle(candidates)
            for c in candidates[:50]:
                valid = find_valid_residues(c, curr_idx, curr_res)
                if valid:
                    r = random.choice(valid)
                    new_idx = curr_idx + [c]
                    new_res = curr_res + [r]
                    new_score = get_score_float(new_idx)
                    if new_score > best_score:
                        best_score = new_score
                        best_idx = new_idx
                        best_res = new_res
                        curr_idx = list(new_idx)
                        curr_res = list(new_res)
                        improved = True
                    if time.time() > time_budget:
                        break
            
            # Try removing one modulus and replacing with better
            for i in range(len(curr_idx)):
                if time.time() > time_budget:
                    break
                saved_idx = curr_idx[i]
                saved_res = curr_res[i]
                temp_idx = curr_idx[:i] + curr_idx[i+1:]
                temp_res = curr_res[:i] + curr_res[i+1:]
                
                # Try to add back a different modulus
                for c in range(n_moduli):
                    if c == saved_idx or c in temp_idx:
                        continue
                    valid = find_valid_residues(c, temp_idx, temp_res)
                    if valid:
                        r = random.choice(valid)
                        new_idx = temp_idx + [c]
                        new_res = temp_res + [r]
                        new_score = get_score_float(new_idx)
                        if new_score > best_score:
                            best_score = new_score
                            best_idx = list(new_idx)
                            best_res = list(new_res)
                            curr_idx = list(new_idx)
                            curr_res = list(new_res)
                            improved = True
                            break
                
                curr_idx = temp_idx + [saved_idx]
                curr_res = temp_res + [saved_res]
        
        return best_idx, best_res, best_score
    
    # Main algorithm
    best_overall_idx = []
    best_overall_res = []
    best_overall_score = 0.0
    
    # Phase 1: Multiple greedy initializations with local search
    n_greedy_restarts = 8
    time_per_phase = 35.0
    
    for restart in range(n_greedy_restarts):
        phase_start = time.time()
        phase_deadline = min(phase_start + time_per_phase, deadline)
        
        # Greedy initialization
        if restart % 2 == 0:
            init_idx, init_res = greedy_construction()
        else:
            init_idx, init_res = greedy_randomized(alpha=0.3)
        
        # Local search
        ls_idx, ls_res, ls_score = local_search(init_idx, init_res, phase_deadline)
        
        if ls_score > best_overall_score:
            best_overall_score = ls_score
            best_overall_idx = ls_idx
            best_overall_res = ls_res
        
        if time.time() >= deadline:
            break
    
    # Phase 2: Hill climbing on the best solution
    if best_overall_idx and time.time() < deadline:
        hc_deadline = min(time.time() + 25.0, deadline)
        hc_idx, hc_res, hc_score = hill_climb_2swap(best_overall_idx, best_overall_res, hc_deadline)
        
        if hc_score > best_overall_score:
            best_overall_score = hc_score
            best_overall_idx = hc_idx
            best_overall_res = hc_res
    
    # Phase 3: Final polish with more local search iterations
    if best_overall_idx and time.time() < deadline:
        final_deadline = deadline
        # Run a few more short local searches from perturbed versions
        for polish in range(3):
            if time.time() >= deadline:
                break
            
            # Perturb the best solution
            perturbed_idx = list(best_overall_idx)
            perturbed_res = list(best_overall_res)
            
            # Randomly change a few residues
            for _ in range(min(3, len(perturbed_idx))):
                pos = random.randint(0, len(perturbed_idx) - 1)
                m_idx = perturbed_idx[pos]
                temp_idx = perturbed_idx[:pos] + perturbed_idx[pos+1:]
                temp_res = perturbed_res[:pos] + perturbed_res[pos+1:]
                valid = find_valid_residues(m_idx, temp_idx, temp_res)
                if valid:
                    perturbed_res[pos] = random.choice(valid)
            
            # Short local search
            ls_idx, ls_res, ls_score = local_search(perturbed_idx, perturbed_res, final_deadline)
            
            if ls_score > best_overall_score:
                best_overall_score = ls_score
                best_overall_idx = ls_idx
                best_overall_res = ls_res
    
    # Convert to required format
    result_classes = []
    for i in range(len(best_overall_idx)):
        m_idx = best_overall_idx[i]
        r = best_overall_res[i]
        m = moduli[m_idx]
        result_classes.append([m, r])
    
    return result_classes
