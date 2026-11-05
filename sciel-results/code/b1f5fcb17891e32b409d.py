import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (150s limit - 35s buffer)

    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    
    best_selected = []
    best_score = 0.0

    def is_compatible(m_new, r_new, selected_list):
        for m_old, r_old in selected_list:
            g = math.gcd(m_new, m_old)
            if (r_new - r_old) % g == 0:
                return False
        return True

    def compute_score(selected):
        return sum(Fraction(1, m) for m, r in selected)

    def run_random_greedy():
        selected = []
        for m in moduli:
            valid_residues = []
            for r in range(m):
                if is_compatible(m, r, selected):
                    valid_residues.append(r)
            
            if valid_residues:
                r = random.choice(valid_residues)
                selected.append([m, r])
        
        score = compute_score(selected)
        return selected, float(score)

    def run_local_search_polish(selected, score, time_limit=2.0):
        """Intensive local search to polish a given solution."""
        polish_start = time.time()
        best_selected_ls = list(selected)
        best_score_ls = score
        current_selected = list(selected)
        current_score = score
        
        # Precompute gcd matrix for speed
        n = len(current_selected)
        if n == 0:
            return best_selected_ls, best_score_ls
        
        # Try all possible residue changes for each modulus
        no_improve_count = 0
        max_no_improve = 2000
        
        while no_improve_count < max_no_improve and (time.time() - polish_start) < time_limit:
            improved = False
            # Shuffle the order of indices to avoid bias
            indices = list(range(len(current_selected)))
            random.shuffle(indices)
            
            for idx in indices:
                m, r_old = current_selected[idx]
                # Find all valid residues for this modulus given the rest
                best_r = r_old
                best_delta = 0.0
                
                for r_new in range(m):
                    if r_new == r_old:
                        continue
                    # Check compatibility with all others
                    compatible = True
                    for j, (m_j, r_j) in enumerate(current_selected):
                        if j == idx:
                            continue
                        g = math.gcd(m, m_j)
                        if (r_new - r_j) % g == 0:
                            compatible = False
                            break
                    if compatible:
                        # Score doesn't change with residue, so any valid residue is fine
                        # But we want to explore the space, so we track improvements via swaps
                        # Actually, score only depends on moduli, not residues.
                        # So local search on residues alone won't improve score directly.
                        # We need to add/remove moduli or swap moduli.
                        pass
                
                # Since score is independent of residue choice, we focus on add/remove/swap
                # Let's try removing a modulus
                # Or adding a new one
                pass
            
            # Try adding a new modulus
            added = False
            available_moduli = [m for m in moduli if m not in [s[0] for s in current_selected]]
            random.shuffle(available_moduli)
            for m_new in available_moduli[:50]:  # Limit to keep it fast
                valid_residues = [r for r in range(m_new) if is_compatible(m_new, r, current_selected)]
                if valid_residues:
                    r_new = random.choice(valid_residues)
                    new_selected = current_selected + [[m_new, r_new]]
                    new_score = current_score + Fraction(1, m_new)
                    if new_score > best_score_ls:
                        best_score_ls = new_score
                        best_selected_ls = list(new_selected)
                        current_selected = new_selected
                        current_score = new_score
                        improved = True
                        added = True
                        break
            
            # Try removing a modulus (least useful, i.e., largest m)
            if not improved:
                if len(current_selected) > 1:
                    # Sort by m descending, try removing the largest
                    sorted_indices = sorted(range(len(current_selected)), key=lambda i: -current_selected[i][0])
                    for idx in sorted_indices[:10]:
                        removed = current_selected.pop(idx)
                        new_score = current_score - Fraction(1, removed[0])
                        # This always decreases score, so we only do this if it enables better adds later
                        # For now, skip pure removals in polish, or do them rarely
                        current_selected.insert(idx, removed)
            
            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0
        
        return best_selected_ls, best_score_ls

    def run_simulated_annealing():
        selected, score = run_random_greedy()
        best_selected_sa = list(selected)
        best_score_sa = score
        
        T = 0.1  # Lower initial temperature for more focused search
        cooling_rate = 0.99
        min_temperature = 0.001
        sa_start = time.time()
        sa_time_limit = 5.0  # Limit SA time per run
        
        while T > min_temperature and (time.time() - sa_start) < sa_time_limit:
            move_type = random.random()
            
            if move_type < 0.4:
                # Add a new modulus
                available_moduli = [m for m in moduli if m not in [s[0] for s in selected]]
                if available_moduli:
                    m_new = random.choice(available_moduli)
                    valid_residues = [r for r in range(m_new) if is_compatible(m_new, r, selected)]
                    if valid_residues:
                        r_new = random.choice(valid_residues)
                        selected.append([m_new, r_new])
                        new_score = score + Fraction(1, m_new)
                        delta = float(new_score - score)
                        if delta > 0 or random.random() < math.exp(delta / T):
                            score = new_score
                            if score > best_score_sa:
                                best_score_sa = score
                                best_selected_sa = list(selected)
                        else:
                            selected.pop()
            elif move_type < 0.7:
                # Remove a modulus
                if len(selected) > 1:
                    idx = random.randint(0, len(selected) - 1)
                    removed = selected.pop(idx)
                    new_score = score - Fraction(1, removed[0])
                    delta = float(new_score - score)
                    if delta > 0 or random.random() < math.exp(delta / T):
                        score = new_score
                        if score > best_score_sa:
                            best_score_sa = score
                            best_selected_sa = list(selected)
                    else:
                        selected.insert(idx, removed)
            else:
                # Swap: remove one, add another
                if len(selected) > 1:
                    idx = random.randint(0, len(selected) - 1)
                    removed = selected[idx]
                    selected.pop(idx)
                    available_moduli = [m for m in moduli if m not in [s[0] for s in selected]]
                    if available_moduli:
                        m_new = random.choice(available_moduli)
                        valid_residues = [r for r in range(m_new) if is_compatible(m_new, r, selected)]
                        if valid_residues:
                            r_new = random.choice(valid_residues)
                            selected.append([m_new, r_new])
                            new_score = score - Fraction(1, removed[0]) + Fraction(1, m_new)
                            delta = float(new_score - score)
                            if delta > 0 or random.random() < math.exp(delta / T):
                                score = new_score
                                if score > best_score_sa:
                                    best_score_sa = score
                                    best_selected_sa = list(selected)
                            else:
                                selected.pop()
                                selected.insert(idx, removed)
                        else:
                            selected.insert(idx, removed)
                    else:
                        selected.insert(idx, removed)
            
            T *= cooling_rate
        
        # Polish the best solution found
        if best_selected_sa:
            best_selected_sa, best_score_sa = run_local_search_polish(best_selected_sa, best_score_sa, time_limit=1.0)
        
        return best_selected_sa, best_score_sa

    # Run multiple restarts
    restarts = 0
    while time.time() < deadline - 5.0:
        if restarts % 2 == 0:
            selected, score = run_simulated_annealing()
        else:
            selected, score = run_random_greedy()
            # Polish the greedy result
            if selected:
                selected, score = run_local_search_polish(selected, score, time_limit=0.5)
        
        if score > best_score:
            best_score = score
            best_selected = selected
        restarts += 1
    
    # Final polish of the best solution
    if best_selected:
        best_selected, best_score = run_local_search_polish(best_selected, best_score, time_limit=2.0)

    return best_selected
