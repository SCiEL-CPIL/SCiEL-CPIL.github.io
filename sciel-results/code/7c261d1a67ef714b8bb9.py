"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit
    
    # Moduli in [20, 200]
    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    
    # Precompute gcd matrix for all pairs of moduli
    gcd_table = [[math.gcd(moduli[i], moduli[j]) for j in range(n_moduli)] for i in range(n_moduli)]
    
    def check_pairwise_disjoint(selected):
        """Check if all pairs in selected are disjoint. selected is list of (m_idx, r)."""
        n = len(selected)
        for i in range(n):
            mi, ri = selected[i]
            for j in range(i + 1, n):
                mj, rj = selected[j]
                g = gcd_table[mi][mj]
                if (ri - rj) % g == 0:
                    return False
        return True
    
    def score_solution(selected):
        """Calculate sum of 1/m for the selected classes. selected is list of (m_idx, r)."""
        return sum(1.0 / moduli[mi] for mi, r in selected)
    
    def find_valid_residue(m_idx, selected):
        """Find a residue r for modulus m_idx that is compatible with all in selected."""
        m = moduli[m_idx]
        for r in range(m):
            valid = True
            for mi_old, r_old in selected:
                g = gcd_table[m_idx][mi_old]
                if (r - r_old) % g == 0:
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def greedy_initial():
        """Greedy construction: try moduli in ascending order, pick first valid residue."""
        selected = []
        for mi in range(n_moduli):
            if time.time() > deadline:
                break
            r = find_valid_residue(mi, selected)
            if r is not None:
                selected.append((mi, r))
        return selected
    
    def simulated_annealing(initial_solution, max_time):
        """
        Simulated annealing with three types of moves:
        1. Add a new modulus (if a valid residue exists)
        2. Remove a modulus
        3. Reassign residue of an existing modulus
        
        This allows escaping local minima by temporarily accepting worse states.
        """
        selected = list(initial_solution)
        current_score = score_solution(selected)
        best_selected = list(selected)
        best_score = current_score
        
        T = 0.1  # Initial temperature
        cooling = 0.999
        min_T = 0.001
        
        selected_set = set(mi for mi, r in selected)
        
        while time.time() < max_time and T > min_T:
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.35:
                # Move 1: Add a new modulus
                candidates = [mi for mi in range(n_moduli) if mi not in selected_set]
                if not candidates:
                    T *= cooling
                    continue
                mi = random.choice(candidates)
                # Try to find a valid residue
                r = find_valid_residue(mi, selected)
                if r is None:
                    T *= cooling
                    continue
                new_selected = selected + [(mi, r)]
                new_score = current_score + 1.0 / moduli[mi]
                delta = new_score - current_score
                # Always accept improvements; accept worsenings with probability exp(-delta/T)
                if delta >= 0 or random.random() < math.exp(-delta / T):
                    selected = new_selected
                    current_score = new_score
                    selected_set.add(mi)
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = list(selected)
                        
            elif move_type < 0.65:
                # Move 2: Remove a modulus (preferentially larger ones)
                if not selected:
                    T *= cooling
                    continue
                # Weight removal toward larger moduli (less valuable)
                weights = [1.0 / moduli[mi] for mi, r in selected]
                total_w = sum(weights)
                if total_w == 0:
                    T *= cooling
                    continue
                r_rand = random.random() * total_w
                cum = 0
                remove_idx = len(selected) - 1
                for i, w in enumerate(weights):
                    cum += w
                    if r_rand <= cum:
                        remove_idx = i
                        break
                mi_removed, r_removed = selected[remove_idx]
                new_selected = selected[:remove_idx] + selected[remove_idx + 1:]
                new_score = current_score - 1.0 / moduli[mi_removed]
                delta = new_score - current_score
                if delta >= 0 or random.random() < math.exp(-delta / T):
                    selected = new_selected
                    current_score = new_score
                    selected_set.discard(mi_removed)
                    
            else:
                # Move 3: Reassign residue of an existing modulus
                if not selected:
                    T *= cooling
                    continue
                idx = random.randrange(len(selected))
                mi, old_r = selected[idx]
                m = moduli[mi]
                # Try a random different residue
                new_r = old_r
                attempts = 0
                while new_r == old_r and attempts < 50:
                    new_r = random.randrange(m)
                    attempts += 1
                if new_r == old_r:
                    T *= cooling
                    continue
                # Check if new_r is compatible with all OTHER selected moduli
                valid = True
                for j, (mi_old, r_old) in enumerate(selected):
                    if j == idx:
                        continue
                    g = gcd_table[mi][mi_old]
                    if (new_r - r_old) % g == 0:
                        valid = False
                        break
                if not valid:
                    T *= cooling
                    continue
                # Score doesn't change with residue reassignment
                delta = 0.0
                # Accept if it doesn't break feasibility (which we already checked)
                # But we should only accept if it potentially helps future additions
                # For now, accept with some probability to explore
                if random.random() < 0.5:
                    selected[idx] = (mi, new_r)
                    # current_score unchanged
                    
            T *= cooling
            
            # Periodic greedy fill: try to add any missing modulus that fits
            if random.random() < 0.1:
                for mi in range(n_moduli):
                    if mi not in selected_set:
                        r = find_valid_residue(mi, selected)
                        if r is not None:
                            selected.append((mi, r))
                            selected_set.add(mi)
                            current_score += 1.0 / moduli[mi]
                            if current_score > best_score:
                                best_score = current_score
                                best_selected = list(selected)
        
        return best_selected, best_score
    
    # Phase 1: Greedy initial solution
    best_solution = greedy_initial()
    best_score = score_solution(best_solution)
    
    # Phase 2: Simulated annealing from the greedy solution
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        sa_solution, sa_score = simulated_annealing(best_solution, deadline - 5)
        if sa_score > best_score:
            best_solution = sa_solution
            best_score = sa_score
    
    # Phase 3: Multiple random restarts with different greedy orders
    if time.time() < deadline - 15:
        restart_attempts = 0
        while time.time() < deadline - 5:
            restart_attempts += 1
            # Randomized greedy: shuffle moduli, but bias toward smaller ones
            shuffled = list(range(n_moduli))
            # Partial shuffle: keep small moduli mostly in order
            random.shuffle(shuffled)
            # Re-sort with some randomness: small moduli get priority but with noise
            shuffled.sort(key=lambda mi: moduli[mi] + random.random() * 10)
            
            selected = []
            for mi in shuffled:
                if time.time() > deadline - 5:
                    break
                r = find_valid_residue(mi, selected)
                if r is not None:
                    selected.append((mi, r))
            
            sc = score_solution(selected)
            if sc > best_score:
                best_score = sc
                best_solution = list(selected)
            
            # Quick SA on this restart
            remaining = deadline - time.time()
            if remaining > 5:
                sa_sol, sa_sc = simulated_annealing(selected, deadline - 3)
                if sa_sc > best_score:
                    best_score = sa_sc
                    best_solution = sa_sol
    
    # Convert to required format [[m, r], ...]
    result = [[moduli[mi], r] for mi, r in best_solution]
    return result
# EVOLVE-BLOCK-END
