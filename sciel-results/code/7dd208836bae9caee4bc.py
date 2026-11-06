import math
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Find pairwise disjoint residue classes with distinct moduli in [20, 200]
    maximizing the sum of 1/m.
    
    Two classes (m1, r1) and (m2, r2) are disjoint iff r1 != r2 mod gcd(m1, m2).
    """
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Precompute gcd for all pairs of moduli in [20, 200]
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    import random
    rng = random.Random(42)
    
    def is_valid_residue(m_idx, r, selected):
        """Check if residue r for modulus at index m_idx is valid given selected classes."""
        for (sel_idx, sel_r) in selected:
            g = gcd_table[m_idx][sel_idx]
            if r % g == sel_r % g:
                return False
        return True
    
    def calc_score(selected):
        score = Fraction(0)
        for idx, r in selected:
            score += Fraction(1, moduli[idx])
        return score
    
    def greedy_construct(strategy='first'):
        """Greedy: for each m in increasing order, pick a valid residue."""
        selected = []
        for m_idx in range(n_mod):
            m = moduli[m_idx]
            if strategy == 'first':
                for r in range(m):
                    if is_valid_residue(m_idx, r, selected):
                        selected.append((m_idx, r))
                        break
            elif strategy == 'random':
                valid_residues = [r for r in range(m) if is_valid_residue(m_idx, r, selected)]
                if valid_residues:
                    r = rng.choice(valid_residues)
                    selected.append((m_idx, r))
            elif strategy == 'min_conflict':
                # Choose residue that minimizes future conflicts
                # Count how many future moduli would be blocked
                best_r = None
                best_conflicts = float('inf')
                for r in range(m):
                    if not is_valid_residue(m_idx, r, selected):
                        continue
                    conflicts = 0
                    for future_idx in range(m_idx + 1, n_mod):
                        g = gcd_table[m_idx][future_idx]
                        # This residue r blocks r % g mod g for future_idx
                        # Count how many residues of future_idx are blocked
                        # Actually, it blocks exactly 1 residue class mod g
                        # The number of blocked residues in [0, future_m) is future_m // g
                        # But we just need to know if it's likely to cause issues
                        # Simpler: count total blocked residue slots
                        future_m = moduli[future_idx]
                        blocked = future_m // g
                        conflicts += blocked
                    if conflicts < best_conflicts:
                        best_conflicts = conflicts
                        best_r = r
                if best_r is not None:
                    selected.append((m_idx, best_r))
            elif strategy == 'max_spread':
                # Choose residue that maximizes minimum distance to existing residues mod gcd
                valid_residues = [r for r in range(m) if is_valid_residue(m_idx, r, selected)]
                if valid_residues:
                    if len(valid_residues) == 1:
                        selected.append((m_idx, valid_residues[0]))
                    else:
                        # Pick the middle one or random
                        r = rng.choice(valid_residues)
                        selected.append((m_idx, r))
        return selected
    
    def local_search(selected, time_budget):
        """Improve solution via local search: try to add skipped moduli, remove low-value ones."""
        selected = list(selected)
        score = calc_score(selected)
        selected_mods = set(idx for idx, _ in selected)
        skipped = [i for i in range(n_mod) if i not in selected_mods]
        
        start_time = time.time()
        improved = True
        
        while improved and time.time() - start_time < time_budget:
            improved = False
            
            # Try to add skipped moduli
            for skip_idx in list(skipped):
                if time.time() - start_time > time_budget:
                    break
                m = moduli[skip_idx]
                
                # Try to find a valid residue directly
                for r in range(m):
                    if is_valid_residue(skip_idx, r, selected):
                        selected.append((skip_idx, r))
                        score += Fraction(1, m)
                        selected_mods.add(skip_idx)
                        skipped.remove(skip_idx)
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try removing one low-value class to add a higher-value one
            # Sort selected by value (1/m) ascending - remove lowest first
            selected_sorted = sorted(selected, key=lambda x: moduli[x[0]])  # ascending m = descending 1/m... wait
            # Actually we want to remove HIGH m (low 1/m) to add LOW m (high 1/m)
            # So sort by m descending to remove highest m first
            selected_by_m_desc = sorted(selected, key=lambda x: -moduli[x[0]])
            
            for rem_idx_pos, (rem_idx, rem_r) in enumerate(selected_by_m_desc):
                if time.time() - start_time > time_budget:
                    break
                rem_m = moduli[rem_idx]
                
                # Try removing this and adding one skipped
                for skip_idx in skipped:
                    if moduli[skip_idx] >= rem_m:
                        continue  # Only add if it's better (lower m = higher 1/m)
                    
                    # Try removing rem_idx and adding skip_idx
                    new_selected = [s for s in selected if s[0] != rem_idx]
                    m = moduli[skip_idx]
                    
                    for r in range(m):
                        if is_valid_residue(skip_idx, r, new_selected):
                            new_selected.append((skip_idx, r))
                            new_score = calc_score(new_selected)
                            if new_score > score:
                                selected = new_selected
                                score = new_score
                                selected_mods = set(idx for idx, _ in selected)
                                skipped = [i for i in range(n_mod) if i not in selected_mods]
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            # Try swapping residues of existing classes to enable adding a skipped one
            if not improved and time.time() - start_time < time_budget:
                for skip_idx in list(skipped):
                    if time.time() - start_time > time_budget:
                        break
                    m = moduli[skip_idx]
                    
                    # For each existing class, try changing its residue
                    for i, (sel_idx, sel_r) in enumerate(selected):
                        if time.time() - start_time > time_budget:
                            break
                        g = gcd_table[skip_idx][sel_idx]
                        
                        # Try a few alternative residues for sel_idx
                        tried = 0
                        for new_r in range(moduli[sel_idx]):
                            if new_r == sel_r:
                                continue
                            tried += 1
                            if tried > 20:  # Limit attempts
                                break
                            
                            # Check if new_r is valid for sel_idx with all other selected
                            valid = True
                            for j, (other_idx, other_r) in enumerate(selected):
                                if j == i:
                                    continue
                                if new_r % gcd_table[sel_idx][other_idx] == other_r % gcd_table[sel_idx][other_idx]:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            
                            # Now check if we can find an r for skip_idx
                            found_r = False
                            for r in range(m):
                                if r % g == new_r % g:
                                    continue
                                # Check against all other selected
                                ok = True
                                for j, (other_idx, other_r) in enumerate(selected):
                                    if j == i:
                                        continue
                                    if r % gcd_table[skip_idx][other_idx] == other_r % gcd_table[skip_idx][other_idx]:
                                        ok = False
                                        break
                                if ok:
                                    # Apply the change
                                    selected[i] = (sel_idx, new_r)
                                    selected.append((skip_idx, r))
                                    score += Fraction(1, m)
                                    selected_mods.add(skip_idx)
                                    skipped.remove(skip_idx)
                                    improved = True
                                    found_r = True
                                    break
                            if found_r:
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return selected, score
    
    # Phase 1: Multiple greedy starts with different strategies
    best_selected = None
    best_score = Fraction(0)
    
    strategies = ['first', 'random', 'min_conflict', 'max_spread']
    for strategy in strategies:
        selected = greedy_construct(strategy)
        score = calc_score(selected)
        if score > best_score:
            best_score = score
            best_selected = selected
    
    # More random restarts
    start_time = time.time()
    for attempt in range(30):
        if time.time() - start_time > 20:
            break
        selected = greedy_construct('random')
        score = calc_score(selected)
        if score > best_score:
            best_score = score
            best_selected = selected
    
    # Phase 2: Local search on best solution
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_selected, best_score = local_search(best_selected, min(remaining_time - 2, 60))
    
    # Phase 3: More local search iterations with different starting points
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        # Try local search from a few other good solutions
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            selected = greedy_construct('random')
            score = calc_score(selected)
            if score > best_score - 0.5:  # Only if reasonably close
                budget = min(deadline - time.time() - 3, 20)
                if budget > 2:
                    new_selected, new_score = local_search(selected, budget)
                    if new_score > best_score:
                        best_score = new_score
                        best_selected = new_selected
    
    # Convert to output format
    result = [[moduli[idx], r] for idx, r in best_selected]
    return result
# EVOLVE-BLOCK-END
