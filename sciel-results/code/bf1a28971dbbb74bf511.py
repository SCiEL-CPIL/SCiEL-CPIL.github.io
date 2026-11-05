import math
import time
import heapq
from fractions import Fraction

def construct():
    deadline = time.time() + 130  # safety margin
    
    # Precompute all valid (m, r) pairs and their compatibility
    # Node: (m, r) with weight 1/m
    # Two nodes (m1, r1) and (m2, r2) are incompatible if r1 ≡ r2 (mod gcd(m1, m2))
    
    all_moduli = list(range(20, 201))
    num_moduli = len(all_moduli)
    
    # For each modulus m, generate all residues
    # Total nodes = sum(m for m in 20..200) = quite large
    # But we can be smart: for each m, we only need to consider one r at a time
    
    # Strategy: Process moduli in order of decreasing weight (increasing m is bad, so decreasing m first)
    # Actually, weight is 1/m, so smaller m has higher weight. Process in order of increasing m.
    
    # Precompute gcd table
    gcd_table = [[0] * (num_moduli + 1) for _ in range(num_moduli + 1)]
    for i, m1 in enumerate(all_moduli):
        for j, m2 in enumerate(all_moduli):
            gcd_table[i][j] = math.gcd(m1, m2)
    
    # For branch and bound:
    # State: which moduli have been processed, and for each processed modulus, which residue was chosen (or skipped)
    # This is too large for explicit state space.
    
    # Better approach: Greedy + local search with structured moves
    # Or: Use the fact that we can process moduli one by one, and for each m, choose r or skip
    
    # Let's use a different structural approach:
    # The key insight is that the constraint is r1 ≢ r2 (mod gcd(m1, m2))
    # For prime moduli, gcd = 1, so all residues are compatible with everything (no constraint)
    # For composite moduli sharing factors, constraints arise
    
    # Approach: Start with a good greedy solution, then use swap-based local search
    # with a focus on adding high-value (small m) moduli that were skipped
    
    # First, get a greedy baseline
    selected = {}  # m -> r
    for m in range(20, 201):
        if time.time() > deadline:
            break
        for r in range(m):
            valid = True
            for m2, r2 in selected.items():
                g = math.gcd(m, m2)
                if r % g == r2 % g:
                    valid = False
                    break
            if valid:
                selected[m] = r
                break
    
    # Now do local search: try to improve by swapping residues or adding/removing moduli
    def get_score(sel):
        return sum(1.0 / m for m in sel)
    
    current_score = get_score(selected)
    best_score = current_score
    best_selected = dict(selected)
    
    # Local search with multiple restarts and different strategies
    import random
    random.seed(42)
    
    while time.time() < deadline - 5:
        # Try various moves
        improved = False
        
        # Move 1: Try to add a missing modulus by changing some existing residues
        missing = [m for m in range(20, 201) if m not in selected]
        random.shuffle(missing)
        
        for m_new in missing:
            if time.time() > deadline:
                break
            if 1.0 / m_new < 1e-6:
                continue
            
            # Try to find a residue for m_new that's compatible with current selection
            # If not, try to swap: remove one modulus and add m_new
            found_direct = False
            for r_new in range(m_new):
                valid = True
                for m2, r2 in selected.items():
                    g = math.gcd(m_new, m2)
                    if r_new % g == r2 % g:
                        valid = False
                        break
                if valid:
                    selected[m_new] = r_new
                    new_score = get_score(selected)
                    if new_score > current_score:
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best_selected = dict(selected)
                    else:
                        del selected[m_new]
                    found_direct = True
                    break
            
            if found_direct:
                improved = True
                continue
            
            # Try swap: remove m_old, add m_new
            # Only try if 1/m_new > 1/m_old (i.e., m_new < m_old)
            candidates_to_remove = [m_old for m_old in selected if m_old > m_new]
            random.shuffle(candidates_to_remove)
            
            for m_old in candidates_to_remove[:5]:  # Limit work
                if time.time() > deadline:
                    break
                # Temporarily remove m_old
                saved_r_old = selected[m_old]
                del selected[m_old]
                
                for r_new in range(m_new):
                    valid = True
                    for m2, r2 in selected.items():
                        g = math.gcd(m_new, m2)
                        if r_new % g == r2 % g:
                            valid = False
                            break
                    if valid:
                        selected[m_new] = r_new
                        new_score = get_score(selected)
                        if new_score > current_score:
                            current_score = new_score
                            if new_score > best_score:
                                best_score = new_score
                                best_selected = dict(selected)
                            improved = True
                        else:
                            del selected[m_new]
                        break
                
                # Restore if we didn't keep the new state
                if m_new not in selected:
                    selected[m_old] = saved_r_old
                elif m_old not in selected:
                    # We swapped
                    pass
                else:
                    # Both present? That shouldn't happen
                    selected[m_old] = saved_r_old
        
        if not improved:
            # Try perturbation: randomly change some residues
            if len(selected) > 5:
                keys = list(selected.keys())
                random.shuffle(keys)
                for m in keys[:3]:
                    old_r = selected[m]
                    for _ in range(10):
                        new_r = random.randint(0, m - 1)
                        if new_r == old_r:
                            continue
                        valid = True
                        for m2, r2 in selected.items():
                            if m2 == m:
                                continue
                            g = math.gcd(m, m2)
                            if new_r % g == r2 % g:
                                valid = False
                                break
                        if valid:
                            selected[m] = new_r
                            new_score = get_score(selected)
                            if new_score > current_score:
                                current_score = new_score
                                if new_score > best_score:
                                    best_score = new_score
                                    best_selected = dict(selected)
                            else:
                                selected[m] = old_r
                            break
    
    # Return best found
    return [[m, r] for m, r in sorted(best_selected.items())]
