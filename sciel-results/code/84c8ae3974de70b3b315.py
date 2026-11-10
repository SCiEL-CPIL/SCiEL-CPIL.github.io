import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    def score(sel):
        return sum(1.0 / m for m, r in sel)
    
    def find_one_valid_residue(m, sel):
        """Finds one valid residue for m given sel, or returns None."""
        for r in range(m):
            ok = True
            for em, er in sel:
                g = math.gcd(m, em)
                if r % g == er % g:
                    ok = False
                    break
            if ok:
                return r
        return None

    def greedy_construction(order):
        sel = []
        for m in order:
            if time.time() > deadline:
                break
            r = find_one_valid_residue(m, sel)
            if r is not None:
                sel.append((m, r))
        return sel

    def local_search(sel, time_limit):
        best = list(sel)
        best_s = score(best)
        all_moduli = list(range(20, 201))
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try to add any unselected modulus (prefer smaller)
            selected_m = set(m for m, r in best)
            unselected = [m for m in all_moduli if m not in selected_m]
            unselected.sort() # Smallest first
            
            for m_new in unselected:
                if time.time() > time_limit:
                    break
                r_new = find_one_valid_residue(m_new, best)
                if r_new is not None:
                    new_sel = best + [(m_new, r_new)]
                    s = score(new_sel)
                    if s > best_s + 1e-12:
                        best = new_sel
                        best_s = s
                        improved = True
                        break # Restart loop with new state
            
            # 2. Try to swap: remove large m_old, add small m_new
            if not improved:
                # Sort current by m descending to try removing largest first
                current_ms = sorted([m for m, r in best], reverse=True)
                
                for m_old in current_ms:
                    if time.time() > time_limit:
                        break
                    
                    # Find index of m_old
                    idx = next((i for i, (m, r) in enumerate(best) if m == m_old), None)
                    if idx is None: continue
                    remaining = [x for i, x in enumerate(best) if i != idx]
                    
                    # Try to add any m_new < m_old that is not in remaining
                    selected_in_rem = set(m for m, r in remaining)
                    candidates = [m for m in all_moduli if m < m_old and m not in selected_in_rem]
                    candidates.sort() # Smallest first
                    
                    for m_new in candidates:
                        if time.time() > time_limit:
                            break
                        r_new = find_one_valid_residue(m_new, remaining)
                        if r_new is not None:
                            new_sel = remaining + [(m_new, r_new)]
                            s = score(new_sel)
                            if s > best_s + 1e-12:
                                best = new_sel
                                best_s = s
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best, best_s

    best_solution = []
    best_score = 0.0
    
    # Phase 1: Greedy with ascending moduli (favors small m)
    base_order = list(range(20, 201))
    
    # A few random restarts with slight perturbations
    for restart in range(50):
        if time.time() > deadline - 20:
            break
        
        rng = random.Random(restart)
        # Perturb the order slightly or just use ascending
        order = base_order[:]
        # To add diversity, we can shuffle the first few or last few
        # But ascending is strong. Let's try different residue choices by shuffling the 'r' selection?
        # find_one_valid_residue picks the first one. 
        # We can modify it to pick randomly from valid ones if we want diversity.
        # But for speed, first one is fine.
        
        sel = greedy_construction(order)
        s = score(sel)
        
        if s > best_score:
            best_score = s
            best_solution = sel
    
    # Phase 2: Local Search
    if best_solution and time.time() < deadline - 5:
        refined, refined_score = local_search(best_solution, deadline - 5)
        if refined_score > best_score:
            best_score = refined_score
            best_solution = refined

    return [[m, r] for m, r in best_solution]
