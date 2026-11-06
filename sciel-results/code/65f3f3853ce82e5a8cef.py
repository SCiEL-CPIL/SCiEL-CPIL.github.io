import time
import math
import random

def construct():
    deadline = time.time() + 110  # Safety margin
    
    moduli = list(range(20, 201))
    
    # Precompute GCDs for speed
    gcd_table = {}
    for m1 in moduli:
        for m2 in moduli:
            gcd_table[(m1, m2)] = math.gcd(m1, m2)
            
    def min_conflicts_solve(state, max_iter=100):
        """
        Attempts to resolve conflicts in state using Min-Conflicts heuristic.
        state: dict {m: r}
        Returns True if solved (no conflicts), False otherwise.
        Modifies state in-place if successful.
        """
        current_state = dict(state)
        ms = list(current_state.keys())
        
        # Helper to count conflicts for a specific variable
        def count_conflicts_for(m, r, state_dict):
            count = 0
            g_row = gcd_table.get((m, m), 1) # placeholder
            for m2, r2 in state_dict.items():
                if m2 == m: continue
                g = gcd_table.get((m, m2), math.gcd(m, m2))
                if r % g == r2 % g:
                    count += 1
            return count

        for _ in range(max_iter):
            if time.time() > deadline:
                return False
                
            # Check if solved
            solved = True
            # We can just check one random pair or all. 
            # For efficiency in the loop, we rely on the variable update step.
            # But we need to know when to stop.
            # Let's just check if any variable has conflicts.
            
            # Optimization: Track conflicts count globally?
            # Simpler: Just pick a variable and try to improve it.
            # If we can't improve any variable, we might be stuck, but we do random kicks.
            
            # Pick a random variable
            m_var = random.choice(ms)
            r_var = current_state[m_var]
            
            # Find best residue for m_var
            best_r = r_var
            min_conf = count_conflicts_for(m_var, r_var, current_state)
            
            # Try a few random residues
            # If m_var is small, try all.
            if m_var < 50:
                candidates = list(range(m_var))
            else:
                candidates = [random.randint(0, m_var - 1) for _ in range(20)]
                if r_var not in candidates:
                    candidates.append(r_var)
            
            for r in candidates:
                if r == r_var: continue
                c = count_conflicts_for(m_var, r, current_state)
                if c < min_conf:
                    min_conf = c
                    best_r = r
            
            current_state[m_var] = best_r
            
            # Check if fully solved
            # This is O(N^2) which is expensive if done every iteration.
            # We can check less frequently.
            if _ % 10 == 0 or _ == max_iter - 1:
                # Check all conflicts
                all_conflicts = 0
                for i in range(len(ms)):
                    m1 = ms[i]
                    r1 = current_state[m1]
                    for j in range(i+1, len(ms)):
                        m2 = ms[j]
                        r2 = current_state[m2]
                        g = gcd_table.get((m1, m2), math.gcd(m1, m2))
                        if r1 % g == r2 % g:
                            all_conflicts += 1
                            break
                    if all_conflicts > 0:
                        break
                
                if all_conflicts == 0:
                    # Success
                    state.update(current_state)
                    return True
                    
        return False

    current_state = {}
    
    for m in moduli:
        if time.time() > deadline:
            break
            
        # 1. Try to find a consistent residue for m given current_state
        found = False
        # Try residues in random order to break symmetry? 
        # Or deterministic? Deterministic is faster if it works.
        # Let's try deterministic first (0 to m-1).
        for r in range(m):
            valid = True
            for m2, r2 in current_state.items():
                g = gcd_table.get((m, m2), math.gcd(m, m2))
                if r % g == r2 % g:
                    valid = False
                    break
            if valid:
                current_state[m] = r
                found = True
                break
        
        if not found:
            # 2. Try Local Search
            temp_state = dict(current_state)
            temp_state[m] = random.randint(0, m - 1)
            
            if min_conflicts_solve(temp_state, max_iter=50):
                current_state = temp_state
                found = True
        
        # If not found, skip m

    # Convert to list of lists
    result = [[m, r] for m, r in current_state.items()]
    return result
