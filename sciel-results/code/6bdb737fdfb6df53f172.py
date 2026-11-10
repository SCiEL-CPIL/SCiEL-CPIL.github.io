import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    best_score = 0.0
    best_selection = []
    
    # We will run multiple randomized greedy passes
    # Each pass tries to build a solution.
    # To improve over the deterministic greedy, we randomize the choice of residue r
    # among valid ones, or randomize the order of moduli slightly.
    
    # Strategy: 
    # 1. Deterministic greedy (ascending m, smallest r)
    # 2. Randomized greedy (ascending m, random valid r)
    # 3. Randomized greedy (shuffled m, random valid r) - might be worse for 1/m sum, so stick to ascending m but random r.
    
    def greedy_pass(randomize_r=False, shuffle_moduli=False):
        selected = []
        candidates = list(range(20, 201))
        if shuffle_moduli:
            # Only shuffle slightly to keep small moduli prioritized? 
            # No, shuffling breaks the 1/m priority. 
            # Let's stick to ascending m, but randomize r.
            pass 
            
        for m in candidates:
            if time.time() > deadline:
                break
            valid_rs = []
            for r in range(m):
                valid = True
                for (m_j, r_j) in selected:
                    g = math.gcd(m, m_j)
                    if r % g == r_j % g:
                        valid = False
                        break
                if valid:
                    valid_rs.append(r)
            
            if valid_rs:
                if randomize_r:
                    r = random.choice(valid_rs)
                else:
                    r = valid_rs[0]
                selected.append([m, r])
        
        score = sum(1.0 / m for m, r in selected)
        return selected, score

    # Run deterministic pass
    sel, score = greedy_pass(randomize_r=False)
    if score > best_score:
        best_score = score
        best_selection = sel
        
    # Run randomized passes
    num_passes = 200
    for i in range(num_passes):
        if time.time() > deadline - 5.0:
            break
        sel, score = greedy_pass(randomize_r=True)
        if score > best_score:
            best_score = score
            best_selection = sel

    return best_selection
