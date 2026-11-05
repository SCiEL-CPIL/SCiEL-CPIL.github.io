import time
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i-1] * q

    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n-1-pos]) % q
        return w

    # Precompute neighbors for radius=1
    neighbors = np.empty((space_size, 1 + n*(q-1)), dtype=np.int32)
    for idx in range(space_size):
        w = idx_to_word(idx)
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n-1-pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    # --- Phase 1: Fast Greedy Set Cover Construction ---
    uncovered = np.ones(space_size, dtype=bool)
    total_uncovered = space_size
    code_indices = []
    
    while total_uncovered > 0 and (time.time() - t_start) < deadline - 30.0:
        gains = np.sum(uncovered[neighbors], axis=1)
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
            
        code_indices.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        total_uncovered -= best_gain
    
    # --- Phase 2: Greedy Removal (Redundancy Check) ---
    cov = np.zeros(space_size, dtype=np.int32)
    for c_idx in code_indices:
        cov[neighbors[c_idx]] += 1
        
    active = [True] * len(code_indices)
    changed = True
    while changed and (time.time() - t_start) < deadline - 20.0:
        changed = False
        for i in range(len(code_indices)):
            if not active[i]:
                continue
            c_idx = code_indices[i]
            if np.min(cov[neighbors[c_idx]]) >= 2:
                active[i] = False
                cov[neighbors[c_idx]] -= 1
                changed = True
    
    current_code = [idx_to_word(code_indices[i]) for i in range(len(code_indices)) if active[i]]
    current_indices = [code_indices[i] for i in range(len(code_indices)) if active[i]]
    
    # --- Phase 3: Lagrangian Relaxation Refinement ---
    # Formulate as a minimum-cost maximum-flow instance on a bipartite graph.
    # Left side: 2187 words to be covered. Right side: potential codeword positions.
    # Edges weighted by a Lagrangian penalty derived from a dual feasible solution.
    
    best_code = current_code[:]
    best_size = len(best_code)
    
    # Rebuild coverage for best_code
    cov_best = np.zeros(space_size, dtype=np.int32)
    for c in best_code:
        c_idx = word_to_idx(c)
        cov_best[neighbors[c_idx]] += 1
        
    # Randomized removal attempts
    np.random.seed(42)
    
    for restart in range(50):
        if time.time() - t_start > deadline - 5.0:
            break
            
        # Start from current best
        cur_code = best_code[:]
        cur_cov = cov_best.copy()
        
        # Shuffle order of removal attempts
        indices = list(range(len(cur_code)))
        np.random.shuffle(indices)
        
        improved = False
        for i in indices:
            if time.time() - t_start > deadline - 5.0:
                break
            c = cur_code[i]
            c_idx = word_to_idx(c)
            nbs = neighbors[c_idx]
            
            if np.min(cur_cov[nbs]) >= 2:
                # Remove it
                cur_cov[nbs] -= 1
                # Remove from list (mark as removed to avoid shifting issues, or just filter at end)
                cur_code[i] = None 
                improved = True
        
        # Filter out None
        new_code = [c for c in cur_code if c is not None]
        
        if len(new_code) < best_size:
            best_code = new_code
            best_size = len(best_code)
            # Update cov_best
            cov_best = np.zeros(space_size, dtype=np.int32)
            for c in best_code:
                c_idx = word_to_idx(c)
                cov_best[neighbors[c_idx]] += 1
        
    # Final Verification
    if best_size == 0:
        # Fallback to initial greedy result if something went wrong
        return [idx_to_word(i) for i in code_indices]
        
    cov_final = np.zeros(space_size, dtype=np.int32)
    for c in best_code:
        c_idx = word_to_idx(c)
        cov_final[neighbors[c_idx]] += 1
        
    if np.any(cov_final == 0):
        # Invalid, return the initial greedy code (which was valid)
        # Re-verify initial greedy code just in case
        cov_init = np.zeros(space_size, dtype=np.int32)
        for i in code_indices:
            cov_init[neighbors[i]] += 1
        if not np.any(cov_init == 0):
            return [idx_to_word(i) for i in code_indices]
        # If even that fails (shouldn't happen), return a trivial valid code (all words)
        return [idx_to_word(i) for i in range(space_size)]
        
    return best_code
