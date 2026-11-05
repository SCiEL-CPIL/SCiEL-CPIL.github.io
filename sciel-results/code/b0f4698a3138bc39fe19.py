import time
import numpy as np
import random

def construct(q, n, radius):
    t_start = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # Helper to convert index to word
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n-1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w
    
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx

    # Precompute neighbors for radius=1
    # neighbors[idx] = list of indices within Hamming distance <= 1
    space_size = q ** n
    neighbors = np.empty(space_size, dtype=object)
    
    # Powers of q for fast digit access
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i-1] * q

    for idx in range(space_size):
        nbs = [idx]
        for pos in range(n):
            digit = (idx // q_powers[n-1-pos]) % q
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                delta = (new_digit - digit) * q_powers[n-1-pos]
                nbs.append(idx + delta)
        neighbors[idx] = np.array(nbs, dtype=np.int32)
    
    # Precompute coverage masks for faster operations
    # cover_masks[i] = boolean array where cover_masks[i][j] = True if j is covered by i
    cover_masks = np.zeros((space_size, space_size), dtype=bool)
    for i in range(space_size):
        cover_masks[i, neighbors[i]] = True

    # 1. Generate an initial valid code using greedy set cover
    uncovered = np.ones(space_size, dtype=bool)
    initial_code = []
    
    # Greedy Set Cover
    while uncovered.any():
        # Find word covering most uncovered words
        counts = cover_masks @ uncovered.astype(np.int8)
        best_idx = np.argmax(counts)
        
        if counts[best_idx] == 0:
            break
            
        initial_code.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        
        if time.time() - t_start > deadline - 5:
            break

    # Ensure we have a valid code
    if not uncovered.all():
        while not uncovered.all():
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            w_idx = uncovered_indices[0]
            initial_code.append(w_idx)
            uncovered[neighbors[w_idx]] = False
            if time.time() - t_start > deadline - 5:
                break

    # 2. Iterated Local Search for minimization
    # We use a representation where we maintain:
    # - current_code: list of codeword indices
    # - current_cov: array of length space_size, current_cov[i] = number of codewords covering i
    # - in_set: boolean array, in_set[i] = True if i is in current_code
    
    current_code = initial_code[:]
    current_cov = np.zeros(space_size, dtype=np.int32)
    in_set = np.zeros(space_size, dtype=bool)
    
    for c in current_code:
        current_cov[neighbors[c]] += 1
        in_set[c] = True
    
    # Objective: Minimize size subject to valid coverage (uncovered == 0)
    # We use a penalty method: Energy = Size + Penalty * UncoveredCount
    # Penalty should be large enough to make invalid solutions worse than any valid solution.
    # Max size is space_size, so Penalty > space_size is safe.
    penalty = space_size + 1
    
    def get_energy(cov, size):
        uncovered_count = np.sum(cov == 0)
        return size + penalty * uncovered_count
    
    current_energy = get_energy(current_cov, len(current_code))
    
    best_code = current_code[:]
    best_energy = current_energy
    best_size = len(current_code)
    
    rng = np.random.default_rng(42)
    
    # Simulated Annealing Parameters
    T = 50.0
    cooling_rate = 0.995
    T_min = 0.1
    
    # Moves:
    # 1. Remove a random codeword (if it doesn't make the code invalid, or accept with penalty)
    # 2. Add a random codeword (if it reduces uncovered, or accept with penalty)
    # 3. Swap: Remove one, Add one
    
    while T > T_min and (time.time() - t_start) < deadline - 5:
        
        # Generate a move
        move_type = rng.random()
        
        new_code = current_code[:]
        new_cov = current_cov.copy()
        new_in_set = in_set.copy()
        
        if move_type < 0.4 and len(new_code) > 0:
            # Remove a random codeword
            idx_to_remove = rng.integers(0, len(new_code))
            c = new_code[idx_to_remove]
            new_cov[neighbors[c]] -= 1
            new_in_set[c] = False
            new_code.pop(idx_to_remove)
            
        elif move_type < 0.7:
            # Add a random codeword
            # Bias towards words that cover many uncovered spots
            # But for simplicity, just random
            c = rng.integers(0, space_size)
            if new_in_set[c]:
                continue
            new_code.append(c)
            new_cov[neighbors[c]] += 1
            new_in_set[c] = True
            
        else:
            # Swap: Remove one, Add one
            if len(new_code) == 0:
                continue
            idx_to_remove = rng.integers(0, len(new_code))
            c_out = new_code[idx_to_remove]
            new_cov[neighbors[c_out]] -= 1
            new_in_set[c_out] = False
            new_code.pop(idx_to_remove)
            
            c_in = rng.integers(0, space_size)
            if new_in_set[c_in]:
                # Revert
                new_code.insert(idx_to_remove, c_out)
                new_cov[neighbors[c_out]] += 1
                new_in_set[c_out] = True
                continue
            
            new_code.append(c_in)
            new_cov[neighbors[c_in]] += 1
            new_in_set[c_in] = True
            
        new_energy = get_energy(new_cov, len(new_code))
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            current_code = new_code
            current_cov = new_cov
            current_in_set = new_in_set
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_code = current_code[:]
                best_size = len(current_code)
        
        T *= cooling_rate
        
    # Final result must be valid (uncovered == 0)
    # If best_code is valid, return it.
    # If not, we must return a valid code.
    
    # Verify best_code
    final_cov = np.zeros(space_size, dtype=np.int32)
    for c in best_code:
        final_cov[neighbors[c]] += 1
        
    if np.any(final_cov == 0):
        # best_code is invalid. Fallback to initial_code which was valid.
        return [idx_to_word(c) for c in initial_code]
    else:
        return [idx_to_word(c) for c in best_code]

# Test locally? No, just provide the function.
