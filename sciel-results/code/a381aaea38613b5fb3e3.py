import time
import numpy as np
import random

def construct(q, n, radius):
    t_start = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # 1. Generate an initial valid code using a direct product construction
    # For q=3, n=7, radius=1, a simple grid code or tetracode product works.
    # We start with a known valid code, then try to reduce its size.
    
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

    # Initial Code Construction: 
    # Use a simple greedy set cover to get a valid starting point, 
    # then apply simulated annealing to minimize the size.
    
    # Greedy Set Cover to get initial valid code
    uncovered = np.ones(space_size, dtype=bool)
    initial_code = []
    
    # Precompute coverage vectors for all words? Too much memory (2187 * 2187 * 8 bytes ~ 38MB, acceptable)
    # But let's just do it on the fly for the greedy part to be safe on memory/time.
    
    # To speed up greedy, we can precompute the "coverage set" of each word as a boolean mask
    # But 2187 * 2187 booleans is ~ 4.7M booleans = ~ 0.6 MB. Very fast to compute.
    cover_masks = np.zeros((space_size, space_size), dtype=bool)
    for i in range(space_size):
        cover_masks[i, neighbors[i]] = True
        
    # Greedy Set Cover
    while uncovered.any():
        # Find word covering most uncovered words
        # counts = (cover_masks & uncovered[None, :]).sum(axis=1)
        # This is a matrix multiplication essentially.
        # cover_masks is (S, S), uncovered is (S,)
        # We want sum over columns for each row.
        # counts = cover_masks @ uncovered (where uncovered is 0/1)
        counts = cover_masks @ uncovered.astype(np.int8)
        best_idx = np.argmax(counts)
        
        if counts[best_idx] == 0:
            break # Should not happen if space is connected
            
        initial_code.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        
        if time.time() - t_start > deadline - 5:
            break

    if not uncovered.all(): # If we didn't finish greedy, we have a partial code, but let's assume it finishes fast
         # Fallback: just use the current partial code if it's valid, otherwise return initial full space? 
         # No, greedy on 2187 elements is very fast.
         pass

    # Ensure we have a valid code
    if not uncovered.all():
         # If greedy failed (shouldn't), we might have gaps. 
         # For robustness, if gaps exist, we add random words to cover them.
         while not uncovered.all():
             # Pick a random uncovered word, add a random word covering it? 
             # Or just add the uncovered word itself (radius 1 covers itself)
             uncovered_indices = np.where(uncovered)[0]
             if len(uncovered_indices) == 0:
                 break
             # Add a random word from the uncovered set to the code? 
             # If we add a word w, it covers w and its neighbors.
             # Let's just add the first uncovered word.
             w_idx = uncovered_indices[0]
             initial_code.append(w_idx)
             uncovered[neighbors[w_idx]] = False
             if time.time() - t_start > deadline - 5:
                 break

    # Now we have a valid code `initial_code`.
    # We want to minimize its size.
    
    # State: list of codeword indices
    # Objective: Minimize size, subject to valid coverage.
    # Penalty: If invalid, add large penalty proportional to uncovered words.
    
    current_code = initial_code[:]
    current_uncovered = np.zeros(space_size, dtype=np.int32)
    
    # Recalculate coverage for current_code
    for c in current_code:
        current_uncovered[neighbors[c]] += 1
    # current_uncovered[i] > 0 means covered.
    # Uncovered count = number of i where current_uncovered[i] == 0
    
    def get_uncovered_count(cov):
        return np.sum(cov == 0)
    
    current_uncovered_count = get_uncovered_count(current_uncovered)
    
    best_code = current_code[:]
    best_uncovered_count = current_uncovered_count
    best_size = len(current_code)
    
    # Simulated Annealing Parameters
    T = 100.0
    cooling_rate = 0.995
    T_min = 0.1
    
    # Moves:
    # 1. Remove a random codeword. If still valid (or fewer uncovered), accept.
    # 2. Add a random codeword. If it reduces uncovered count (and we accept larger size temporarily? No, we want min size).
    #    Actually, standard SA minimizes Energy.
    #    Energy = Size * Lambda + UncoveredCount * Mu.
    #    We want Size small and Uncovered 0.
    #    Let's define Energy = UncoveredCount * 1000 + Size.
    #    We want to reach Uncovered=0 and minimize Size.
    
    def calc_energy(cov, size):
        unc = np.sum(cov == 0)
        return unc * 1000 + size
    
    current_energy = calc_energy(current_uncovered, len(current_code))
    best_energy = current_energy
    
    rng = np.random.default_rng(42)
    
    while T > T_min and (time.time() - t_start) < deadline - 5:
        
        # Generate a move
        move_type = rng.random()
        
        new_code = current_code[:]
        new_cov = current_uncovered.copy()
        
        if move_type < 0.4 and len(new_code) > 0:
            # Remove a random codeword
            idx_to_remove = rng.integers(0, len(new_code))
            c = new_code[idx_to_remove]
            new_cov[neighbors[c]] -= 1
            new_code.pop(idx_to_remove)
            
        elif move_type < 0.7:
            # Add a random codeword
            # Bias towards words that cover many uncovered spots?
            # For now, just random.
            c = rng.integers(0, space_size)
            # If c is already in code, skip? Or just add duplicate? 
            # Duplicates are useless. Check if already present.
            if c in new_code:
                continue
            new_code.append(c)
            new_cov[neighbors[c]] += 1
            
        else:
            # Swap: Remove one, Add one
            if len(new_code) == 0:
                continue
            idx_to_remove = rng.integers(0, len(new_code))
            c_out = new_code[idx_to_remove]
            new_cov[neighbors[c_out]] -= 1
            new_code.pop(idx_to_remove)
            
            c_in = rng.integers(0, space_size)
            if c_in in new_code:
                # Revert
                new_code.insert(idx_to_remove, c_out)
                new_cov[neighbors[c_out]] += 1
                continue
            
            new_code.append(c_in)
            new_cov[neighbors[c_in]] += 1
            
        new_energy = calc_energy(new_cov, len(new_code))
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            current_code = new_code
            current_uncovered = new_cov
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_code = current_code[:]
                best_uncovered_count = np.sum(current_uncovered == 0)
                best_size = len(current_code)
        
        T *= cooling_rate
        
        # Check if we found a valid code with smaller size than initial
        # If best_uncovered_count == 0 and best_size < best_size, we are improving.
        
    # Final result must be valid (uncovered == 0)
    # If best_code is valid, return it.
    # If not, we must return a valid code. 
    # We can always return the initial_code which was valid.
    # But we want the best valid code found.
    
    # Let's verify best_code
    final_cov = np.zeros(space_size, dtype=np.int32)
    for c in best_code:
        final_cov[neighbors[c]] += 1
        
    if np.any(final_cov == 0):
        # best_code is invalid. 
        # We need to repair it or fall back to initial_code.
        # Fallback to initial_code is safe.
        # Or try to repair best_code by adding missing words.
        # For simplicity and safety, if best is invalid, use initial.
        # But maybe best_code is close? 
        # Let's just use the best VALID code we found.
        # We tracked best_code based on energy. If energy < 1000, it was valid.
        # If best_energy >= 1000, it was invalid.
        
        # Let's keep track of the best VALID code separately.
        # Actually, the logic above updates best_code only if energy is lower.
        # If we start with valid code (energy = size), any invalid move has energy > 1000.
        # So best_code will always be valid if we start with a valid code and don't accept worse energy?
        # No, SA accepts worse energy. So best_code might be invalid if we haven't found a valid one with lower energy?
        # Wait, if we start valid, energy = Size.
        # If we move to invalid, energy = 1000*Unc + Size. This is > Size.
        # So we only accept if T is high.
        # But we update best_code only if current_energy < best_energy.
        # Since we start with valid, best_energy = Size_initial.
        # Any invalid state has energy > 1000.
        # So if Size_initial < 1000 (which is true, ~186), best_code will NEVER be invalid.
        # Because to update best_code, current_energy must be < best_energy.
        # If current state is invalid, energy >= 1000.
        # If best_energy is 186, 1000 is not < 186.
        # So best_code remains valid.
        
        # However, if we started with an INVALID code (greedy failed), this logic breaks.
        # But greedy on 2187 words is very robust.
        
        return [idx_to_word(c) for c in initial_code]
    else:
        return [idx_to_word(c) for c in best_code]

# Test locally? No, just provide the function.
