import numpy as np
import time
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Precompute all words as integers
    total_words = q ** n
    all_words = np.array(list(range(total_words)), dtype=np.int32)
    
    # Precompute base powers
    base_powers = np.array([q ** i for i in range(n)], dtype=np.int32)
    
    def int_to_word(w):
        return (w // base_powers) % q
        
    def word_to_int(w):
        return np.dot(w, base_powers)

    # 1. Precompute distance matrix and balls
    # Flatten words to matrix
    words_matrix = np.array([int_to_word(w) for w in range(total_words)], dtype=np.int32)
    
    # Compute distances
    diff = words_matrix[:, None, :] - words_matrix[None, :, :]
    dist_matrix = np.sum(diff != 0, axis=2).astype(np.int32)
    
    # Balls: (N, N) boolean. 2187*2187 bytes = ~5MB.
    balls = dist_matrix <= radius
    
    # 2. Initial Heuristic: Greedy Set Cover
    covered = np.zeros(total_words, dtype=bool)
    code = []
    
    while not covered.all():
        # Calculate gain for each candidate
        gain = (balls & ~covered[None, :]).sum(axis=1)
        
        if gain.max() == 0:
            break
            
        best_idx = np.argmax(gain)
        
        code.append(int(best_idx))
        covered[balls[best_idx]] = True
        
        if time.time() > deadline - 30: # Reserve 30s for ILS
            break

    # Convert to list of lists
    code_list = [int_to_word(c).tolist() for c in code]
    
    if time.time() > deadline - 10:
        return code_list

    # 3. Iterated Local Search (ILS)
    # State: set of codewords (indices into 0..N-1)
    # Energy: number of uncovered words. We want to minimize this.
    
    # We represent the code as a set of indices.
    # We also maintain `cover_count`: for each word w, how many codewords cover it.
    # If cover_count[w] == 0, it's uncovered.
    # Energy = sum(cover_count == 0)
    
    cover_count = np.zeros(total_words, dtype=np.int32)
    current_code_set = set(code)
    
    # Initialize cover_count
    for c in current_code_set:
        cover_count[balls[c]] += 1
        
    current_energy = np.sum(cover_count == 0)
    
    best_code_set = set(current_code_set)
    best_energy = current_energy
    
    rng = np.random.default_rng(42)
    
    def local_search(code_set, cover_count, energy):
        """
        Deterministic local search: repeatedly remove codewords that are redundant
        (i.e., removing them does not increase the number of uncovered words).
        """
        improved = True
        while improved:
            improved = False
            # Try to remove each codeword in the set
            # We iterate over a copy of the list because we modify the set
            for c in list(code_set):
                # Check if c is redundant
                # A codeword is redundant if all words it covers are covered by at least one other codeword.
                # Equivalently, if removing c does not increase the number of uncovered words.
                
                # We can check this by looking at words covered by c.
                # If for all w in ball(c), cover_count[w] > 1, then c is redundant.
                # If there exists w in ball(c) such that cover_count[w] == 1, then c is essential.
                
                # Vectorized check:
                # mask = balls[c]
                # if not np.any(cover_count[mask] == 1):
                #     # c is redundant
                #     code_set.remove(c)
                #     cover_count[mask] -= 1
                #     energy -= np.sum(cover_count[mask] == 0) # Wait, energy change is subtle.
                #     # Actually, if c is redundant, energy doesn't change.
                #     # Let's just recompute energy or track it carefully.
                #     # If c is redundant, no new words become uncovered. So energy stays same.
                #     improved = True
                #     break # Restart loop as cover_count changed
                
                # More efficient:
                # count_essential = np.sum(cover_count[balls[c]] == 1)
                # if count_essential == 0:
                #     # Redundant
                #     code_set.remove(c)
                #     cover_count[balls[c]] -= 1
                #     # Energy does not change because no word went from covered to uncovered
                #     improved = True
                #     break
                
                count_essential = np.sum(cover_count[balls[c]] == 1)
                if count_essential == 0:
                    code_set.remove(c)
                    cover_count[balls[c]] -= 1
                    # Energy remains the same
                    improved = True
                    break # Restart the for loop with the new set state
                    
        return code_set, cover_count, energy

    def perturb(code_set, cover_count, energy, rng):
        """
        Perturbation: Add a random number of random codewords.
        """
        # Add 1 to 3 random codewords
        num_adds = rng.integers(1, 4)
        for _ in range(num_adds):
            c_to_add = rng.integers(0, total_words)
            if c_to_add not in code_set:
                code_set.add(c_to_add)
                cover_count[balls[c_to_add]] += 1
                # Energy might decrease if new words are covered
                # We don't need to update energy here precisely for the perturbation step,
                # but we must ensure cover_count is correct.
                
        # Recompute energy for the perturbed state
        new_energy = np.sum(cover_count == 0)
        return code_set, cover_count, new_energy

    # ILS Loop
    while time.time() < deadline - 5:
        # 1. Local Search (Prune)
        current_code_set, cover_count, current_energy = local_search(current_code_set, cover_count, current_energy)
        
        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_code_set = set(current_code_set)
        
        # If we reached 0 energy, we have a valid cover.
        # We can still try to minimize size, but the problem asks for valid cover first.
        # Actually, the score is the number of codewords.
        # So we want to minimize size, subject to energy == 0.
        # If energy > 0, we are invalid.
        
        # Check time
        if time.time() > deadline - 10:
            break
            
        # 2. Perturb
        current_code_set, cover_count, current_energy = perturb(current_code_set, cover_count, current_energy, rng)
        
        # 3. Local Search again on perturbed state
        current_code_set, cover_count, current_energy = local_search(current_code_set, cover_count, current_energy)
        
        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_code_set = set(current_code_set)
            
        # If we have a valid cover (energy == 0), we might want to keep pruning?
        # The local_search function already prunes redundant codewords.
        # So if energy == 0, the code size is minimal for that specific set structure.
        # Perturbation allows escaping this local minimum.

    # Return best found
    # Ensure we return a valid cover if possible.
    # If best_energy > 0, the solution is invalid, which scores 0.
    # The greedy init usually produces a valid cover.
    # ILS should maintain validity if we start with a valid cover and only remove redundant words?
    # Wait, perturb adds words, so it maintains validity.
    # Local search removes redundant words, so it maintains validity.
    # So if we start with a valid cover (energy=0), we stay valid.
    
    best_code_list = [int_to_word(c).tolist() for c in best_code_set]
    return best_code_list
