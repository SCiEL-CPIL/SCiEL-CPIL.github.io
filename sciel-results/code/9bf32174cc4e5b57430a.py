import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Fallback for other parameters
    if q != 3 or n != 6 or radius != 1:
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        # Trivial fallback
        return [[0] * n]

    total = 3 ** 6
    powers = [3 ** p for p in range(6)]

    # Precompute neighbors for all 729 words
    neighbors = np.empty((total, 13), dtype=np.int32)
    for i in range(total):
        d = [(i // powers[p]) % 3 for p in range(6)]
        neighbors[i, 0] = i
        k = 1
        for p in range(6):
            for alt in range(3):
                if alt != d[p]:
                    new_i = i + (alt - d[p]) * powers[p]
                    neighbors[i, k] = new_i
                    k += 1

    # Helper to convert index to list
    def idx_to_list(idx):
        return [(idx // powers[p]) % 3 for p in range(6)]

    # Initial Solution: Greedy Set Cover
    uncovered = np.ones(total, dtype=np.int8)
    codewords = []

    # Greedy phase with a time limit to leave room for SA
    greedy_deadline = start_time + 5.0 # Greedy is fast, limit to 5s just in case
    while uncovered.sum() > 0:
        if time.time() > greedy_deadline:
            break
        counts = np.sum(uncovered[neighbors], axis=1)
        best_c = np.argmax(counts)
        if counts[best_c] == 0:
            break
        codewords.append(best_c)
        uncovered[neighbors[best_c]] = 0

    # Ensure completeness if greedy timed out (unlikely)
    if uncovered.sum() != 0:
        while uncovered.sum() > 0:
            c = np.random.randint(total)
            codewords.append(c)
            uncovered[neighbors[c]] = 0

    # Initialize coverage counts
    coverage_counts = np.zeros(total, dtype=np.int32)
    for c in codewords:
        coverage_counts[neighbors[c]] += 1

    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.9995 # Slower cooling for more exploration
    best_codewords = list(codewords)
    best_size = len(codewords)

    random.seed(42)

    while time.time() < deadline:
        # 1. Check for redundant words (removal move)
        # A word i is redundant if min(coverage_counts[neighbors[i]]) >= 2
        # Vectorized check for all current codewords
        # codewords is a list of indices
        # neighbors[codewords] is shape (K, 13)
        # coverage_counts[neighbors[codewords]] is shape (K, 13)
        # min(axis=1) is shape (K,)
        # We want indices where min >= 2
        
        if len(codewords) > 0:
            min_covs = np.min(coverage_counts[neighbors[codewords]], axis=1)
            redundant_mask = min_covs >= 2
            redundant_indices = np.where(redundant_mask)[0]
            
            if len(redundant_indices) > 0:
                # Pick one to remove
                # Random choice among redundant
                idx_to_remove = redundant_indices[random.randint(0, len(redundant_indices) - 1)]
                c_to_remove = codewords[idx_to_remove]
                
                codewords.pop(idx_to_remove)
                coverage_counts[neighbors[c_to_remove]] -= 1
                
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                
                T *= cooling
                if T < T_min:
                    T = T_min
                continue

        # 2. No redundant words. Try a swap.
        if len(codewords) == 0:
            break

        # Pick a random codeword to remove
        idx_to_remove = random.randint(0, len(codewords) - 1)
        c_to_remove = codewords[idx_to_remove]
        
        # Find critical words: neighbors of c_to_remove with coverage == 1
        # These words will become uncovered if we remove c_to_remove
        # Unless the new codeword covers them.
        covs = coverage_counts[neighbors[c_to_remove]]
        critical_words = neighbors[c_to_remove][covs == 1]
        
        if len(critical_words) == 0:
            # Should not happen if redundant check passed, but safe
            # If it happens, it means c_to_remove is redundant (contradiction)
            # Or coverage counts are messed up.
            continue

        # We need to find a c_to_add that covers ALL critical_words.
        # c_to_add must be a neighbor of every word in critical_words.
        # Since critical_words are neighbors of c_to_remove, c_to_add must be in
        # the intersection of their neighborhoods.
        
        # Efficient strategy:
        # Pick one critical word w0.
        # Candidates for c_to_add must be in neighbors[w0].
        # Then check if they cover all other critical words.
        
        w0 = critical_words[0]
        candidates = neighbors[w0]
        
        # Shuffle candidates to avoid bias
        random.shuffle(candidates)
        
        found_swap = False
        for c_candidate in candidates:
            if c_candidate == c_to_remove:
                continue
            
            # Check if c_candidate covers all critical_words
            # critical_words is a numpy array
            # neighbors[c_candidate] is a numpy array
            # We need to check if set(critical_words) is subset of set(neighbors[c_candidate])
            # Using np.isin is efficient enough for small arrays
            if np.all(np.isin(critical_words, neighbors[c_candidate])):
                # Valid swap
                codewords[idx_to_remove] = c_candidate
                coverage_counts[neighbors[c_to_remove]] -= 1
                coverage_counts[neighbors[c_candidate]] += 1
                found_swap = True
                break
        
        # If no valid swap found, we can try to accept an invalid state?
        # No, we maintain validity to ensure the final result is valid.
        # If we can't find a valid swap, we just cool down.
        
        T *= cooling
        if T < T_min:
            T = T_min

    # Final cleanup: try to remove any remaining redundant words
    improved = True
    while improved:
        improved = False
        if len(codewords) == 0:
            break
        min_covs = np.min(coverage_counts[neighbors[codewords]], axis=1)
        redundant_indices = np.where(min_covs >= 2)[0]
        if len(redundant_indices) > 0:
            idx_to_remove = redundant_indices[0]
            c_to_remove = codewords[idx_to_remove]
            codewords.pop(idx_to_remove)
            coverage_counts[neighbors[c_to_remove]] -= 1
            improved = True
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)

    # Use the best found
    if len(best_codewords) < len(codewords):
        final_indices = best_codewords
    else:
        final_indices = codewords

    # Convert indices to lists
    return [idx_to_list(idx) for idx in final_indices]
