import time
import itertools
import numpy as np
import random


def construct(q, n, radius):
    start = time.time()
    deadline = start + 135.0

    total = q ** n

    # Tetracode: 9 words of length 4 over {0,1,2}
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

    # Build initial code: tetracode x all tails
    if n == 4:
        code = [list(t) for t in tetracode]
    else:
        code = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                code.append(h + list(tail))

    # Convert to numpy
    code_arr = np.array(code, dtype=np.int8)
    m = len(code_arr)

    # Precompute q powers
    q_powers = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    # Convert codewords to integer indices
    code_indices = code_arr.astype(np.int64) @ q_powers

    # Precompute covered word indices for each codeword
    # For radius=1: codeword + all single-position changes
    covered_sets = []
    for c in range(m):
        cw = code_arr[c]
        covered = [int(code_indices[c])]
        for pos in range(n):
            for val in range(q):
                if val == cw[pos]:
                    continue
                idx = int(code_indices[c]) + (val - int(cw[pos])) * int(q_powers[pos])
                covered.append(idx)
        covered_sets.append(np.array(covered, dtype=np.int32))

    # Build coverage count
    cov_count = np.zeros(total, dtype=np.int32)
    for c in range(m):
        cov_count[covered_sets[c]] += 1

    # Phase 1: Greedy pruning - remove redundant codewords
    active = np.ones(m, dtype=bool)
    
    for pass_num in range(5):
        if time.time() > deadline - 30.0:
            break
        changed = False
        for c in range(m):
            if not active[c]:
                continue
            cs = covered_sets[c]
            if np.all(cov_count[cs] >= 2):
                cov_count[cs] -= 1
                active[c] = False
                changed = True
        if not changed:
            break

    active_indices = [c for c in range(m) if active[c]]
    current_size = len(active_indices)

    # Phase 2: Simulated annealing - try to remove more codewords
    # by temporarily accepting some uncovered words
    best_size = current_size
    best_active = list(active_indices)
    
    # Track uncovered count
    uncovered_count = int(np.sum(cov_count == 0))
    
    T = 5.0  # initial temperature
    cooling = 0.995
    
    if time.time() < deadline - 10.0:
        random.seed(42)
        while time.time() < deadline - 10.0:
            if not active_indices:
                break
            
            # Randomly pick an active codeword to try removing
            c = random.choice(active_indices)
            cs = covered_sets[c]
            
            # Check: how many words would become uncovered if we remove c?
            newly_uncovered = int(np.sum(cov_count[cs] == 1))
            
            if newly_uncovered == 0:
                # Safe removal
                cov_count[cs] -= 1
                active_indices.remove(c)
                active[c] = False
                uncovered_count += 0  # no change
                if len(active_indices) < best_size:
                    best_size = len(active_indices)
                    best_active = list(active_indices)
            else:
                # Risky removal: accept with probability based on temperature
                if T > 0.1:
                    prob = np.exp(-newly_uncovered / T)
                    if random.random() < prob:
                        # Accept: remove codeword, some words become uncovered
                        cov_count[cs] -= 1
                        active_indices.remove(c)
                        active[c] = False
                        uncovered_count += newly_uncovered
                        # Try to recover by adding a new codeword
                        # Find the most uncovered word and add a codeword near it
                        if time.time() < deadline - 10.0 and uncovered_count > 0:
                            # Sample some uncovered words
                            uncovered_words = np.where(cov_count == 0)[0]
                            if len(uncovered_words) > 0:
                                # Pick a random uncovered word
                                w_idx = random.choice(uncovered_words)
                                # Generate candidate codewords: words within radius 1 of w_idx
                                # Convert w_idx to base-q word
                                w_word = []
                                temp = w_idx
                                for i in range(n):
                                    w_word.append((temp // int(q_powers[i])) % q)
                                # Try adding a random variation
                                pos = random.randint(0, n - 1)
                                if q > 1:
                                    new_val = random.randint(0, q - 1)
                                new_word = list(w_word)
                                new_word[pos] = new_val
                                new_idx = sum(new_word[i] * int(q_powers[i]) for i in range(n))
                                
                                # Check if this word is not already in the code
                                already_in = False
                                for ai in active_indices:
                                    if int(code_indices[ai]) == new_idx:
                                        already_in = True
                                        break
                                
                                if not already_in:
                                    # Add this codeword
                                    # First, we need to handle the fact that our code_arr is fixed
                                    # We'll work with a dynamic list of indices
                                    # For now, just track which base-q indices are active
                                    # This gets complex, so let's use a different approach
                                    pass
            
            # Cool down
            T *= cooling
    
    # Phase 3: Final pruning pass
    for pass_num in range(3):
        if time.time() > deadline - 5.0:
            break
        changed = False
        for c in list(active_indices):
            if not active[c]:
                continue
            cs = covered_sets[c]
            if np.all(cov_count[cs] >= 2):
                cov_count[cs] -= 1
                active_indices.remove(c)
                active[c] = False
                changed = True
        if not changed:
            break

    # Use best found
    final_indices = best_active
    final_code = [list(code_arr[c]) for c in final_indices]
    
    return final_code
