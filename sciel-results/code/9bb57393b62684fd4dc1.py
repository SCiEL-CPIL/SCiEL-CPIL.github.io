"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def construct(q, n, radius):
    # Deadline guard: return best result before 150s wall-clock limit (with safety margin)
    deadline = time.time() + 140.0  # 10s safety margin

    # Parameters for the problem
    q_val = q
    n_val = n
    r_val = radius
    total_words = q_val ** n_val

    # Step 1: Build an initial covering code using a greedy approach
    # We'll represent each word as an integer in [0, q^n)
    def word_to_idx(word):
        idx = 0
        for i in range(n_val):
            idx = idx * q_val + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n_val
        for i in range(n_val - 1, -1, -1):
            word[i] = idx % q_val
            idx //= q_val
        return word

    # Precompute all words and their neighbors within radius
    # For efficiency, we'll use a bitset to track coverage
    covered = np.zeros(total_words, dtype=bool)
    code = []

    # Greedy: repeatedly pick the word that covers the most uncovered words
    # For large spaces, this is expensive, so we use a heuristic
    # Start with the tetracode-based construction if n >= 4
    if n_val >= 4 and q_val == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n_val - 4)]
    else:
        # Fallback: start with a random or systematic initial set
        initial_code = []
        for _ in range(min(100, total_words)):
            initial_code.append([random.randint(0, q_val - 1) for _ in range(n_val)])

    code = initial_code

    # Helper: compute coverage of a set of codewords
    def compute_coverage(codewords):
        cov = np.zeros(total_words, dtype=bool)
        for cw in codewords:
            cw_idx = word_to_idx(cw)
            # Mark all words within radius r of cw as covered
            # For radius 1, this is just the word itself plus its neighbors
            if r_val == 1:
                cov[cw_idx] = True
                for pos in range(n_val):
                    orig = cw[pos]
                    for new_val in range(q_val):
                        if new_val == orig:
                            continue
                        cw2 = cw[:]
                        cw2[pos] = new_val
                        cov[word_to_idx(cw2)] = True
            else:
                # General radius: enumerate all words within Hamming distance r
                # This is expensive; for now, only handle radius 1 efficiently
                # For larger radii, we'd need a more sophisticated approach
                import itertools
                for dist in range(r_val + 1):
                    for positions in itertools.combinations(range(n_val), dist):
                        for values in itertools.product(range(q_val), repeat=dist):
                            cw2 = cw[:]
                            for pos, val in zip(positions, values):
                                if val != cw[pos]:
                                    cw2[pos] = val
                            cov[word_to_idx(cw2)] = True
        return cov

    # Check if current code covers everything
    def is_valid(codewords):
        cov = compute_coverage(codewords)
        return np.all(cov)

    # Try to reduce the code size using simulated annealing
    # Move types: remove a codeword, add a codeword, swap a codeword

    def get_uncovered(cov):
        return np.where(~cov)[0]

    # Initial coverage
    cov = compute_coverage(code)
    best_code = code[:]
    best_size = len(code)
    best_cov = cov.copy()

    if is_valid(code):
        best_size = len(code)

    # Simulated annealing
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    rng = np.random.default_rng(42)

    while time.time() < deadline and T > T_min:
        if time.time() > deadline - 5.0:
            break

        # Generate a random move
        move_type = rng.random()
        new_code = code[:]
        accepted = False

        if move_type < 0.3 and len(code) > 1:
            # Remove a random codeword
            idx = rng.integers(0, len(code))
            removed = code[idx]
            new_code.pop(idx)
            
            # Check if still valid
            new_cov = compute_coverage(new_code)
            if np.all(new_cov):
                code = new_code
                cov = new_cov
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                    best_cov = cov.copy()
                accepted = True
            else:
                # Accept with probability based on temperature
                n_uncovered = np.sum(~new_cov)
                energy_diff = n_uncovered
                if rng.random() < np.exp(-energy_diff / T):
                    code = new_code
                    cov = new_cov
                    accepted = True
                else:
                    # Revert
                    code = best_code if False else code  # keep as is

        elif move_type < 0.6 and len(code) < total_words:
            # Add a random codeword (only useful if we're not covering, but we start covered)
            # This is less useful for minimization, so low probability
            new_cw = [int(rng.integers(0, q_val)) for _ in range(n_val)]
            if new_cw not in code:
                new_code.append(new_cw)
                new_cov = compute_coverage(new_code)
                # Adding can only help coverage, but increases size
                # Only accept if it allows future removals - for now, skip
                pass

        else:
            # Swap: remove one, add another
            if len(code) > 1:
                idx = rng.integers(0, len(code))
                new_cw = [int(rng.integers(0, q_val)) for _ in range(n_val)]
                new_code[idx] = new_cw
                new_cov = compute_coverage(new_code)
                if np.all(new_cov):
                    code = new_code
                    cov = new_cov
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                        best_cov = cov.copy()
                    accepted = True
                else:
                    n_uncovered = np.sum(~new_cov)
                    energy_diff = n_uncovered
                    if rng.random() < np.exp(-energy_diff / T):
                        code = new_code
                        cov = new_cov
                        accepted = True

        T *= cooling

        # Periodically reset to best if stuck
        if not accepted and rng.random() < 0.05:
            code = best_code[:]
            cov = best_cov.copy()

    # Return the best code found
    if best_size < len(code):
        return best_code
    else:
        return code
# EVOLVE-BLOCK-END
