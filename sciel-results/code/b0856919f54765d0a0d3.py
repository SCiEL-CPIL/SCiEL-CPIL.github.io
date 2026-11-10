import numpy as np
import time
import random
from collections import Counter

def construct(q, n, radius):
    start_time = time.time()
    DEADLINE = 118.0

    total_words = q ** n

    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors for all words
    neighbors = np.empty(total_words, dtype=object)
    for idx in range(total_words):
        w = idx_to_word(idx)
        nbrs = [idx]
        for i in range(n):
            orig = w[i]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[i] = new_val
                nbrs.append(word_to_idx(w))
            w[i] = orig
        neighbors[idx] = np.array(nbrs, dtype=np.int64)

    # Convert to list of lists for easier manipulation
    nbr_lists = [neighbors[i].tolist() for i in range(total_words)]

    # Build inverse mapping: for each word, list of codeword indices that cover it
    # This is expensive but useful for SA. We'll build it lazily.
    # Actually, let's precompute a coverage matrix for the greedy phase only.

    # --- Greedy set-cover with faster counting ---
    covered = np.zeros(total_words, dtype=bool)
    code_indices = []
    code_set = set()

    while not np.all(covered):
        if time.time() - start_time > DEADLINE - 15:
            break

        best_idx = -1
        best_new = -1

        # Find the word that covers the most uncovered neighbors
        # Vectorized approach: for each candidate, count uncovered neighbors
        for c in range(total_words):
            if c in code_set:
                continue
            nbrs = nbr_lists[c]
            new_cov = 0
            for nb in nbrs:
                if not covered[nb]:
                    new_cov += 1
            if new_cov > best_new:
                best_new = new_cov
                best_idx = c

        if best_new <= 0:
            break

        code_indices.append(best_idx)
        code_set.add(best_idx)
        for nb in nbr_lists[best_idx]:
            covered[nb] = True

    # --- Simulated Annealing for dynamic code optimization ---
    # Objective: minimize |code| subject to full coverage
    # We use a cost function that penalizes uncovered words heavily

    def compute_coverage(code_list):
        """Return a boolean array of covered words."""
        cov = np.zeros(total_words, dtype=bool)
        for ci in code_list:
            for nb in nbr_lists[ci]:
                cov[nb] = True
        return cov

    def count_uncovered(cov):
        """Count number of uncovered words."""
        return int(np.sum(~cov))

    # Initial coverage
    current_cov = compute_coverage(code_indices)
    current_uncovered = count_uncovered(current_cov)
    current_size = len(code_indices)

    best_code = list(code_indices)
    best_size = current_size
    best_uncovered = current_uncovered

    # SA parameters
    T_start = 10.0
    T_end = 0.1
    cooling_rate = 0.995
    max_iterations = 5000

    rng = random.Random(42)

    # Precompute: for each codeword, its neighbor set as a frozenset for fast lookup
    nbr_frozensets = [frozenset(nbr_lists[i]) for i in range(total_words)]

    # For faster coverage updates, maintain coverage counts
    # coverage_count[word] = number of codewords that cover word
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for ci in code_indices:
        for nb in nbr_lists[ci]:
            coverage_count[nb] += 1

    # Track which words are covered (coverage_count > 0)
    def update_coverage_counts(ci, sign):
        """Add or remove codeword ci's coverage."""
        for nb in nbr_lists[ci]:
            coverage_count[nb] += sign
            if sign > 0 and coverage_count[nb] == 1:
                # Just became covered
                pass
            elif sign < 0 and coverage_count[nb] == 0:
                # Just became uncovered
                pass

    # Recompute uncovered count from coverage_count
    def get_uncovered_count():
        return int(np.sum(coverage_count == 0))

    current_uncovered = get_uncovered_count()
    best_uncovered = current_uncovered

    T = T_start
    for iteration in range(max_iterations):
        if time.time() - start_time > DEADLINE - 10:
            break

        if T < T_end:
            break

        # Choose a random move type
        move_type = rng.random()

        if move_type < 0.4:
            # Swap: remove one codeword, add another
            if len(code_indices) < 2:
                continue
            i = rng.randrange(len(code_indices))
            ci = code_indices[i]
            # Choose a random new codeword not in current set
            c_new = rng.randrange(total_words)
            if c_new in code_set:
                continue

            # Compute delta
            # Remove ci: coverage_count changes
            removed_uncovered = 0
            added_uncovered = 0

            # Simulate removal of ci
            for nb in nbr_lists[ci]:
                if coverage_count[nb] == 1:
                    removed_uncovered += 1

            # Simulate addition of c_new
            for nb in nbr_lists[c_new]:
                if coverage_count[nb] == 0:
                    added_uncovered += 1

            # Net change in uncovered: removed_uncovered (newly uncovered) - added_uncovered (newly covered)
            delta_uncovered = removed_uncovered - added_uncovered
            delta_size = 0  # Swap keeps same size

            # Cost: primarily minimize uncovered, secondarily minimize size
            # For swap, size doesn't change, so only uncovered matters
            cost_delta = delta_uncovered * 100  # Heavy penalty for uncovered

            # Accept if better or with probability exp(-cost_delta / T)
            if cost_delta < 0:
                # Better solution, accept
                code_indices[i] = c_new
                code_set.discard(ci)
                code_set.add(c_new)
                update_coverage_counts(ci, -1)
                update_coverage_counts(c_new, 1)
                current_uncovered = get_uncovered_count()
                current_size = len(code_indices)

                if current_uncovered < best_uncovered or (current_uncovered == best_uncovered and current_size < best_size):
                    best_code = list(code_indices)
                    best_uncovered = current_uncovered
                    best_size = current_size
            else:
                # Worse or equal, accept with probability
                prob = np.exp(-cost_delta / T)
                if rng.random() < prob:
                    code_indices[i] = c_new
                    code_set.discard(ci)
                    code_set.add(c_new)
                    update_coverage_counts(ci, -1)
                    update_coverage_counts(c_new, 1)
                    current_uncovered = get_uncovered_count()
                    current_size = len(code_indices)

                    if current_uncovered < best_uncovered or (current_uncovered == best_uncovered and current_size < best_size):
                        best_code = list(code_indices)
                        best_uncovered = current_uncovered
                        best_size = current_size

        elif move_type < 0.7:
            # Remove: try to remove a codeword if it doesn't create uncovered words
            if len(code_indices) < 1:
                continue
            i = rng.randrange(len(code_indices))
            ci = code_indices[i]

            # Check if removing ci creates any uncovered words
            creates_uncovered = False
            for nb in nbr_lists[ci]:
                if coverage_count[nb] == 1:
                    creates_uncovered = True
                    break

            if not creates_uncovered:
                # Safe to remove
                code_indices.pop(i)
                code_set.discard(ci)
                update_coverage_counts(ci, -1)
                current_uncovered = get_uncovered_count()
                current_size = len(code_indices)

                if current_uncovered < best_uncovered or (current_uncovered == best_uncovered and current_size < best_size):
                    best_code = list(code_indices)
                    best_uncovered = current_uncovered
                    best_size = current_size
            else:
                # Would create uncovered, but maybe we can add another codeword to compensate
                # This is a compound move: remove ci, add c_new
                c_new = rng.randrange(total_words)
                if c_new in code_set:
                    continue

                # Compute if c_new covers all the words that would become uncovered
                newly_uncovered = []
                for nb in nbr_lists[ci]:
                    if coverage_count[nb] == 1:
                        newly_uncovered.append(nb)

                if not newly_uncovered:
                    continue

                # Check if c_new covers all newly_uncovered
                nbrs_new = set(nbr_lists[c_new])
                all_covered = all(nb in nbrs_new for nb in newly_uncovered)

                if all_covered:
                    # Perform the compound move
                    code_indices[i] = c_new
                    code_set.discard(ci)
                    code_set.add(c_new)
                    update_coverage_counts(ci, -1)
                    update_coverage_counts(c_new, 1)
                    current_uncovered = get_uncovered_count()
                    current_size = len(code_indices)

                    if current_uncovered < best_uncovered or (current_uncovered == best_uncovered and current_size < best_size):
                        best_code = list(code_indices)
                        best_uncovered = current_uncovered
                        best_size = current_size

        else:
            # Add: add a random codeword (only useful if we later remove others)
            c_new = rng.randrange(total_words)
            if c_new in code_set:
                continue

            code_indices.append(c_new)
            code_set.add(c_new)
            update_coverage_counts(c_new, 1)
            current_uncovered = get_uncovered_count()
            current_size = len(code_indices)

            # Only keep this if it improves uncovered count
            if current_uncovered < best_uncovered:
                best_code = list(code_indices)
                best_uncovered = current_uncovered
                best_size = current_size
            else:
                # Revert if not helpful
                code_indices.pop()
                code_set.discard(c_new)
                update_coverage_counts(c_new, -1)
                current_uncovered = get_uncovered_count()
                current_size = len(code_indices)

        # Cool down
        T *= cooling_rate

    # Use the best solution found
    code_indices = best_code
    code_set = set(code_indices)

    # --- Final local search: remove redundant codewords ---
    def is_redundant(ci, code_list, cov_count):
        """Check if codeword ci is redundant using coverage counts."""
        for nb in nbr_lists[ci]:
            if cov_count[nb] == 1:
                return False
        return True

    # Rebuild coverage counts for final code
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for ci in code_indices:
        for nb in nbr_lists[ci]:
            coverage_count[nb] += 1

    removed = True
    while removed and time.time() - start_time < DEADLINE - 5:
        removed = False
        for i in range(len(code_indices)):
            if is_redundant(code_indices[i], code_indices, coverage_count):
                ci = code_indices.pop(i)
                for nb in nbr_lists[ci]:
                    coverage_count[nb] -= 1
                removed = True
                break

    # Final verification and emergency fill
    final_cov = np.zeros(total_words, dtype=bool)
    for ci in code_indices:
        for nb in nbr_lists[ci]:
            final_cov[nb] = True

    if not np.all(final_cov):
        covered = final_cov.copy()
        code_set = set(code_indices)
        while not np.all(covered) and time.time() - start_time < DEADLINE - 1:
            best_idx = -1
            best_new = -1
            for c in range(total_words):
                if c in code_set:
                    continue
                nbrs = nbr_lists[c]
                new_cov = 0
                for nb in nbrs:
                    if not covered[nb]:
                        new_cov += 1
                if new_cov > best_new:
                    best_new = new_cov
                    best_idx = c
            if best_new <= 0:
                break
            code_indices.append(best_idx)
            code_set.add(best_idx)
            for nb in nbr_lists[best_idx]:
                covered[nb] = True

    # Deduplicate
    seen = set()
    unique_code = []
    for idx in code_indices:
        if idx not in seen:
            seen.add(idx)
            unique_code.append(idx)

    return [idx_to_word(idx) for idx in unique_code]
