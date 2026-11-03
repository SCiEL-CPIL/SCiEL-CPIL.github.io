import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=6, radius=1
    total_words = q ** n

    # Precompute all words as a 2D numpy array
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)

    # Precompute spheres as boolean matrices: spheres[i] is True for words covered by codeword i
    # Sphere of radius 1: word itself + all words at Hamming distance 1
    # For each position and each alternative value, compute the target word
    spheres = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        w = all_words[i]
        spheres[i, i] = True
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                # Compute index of word with w[pos] replaced by val
                # Use base-q representation
                idx = 0
                for j in range(n):
                    v = val if j == pos else w[j]
                    idx = idx * q + v
                spheres[i, idx] = True

    # Initial solution: tetracode-based construction (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_codewords.append(h + list(tail))

    best_codewords = initial_codewords
    best_size = len(best_codewords)

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115  # 5 second margin before 120s limit

    def get_indices(codewords):
        """Convert list of codeword lists to list of indices."""
        indices = []
        for cw in codewords:
            idx = 0
            for i in range(n):
                idx = idx * q + cw[i]
            indices.append(idx)
        return indices

    def compute_coverage(indices):
        """Compute coverage count for each word given a set of codeword indices."""
        if not indices:
            return np.zeros(total_words, dtype=np.int32)
        idx_arr = np.array(indices, dtype=np.int64)
        # Sum spheres rows for selected codewords
        cov = spheres[idx_arr].sum(axis=0)
        return cov

    def is_covered(indices):
        """Check if all words are covered."""
        cov = compute_coverage(indices)
        return np.all(cov >= 1)

    def try_remove_redundant(indices):
        """Try to remove redundant codewords. Returns (new_indices, removed_count)."""
        removed = 0
        improved = True
        while improved:
            improved = False
            for i in range(len(indices)):
                # Temporarily remove indices[i]
                temp = indices[:i] + indices[i+1:]
                if not temp:
                    continue
                cov = compute_coverage(temp)
                if np.all(cov >= 1):
                    indices = temp
                    removed += 1
                    improved = True
                    break
        return indices, removed

    # Phase 1: Start from initial solution and prune
    current_indices = get_indices(best_codewords)
    current_indices, _ = try_remove_redundant(current_indices)
    if len(current_indices) < best_size:
        best_size = len(current_indices)
        best_codewords = [all_words[i].tolist() for i in current_indices]

    # Phase 2: Perturbation + repair loop
    # Strategy: remove k random codewords, then greedily repair coverage
    for restart in range(2000):
        if time.time() > deadline:
            break

        # Create a perturbed version of the current best
        if random.random() < 0.5:
            # Start from best
            base = list(best_codewords)
        else:
            # Start from a random subset of best
            base = random.sample(best_codewords, max(1, len(best_codewords) - random.randint(5, 15)))

        # Remove some random codewords to create gaps
        num_remove = random.randint(3, min(10, len(base) - 5))
        if len(base) <= num_remove + 5:
            num_remove = max(1, len(base) - 5)
        if num_remove < 1:
            continue
        to_remove = random.sample(range(len(base)), num_remove)
        to_remove_set = set(to_remove)
        perturbed = [base[i] for i in range(len(base)) if i not in to_remove_set]

        # Greedy repair: add codewords to cover uncovered words
        perturbed_indices = get_indices(perturbed)
        cov = compute_coverage(perturbed_indices)
        uncovered_count = int(np.sum(cov == 0))

        while uncovered_count > 0:
            # Find codeword that covers most uncovered words
            # For each candidate, count how many uncovered words it covers
            uncovered_mask = (cov == 0)
            # Gain for each candidate = sum of spheres[i] over uncovered positions
            gains = spheres[:, uncovered_mask].sum(axis=1)
            # Only consider candidates not already in set
            in_set = np.zeros(total_words, dtype=bool)
            for idx in perturbed_indices:
                in_set[idx] = True
            gains[in_set] = 0

            best_gain = np.max(gains)
            if best_gain == 0:
                break  # Should not happen

            # Pick from top candidates with some randomness
            top_threshold = max(1, best_gain - 1)
            candidates = np.where(gains >= top_threshold)[0]
            chosen = random.choice(candidates)

            perturbed_indices.append(chosen)
            cov = compute_coverage(perturbed_indices)
            uncovered_count = int(np.sum(cov == 0))

        # Try to prune the repaired solution
        perturbed_indices, _ = try_remove_redundant(perturbed_indices)

        if len(perturbed_indices) < best_size:
            best_size = len(perturbed_indices)
            best_codewords = [all_words[i].tolist() for i in perturbed_indices]

            # Additional pruning pass
            best_indices = get_indices(best_codewords)
            best_indices, _ = try_remove_redundant(best_indices)
            if len(best_indices) < best_size:
                best_size = len(best_indices)
                best_codewords = [all_words[i].tolist() for i in best_indices]

        # If we hit the known optimum, we can stop early
        if best_size <= 73:
            # Continue briefly to see if we can do better
            if time.time() > deadline - 10:
                break

    # Final pruning pass
    best_indices = get_indices(best_codewords)
    best_indices, _ = try_remove_redundant(best_indices)
    if len(best_indices) < best_size:
        best_size = len(best_indices)
        best_codewords = [all_words[i].tolist() for i in best_indices]

    return best_codewords
