import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    total_words = q ** n

    # Precompute all words as tuples and map to indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute neighbors (Hamming ball of radius 1) for each word
    # For q=3, n=6, radius=1: each word has 1 + 6*(3-1) = 13 neighbors
    neighbors = []
    for i, w in enumerate(all_words):
        nbrs = {i}
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_list = list(w)
                    new_list[pos] = val
                    nbrs.add(word_to_idx[tuple(new_list)])
        neighbors.append(nbrs)

    # Convert neighbors to sorted lists for faster intersection
    neighbors_list = [sorted(nbrs) for nbrs in neighbors]

    def coverage_count(code_set):
        """Count how many words are covered by the given set of codeword indices."""
        covered = set()
        for idx in code_set:
            covered.update(neighbors[idx])
        return len(covered)

    def is_fully_covered(code_set):
        """Check if all words are covered."""
        covered = set()
        for idx in code_set:
            covered.update(neighbors[idx])
        return len(covered) == total_words

    def greedy_repair(code_set, uncovered):
        """Add codewords greedily to cover all uncovered words."""
        code_set = set(code_set)
        uncovered = set(uncovered)
        while uncovered and time.time() < deadline - 5:
            best_idx = -1
            best_count = 0
            for i in range(total_words):
                if i in code_set:
                    continue
                count = len(uncovered & neighbors[i])
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            code_set.add(best_idx)
            uncovered -= neighbors[best_idx]
        return code_set, uncovered

    def remove_redundant(code_set):
        """Remove redundant codewords."""
        code_list = list(code_set)
        removed_any = True
        while removed_any and time.time() < deadline - 5:
            removed_any = False
            for i in range(len(code_list) - 1, -1, -1):
                idx = code_list[i]
                others = [j for j in code_list if j != i]
                other_covered = set()
                for j in others:
                    other_covered.update(neighbors[j])
                if neighbors[idx].issubset(other_covered):
                    code_list.pop(i)
                    removed_any = True
                    break
        return set(code_list)

    def local_search_swap(code_set):
        """Try to improve by swapping or removing codewords. Returns (new_code_set, improved)."""
        code_list = list(code_set)
        improved = False

        for i in range(len(code_list)):
            if time.time() > deadline - 10:
                break
            c = code_list[i]
            others = [j for j in code_list if j != i]
            other_covered = set()
            for j in others:
                other_covered.update(neighbors[j])
            only_c = neighbors[c] - other_covered

            if not only_c:
                # c is redundant, remove it
                code_list.pop(i)
                improved = True
                break

            # Try to find a swap: replace c with c' that covers only_c
            # Prioritize candidates that cover many of only_c and have high overlap with others
            candidates = []
            for c_prime in range(total_words):
                if c_prime in code_set or c_prime == c:
                    continue
                if only_c.issubset(neighbors[c_prime]):
                    # Compute overlap with existing code (to prefer dense regions)
                    overlap = 0
                    for j in others:
                        overlap += len(neighbors[c_prime] & neighbors[j])
                    candidates.append((overlap, c_prime))

            if candidates:
                # Pick the one with highest overlap (most "central")
                candidates.sort(reverse=True)
                best_c_prime = candidates[0][1]
                code_list[i] = best_c_prime
                improved = True
                break

        return set(code_list), improved

    # Phase 1: Start with a good initial code using tetracode-inspired construction
    # The ternary tetracode has 16 words of length 4 covering radius 1.
    # We'll use a product-like construction and then optimize.

    # Generate a random initial code of size ~75 and repair
    random.seed(42)

    best_code = None
    best_size = total_words + 1

    # Multiple random restarts
    for restart in range(30):
        if time.time() > deadline - 15:
            break

        # Random initial code
        code_set = set()
        uncovered = set(range(total_words))

        # Add random codewords
        num_initial = random.randint(50, 80)
        for _ in range(num_initial):
            idx = random.randint(0, total_words - 1)
            if idx not in code_set:
                code_set.add(idx)
                uncovered -= neighbors[idx]

        # Repair with greedy
        code_set, uncovered = greedy_repair(code_set, uncovered)

        # Remove redundant
        code_set = remove_redundant(code_set)

        # Local search iterations
        for iteration in range(200):
            if time.time() > deadline - 10:
                break
            code_set, improved = local_search_swap(code_set)
            if not improved:
                break

        # Try to further reduce by checking if we can remove any
        code_set = remove_redundant(code_set)

        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = set(code_set)
            if best_size <= 73:
                break  # Can't do better than known best

    # If we didn't find anything, fall back to pure greedy
    if best_code is None:
        code_set = set()
        uncovered = set(range(total_words))
        # Fix first codeword to all zeros for symmetry breaking
        first_idx = word_to_idx[(0,) * n]
        code_set.add(first_idx)
        uncovered -= neighbors[first_idx]
        code_set, uncovered = greedy_repair(code_set, uncovered)
        code_set = remove_redundant(code_set)
        best_code = code_set
        best_size = len(best_code)

    # Final validation and conversion
    final_covered = set()
    for idx in best_code:
        final_covered.update(neighbors[idx])

    if final_covered != set(range(total_words)):
        # Fallback: pure greedy
        code_set = set()
        uncovered = set(range(total_words))
        first_idx = word_to_idx[(0,) * n]
        code_set.add(first_idx)
        uncovered -= neighbors[first_idx]
        while uncovered and time.time() < deadline:
            best_idx = -1
            best_count = 0
            for i in range(total_words):
                if i in code_set:
                    continue
                count = len(uncovered & neighbors[i])
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            code_set.add(best_idx)
            uncovered -= neighbors[best_idx]
        best_code = code_set

    code = [list(all_words[i]) for i in best_code]

    # Ensure distinctness
    seen = set()
    unique_code = []
    for c in code:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            unique_code.append(c)

    return unique_code
