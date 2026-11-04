"""Combined approach: structured init + coverage-count local search + bitmask validation."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _int_to_word(idx, q, n):
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _word_to_int(word, q):
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _build_neighbors(q, n, radius):
    """Build neighbor list for each word index (Hamming distance <= radius)."""
    total = q ** n
    neighbors = []
    for i in range(total):
        word = _int_to_word(i, q, n)
        nbrs = {i}
        for r in range(1, radius + 1):
            for positions in itertools.combinations(range(n), r):
                for values in itertools.product(range(1, q), repeat=r):
                    new_word = list(word)
                    for pos, val in zip(positions, values):
                        new_word[pos] = (word[pos] + val) % q
                    nbrs.add(_word_to_int(new_word, q))
        neighbors.append(sorted(nbrs))
    return neighbors


def _coverage_count_valid(coverage_count, neighbors, selected):
    """Check if coverage is valid using coverage counts."""
    for pos in selected:
        if coverage_count[pos] == 0:
            return False
    return True


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    total_words = q ** n
    sphere_size = 1 + n * (q - 1)  # For radius=1

    # Precompute neighbors
    neighbors = _build_neighbors(q, n, radius)
    neighbors_arr = np.array(neighbors, dtype=np.int32)

    # Build bitmask representation for validation
    num_chunks = (total_words + 63) // 64

    def mask_for_word(w_idx):
        chunks = np.zeros(num_chunks, dtype=np.uint64)
        for nbr in neighbors[w_idx]:
            chunks[nbr // 64] |= (1 << (nbr % 64))
        return chunks

    # Precompute masks for all words
    all_masks = [mask_for_word(i) for i in range(total_words)]

    def is_full_coverage(covered_chunks):
        for c in range(num_chunks):
            expected = (1 << 64) - 1
            if c == num_chunks - 1 and (total_words % 64) != 0:
                expected = (1 << (total_words % 64)) - 1
            if covered_chunks[c] != expected:
                return False
        return True

    def compute_coverage(solution):
        cov = np.zeros(total_words, dtype=np.int32)
        for w in solution:
            cov[neighbors_arr[w]] += 1
        return cov

    def greedy_cover(start_indices=None):
        """Greedy set cover with optional fixed starting codewords."""
        covered = np.zeros(total_words, dtype=bool)
        code = []
        
        if start_indices:
            for si in start_indices:
                if si not in code:
                    code.append(si)
                    covered[neighbors_arr[si]] = True
        
        while time.time() < deadline - 30.0:
            gains = (~covered)[neighbors_arr].sum(axis=1)
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            if best_idx not in code:
                code.append(best_idx)
                covered[neighbors_arr[best_idx]] = True
        return code

    def local_search_remove(code, coverage):
        """Remove redundant codewords using coverage counts."""
        changed = True
        while changed and time.time() < deadline - 20.0:
            changed = False
            for i in range(len(code)):
                if time.time() > deadline - 20.0:
                    break
                w = code[i]
                coverage[neighbors_arr[w]] -= 1
                if np.any(coverage[neighbors_arr[w]] == 0):
                    coverage[neighbors_arr[w]] += 1
                else:
                    code.pop(i)
                    changed = True
                    break
        return code

    def local_search_swap(code, coverage, max_iters=5000):
        """Try swap moves: remove one, add another, then try to remove more."""
        improved = True
        iters = 0
        while improved and time.time() < deadline - 15.0 and iters < max_iters:
            improved = False
            iters += 1
            for i in range(len(code)):
                if time.time() > deadline - 15.0:
                    break
                w_remove = code[i]
                coverage[neighbors_arr[w_remove]] -= 1
                
                # Find best word to add that covers most newly-uncovered
                uncovered = (coverage == 0)
                if not np.any(uncovered):
                    coverage[neighbors_arr[w_remove]] += 1
                    # Redundant, just remove
                    code.pop(i)
                    improved = True
                    break
                
                gains = uncovered[neighbors_arr].sum(axis=1)
                gains[w_remove] = -1  # Don't re-add same word
                best_add = int(np.argmax(gains))
                
                if gains[best_add] <= 0:
                    coverage[neighbors_arr[w_remove]] += 1
                    continue
                
                # Try the swap
                coverage[neighbors_arr[best_add]] += 1
                new_len = len(code)  # Same length after swap
                code[i] = best_add
                
                # Try to remove more
                for j in range(len(code)):
                    if j == i:
                        continue
                    w_test = code[j]
                    coverage[neighbors_arr[w_test]] -= 1
                    if np.any(coverage[neighbors_arr[w_test]] == 0):
                        coverage[neighbors_arr[w_test]] += 1
                    else:
                        code.pop(j)
                        improved = True
                        break
                
                if not improved:
                    # Roll back swap if it didn't help
                    coverage[neighbors_arr[best_add]] -= 1
                    coverage[neighbors_arr[w_remove]] += 1
                    code[i] = w_remove
        return code

    def validate_solution(code):
        """Validate using bitmask representation."""
        cov = np.zeros(num_chunks, dtype=np.uint64)
        for w in code:
            for c in range(num_chunks):
                cov[c] |= all_masks[w][c]
        return is_full_coverage(cov)

    best_code = None
    best_size = total_words + 1

    # Strategy 1: Standard greedy
    code1 = greedy_cover()
    cov1 = compute_coverage(code1)
    code1 = local_search_remove(code1, cov1)
    code1 = local_search_swap(code1, cov1)
    if validate_solution(code1) and len(code1) < best_size:
        best_size = len(code1)
        best_code = code1[:]

    # Strategy 2: Greedy with tetracode-based structured start
    if time.time() < deadline - 50.0:
        # Tetracode: 9 words covering 3^4 with radius 1
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetra_indices = [_word_to_int(w, q, 4) for w in tetracode]
        
        # For n=6: combine tetracode (4 coords) with all 9 words on remaining 2 coords
        # This gives 81 words initially, then optimize
        if n >= 4:
            start_idx = []
            for ti in tetra_indices:
                tetra_word = _int_to_word(ti, q, 4)
                for tail in itertools.product(range(q), repeat=n - 4):
                    full_word = tetra_word + list(tail)
                    start_idx.append(_word_to_int(full_word, q))
            
            code2 = greedy_cover(start_indices=start_idx[:20])  # Use subset as hint
            cov2 = compute_coverage(code2)
            code2 = local_search_remove(code2, cov2)
            code2 = local_search_swap(code2, cov2, max_iters=3000)
            if validate_solution(code2) and len(code2) < best_size:
                best_size = len(code2)
                best_code = code2[:]

    # Strategy 3: Multiple random restarts with different seeds
    for attempt in range(10):
        if time.time() > deadline - 25.0:
            break
        
        rng = random.Random(1000 + attempt * 7)
        
        # Random initial selection of a few codewords
        initial = rng.sample(range(total_words), min(5, total_words))
        code3 = greedy_cover(start_indices=initial)
        cov3 = compute_coverage(code3)
        code3 = local_search_remove(code3, cov3)
        code3 = local_search_swap(code3, cov3, max_iters=2000)
        if validate_solution(code3) and len(code3) < best_size:
            best_size = len(code3)
            best_code = code3[:]

    # Strategy 4: Aggressive local search on best solution found
    if best_code is not None and time.time() < deadline - 15.0:
        cov_best = compute_coverage(best_code)
        # Multiple passes of removal
        for _ in range(3):
            if time.time() > deadline - 15.0:
                break
            best_code = local_search_remove(best_code, cov_best)
            cov_best = compute_coverage(best_code)
        # Swap optimization
        best_code = local_search_swap(best_code, cov_best, max_iters=5000)
        if validate_solution(best_code):
            best_size = len(best_code)

    # Fallback: direct sum construction
    if best_code is None:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    return [_int_to_word(w, q, n) for w in best_code]
# EVOLVE-BLOCK-END
