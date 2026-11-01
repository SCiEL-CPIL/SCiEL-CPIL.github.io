"""Improved ternary covering code K_3(8,1): greedy init + aggressive local search."""
import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallback: direct sum with tetracode for general case
    import itertools
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Generic fallback: brute force all words
    import itertools
    return [list(c) for c in itertools.product(range(q), repeat=n)]


def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561

    # Precompute coverage: for each word, the 17 words it covers (itself + 16 neighbors)
    covered_by = np.zeros((total, 17), dtype=np.int32)
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    for w in range(total):
        covered_by[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                covered_by[w, idx] = new_w
                idx += 1

    # Greedy set cover initialization
    uncovered = np.ones(total, dtype=bool)
    uncovered_count = total
    selected = []
    
    while uncovered_count > 0 and time.time() < deadline - 40.0:
        cov_uncovered = uncovered[covered_by]
        coverage_counts = cov_uncovered.sum(axis=1)
        
        best_idx = np.argmax(coverage_counts)
        best_count = coverage_counts[best_idx]
        
        if best_count == 0:
            break
        
        selected.append(best_idx)
        new_cov = covered_by[best_idx]
        uncovered[new_cov] = False
        uncovered_count -= best_count
    
    # Fix any remaining uncovered
    if uncovered_count > 0 and time.time() < deadline - 15.0:
        remaining = np.where(uncovered)[0]
        for w in remaining:
            if time.time() > deadline - 10.0:
                break
            if w not in selected:
                selected.append(w)
                uncovered[covered_by[w]] = False
                uncovered_count -= 17  # approximate

    def to_digits(w):
        digits = []
        for i in range(n):
            digits.append((w // weights[n - 1 - i]) % 3)
        return digits

    # Build cover_count: for each word in space, how many codewords cover it
    cover_count = np.zeros(total, dtype=np.int32)
    for c in selected:
        cover_count[covered_by[c]] += 1

    # Local search: iterative removal
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        indices = list(range(len(selected)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 5.0:
                break
            if i >= len(selected):
                continue
            c = selected[i]
            ball = covered_by[c]
            if np.all(cover_count[ball] >= 2):
                cover_count[ball] -= 1
                selected.pop(i)
                # Fix indices
                indices = [j for j in indices if j != i]
                indices = [j - 1 if j > i else j for j in indices]
                improved = True

    # Try 2-for-1 exchange: find two codewords that can be replaced by one
    # This is complex, so let's try a simpler approach: try adding a random codeword,
    # then do removal, and if the net effect is positive, keep it.
    
    # Actually, let's try multiple passes of removal with different random seeds
    for seed in range(10):
        if time.time() > deadline - 5.0:
            break
        random.seed(seed)
        # Rebuild cover_count
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[covered_by[c]] += 1
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 5.0:
                    break
                if i >= len(selected):
                    continue
                c = selected[i]
                ball = covered_by[c]
                if np.all(cover_count[ball] >= 2):
                    cover_count[ball] -= 1
                    selected.pop(i)
                    indices = [j for j in indices if j != i]
                    indices = [j - 1 if j > i else j for j in indices]
                    improved = True

    code = [to_digits(w) for w in selected]
    
    # Ensure distinctness
    seen = set()
    unique_code = []
    for c in code:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            unique_code.append(c)
    
    return unique_code
