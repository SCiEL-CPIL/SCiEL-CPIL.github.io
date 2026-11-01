"""Improved covering code construction for K_3(6,1)."""
import time
import itertools
import numpy as np
from collections import deque


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Base solution: direct sum of ternary tetracode with all tails
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    base_solution = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Encode words as integers for fast coverage checking
    total_words = q ** n
    # Precompute neighbors for each word (word itself + all words at distance 1)
    # For n=6, q=3: each word has 1 + 6*2 = 13 neighbors
    neighbors = {}
    for w in range(total_words):
        arr = []
        for i in range(n):
            arr.append((w // (q ** (n - 1 - i))) % q)
        arr = tuple(arr)
        nbrs = [w]
        for pos in range(n):
            for delta in range(1, q):
                new_val = (arr[pos] + delta) % q
                # Compute new integer value
                new_w = 0
                for j in range(n):
                    val = new_val if j == pos else arr[j]
                    new_w += val * (q ** (n - 1 - j))
                nbrs.append(new_w)
        neighbors[w] = nbrs
    
    # Build coverage: for each word, which codeword covers it
    # We'll use a greedy + local search approach
    
    def coverage_count(codewords):
        """Count how many words are covered by the given codewords."""
        covered = set()
        for cw in codewords:
            arr = []
            for i in range(n):
                arr.append((cw // (q ** (n - 1 - i))) % q)
            cw_int = 0
            for j in range(n):
                cw_int += arr[j] * (q ** (n - 1 - j))
            for nb in neighbors[cw_int]:
                covered.add(nb)
        return len(covered)
    
    def is_valid(codewords):
        """Check if codewords cover all words."""
        covered = set()
        for cw in codewords:
            arr = []
            for i in range(n):
                arr.append((cw // (q ** (n - 1 - i))) % q)
            cw_int = 0
            for j in range(n):
                cw_int += arr[j] * (q ** (n - 1 - j))
            for nb in neighbors[cw_int]:
                covered.add(nb)
        return len(covered) == total_words
    
    # Start with base solution and try to improve via local search
    best_solution = list(base_solution)
    best_count = len(best_solution)
    
    # Greedy removal: try removing each codeword if coverage is maintained
    improved = True
    while improved:
        improved = False
        for i in range(len(best_solution)):
            if time.time() > deadline:
                break
            candidate = best_solution[:i] + best_solution[i+1:]
            if coverage_count(candidate) == total_words:
                best_solution = candidate
                best_count = len(best_solution)
                improved = True
                break  # restart the scan since indices shifted
    
    # Local search: try swapping codewords
    if time.time() < deadline:
        for iteration in range(100):
            if time.time() > deadline:
                break
            # Try to replace one codeword with another
            for i in range(len(best_solution)):
                if time.time() > deadline:
                    break
                # Try a random replacement
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    # Generate a random candidate
                    new_cw = [np.random.randint(0, q) for _ in range(n)]
                    candidate = best_solution[:i] + [new_cw] + best_solution[i+1:]
                    if coverage_count(candidate) == total_words:
                        best_solution = candidate
                        best_count = len(best_solution)
    
    return best_solution
