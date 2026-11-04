"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def get_direct_sum():
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [[0] * n]

    # For the specific case q=3, n=6, radius=1, use optimized greedy set cover
    if q == 3 and n == 6 and radius == 1:
        total = 3 ** 6
        powers = [3 ** p for p in range(n)]

        # Precompute for each word, the 13 words within Hamming distance 1
        neighbors = np.empty((total, 13), dtype=np.int32)
        for i in range(total):
            d = [(i // powers[p]) % 3 for p in range(n)]
            neighbors[i, 0] = i
            k = 1
            for p in range(n):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        neighbors[i, k] = new_i
                        k += 1

        # Now do greedy: prefer candidates that cover the most uncovered words
        uncovered = np.ones(total, dtype=np.int8)
        codewords = []

        while uncovered.sum() > 0:
            if time.time() > deadline:
                break

            # Find candidate that covers most uncovered words
            # Sum of uncovered neighbors for each candidate
            # neighbors shape (729, 13), uncovered shape (729,)
            # uncovered[neighbors] shape (729, 13)
            counts = np.sum(uncovered[neighbors], axis=1)
            
            best_candidate = np.argmax(counts)
            
            if counts[best_candidate] == 0:
                break

            # Add this codeword index
            codewords.append(best_candidate)
            
            # Mark covered
            uncovered[neighbors[best_candidate]] = 0

        # Verify we have a valid covering
        if len(codewords) > 0 and uncovered.sum() == 0:
            # Local Search: Try to remove redundant codewords
            # Structural improvement: post-optimization via local search
            improved = True
            while improved:
                improved = False
                # Rebuild coverage counts
                coverage_counts = np.zeros(total, dtype=np.int32)
                for idx_c in codewords:
                    coverage_counts[neighbors[idx_c]] += 1
                
                # Find a codeword that can be removed
                # A codeword idx_c can be removed if all its neighbors are covered by >= 2 codewords
                for i, idx_c in enumerate(codewords):
                    if np.min(coverage_counts[neighbors[idx_c]]) >= 2:
                        # Safe to remove
                        codewords.pop(i)
                        improved = True
                        break # Restart the while loop

            # Convert indices back to codeword lists
            final_codewords = []
            for idx_c in codewords:
                final_codewords.append([(idx_c // powers[p]) % 3 for p in range(n)])
            
            ds = get_direct_sum()
            if len(final_codewords) <= len(ds):
                return final_codewords
            else:
                return ds
        else:
            return get_direct_sum()

    # General case: fallback to direct sum
    return get_direct_sum()
# EVOLVE-BLOCK-END
