"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct(q, n, radius):
    deadline = time.time() + 115.0  # safety margin before 120s limit
    
    # Generate all words in the space as a flat array
    # For q=3, n=8: 6561 words, manageable
    if q * n == 24 and q == 3:
        # Specialized fast path for the main task
        all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int8)
        total_words = q ** n  # 6561
        num_coords = n  # 8
        radius_val = radius  # 1
        
        # Precompute neighbors for each word (word itself + all single-coordinate changes)
        # For radius 1, q=3: each word has 1 + 8*2 = 17 neighbors
        neighbor_indices = np.zeros((total_words, 1 + num_coords * (q - 1)), dtype=np.int32)
        for i, w in enumerate(all_words):
            neighbor_indices[i, 0] = i
            pos = 1
            for coord in range(num_coords):
                for val in range(q):
                    if val != w[coord]:
                        # Compute index of word with coord changed to val
                        idx = i
                        # Convert to base-q, change one digit, convert back
                        # Faster: precompute powers
                        pass
            # Actually let's build neighbors differently for speed
        # Rebuild neighbors more efficiently
        # Each word index i corresponds to a base-q number
        # Changing coordinate c to value v: new index = i - old_digit * q^(n-1-c) + v * q^(n-1-c)
        powers = np.array([q ** (n - 1 - c) for c in range(n)], dtype=np.int32)
        neighbor_indices = np.zeros((total_words, 1 + num_coords * (q - 1)), dtype=np.int32)
        digits = np.zeros((total_words, n), dtype=np.int8)
        for c in range(n):
            digits[:, c] = (np.arange(total_words) // powers[c]) % q
        
        for i in range(total_words):
            neighbor_indices[i, 0] = i
            pos = 1
            for c in range(n):
                old_val = digits[i, c]
                for v in range(q):
                    if v != old_val:
                        neighbor_indices[i, pos] = i - old_val * powers[c] + v * powers[c]
                        pos += 1
        
        # Greedy set cover: repeatedly pick the word covering the most uncovered words
        covered = np.zeros(total_words, dtype=bool)
        code = []
        
        # Precompute for each candidate center, which words it covers
        # Use the neighbor_indices table
        while not covered.all() and time.time() < deadline:
            # Count uncovered words each candidate covers
            # For each candidate, sum over its neighbors of (not covered)
            # This is expensive per iteration, so we need to be smart
            
            # Vectorized: for each word j, it's covered by candidate i if j is in neighbors of i
            # Equivalently, for each candidate i, covered_count[i] = sum(covered[neighbors[i]])
            # But we want UNcovered, so: uncovered_count[i] = total_neighbors - covered_count[i]
            
            # Compute covered_count for all candidates
            # For each candidate i, it covers neighbor_indices[i, :]
            # covered_count[i] = covered[neighbor_indices[i, :]].sum()
            # This can be done with matrix multiply if we build the incidence matrix,
            # but 6561 x 17 is small enough to just do it directly
            
            # Build incidence: for each candidate, list of neighbors
            # Use numpy advanced indexing
            all_neighbors_flat = neighbor_indices.reshape(-1)
            neighbor_starts = np.arange(total_words) * (1 + num_coords * (q - 1))
            
            # For each candidate, compute how many of its neighbors are already covered
            # covered_count[i] = sum over j in neighbors[i] of covered[j]
            # Using the fact that neighbors[i] = neighbor_indices[i, :]
            covered_count = np.array([covered[nb].sum() for nb in neighbor_indices], dtype=np.int32)
            # This is 6561 iterations of 17-element sum - might be slow
            
            # Faster approach: use the incidence matrix
            # M[i, j] = 1 if j is a neighbor of i
            # Then covered_count = M @ covered
            # Build M once
            if not hasattr(construct, '_M_cached'):
                M = np.zeros((total_words, total_words), dtype=np.float32)
                for i in range(total_words):
                    M[i, neighbor_indices[i]] = 1.0
                construct._M_cached = M
            
            M = construct._M_cached
            covered_float = covered.astype(np.float32)
            covered_count = (M @ covered_float).astype(np.int32)
            
            # Each candidate covers 17 words total
            num_neighbors = 1 + num_coords * (q - 1)
            uncovered_count = num_neighbors - covered_count
            
            # Pick candidate with max uncovered
            best_idx = uncovered_count.argmax()
            if uncovered_count[best_idx] == 0:
                break
            
            code.append(all_words[best_idx].tolist())
            # Mark all neighbors of best_idx as covered
            covered[neighbor_indices[best_idx]] = True
        
        if time.time() >= deadline:
            # Return what we have so far - but it might not be a valid cover
            # We need to ensure validity. If we haven't finished, fall back to direct sum.
            pass
        
        # Verify we have a complete cover
        if not covered.all():
            # Fall back to the direct sum construction
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]
        
        return code
    else:
        # General fallback: direct sum construction for q=3
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]
        else:
            # Very general fallback: just use all words (trivially valid)
            return [list(w) for w in product(range(q), repeat=n)]
# EVOLVE-BLOCK-END
