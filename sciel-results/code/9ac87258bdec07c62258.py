import itertools
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Fallback for non-specific cases
    def get_direct_sum():
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [[0] * n]

    # Specific optimization for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        total = 3 ** n
        powers = [3 ** p for p in range(n)]
        
        # Precompute neighbors: for each word i, list of 13 neighbors (including self)
        # Shape: (729, 13)
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
        
        best_codewords = None
        best_size = float('inf')
        
        # Run multiple randomized greedy attempts
        # The greedy algorithm picks the word covering the most uncovered words.
        # Randomizing tie-breaking or adding noise can lead to different local optima.
        
        for attempt in range(20): # Try a few times
            if time.time() > deadline:
                break
                
            uncovered = np.ones(total, dtype=np.int8)
            codewords = []
            
            while uncovered.sum() > 0:
                if time.time() > deadline:
                    break
                
                # Calculate coverage for all candidates
                # uncovered[neighbors] has shape (729, 13)
                # Summing along axis 1 gives count of uncovered neighbors for each candidate
                counts = np.sum(uncovered[neighbors], axis=1)
                
                # Find max count
                max_count = np.max(counts)
                if max_count == 0:
                    break
                
                # Get all candidates with max count
                candidates = np.where(counts == max_count)[0]
                
                # Randomize selection among ties to explore different paths
                # If we always pick the first, we get a deterministic result.
                # Picking randomly helps escape the "standard" greedy trap.
                best_candidate = candidates[np.random.randint(len(candidates))]
                
                codewords.append(best_candidate)
                uncovered[neighbors[best_candidate]] = 0
            
            if len(codewords) > 0 and uncovered.sum() == 0:
                # Local Search: Remove redundancies
                # Rebuild coverage counts
                coverage_counts = np.zeros(total, dtype=np.int32)
                for idx_c in codewords:
                    coverage_counts[neighbors[idx_c]] += 1
                
                improved = True
                while improved:
                    improved = False
                    # Try to remove a codeword if it's redundant
                    # A codeword i is redundant if min(coverage_counts[neighbors[i]]) >= 2
                    # We can check this for all codewords in the list
                    for i in range(len(codewords) - 1, -1, -1):
                        idx_c = codewords[i]
                        if np.min(coverage_counts[neighbors[idx_c]]) >= 2:
                            # Remove it
                            coverage_counts[neighbors[idx_c]] -= 1
                            codewords.pop(i)
                            improved = True
                            break # Restart the while loop to re-evaluate others
                
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = codewords[:]
                    # If we hit the theoretical lower bound (57) or best known (71/72), we could stop early
                    # But let's keep trying to see if we can go lower.
                    if best_size <= 71:
                        break

        # Convert best codewords to list of lists
        if best_codewords is not None:
            final_codewords = []
            for idx_c in best_codewords:
                final_codewords.append([(idx_c // powers[p]) % 3 for p in range(n)])
            return final_codewords
        else:
            return get_direct_sum()

    return get_direct_sum()
