import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    limit = 148.0  # Safety margin before 150s kill

    total_words = q ** n

    # Precompute base powers for index conversion
    base_powers = [q ** i for i in range(n)]

    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits

    def digits_to_idx(digits):
        return sum(d * b for d, b in zip(digits, base_powers))

    # Precompute neighbor lists for each word
    # For radius 1: each word has 1 + n*(q-1) neighbors (including itself)
    neighbor_lists = [None] * total_words
    for i in range(total_words):
        digits = idx_to_digits(i)
        neighbors = {i}
        if radius >= 1:
            for pos in range(n):
                orig = digits[pos]
                for new_val in range(q):
                    if new_val != orig:
                        new_digits = digits.copy()
                        new_digits[pos] = new_val
                        neighbors.add(digits_to_idx(new_digits))
        neighbor_lists[i] = list(neighbors)

    # Build sparse adjacency: for each word, list of neighbors
    # Use numpy for fast coverage counting
    # M[i] is a boolean array of size total_words where M[i,j]=1 if j is neighbor of i
    # This is dense and memory-heavy for large N, but for N=6561 it's ~43M booleans = ~43MB, acceptable.
    
    # Alternative: use a list of sets and count overlaps manually.
    # Let's try a smarter greedy: maintain uncovered set, and for each candidate count how many uncovered it covers.
    
    uncovered = np.ones(total_words, dtype=bool)
    solution = []

    # Precompute neighbor arrays for fast access
    neighbor_arrays = [np.zeros(total_words, dtype=bool) for _ in range(total_words)]
    for i in range(total_words):
        neighbor_arrays[i][neighbor_lists[i]] = True

    while True:
        if time.time() - start_time > limit:
            break
        if not np.any(uncovered):
            break

        # Find the codeword that covers the most uncovered words
        # counts[i] = sum(uncovered & neighbor_arrays[i])
        # This is O(N^2) per step which is 6561^2 ~ 43M ops per step.
        # With ~400 steps, that's ~17B ops. Too slow in pure Python/NumPy.
        
        # Better: maintain a "coverage count" array and update incrementally.
        # When we add codeword c, for each neighbor j of c, if uncovered[j] was True, 
        # then for every other candidate i that also covers j, its count decreases by 1.
        # This is still expensive.
        
        # Alternative approach: use the fact that each word has only ~17 neighbors.
        # For each uncovered word u, it is covered by a small set of candidates.
        # We can maintain for each candidate i, the number of uncovered neighbors.
        # When we pick candidate c and mark its neighbors as covered, we update counts for all candidates that share neighbors with c.
        
        # Let's precompute: for each pair of candidates (i, j), how many common neighbors they have.
        # This is a 6561x6561 matrix, too large.
        
        # Simpler: just do the naive O(N * deg) count per step.
        # deg ~ 17, N = 6561, so ~111k ops per step. With 400 steps, ~44M ops. Should be fast enough.
        
        best_idx = -1
        best_count = 0
        for i in range(total_words):
            if not uncovered[i]:
                continue
            # Count uncovered neighbors of i
            cnt = 0
            for nb in neighbor_lists[i]:
                if uncovered[nb]:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        
        if best_idx == -1 or best_count == 0:
            break
        
        solution.append(best_idx)
        for nb in neighbor_lists[best_idx]:
            uncovered[nb] = False

    # Local search: try to remove redundant codewords
    cover_count = np.zeros(total_words, dtype=np.int32)
    for idx in solution:
        for nb in neighbor_lists[idx]:
            cover_count[nb] += 1

    new_solution = list(solution)
    changed = True
    while changed:
        if time.time() - start_time > limit:
            break
        changed = False
        for i in range(len(new_solution) - 1, -1, -1):
            idx = new_solution[i]
            min_cover = min(cover_count[nb] for nb in neighbor_lists[idx])
            if min_cover > 1:
                for nb in neighbor_lists[idx]:
                    cover_count[nb] -= 1
                new_solution.pop(i)
                changed = True
                break

    # Try swaps: replace one codeword with another to enable further removals
    if time.time() - start_time < limit - 5:
        improved = True
        while improved and time.time() - start_time < limit - 2:
            improved = False
            for i in range(len(new_solution)):
                if time.time() - start_time > limit - 2:
                    break
                idx = new_solution[i]
                # Try removing idx and see if we can add a different codeword that covers what idx covered
                # Actually, try swapping idx with some j not in solution
                # Check if solution - {idx} + {j} is still a valid cover
                # This is expensive. Let's skip for now.
                pass

    return [idx_to_digits(idx) for idx in new_solution]
