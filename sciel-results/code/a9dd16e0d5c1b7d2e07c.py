import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    N = q ** n
    
    # Precompute neighbors for each word index
    # neighbors[i] is a list of indices j such that HammingDist(i, j) <= radius
    # For radius 1, this is fast to compute.
    # We can generalize for small radius, but task implies radius 1 typically for these sizes.
    # Let's assume radius is small (1 or 2).
    
    # Helper to convert index to word
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w
        
    # Helper to convert word to index
    def word_to_idx(w):
        idx = 0
        base = 1
        for i in range(n):
            idx += w[i] * base
            base *= q
        return idx

    # Precompute neighbors
    neighbors = [[] for _ in range(N)]
    if radius == 1:
        for idx in range(N):
            w = idx_to_word(idx)
            neighbors[idx].append(idx)
            for pos in range(n):
                orig = w[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    w[pos] = val
                    neighbors[idx].append(word_to_idx(w))
                w[pos] = orig
    elif radius == 0:
        for idx in range(N):
            neighbors[idx].append(idx)
    else:
        # Fallback for radius > 1: brute force check or BFS
        # For this task, radius is likely 1.
        for idx in range(N):
            w1 = idx_to_word(idx)
            for jdx in range(N):
                w2 = idx_to_word(jdx)
                dist = sum(1 for a, b in zip(w1, w2) if a != b)
                if dist <= radius:
                    neighbors[idx].append(jdx)
                    
    best_code = None
    best_size = N + 1
    
    # Greedy Set Cover
    def greedy_solve(seed):
        random.seed(seed)
        coverage = np.zeros(N, dtype=np.int32)
        code = []
        
        # To speed up, we can maintain a list of uncovered words?
        # Or just scan all candidates each time.
        # Scanning all N candidates each step is O(N * BallSize) per step.
        # Total steps ~ N / BallSize.
        # Total ~ N^2 / BallSize.
        
        while True:
            if time.time() > deadline:
                return None
            
            best_c = -1
            max_uncovered = -1
            
            # Iterate through all potential centers
            for c in range(N):
                # Count how many words in neighbors[c] are currently uncovered
                # Using numpy for this specific check might be slow due to overhead per iteration
                # Let's try a pure python loop first, it might be fast enough for N=6561
                cnt = 0
                for nb in neighbors[c]:
                    if coverage[nb] == 0:
                        cnt += 1
                        # Early exit if we know we can't beat current best?
                        # if cnt + (len(neighbors[c]) - current_pos) <= max_uncovered: break
                        # But we don't have current_pos easily.
                        
                if cnt > max_uncovered:
                    max_uncovered = cnt
                    best_c = c
                    # If we cover 0, we are stuck (should not happen if coverage < N)
                    if max_uncovered == 0:
                        break
            
            if best_c == -1 or max_uncovered == 0:
                break
                
            code.append(best_c)
            
            # Update coverage
            for nb in neighbors[best_c]:
                coverage[nb] += 1
            
            # Optimization: If all covered, stop
            # Checking if all covered takes O(N).
            # We can track a count of uncovered words.
            # Let's add an uncovered_count variable.
            
        return code

    # The above greedy doesn't track uncovered_count efficiently to stop early.
    # Let's refine the greedy loop.
    
    def greedy_solve_optimized(seed):
        random.seed(seed)
        coverage = np.zeros(N, dtype=np.int32)
        uncovered_count = N
        code = []
        
        while uncovered_count > 0:
            if time.time() > deadline:
                return None
            
            best_c = -1
            max_new_covers = -1
            
            for c in range(N):
                new_covers = 0
                for nb in neighbors[c]:
                    if coverage[nb] == 0:
                        new_covers += 1
                        
                if new_covers > max_new_covers:
                    max_new_covers = new_covers
                    best_c = c
                    if max_new_covers == 0:
                        break
            
            if best_c == -1 or max_new_covers == 0:
                # Should not happen if uncovered_count > 0
                break
                
            code.append(best_c)
            
            # Update coverage and uncovered_count
            for nb in neighbors[best_c]:
                if coverage[nb] == 0:
                    uncovered_count -= 1
                coverage[nb] += 1
                
        return code

    # Run multiple restarts
    num_restarts = 10
    for i in range(num_restarts):
        code = greedy_solve_optimized(i)
        if code is None:
            break
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
    if best_code is None:
        # Fallback to direct sum if greedy failed or timed out completely
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        best_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return best_code
