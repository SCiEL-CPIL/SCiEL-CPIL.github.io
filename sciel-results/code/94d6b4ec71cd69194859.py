import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    # For this specific task, q=3, n=7, radius=1
    # But we should handle general inputs if possible, or at least not crash.
    # The task contract says the evaluator passes q, n, radius.
    # If n is large, this algorithm won't work. We assume the test case is the one described.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to the baseline or a simple construction if it's not the specific case
        # The baseline was:
        # tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Fallback for small n
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    N = q ** n
    # Map word to index
    def to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def from_idx(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute neighbors for all N words
    # neighbors[i] is a list of indices covered by codeword i
    neighbors = []
    for i in range(N):
        w = from_idx(i)
        neigh = [i]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w[pos] = val
                neigh.append(to_idx(w))
            w[pos] = orig
        neighbors.append(neigh)
    
    # Convert to numpy arrays for faster intersection/counting if needed, 
    # but lists are fine for N=2187.
    
    best_codewords = None
    best_size = N # Upper bound
    
    # Run multiple rounds of greedy
    # Round 0: Deterministic Greedy
    # Round 1..K: Randomized Greedy
    
    def run_greedy(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        
        covered = np.zeros(N, dtype=bool)
        codewords = []
        
        # Initial gains
        # gain[i] = number of uncovered words in neighbors[i]
        # Initially all are uncovered, so gain[i] = len(neighbors[i])
        gains = np.array([len(neigh) for neigh in neighbors], dtype=np.int32)
        
        while not np.all(covered):
            # Find candidate with max gain
            if randomize:
                # Pick from top 10
                top_k = 10
                # Get indices of top k gains
                # np.argpartition is efficient
                top_indices = np.argpartition(gains, -top_k)[-top_k:]
                # Pick one randomly
                c = top_indices[random.randint(0, top_k - 1)]
            else:
                c = np.argmax(gains)
                
            if gains[c] == 0:
                # Should not happen if not all covered, but safety check
                break
                
            codewords.append(c)
            
            # Update covered
            for idx in neighbors[c]:
                covered[idx] = True
            
            # Update gains for all other codewords
            # gain[j] -= number of newly covered words in neighbors[j]
            # Newly covered words are neighbors[c] that were previously uncovered.
            # Actually, it's easier to just recompute gains for all j?
            # N=2187, len(neigh)=15. 
            # Loop over N codewords, for each count uncovered in its neighbor list.
            # 2187 * 15 = ~32k ops per step.
            # 200 steps * 32k = 6.4M ops. Fast enough.
            
            for j in range(N):
                if j == c:
                    gains[j] = 0
                    continue
                # Count uncovered in neighbors[j]
                # Using numpy sum on a boolean slice?
                # neighbors[j] is a list. 
                # We can use: gains[j] = np.sum(covered[neighbors[j]] == False)
                # But converting list to array every time is slow.
                # Let's pre-store neighbors as numpy arrays.
                
            # Optimization: Pre-store neighbors as numpy arrays
            # We did `neighbors.append(neigh)` where neigh is a list.
            # Let's convert once.
            
        return codewords

    # Pre-store neighbors as numpy arrays for speed
    neighbors_np = [np.array(neigh, dtype=np.int32) for neigh in neighbors]
    
    def run_greedy_optimized(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        
        covered = np.zeros(N, dtype=bool)
        codewords = []
        
        # gains[j] = number of uncovered words in neighbors_np[j]
        # Initialize
        gains = np.array([len(neigh) for neigh in neighbors_np], dtype=np.int32)
        
        while not np.all(covered):
            if randomize:
                top_k = min(10, N)
                top_indices = np.argpartition(gains, -top_k)[-top_k:]
                c = top_indices[random.randint(0, top_k - 1)]
            else:
                c = np.argmax(gains)
                
            if gains[c] == 0:
                break
                
            codewords.append(c)
            
            # Update covered
            covered[neighbors_np[c]] = True
            
            # Update gains
            # For each j, gains[j] = count of uncovered in neighbors_np[j]
            # We can vectorize this?
            # gains = np.array([np.sum(~covered[nb]) for nb in neighbors_np])
            # This is a list comprehension over N=2187 items, each doing a numpy sum of 15 items.
            # This is fast.
            gains = np.array([np.sum(~covered[nb]) for nb in neighbors_np], dtype=np.int32)
            
        return codewords

    # Run deterministic first
    cw = run_greedy_optimized(randomize=False)
    if len(cw) < best_size:
        best_size = len(cw)
        best_codewords = cw
        
    # Run randomized variants
    num_rounds = 50
    for i in range(num_rounds):
        if time.time() - start_time > timeout:
            break
        cw = run_greedy_optimized(randomize=True, seed=i)
        if len(cw) < best_size:
            best_size = len(cw)
            best_codewords = cw
            
    # Convert indices back to lists of integers
    if best_codewords is None:
        # Fallback
        return [list(t) for t in itertools.product(range(q), repeat=n)]
        
    return [from_idx(c) for c in best_codewords]

# EVOLVE-BLOCK-END
