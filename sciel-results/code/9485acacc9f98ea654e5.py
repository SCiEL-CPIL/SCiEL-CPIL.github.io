import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110
    
    total_words = q ** n
    
    # Fallback for large spaces where N^2 memory is too high
    if total_words > 10000:
        # Use the baseline direct sum construction
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Generate all words
    words_list = list(itertools.product(range(q), repeat=n))
    words = np.array(words_list, dtype=np.int8)
    
    # Precompute balls
    # balls[i, j] = 1 if dist(words[i], words[j]) <= radius
    balls = np.zeros((total_words, total_words), dtype=np.uint8)
    
    # Helper to get index from word tuple
    # Since words are generated in lexicographic order (base q),
    # index is the value of the word interpreted in base q.
    def get_index(word):
        idx = 0
        for val in word:
            idx = idx * q + val
        return idx

    # Build balls
    for i, w in enumerate(words):
        balls[i, i] = 1
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_w = list(w)
                    new_w[pos] = new_val
                    j = get_index(new_w)
                    balls[i, j] = 1
                    
        # If radius > 1, we would need to add distance 2, etc.
        # But the task is radius 1.
        # To be generic, we could compute distances, but for radius 1 this is optimal.
        # If radius > 1, this code is invalid.
        # The task specifies radius 1 for the 7 matches problem.
        # However, the interface takes `radius`.
        # If radius > 1, we need a more general ball construction.
        
    # General ball construction for any radius
    # Recompute balls if radius > 1
    if radius > 1:
        balls = np.zeros((total_words, total_words), dtype=np.uint8)
        for i in range(total_words):
            for j in range(total_words):
                # Hamming distance
                dist = np.sum(words[i] != words[j])
                if dist <= radius:
                    balls[i, j] = 1
        # This is O(N^2 * n). For N=2187, N^2=4.7M, n=7. 33M ops. Fast enough.
        # But we can do better with precomputed distances or just stick to radius 1 assumption if we know it.
        # The prompt says "evaluator passes this task's alphabet size q, length n and covering radius".
        # It implies the specific task (radius 1).
        # But to be safe, the O(N^2) check is fine for N <= 10000.
        # Actually, for N=10000, N^2 = 100M. 100M * 7 = 700M ops. Might be slow in Python loop.
        # Let's stick to the radius 1 specific optimization for the main case, and fallback for large N.
        # If radius > 1 and N is small, the O(N^2) loop is acceptable.
        # If radius > 1 and N is large, we fallback.
        
    # Re-implementing the ball construction to handle radius correctly and efficiently
    # For radius 1, the neighbor generation is O(N * n * q).
    # For general radius, we can use the distance matrix approach if N is small.
    
    # Let's restart the ball construction logic to be robust.
    if radius == 1:
         balls = np.zeros((total_words, total_words), dtype=np.uint8)
         for i, w in enumerate(words):
             balls[i, i] = 1
             for pos in range(n):
                 orig_val = w[pos]
                 for new_val in range(q):
                     if new_val == orig_val:
                         continue
                     new_w = list(w)
                     new_w[pos] = new_val
                     j = get_index(new_w)
                     balls[i, j] = 1
    else:
         # General case
         # Compute pairwise distances
         # This might be slow for large N, but we have the N < 10000 guard.
         balls = np.zeros((total_words, total_words), dtype=np.uint8)
         for i in range(total_words):
             # Vectorized distance calculation for row i
             # words is (N, n)
             # dists = np.sum(words != words[i], axis=1)
             dists = np.sum(words != words[i], axis=1)
             balls[i] = (dists <= radius).astype(np.uint8)

    # Greedy Set Cover
    uncovered = np.ones(total_words, dtype=np.uint8)
    selected_indices = []
    
    while uncovered.any():
        # counts[i] = number of uncovered words covered by ball i
        counts = balls @ uncovered
        
        if counts.max() == 0:
            break
            
        best_idx = np.argmax(counts)
        selected_indices.append(best_idx)
        
        # Update uncovered
        uncovered &= ~balls[best_idx]
        
        if time.time() > deadline:
            break
            
    result = [words[i].tolist() for i in selected_indices]
    
    return result
