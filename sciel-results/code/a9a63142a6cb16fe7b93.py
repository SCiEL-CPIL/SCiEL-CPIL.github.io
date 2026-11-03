import time
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 110s out of 120s
    
    # For the specific case of q=3, n=7, radius=1, we can use a highly optimized greedy.
    # If parameters are different, we fall back to a standard greedy or the seed.
    # Given the task description, it is highly likely q=3, n=7, radius=1.
    
    if q == 3 and n == 7 and radius == 1:
        return _construct_optimized_3_7_1()
    else:
        # Fallback: Standard greedy for general small cases
        return _construct_general_greedy(q, n, radius, start_time, deadline)

def _construct_optimized_3_7_1():
    q = 3
    n = 7
    radius = 1
    N = q ** n  # 2187
    
    # Precompute distance matrix
    # dist[i][j] = Hamming distance between i and j
    # We can compute this by converting to base-q digits.
    # To save memory and time, we can compute the intersection sizes directly.
    
    # Intersection size for radius 1:
    # d=0: 1 + n(q-1) = 15
    # d=1: q = 3
    # d=2: 2
    # d>2: 0
    
    # We need to know the distance between any pair to update counts.
    # Precomputing a full N x N distance matrix (2187x2187) is ~4.7M bytes (uint8).
    # This is fast and memory efficient.
    
    # Generate all words as tuples of digits
    words = list(itertools.product(range(q), repeat=n))
    
    # Map index to word
    # index is the base-q number
    
    # Precompute distances
    # dist_mat[i][j]
    dist_mat = np.zeros((N, N), dtype=np.uint8)
    
    # Optimization: Compute distances on the fly in the update loop?
    # No, precomputation is better.
    # But generating the matrix takes time.
    # Let's compute digits for all words first.
    digits = np.zeros((N, n), dtype=np.uint8)
    for i in range(N):
        val = i
        for j in range(n):
            digits[i, j] = val % q
            val //= q
            
    # Compute pairwise distances
    # This is N^2 * n operations. 2187^2 * 7 ~ 33 million ops.
    # In Python, this might take a few seconds.
    # Let's try to be smarter.
    
    # Alternative: Don't precompute full matrix.
    # In the update step, for the chosen w*, we need to update count[w] for all w.
    # count[w] -= inter_size(dist(w, w*))
    # We can compute dist(w, w*) on the fly.
    # dist(w, w*) requires comparing n digits.
    # Update loop is N iterations. Each takes n ops.
    # Total per step: N * n = 2187 * 7 ~ 15k ops.
    # Total steps ~ 186.
    # Total ops ~ 186 * 15k ~ 2.7 million.
    # This is very fast in Python if we use lists/tuples.
    # Using numpy for the inner loop might be overkill or slower due to overhead.
    # Let's use pure Python lists for digits.
    
    digits_list = [tuple(d) for d in digits]
    
    # Initialize counts
    # Initial count for any w is |B(w)| = 15
    counts = [15] * N
    
    # Uncovered set
    # We don't strictly need the set if we use counts.
    # But we need to know when we are done.
    # We are done when max(counts) == 0.
    
    code = []
    
    # Precompute intersection sizes based on distance
    # inter_size[d]
    # d=0: 15
    # d=1: 3
    # d=2: 2
    # d>=3: 0
    inter_sizes = [0] * (n + 1)
    inter_sizes[0] = 1 + n * (q - 1)
    inter_sizes[1] = q
    inter_sizes[2] = 2
    # for d > 2, it's 0
    
    for _ in range(N):  # Max possible codewords
        # Find max count
        max_count = max(counts)
        if max_count == 0:
            break
            
        # Find index of max
        # To make it deterministic and potentially faster, we can just take the first max
        best_idx = counts.index(max_count)
        
        code.append(best_idx)
        
        # Update counts
        # For all w, counts[w] -= inter_sizes[dist(w, best_idx)]
        best_digits = digits_list[best_idx]
        
        for w in range(N):
            # Compute distance
            d = 0
            w_digits = digits_list[w]
            for k in range(n):
                if w_digits[k] != best_digits[k]:
                    d += 1
                    if d > 2:
                        break
            if d <= 2:
                counts[w] -= inter_sizes[d]
                
        # Time check
        if time.time() > deadline:
            break
            
    # Convert indices to lists of integers
    result = [list(digits_list[i]) for i in code]
    return result

def _construct_general_greedy(q, n, radius, start_time, deadline):
    N = q ** n
    if N > 10000:
        # Fallback to seed if too large
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
            
    # General greedy
    words = list(itertools.product(range(q), repeat=n))
    words_set = set(words)
    
    # Precompute balls
    balls = {}
    for w in words:
        ball = {w}
        for i in range(n):
            for val in range(q):
                if val != w[i]:
                    new_w = list(w)
                    new_w[i] = val
                    ball.add(tuple(new_w))
        balls[w] = ball
        
    uncovered = set(words)
    code = []
    
    while uncovered:
        best_word = None
        max_cov = -1
        for w in words:
            cov = len(balls[w] & uncovered)
            if cov > max_cov:
                max_cov = cov
                best_word = w
                if cov == len(uncovered):
                    break
        code.append(best_word)
        uncovered -= balls[best_word]
        
        if time.time() > deadline:
            break
            
    return [list(w) for w in code]
