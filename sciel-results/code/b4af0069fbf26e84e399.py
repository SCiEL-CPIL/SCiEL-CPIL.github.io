import time
import random
import numpy as np

def _score(A):
    """
    Compute max(|A+A|, |A*A|) efficiently.
    A is a list or numpy array of 20 distinct positive integers.
    """
    A = np.asarray(A)
    # A+A: outer sum
    # A*A: outer product
    # We need the number of unique elements.
    # For n=20, n^2 = 400 elements. 
    # Converting to a set of Python integers is usually faster than 
    # complex numpy unique operations for small n, but let's try 
    # a hybrid or just optimized python sets if numpy overhead is high.
    
    # Actually, for n=20, pure Python sets are extremely fast.
    # Let's stick to pure Python for the core scoring to avoid 
    # numpy conversion overhead which might dominate for small n.
    # But we can use numpy to generate the pairs quickly?
    # No, pure python nested loops for 400 items is < 1ms.
    
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(n):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    return max(len(sums), len(prods))

def _is_valid(A, n=20, maximum=10000):
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 for x in A):
        return False
    if any(x > maximum for x in A):
        return False
    return True

def construct():
    start_time = time.time()
    # Deadline guard: 60s limit, aim for 55s safety
    deadline = start_time + 55
    
    rng = random.Random(12345)
    
    best_A = None
    best_score = float('inf')
    
    def try_candidate(A):
        nonlocal best_A, best_score
        if not _is_valid(A, 20, 10000):
            return
        s = _score(A)
        if s < best_score:
            best_score = s
            best_A = sorted(A)

    # 1. Generate diverse structural initial candidates
    start_points = []
    
    # Arithmetic Progressions
    for start in [1, 2, 10, 100]:
        for step in [1, 2, 5, 10, 50, 100]:
            A = [start + i*step for i in range(20)]
            if max(A) <= 10000:
                start_points.append(A)
                
    # Geometric Progressions (Powers of 2, 3, 4)
    for base in [2, 3, 4]:
        A = [base**i for i in range(20)]
        if max(A) <= 10000:
            start_points.append(A)
        else:
            # Truncate and fill with large distinct numbers
            A = [base**i for i in range(13)] # 2^12 = 4096
            # Fill remaining 7 with large numbers to keep product set large? 
            # Or small numbers? 
            # If we want to minimize max(sum, prod), we need to balance.
            # Let's just append large randoms for now
            remaining = rng.sample(range(max(A)+1, 10001), 7)
            A.extend(remaining)
            start_points.append(A)

    # "Small Core, Large Tail"
    # Small numbers keep sums small, large numbers keep products distinct?
    # Actually, large numbers make products huge, but if they are powers, 
    # products are unique.
    # Try: 1..10 and 100, 200, 300...
    for tail_start in [100, 1000, 5000]:
        A = list(range(1, 11))
        A.extend([tail_start + i*10 for i in range(10)])
        if max(A) <= 10000:
            start_points.append(A)
            
    # Random Samples
    for _ in range(20):
        A = rng.sample(range(1, 10001), 20)
        start_points.append(A)
        
    # Evaluate all start points
    for A in start_points:
        try_candidate(A)
        
    if best_A is None:
        return list(range(1, 21))

    # 2. Simulated Annealing from the best start point
    # We will run SA from multiple top-performing start points
    
    top_starts = []
    seen_scores = {}
    for A in start_points:
        if _is_valid(A, 20, 10000):
            s = _score(A)
            if s not in seen_scores:
                seen_scores[s] = A
    # Get top 5 distinct scores
    sorted_scores = sorted(seen_scores.keys())
    for s in sorted_scores[:5]:
        top_starts.append(seen_scores[s])
        
    if not top_starts:
        top_starts = [best_A]

    T = 1000.0
    T_min = 1.0
    alpha = 0.9995 # Cooling rate
    
    for start_A in top_starts:
        current_A = sorted(start_A[:])
        current_score = _score(current_A)
        T = 1000.0
        
        it = 0
        max_iters = 20000
        while it < max_iters and time.time() < deadline - 2:
            it += 1
            
            # Perturbation: pick a random index, change value
            i = rng.randint(0, 19)
            # Choose a delta. Mix of small and large jumps.
            if rng.random() < 0.8:
                delta = rng.randint(-5, 5)
            else:
                delta = rng.randint(-100, 100)
                
            new_val = current_A[i] + delta
            
            if 1 <= new_val <= 10000:
                # Check distinctness
                if new_val not in current_A or new_val == current_A[i]:
                    # If new_val is already in A (and not at i), it's invalid
                    if new_val != current_A[i] and new_val in current_A:
                        pass # Skip
                    else:
                        new_A = current_A[:]
                        new_A[i] = new_val
                        new_A.sort()
                        
                        # Quick check: if we just changed one element, 
                        # can we update score incrementally? 
                        # For now, full recompute is fast enough (400 ops).
                        new_score = _score(new_A)
                        
                        # Acceptance criterion
                        delta_E = new_score - current_score
                        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                            current_A = new_A
                            current_score = new_score
                            
                            if current_score < best_score:
                                best_score = current_score
                                best_A = current_A[:]
                            
            T *= alpha
            if T < T_min:
                T = T_min

    return sorted(best_A)
