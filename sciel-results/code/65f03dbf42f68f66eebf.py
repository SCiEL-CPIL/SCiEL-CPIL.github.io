import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0
    
    # Try to find a valid tournament for n=67
    result = try_find_tournament(67, deadline - start_time)
    if result is not None:
        return result
    
    # Fallback to known valid Paley tournament for n=83
    return pale_tournament(83)

def pale_tournament(p):
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    
    return {'n': p, 'orientation': orientation}

def try_find_tournament(n, time_limit):
    # Full Re-evaluation Hill Climbing
    # This is simpler and less error-prone.
    # n=67, ~600k sets.
    # Python can do ~1M simple ops per second?
    # 600k checks might take 0.5s - 1s.
    # So we can do ~100-200 restarts.
    
    best_bad_count = float('inf')
    best_A = None
    
    while time.time() < time.time() + time_limit: # Note: time_limit is remaining
        # Initialize random A
        A = np.random.randint(0, 2, (n, n))
        for i in range(n):
            A[i, i] = 0
            for j in range(i + 1, n):
                if A[i, j] == A[j, i]:
                    if A[i, j] == 1:
                        A[i, j] = 0
                        A[j, i] = 1
                    else:
                        A[i, j] = 1
                        A[j, i] = 0
        
        # Hill Climbing
        current_bad_count = count_bad_sets(A, n)
        
        if current_bad_count == 0:
            return format_result(n, A)
        
        stuck = 0
        while time.time() < time.time() + time_limit:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            # Flip
            A[i, j] = 1 - A[i, j]
            A[j, i] = 1 - A[i, j]
            
            new_bad_count = count_bad_sets(A, n)
            
            if new_bad_count <= current_bad_count:
                current_bad_count = new_bad_count
                stuck = 0
                if current_bad_count == 0:
                    return format_result(n, A)
            else:
                # Revert
                A[i, j] = 1 - A[i, j]
                A[j, i] = 1 - A[j, i]
                stuck += 1
                if stuck > 1000:
                    break
        
        # Track best
        if current_bad_count < best_bad_count:
            best_bad_count = current_bad_count
            best_A = A.copy()

    return None

def count_bad_sets(A, n):
    bad_count = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    od_a = A[a, b] + A[a, c] + A[a, d]
                    od_b = A[b, a] + A[b, c] + A[b, d]
                    od_c = A[c, a] + A[c, b] + A[c, d]
                    od_d = A[d, a] + A[d, b] + A[d, c]
                    if max(od_a, od_b, od_c, od_d) < 3:
                        bad_count += 1
    return bad_count

def format_result(n, A):
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            orientation.append(int(A[i, j]))
    return {'n': n, 'orientation': orientation}
