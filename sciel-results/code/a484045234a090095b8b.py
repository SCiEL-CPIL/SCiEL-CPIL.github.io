import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    best_set = None
    best_score = float('inf')
    
    def get_score(A):
        sums = set()
        prods = set()
        for i in A:
            for j in A:
                sums.add(i + j)
                prods.add(i * j)
        return max(len(sums), len(prods))
    
    def local_search(A, pool, max_iter=1000):
        A = list(A)
        current_score = get_score(A)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(A)):
                # Try swapping A[i] with x in pool
                # To save time, sample pool or use full pool if small
                for x in pool:
                    if x in A: continue
                    new_A = A.copy()
                    new_A[i] = x
                    new_score = get_score(new_A)
                    if new_score < current_score:
                        A = new_A
                        current_score = new_score
                        improved = True
                        # Don't break, try more swaps in this iteration?
                        # Or break to restart?
                        # Usually better to continue finding all improvements in a pass
        return A, current_score

    # Candidate 1: Baseline
    A1 = list(range(1, 21))
    pool1 = list(range(1, 1001)) # Restrict pool to speed up
    A1, s1 = local_search(A1, pool1)
    if s1 < best_score:
        best_score = s1
        best_set = A1
        
    # Candidate 2: Powers of 2 core
    A2 = [2**i for i in range(14)]
    # Greedy add
    pool2 = [x for x in range(1, 1001) if x not in A2]
    # Sort pool by some heuristic? Just random shuffle or small first?
    random.shuffle(pool2)
    # Greedy add 6 elements
    for _ in range(6):
        best_x = None
        best_s = float('inf')
        for x in pool2[:100]: # Only check first 100 candidates for speed
            if x in A2: continue
            new_A = A2 + [x]
            s = get_score(new_A)
            if s < best_s:
                best_s = s
                best_x = x
        if best_x is not None:
            A2.append(best_x)
            pool2.remove(best_x)
    A2, s2 = local_search(A2, pool2)
    if s2 < best_score:
        best_score = s2
        best_set = A2

    # Candidate 3: Powers of 3 core
    A3 = [3**i for i in range(9)]
    pool3 = [x for x in range(1, 1001) if x not in A3]
    random.shuffle(pool3)
    for _ in range(11):
        best_x = None
        best_s = float('inf')
        for x in pool3[:100]:
            if x in A3: continue
            new_A = A3 + [x]
            s = get_score(new_A)
            if s < best_s:
                best_s = s
                best_x = x
        if best_x is not None:
            A3.append(best_x)
            pool3.remove(best_x)
    A3, s3 = local_search(A3, pool3)
    if s3 < best_score:
        best_score = s3
        best_set = A3

    # Candidate 4: Random
    for _ in range(5):
        A4 = random.sample(range(1, 10001), 20)
        pool4 = [x for x in range(1, 1001) if x not in A4]
        A4, s4 = local_search(A4, pool4)
        if s4 < best_score:
            best_score = s4
            best_set = A4

    return best_set
