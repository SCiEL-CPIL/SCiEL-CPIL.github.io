import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = 110.0  # 10 second margin
    
    # Handle non-standard cases with a trivial valid solution
    if q != 3 or n != 8 or radius != 1:
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    N = q ** n
    # Precompute neighbors
    neighbors = []
    pow3 = [1] * n
    for i in range(1, n):
        pow3[i] = pow3[i-1] * 3
        
    for i in range(N):
        digits = [0] * n
        temp = i
        for k in range(n):
            digits[k] = temp % 3
            temp //= 3
        
        nbs = [i]
        for pos in range(n):
            orig = digits[pos]
            for val in range(3):
                if val == orig:
                    continue
                new_i = i + (val - orig) * pow3[pos]
                nbs.append(new_i)
        neighbors.append(nbs)
        
    covered = [False] * N
    uncovered_count = [len(neighbors[i]) for i in range(N)]
    
    code = []
    
    while True:
        if time.time() > deadline:
            # Emergency: add all uncovered words
            for i in range(N):
                if not covered[i]:
                    code.append(i)
                    # Marking covered is not strictly necessary if we return immediately
                    # but good practice if we were to continue
            break
            
        max_count = -1
        max_idx = -1
        for i in range(N):
            if uncovered_count[i] > max_count:
                max_count = uncovered_count[i]
                max_idx = i
                
        if max_count == 0:
            break
            
        code.append(max_idx)
        
        for u in neighbors[max_idx]:
            if not covered[u]:
                covered[u] = True
                for v in neighbors[u]:
                    uncovered_count[v] -= 1
                    
    # Convert to list of lists
    result = []
    for idx in code:
        digits = [0] * n
        temp = idx
        for k in range(n):
            digits[k] = temp % 3
            temp //= 3
        result.append(digits)
        
    return result
