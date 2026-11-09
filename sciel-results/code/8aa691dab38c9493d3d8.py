import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n = 39
    # S = {14, 15, 16, 17, 18, 19}
    # Edges if (j - i) % n in S or (i - j) % n in S
    # Which is equivalent to min((j-i)%n, (i-j)%n) in S
    # Or simply (j-i)%n in S since we iterate i<j and add both?
    # The return format is [[u, v], ...] for u < v.
    # Condition: (j - i) % n in S.
    
    S = {14, 15, 16, 17, 18, 19}
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in S:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
