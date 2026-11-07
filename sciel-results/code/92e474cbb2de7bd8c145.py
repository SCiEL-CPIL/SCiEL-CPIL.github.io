import time

def construct():
    """
    Constructs a graph on vertices 0..11 with minimum degree 3,
    triangle-free, and minimizing the number of distinct simple cycle lengths.
    
    The strategy is to use two disjoint K_{3,3} components.
    K_{3,3} has cycles of length 4 and 6 only.
    Thus, the distinct cycle lengths are {4, 6}.
    Raw value = 2.
    Score = 5.0 / 2.0 = 2.5.
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    edges = []
    
    # Component 1: K_{3,3} on vertices 0,1,2 (left) and 3,4,5 (right)
    for i in range(3):
        for j in range(3):
            u = i
            v = 3 + j
            edges.append([u, v])
    
    # Component 2: K_{3,3} on vertices 6,7,8 (left) and 9,10,11 (right)
    for i in range(3):
        for j in range(3):
            u = 6 + i
            v = 9 + j
            edges.append([u, v])
    
    # Verify types are correct (int, not bool, not numpy)
    verified_edges = []
    for e in edges:
        u, v = e
        if not isinstance(u, int) or isinstance(u, bool):
            u = int(u)
        if not isinstance(v, int) or isinstance(v, bool):
            v = int(v)
        verified_edges.append([u, v])
    
    return verified_edges
