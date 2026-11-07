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
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    edges = []
    
    # Component 1: K_{3,3} on vertices 0,1,2 (left) and 3,4,5 (right)
    for i in range(3):
        for j in range(3):
            u = int(i)
            v = int(3 + j)
            edges.append([u, v])
    
    # Component 2: K_{3,3} on vertices 6,7,8 (left) and 9,10,11 (right)
    for i in range(3):
        for j in range(3):
            u = int(6 + i)
            v = int(9 + j)
            edges.append([u, v])
    
    # Verify deadline (should be well within, but guard just in case)
    if time.time() > deadline:
        # Return best result found (the complete K3,3 x 2 construction)
        pass
    
    # Ensure edges are list of lists of Python ints
    result = [[int(u), int(v)] for u, v in edges]
    
    return result
