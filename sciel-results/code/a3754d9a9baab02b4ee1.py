import time

def construct():
    start_time = time.time()
    deadline = start_time + 59.0
    
    edges = []
    # Component 1: K_{3,3} on vertices 0..5
    # U = {0, 1, 2}, V = {3, 4, 5}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices 6..11
    # U = {6, 7, 8}, V = {9, 10, 11}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    return edges
