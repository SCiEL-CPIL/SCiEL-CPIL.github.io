import time

def construct():
    """
    Constructs a triangle-free, connected graph with diameter at most 4
    by adding edges to the path P_24.
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}
    
    # Hub-based construction
    hub = 12
    leaves = [0, 2, 4, 6, 8, 16, 18, 20, 22]
    
    added_edges = []
    for v in leaves:
        added_edges.append([hub, v])
        
    # Convert to list of lists of ints
    result = [[int(u), int(v)] for u, v in added_edges]
    
    return result
