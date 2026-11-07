import time

def construct():
    deadline = time.time() + 85.0  # 85s safety margin within 90s limit
    n = 15
    
    # Construct complete bipartite graph K_{7,8}
    # Partition A = {0, ..., 6} (size 7)
    # Partition B = {7, ..., 14} (size 8)
    # Degrees in A are all 8, degrees in B are all 7.
    # Any path of length 3 (s-u-v-t) has endpoints in different partitions
    # (since paths alternate between partitions), so endpoints have degrees 8 and 7,
    # which are distinct.
    
    edges = []
    for i in range(7):
        if time.time() > deadline:
            break
        for j in range(7, 15):
            if time.time() > deadline:
                break
            edges.append([int(i), int(j)])
        if time.time() > deadline:
            break
    
    # Ensure all edges are proper Python lists of Python ints
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
