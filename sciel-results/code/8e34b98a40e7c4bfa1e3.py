import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 16
    part_a = list(range(8))
    part_b = list(range(8, 16))
    edges = []
    
    # Construction is O(64), extremely fast.
    # We include the loop structure to allow for future search if needed,
    # but for now it just builds K_{8,8}.
    for u in part_a:
        if time.time() > deadline:
            break
        for v in part_b:
            edges.append([u, v])
            
    return edges
