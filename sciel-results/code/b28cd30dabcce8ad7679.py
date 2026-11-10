import time

# EVOLVE-BLOCK-START
P = {'n': 16}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    n = P['n']
    half = n // 2
    
    # Construct K_{8,8}
    # Vertices 0..7 in set A, 8..15 in set B
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
            
    # Convert to required types if necessary, though lists of ints are fine.
    # The prompt says "Return edges as [u, v] lists".
    # "Types are checked exactly: integers must be Python int".
    # range() produces Python ints in Python 3.
    
    return edges
# EVOLVE-BLOCK-END
