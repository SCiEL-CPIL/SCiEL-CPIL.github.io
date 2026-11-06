"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Neighbour joining on Jukes-Cantor corrected distances (Saitou and Nei 1987).'''
import math
import time


def _jc_distance(a, b):
    """Compute Jukes-Cantor corrected distance between two DNA sequences."""
    n = len(a)
    diff = 0
    for i in range(n):
        if a[i] != b[i]:
            diff += 1
    p = diff / n
    if p >= 0.75:
        return 3.0  # cap to avoid log(0) or negative
    if p == 0:
        return 0.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 170.0  # 170s safety margin within 180s budget
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 1:
            return f"({labels[0]})"
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Build distance matrix using NumPy for speed
    try:
        import numpy as np
        seqs = [sequences[label] for label in labels]
        L = len(seqs[0])
        # Convert to numpy arrays for fast comparison
        # We need space for original n nodes + (n-3) new nodes = 2n-3 nodes.
        # Allocate 2n x 2n to be safe.
        D = np.zeros((2 * n, 2 * n))
        
        # Vectorized distance calculation is complex for DNA strings directly in numpy
        # without converting to numbers. We will use a loop but optimize slightly.
        # For n=32, n^2/2 = 512 pairs. L up to 2000. 1M comparisons. Python is slow.
        # Let's try to map A,C,G,T to 0,1,2,3 and use numpy arrays.
        # But sequences are strings.
        
        # Faster approach: convert each sequence to a numpy array of integers?
        # Or just use the loop, it might be fast enough for n=32.
        # Let's stick to the loop but ensure D is big enough.
        
        for i in range(n):
            for j in range(i + 1, n):
                # Using zip and sum might be faster than np.sum on strings in some cases
                # but np.sum on boolean array from string comparison is okay.
                diff = np.sum(seqs[i] != seqs[j])
                p = diff / L
                if p >= 0.75:
                    d = 3.0
                elif p == 0:
                    d = 0.0
                else:
                    d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
                D[i, j] = d
                D[j, i] = d
    except ImportError:
        # Fallback to pure Python
        D = [[0.0] * (2 * n) for _ in range(2 * n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
                D[i][j] = d
                D[j][i] = d
        D = np.array(D)
    
    # Neighbour Joining
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    counter = n
    
    while len(active) > 3:
        if time.time() > deadline:
            break
        
        m = len(active)
        # Compute net divergence r
        r = np.zeros(m)
        for i in range(m):
            for j in range(m):
                if i != j:
                    r[i] += D[active[i], active[j]]
        
        best_criterion = float('inf')
        best_i, best_j = 0, 1
        
        for i in range(m):
            for j in range(i + 1, m):
                criterion = (m - 2) * D[active[i], active[j]] - r[i] - r[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
        
        a, b = active[best_i], active[best_j]
        joined = f"({trees[a]},{trees[b]})"
        new_node = counter
        counter += 1
        trees[new_node] = joined
        
        rest = [x for k, x in enumerate(active) if k != best_i and k != best_j]
        for x in rest:
            D[new_node, x] = D[x, new_node] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
        
        active = rest + [new_node]
    
    # Build final tree
    if len(active) == 3:
        result = f"({','.join(trees[x] for x in active)})"
    else:
        result = f"({','.join(trees[x] for x in active)})"
    
    return result + ";"
# EVOLVE-BLOCK-END
