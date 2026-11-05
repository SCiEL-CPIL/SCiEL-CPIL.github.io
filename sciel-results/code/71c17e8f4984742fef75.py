import numpy as np

def construct():
    n, m, p = 3, 4, 5
    
    # We will build a decomposition by splitting the dimensions.
    # Split n=3 into n1=2, n2=1.
    # Split m=4 into m1=2, m2=2.
    # Split p=5 into p1=2, p2=3.
    
    # We'll accumulate U, V, W blocks.
    # Each block corresponds to a sub-problem M_{ni, mj, pk}.
    # We use trivial decomposition for each sub-problem.
    
    # List of (ni, mj, pk, i_offset, j_offset, k_offset)
    splits = [
        (2, 2, 2, 0, 0, 0),
        (2, 2, 3, 0, 0, 2),
        (2, 2, 2, 0, 2, 0),
        (2, 2, 3, 0, 2, 2),
        (1, 2, 2, 2, 0, 0),
        (1, 2, 3, 2, 0, 2),
        (1, 2, 2, 2, 2, 0),
        (1, 2, 3, 2, 2, 2),
    ]
    
    # Calculate total rank
    total_rank = sum(ni*mj*pk for ni, mj, pk, _, _, _ in splits)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for ni, mj, pk, io, jo, ko in splits:
        for i in range(ni):
            for j in range(mj):
                for k in range(pk):
                    # Global indices
                    gi = io + i
                    gj = jo + j
                    gk = ko + k
                    
                    U[gi * m + gj, col] = 1
                    V[gj * p + gk, col] = 1
                    W[gk * n + gi, col] = 1
                    col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
