def construct():
    n, m, p = 2, 3, 9
    total_rank = n * m * p  # 54
    
    # Initialize U, V, W with zeros
    # U: (n*m) x total_rank
    # V: (m*p) x total_rank
    # W: (p*n) x total_rank
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # The decomposition is a direct sum of n*m*p rank-1 tensors
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} E_{i,j} (A) otimes E_{j,k} (B) otimes E_{i,k} (C)
    # Note: C index is (i, k) where i is row, k is col.
    # In the problem statement: W[k*n + i, r]
    # So C row index is k? No.
    # "W[k*n+i,r]"
    # Usually C is n x p.
    # If C is stored as W, and W has shape (p*n, rank), then the index is flattened.
    # Let's check the convention in the prompt.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This implies the index of W is `idx = k * n + i`.
    # Here `i` is likely the row index of C (0 to n-1) and `k` is the column index of C (0 to p-1)?
    # Or `i` is column and `k` is row?
    # Standard C is n x p.
    # If we flatten row-major: index = row * p + col.
    # If we flatten col-major: index = col * n + row.
    # The prompt says `W[k*n + i, r]`.
    # This looks like col-major if `k` is col and `i` is row.
    # Let's assume `i` is the row index of A/C (0..n-1) and `k` is the column index of B/C (0..p-1).
    # In the multiplication $C = A B$:
    # $C_{i, k} = \sum_{j} A_{i, j} B_{j, k}$.
    # So `i` is row of A and C. `k` is col of B and C. `j` is col of A and row of B.
    # U index: `i * m + j`. (Row-major for A)
    # V index: `j * p + k`. (Row-major for B)
    # W index: `k * n + i`. (Col-major for C? Or just the specific convention given)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Determine column index in the factor matrices
                # We can assign a unique column for each (i, j, k) tuple
                # total_rank = n * m * p
                col = (i * m + j) * p + k
                
                # U[i*m + j, col] = 1
                U[i * m + j][col] = 1
                
                # V[j*p + k, col] = 1
                V[j * p + k][col] = 1
                
                # W[k*n + i, col] = 1
                W[k * n + i][col] = 1
                
    return {"U": U, "V": V, "W": W}
