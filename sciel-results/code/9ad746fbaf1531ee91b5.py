"""Matrix multiplication 4x4x7 tensor decomposition via block schoolbook."""
import time

DIMENSIONS = [4, 4, 7]


def construct():
    """Construct a rank-minimal (within schoolbook) decomposition for 4x4x7.
    
    Strategy: Split B (4x7) into B1 (4x4) and B2 (4x3).
    C = A*B1 + A*B2.
    
    For each block, use standard schoolbook:
      (A*B1)_{i,j} = sum_k A_{i,k} * B1_{k,j}  for j=0..3
      (A*B2)_{i,j} = sum_k A_{i,k} * B2_{k,j}  for j=4..6
    
    Rank for 4x4x4 block: 4*4 = 16
    Rank for 4x4x3 block: 4*3 = 12
    Total rank: 16 + 12 = 28
    
    Output convention: W[k*n + i, r] where n=4.
    So row index in W is k*n + i = 4*k + i.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Block sizes
    p1 = 4  # first 4 columns of B
    p2 = 3  # remaining 3 columns of B
    
    # Total rank
    rank1 = n * p1  # 16
    rank2 = n * p2  # 12
    total_rank = rank1 + rank2  # 28
    
    # Initialize factor matrices
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 28
    V = [[0] * total_rank for _ in range(m * p)]  # 28 x 28
    W = [[0] * total_rank for _ in range(p * n)]  # 28 x 28
    
    # Block 1: B1 is columns 0..3 of B
    # For each i in 0..3 (row of A), j in 0..3 (col of B1), k in 0..3 (inner)
    # U[i*m + k, col] = 1  (A_{i,k})
    # V[k*p + j, col] = 1  (B_{k,j})
    # W[j*n + i, col] = 1  (C_{i,j})
    
    col = 0
    for i in range(n):
        for j in range(p1):
            for k in range(m):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    
    # Block 2: B2 is columns 4..6 of B
    # For each i in 0..3, j in 4..6, k in 0..3
    for i in range(n):
        for j in range(p1, p):
            for k in range(m):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    
    elapsed = time.time() - start_time
    # Safety margin: return before 120s
    if elapsed > 110:
        pass  # Already done
    
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    U, V, W = result["U"], result["V"], result["W"]
    n, m, p = DIMENSIONS
    
    # Verify: C[i,j] should equal sum_r U[i*m+k... wait, let me check the convention
    
    # The tensor decomposition says:
    # C_{i,j} = sum_{k, r} U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # Actually, the standard convention is:
    # C_{i,j} = sum_{k,r} U[i*m+k, r] * V[k*p+j, r] * W[r, j*n+i]
    # But the task says W[k*n+i, r] which is unusual.
    
    # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    # This means W has rows indexed by (k, i) where k is the column index of B / row index of C's second dim
    # and i is the row index of A / first dim of C.
    
    # So W[j*n + i, r] where j is the column of C (and B), i is the row of C (and A).
    # Wait, that's what I have: W[j*n + i, r].
    
    # Let me verify with a simple case.
    # For schoolbook: C_{i,j} = sum_k A_{i,k} B_{k,j}
    # We need: C_{i,j} = sum_r U[i*m+k, r] V[k*p+j, r] W[?, r]
    
    # Actually the tensor T_{i,k,j} = 1 if we're doing A*B, where the multiplication is
    # C_{i,j} = sum_k A_{i,k} B_{k,j}
    
    # The decomposition is T = sum_r U_r * V_r * W_r (outer products)
    # U_r is a vector of length n*m (indexed by i,k)
    # V_r is a vector of length m*p (indexed by k,j)  
    # W_r is a vector of length p*n (indexed by j,i)
    
    # So T_{i,k,j} = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    
    # For schoolbook with one term per (i,k,j):
    # U[i*m+k, r] = 1, V[k*p+j, r] = 1, W[j*n+i, r] = 1
    # Then T_{i,k,j} = 1 * 1 * 1 = 1. Correct!
    
    # Let's do a numerical check
    import numpy as np
    
    rank = total_rank
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                for r in range(rank):
                    T[i, k, j] += U_np[i*m+k, r] * V_np[k*p+j, r] * W_np[j*n+i, r]
    
    # The true tensor for matrix mult is T_{i,k,j} = 1 for all i,k,j (since C_{i,j} = sum_k A_{i,k}B_{k,j})
    expected = np.ones((n, m, p))
    
    if np.allclose(T, expected):
        print(f"Verification PASSED. Rank = {rank}")
    else:
        print(f"Verification FAILED. Max diff = {np.max(np.abs(T - expected))}")
        print("T sample:", T[0, :, :])
