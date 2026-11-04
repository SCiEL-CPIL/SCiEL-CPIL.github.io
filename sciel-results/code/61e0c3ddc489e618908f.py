import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill
    
    # Start with the known rank-55 decomposition
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen coefficients
    L = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    R = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    Wc = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {1: 1, 0: -1},
        {3: 1},
        {0: 1}
    ]
    
    for block in range(5):
        k_off = block * 2
        for s in range(7):
            for li, val in L[s].items():
                i, j = li // 2, li % 2
                U[i * m + j][col] = val
            for li, val in R[s].items():
                j, k = li // 2, li % 2
                V[j * p + k_off + k][col] = val
            for li, val in Wc[s].items():
                i, k = li // 2, li % 2
                W[(k_off + k) * n + i][col] = val
            col += 1
    
    for i in range(2):
        for k in range(10):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Try to reduce rank by attempting to eliminate one term
    # Check if any term is a linear combination of others
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # For each column r, check if the rank-1 tensor U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # is in the span of the others. This is expensive, so try a heuristic:
    # Check if removing column r still gives correct tensor.
    
    def check_valid(U_m, V_m, W_m, r):
        """Check if decomposition without column r is still valid."""
        # Build the full tensor and compare
        # Tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # We need T[i,j,k] = delta(i,j,k) for matrix mult: T[i,j,k] = 1 iff... 
        # Actually T[i,j,k] should equal the matrix multiplication tensor.
        # For 2x3x10: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # The tensor is: T[i,j,k] = 1 for all i,j,k (it's the identity tensor for matmul)
        # Wait no. The matrix multiplication tensor M_{n,m,p} has entries:
        # M[i,j,k] = 1 if... it's defined as the bilinear map (A,B) -> A*B
        # M[i,j,k] = 1 for all valid i,j,k. Yes, it's all ones.
        # Actually no. Let me reconsider.
        # The tensor T has T[i,j,k] such that C[i,k] = sum_j A[i,j] * B[j,k]
        # In tensor form: C[i,k] = sum_{j,r} U[i,r]*V[j,r]*W[k,r] * A[i,j]*B[j,k]... 
        # No. The decomposition says: for each r, we compute product P_r = (sum_i U[i,r] A[i,j]) * (sum_k V[j,r] B[j,k])
        # and then C[i,k] = sum_r W[i,k,r] * P_r... 
        # Let me re-read the convention.
        # W[k*n+i, r] is the coefficient. So C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k])
        # For this to equal sum_j A[i,j] B[j,k], we need:
        # sum_r W[k*n+i,r] * U[i*m+j,r] * V[j*p+k,r] = 1 for all i,j,k (the matmul tensor is all 1s)
        # Wait, that's not right either. Let me think again.
        # C[i,k] = (A*B)[i,k] = sum_j A[i,j] B[j,k]
        # The decomposition: C[i,k] = sum_r W[k*n+i,r] * prod_r * ... 
        # Actually the standard tensor decomposition of matrix multiplication:
        # M_{n,m,p}[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # where M[i,j,k] = 1 for all i,j,k (the matrix multiplication tensor is the all-ones tensor).
        # So we need: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        
        # Check without column r
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r2 in range(rank):
                        if r2 == r:
                            continue
                        s += U_np[i*m+j, r2] * V_np[j*p+k, r2] * W_np[k*n+i, r2]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try removing each column
    best_rank = rank
    for r in range(rank):
        if time.time() > deadline:
            break
        if check_valid(U_np, V_np, W_np, r):
            best_rank = rank - 1
            # Remove column r
            U_np = np.delete(U_np, r, axis=1)
            V_np = np.delete(V_np, r, axis=1)
            W_np = np.delete(W_np, r, axis=1)
            rank = best_rank
            break
    
    # Try one more removal
    if time.time() < deadline - 10:
        for r in range(rank):
            if time.time() > deadline:
                break
            if check_valid(U_np, V_np, W_np, r):
                U_np = np.delete(U_np, r, axis=1)
                V_np = np.delete(V_np, r, axis=1)
                W_np = np.delete(W_np, r, axis=1)
                rank -= 1
                break
    
    # Convert back to lists
    U_out = U_np.tolist()
    V_out = V_np.tolist()
    W_out = W_np.tolist()
    
    # Ensure entries are integers or half-integers
    for mat in [U_out, V_out, W_out]:
        for row in mat:
            for idx, val in enumerate(row):
                # Round to nearest 0.5
                row[idx] = round(val * 2) / 2
    
    return {"U": U_out, "V": V_out, "W": W_out}
