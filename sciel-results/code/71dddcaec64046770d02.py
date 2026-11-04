"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_cells = n * m * p  # 48
    nm, mp, pn = n*m, m*p, p*n
    
    def make_schoolbook():
        U = [[0]*total_cells for _ in range(nm)]
        V = [[0]*total_cells for _ in range(mp)]
        W = [[0]*total_cells for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W
    
    # Try to find a decomposition with rank < 48
    # Use a randomized approach: start with random vectors and try to match the tensor
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if we multiply A[i,j] * B[j,k]
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # But that's not right. The standard decomposition is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = 1 for all (i,j,k)
    # And we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # Wait, that's not the standard way. Let me reconsider.
    
    # The matrix multiplication tensor is T[i,j,k] = 1 for all i,j,k.
    # A rank-1 decomposition would be: U[i,j] * V[j,k] * W[k,i]
    # But we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # Actually, the standard decomposition is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is: T[i,j,k] = 1 (since the product is A[i,j]*B[j,k])
    # And the decomposition is:
    # C[i,k] = sum_c (sum_j U[i,j,c] * A[i,j]) * (sum_k V[j,k,c] * B[j,k]) * W[i,k,c]
    # Hmm, that's not right either.
    
    # Let me think more carefully. The standard tensor decomposition for matrix multiplication:
    # We have matrices A (n x m), B (m x p), C (n x p).
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1.
    # A rank-r decomposition is:
    # T[i,j,k] = sum_{c=1}^r U[i,j,c] * V[j,k,c] * W[i,k,c]
    # But wait, that would give T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # For this to equal 1 for all (i,j,k), we need something like:
    # U[i,j,c] = delta_{i,j,c} * something...
    
    # Actually, the standard way is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We can write this as:
    # C[i,k] = sum_c (sum_j U[i,j,c] * A[i,j]) * (sum_k V[j,k,c] * B[j,k]) * W[i,k,c]
    # No, that's not right.
    
    # Let me use the correct formulation:
    # The matrix multiplication tensor is:
    # T[i,j,k] = 1
    # And we want:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # But that would mean:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # The schoolbook decomposition has rank n*m*p = 48.
    # U[i*m+j, c] = 1 if c corresponds to (i,j,k), else 0
    # V[j*p+k, c] = 1 if c corresponds to (i,j,k), else 0
    # W[k*n+i, c] = 1 if c corresponds to (i,j,k), else 0
    # Then sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # So the tensor we're decomposing is:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And we need T[i,j,k] = 1 for all (i,j,k).
    
    # To find a lower rank decomposition, we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # This is a 3-tensor with dimensions (n*m) x (m*p) x (p*n) = 12 x 16 x 12.
    # We need to find a rank-r decomposition.
    
    # Let's try a randomized approach:
    # 1. Start with random U, V, W of rank r.
    # 2. Compute T_approx = sum_c U[:, c] * V[:, c] * W[:, c]
    # 3. Try to minimize ||T_approx - T_target||
    
    # But this is a non-convex problem. Let's try a simpler approach:
    # Try to find a decomposition with rank 47, 46, etc.
    
    # Given the time constraint, let's just try a few random initializations and see if we can find a lower rank.
    
    deadline = time.time() + 120  # 120 seconds
    
    def verify(U, V, W, rank):
        # Check if the decomposition is valid
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Should equal 1 for all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try rank 47
    for target_rank in range(47, 30, -1):
        if time.time() > deadline - 10:
            break
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            # Random initialization
            U = [[random.gauss(0, 1) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.gauss(0, 1) for _ in range(target_rank)] for _ in range(mp)]
            W = [[random.gauss(0, 1) for _ in range(target_rank)] for _ in range(pn)]
            
            # Try ALS (Alternating Least Squares)
            for iteration in range(100):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                for idx in range(nm):
                    i, j = idx // m, idx % m
                    # We need: sum_c U[idx, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                    # This is a linear system: sum_c U[idx, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for all k
                    # Let's build the system
                    A_mat = [[0]*target_rank for _ in range(p)]
                    b_vec = [1.0] * p
                    for k in range(p):
                        for c in range(target_rank):
                            A_mat[k][c] = V[j*p + k][c] * W[k*n + i][c]
                    # Solve A_mat * U[idx] = b_vec
                    # Use simple Gaussian elimination or just use numpy if available
                    try:
                        import numpy as np
                        A_np = np.array(A_mat)
                        b_np = np.array(b_vec)
                        # Least squares
                        U_np, residuals, rank, sv = np.linalg.lstsq(A_np, b_np, rcond=None)
                        U[idx] = U_np.tolist()
                    except:
                        pass
                
                # Update V
                for idx in range(mp):
                    j, k = idx // p, idx % p
                    A_mat = [[0]*target_rank for _ in range(n)]
                    b_vec = [1.0] * n
                    for i in range(n):
                        for c in range(target_rank):
                            A_mat[i][c] = U[i*m + j][c] * W[k*n + i][c]
                    try:
                        import numpy as np
                        A_np = np.array(A_mat)
                        b_np = np.array(b_vec)
                        V_np, residuals, rank, sv = np.linalg.lstsq(A_np, b_np, rcond=None)
                        V[idx] = V_np.tolist()
                    except:
                        pass
                
                # Update W
                for idx in range(pn):
                    k, i = idx // n, idx % n
                    A_mat = [[0]*target_rank for _ in range(m)]
                    b_vec = [1.0] * m
                    for j in range(m):
                        for c in range(target_rank):
                            A_mat[j][c] = U[i*m + j][c] * V[j*p + k][c]
                    try:
                        import numpy as np
                        A_np = np.array(A_mat)
                        b_np = np.array(b_vec)
                        W_np, residuals, rank, sv = np.linalg.lstsq(A_np, b_np, rcond=None)
                        W[idx] = W_np.tolist()
                    except:
                        pass
            
            if verify(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = make_schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
