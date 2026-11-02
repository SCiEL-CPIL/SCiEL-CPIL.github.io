import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = 2, 3, 6
    
    # Schoolbook fallback: rank = n*m*p = 36
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True
    
    # Strategy: Model as linear algebra problem
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Key insight: Fix W, then the problem becomes finding U,V such that
    # for each (i,j,k): sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
    # 
    # Alternative: Think of it as a matrix factorization problem.
    # Let's construct the "flattened" tensor and find its rank.
    
    # Construct the 36x36 matrix M where:
    # M[(i,j), (k,l)] represents... actually let's think differently.
    
    # Better approach: The multiplication tensor has a natural structure.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a bilinear map: (A, B) -> C
    # The tensor T in the space U(A) ⊗ U(B) ⊗ U(C) has entries T[i,j,k] = 1
    
    # For rank-r decomposition, we need:
    # T = sum_{r=1}^{r} u_r ⊗ v_r ⊗ w_r
    # where u_r is in U(A) (dimension 6), v_r in U(B) (dimension 18), w_r in U(C) (dimension 12)
    
    # The minimal rank is the tensor rank, which is hard to compute exactly.
    # But we can try to find a low-rank decomposition via:
    # 1. Random initialization + alternating least squares (current approach)
    # 2. Structured approaches based on matrix multiplication algorithms
    
    # Let's try a smarter approach: exploit the structure of 2x3x6 multiplication.
    # 2x3x6 can be seen as 3 blocks of 2x3x2.
    # If we can find a rank-r decomposition for 2x3x2, we can extend it.
    
    # First, let's try to find the rank of the "slice" matrices.
    # For fixed i, the tensor slice T[i, :, :] is a 3x6 matrix of ones.
    # For fixed j, T[:, j, :] is a 2x6 matrix of ones.
    # For fixed k, T[:, :, k] is a 2x3 matrix of ones.
    
    # The tensor rank of a 2x3x6 all-ones tensor...
    # Actually, the multiplication tensor is NOT all-ones in the standard basis.
    # Let me reconsider.
    
    # The standard matrix multiplication tensor M_{n,m,p} has:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # In the standard basis, T[i,j,k] = 1 for all i,j,k (since each product A[i,j]B[j,k] contributes to C[i,k])
    
    # So yes, T[i,j,k] = 1 for all i=0..1, j=0..2, k=0..5.
    
    # The rank of this tensor is what we're looking for.
    # For M_{2,3,6}, the known bounds are:
    # Lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # Upper bound: n*m*p = 36 (schoolbook)
    
    # Let's try to find a rank-18 decomposition first (the lower bound).
    # If that fails, try 19, 20, ... up to 35.
    
    # Approach: Use the structure of the problem.
    # C = A * B where A is 2x3, B is 3x6, C is 2x6.
    # We can write B as [B1, B2, B3] where each B_i is 3x2.
    # Then C = [A*B1, A*B2, A*B3].
    # Each A*B_i is a 2x2 matrix multiplication.
    
    # The rank of M_{2,3,2} is at most 7 (by known results for 2x2x2 with an extra dimension).
    # Actually, M_{2,3,2} has rank at most 3*2*2 = 12, but can be done better.
    
    # Let's try a different approach: use the fact that 6 = 3*2.
    # If we can find a rank-r decomposition for M_{2,3,2}, we can get rank 3r for M_{2,3,6}.
    
    # Known: rank(M_{2,2,2}) = 7 (Strassen)
    # rank(M_{2,3,2}) <= ?
    
    # Let's just try random search with alternating least squares for various ranks.
    
    def solve_for_W(U, V, r):
        W = np.zeros((p * n, r))
        for i in range(n):
            for k in range(p):
                A_mat = np.zeros((m, r))
                for j in range(m):
                    A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                try:
                    w, _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                except:
                    w = np.zeros(r)
                W[k * n + i, :] = w
        return W
    
    def solve_for_U(V, W, r):
        U = np.zeros((n * m, r))
        for i in range(n):
            for j in range(m):
                A_mat = np.zeros((p, r))
                for k in range(p):
                    A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p)
                try:
                    u, _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                except:
                    u = np.zeros(r)
                U[i * m + j, :] = u
        return U
    
    def solve_for_V(U, W, r):
        V = np.zeros((m * p, r))
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((n, r))
                for i in range(n):
                    A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n)
                try:
                    v, _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                except:
                    v = np.zeros(r)
                V[j * p + k, :] = v
        return V
    
    def compute_error(U, V, W):
        r = U.shape[1]
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        return err
    
    # Try to find a low-rank decomposition using alternating least squares.
    # Start from the lower bound and work up.
    
    # Lower bound is 18, but let's start a bit higher to be safe.
    # Try ranks from 18 to 35.
    
    for rank in range(18, 36):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try multiple random initializations
        n_inits = 50 if rank < 25 else 20
        for init in range(n_inits):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Normalize
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            prev_err = float('inf')
            for it in range(200):
                W = solve_for_W(U, V, rank)
                U = solve_for_U(V, W, rank)
                V = solve_for_V(U, W, rank)
                
                err = compute_error(U, V, W)
                
                if err < 1e-10:
                    if verify(U, V, W):
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = rank
                        found = True
                        break
                
                if abs(prev_err - err) < 1e-12 and it > 10:
                    break
                prev_err = err
            
            if found:
                break
        
        if found:
            break
    
    return best_result
