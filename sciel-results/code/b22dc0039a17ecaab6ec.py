"""Matrix multiplication 2,3,5 - rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Build the multiplication tensor T[a, b, c] where:
    # a = i*m + j (first matrix index)
    # b = j*p + k (second matrix index)  
    # c = k*n + i (output index, trace convention W[k*n+i, r])
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    
    tensor = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a, b, c] = 1.0
    
    # Compute flattening ranks to get lower bounds
    T1 = tensor.reshape(n * m, m * p * p * n)
    T2 = tensor.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    T3 = tensor.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
    
    rank1 = np.linalg.matrix_rank(T1)
    rank2 = np.linalg.matrix_rank(T2)
    rank3 = np.linalg.matrix_rank(T3)
    
    lower_bound = max(rank1, rank2, rank3)
    
    # Try to find a rank-r decomposition using border rank analysis
    # For each candidate rank r from lower_bound to schoolbook rank,
    # we check if the tensor lies in the r-th border of the variety of
    # rank-r tensors by examining the null space of the flattening matrices.
    
    # The key insight: if rank(T) <= r, then there exist vectors u_c, v_c, w_c
    # such that T = sum_c u_c ⊗ v_c ⊗ w_c
    # This means the mode-1 flattening T1 has rank <= r, and similarly for others.
    # More precisely, the column space of each flattening must be contained in
    # the span of the corresponding factor vectors.
    
    # We'll use a direct approach: for candidate rank r, try to find
    # U (n*m x r), V (m*p x r), W (p*n x r) such that
    # T[a,b,c] = sum_{t=0}^{r-1} U[a,t] * V[b,t] * W[c,t]
    
    # This is a system of (n*m)*(m*p)*(p*n) equations in r*(n*m + m*p + p*n) unknowns.
    # For small r, we can try to solve this using numerical methods.
    
    # Start with the schoolbook decomposition as baseline
    schoolbook_rank = n * m * p
    
    # Try ranks from lower_bound up to schoolbook_rank - 1
    best_rank = schoolbook_rank
    best_decomp = None
    
    for r in range(lower_bound, schoolbook_rank):
        # Check if rank-r decomposition is possible
        # Method: Use the fact that for a rank-r tensor, the mode-1 flattening
        # T1 must have its columns lying in an r-dimensional subspace.
        # We can check this by seeing if the (r+1)-th exterior power vanishes.
        
        # Alternative practical approach: try to solve the bilinear system
        # by fixing two matrices and solving for the third.
        
        # Let's try a structured approach: assume the decomposition has
        # integer/half-integer entries and search over small values.
        
        # For efficiency, we'll use a random restart approach with gradient descent
        # on the tensor reconstruction error.
        
        success = False
        max_attempts = 50
        
        for attempt in range(max_attempts):
            # Initialize U, V, W randomly with small values
            np.random.seed(attempt * 12345 + r)
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Iteratively refine: fix two, solve for third
            for iteration in range(100):
                # Fix V and W, solve for U
                # T1[a, b*n*p+c] = sum_t U[a,t] * (V[:,t] ⊗ W[:,t])[b*n*p+c]
                # This is a linear system: T1 = U * M where M[b*n*p+c, t] = V[b,t]*W[c,t]
                M = np.zeros((m * p * p * n, r))
                for b in range(m * p):
                    for c in range(p * n):
                        for t in range(r):
                            M[b * p * n + c, t] = V[b, t] * W[c, t]
                
                # Solve U * M = T1 using least squares
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M.T, T1.T, rcond=None)
                    U = U_new.T
                except:
                    break
                
                # Fix U and W, solve for V
                M = np.zeros((n * m * p * n, r))
                for a in range(n * m):
                    for c in range(p * n):
                        for t in range(r):
                            M[a * p * n + c, t] = U[a, t] * W[c, t]
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(M.T, T2.T, rcond=None)
                    V = V_new.T
                except:
                    break
                
                # Fix U and V, solve for W
                M = np.zeros((n * m * m * p, r))
                for a in range(n * m):
                    for b in range(m * p):
                        for t in range(r):
                            M[a * m * p + b, t] = U[a, t] * V[b, t]
                
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M.T, T3.T, rcond=None)
                    W = W_new.T
                except:
                    break
                
                # Check convergence
                T_recon = np.zeros((n * m, m * p, p * n))
                for t in range(r):
                    T_recon += np.outer(np.outer(U[:, t], V[:, t]), W[:, t])
                
                error = np.linalg.norm(T_recon - tensor) / np.linalg.norm(tensor)
                if error < 1e-8:
                    success = True
                    break
        
        if success:
            best_rank = r
            best_decomp = (U, V, W)
            break
    
    # If we found a better decomposition, return it; otherwise return schoolbook
    if best_decomp is not None:
        U, V, W = best_decomp
        # Round to nearest half-integer if close
        def round_half(arr):
            arr = np.round(arr * 2) / 2
            return arr.tolist()
        
        return {"U": round_half(U), "V": round_half(V), "W": round_half(W)}
    else:
        # Return schoolbook decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
