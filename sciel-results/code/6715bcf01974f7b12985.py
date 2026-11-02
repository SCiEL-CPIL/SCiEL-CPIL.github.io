"""Rewritten program: linear-algebraic rank minimization for 2x3 @ 3x9 tensor."""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor T has shape (n*m, m*p, p*n)
    # We want to find a rank-r decomposition:
    #   T[i, j, k] = sum_{c=0}^{r-1} U[i,c] * V[j,c] * W[k,c]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    #
    # The "trace convention" says W[k*n+i, r], so W is indexed by (k,i) with i in [0,n), k in [0,p)
    # and the output C[i,k] = sum_{j} A[i,j] B[j,k].
    #
    # We can view the bilinear map as:
    #   f(A, B) = A @ B, a map from F^{n*m} x F^{m*p} -> F^{p*n}
    #
    # We seek a minimum-rank decomposition. One approach:
    # 1. Build the full multiplication tensor T of shape (n*m, m*p, p*n).
    # 2. Try to find a rank-r decomposition by solving for U, V, W.
    #
    # For a given rank r, the problem is: find U, V, W such that
    #   T[a, b, c] = sum_{t} U[a,t] V[b,t] W[c,t]
    #
    # This is a 3-way tensor decomposition problem. We can use a simple iterative
    # approach: fix two matrices and solve for the third.
    #
    # However, a cleaner approach for this specific case:
    # Since n=2, m=3, p=9, the schoolbook rank is 54.
    # We can potentially reduce this by exploiting the structure.
    #
    # Let's try a direct approach: use the fact that the tensor can be seen as
    # a linear map and find its "rank" in the appropriate sense.
    #
    # Actually, let's just implement a rank-reduction search:
    # For each candidate rank r from some lower bound up to 54, try to find
    # a valid decomposition. If we find one, return it.
    #
    # A lower bound on the rank of matrix multiplication A(n,m) x B(m,p) is
    # at least max(n*p, ... ) but let's just search.
    #
    # For small dimensions, we can try to find a decomposition by:
    # 1. Randomly initialize U, V, W with r columns.
    # 2. Iterate: fix U and V, solve for W (least squares); fix U and W, solve for V; etc.
    # 3. Check if the residual is close to zero.
    #
    # But for exact decomposition with integer/half-integer entries, we need exact arithmetic.
    #
    # Let me think about this more carefully. The tensor T is defined by:
    #   T[i*m+j, j'*p+k, k'*n+i'] = 1 if j=j' and i=i' and k=k', else 0
    # Wait, no. Let's be precise.
    #
    # A is n x m, indexed by (i, j) with i in [0,n), j in [0,m).
    # B is m x p, indexed by (j', k) with j' in [0,m), k in [0,p).
    # C is n x p, indexed by (i, k) with i in [0,n), k in [0,p).
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # So the tensor T has indices (i,j) for A, (j',k) for B, and (i,k) for C.
    # T[(i,j), (j',k), (i'',k'')] = 1 if i=i'' and k=k'' and j=j', else 0.
    #
    # In flattened form:
    # A index: i*m + j
    # B index: j'*p + k
    # C index: k'*n + i'' (using the trace convention W[k*n+i])
    #
    # So T[a, b, c] = 1 if:
    #   a = i*m + j, b = j'*p + k, c = k'*n + i''
    #   and i = i'', k = k'', j = j'
    #
    # Let's build T and then try to find a low-rank decomposition.
    
    # Build the tensor T
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try ranks from a reasonable lower bound up to 54
    # Lower bound: the rank must be at least the maximum of the ranks of
    # any matricization. For a (I, J, K) tensor, the rank is at least
    # max(rank(T_{I;JK}), rank(T_{J;IK}), rank(T_{K;IJ}))
    # Let's compute these.
    
    # Matricize T along different axes
    T_I = T.reshape(n * m, m * p * p * n)  # I x (J*K)
    T_J = T.reshape(m * p, n * m * p * n)  # J x (I*K)
    T_K = T.reshape(p * n, n * m * m * p)  # K x (I*J)
    
    lb = max(np.linalg.matrix_rank(T_I), np.linalg.matrix_rank(T_J), np.linalg.matrix_rank(T_K))
    
    # Try ranks from lb up to 54
    for r in range(lb, n * m * p + 1):
        # Try to find a decomposition with rank r
        # Use a simple iterative method with random restarts
        for attempt in range(20):
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Iterate
            for it in range(100):
                # Fix U and V, solve for W
                # T[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
                # For each (a,b), T[a,b,:] = W @ diag(U[a,:] * V[b,:])^T
                # So W @ (U[a,:] * V[b,:])^T = T[a,b,:]
                # Let M[c,a,b] = U[a,:] * V[b,:] (element-wise)
                # Then for each (a,b): sum_c' W[c', t] * U[a,t] V[b,t] = T[a,b,c]
                # This is: W^T @ (U[a,:] * V[b,:]) = T[a,b,:]^T
                # Or: W @ diag(U[a,:] * V[b,:]) = T[a,b,:]^T ... let me be careful.
                
                # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
                # Let D[a,b,t] = U[a,t] * V[b,t]
                # Then T[a,b,c] = sum_t D[a,b,t] * W[c,t]
                # So for fixed (a,b): T[a,b,:] = W @ D[a,b,:]^T
                # i.e., T[a,b,c] = sum_t W[c,t] D[a,b,t]
                # This is a linear system: W @ D[a,b,:]^T = T[a,b,:]
                # In matrix form: if we stack all (a,b), we get:
                # T_flat = W @ D_flat^T where D_flat has shape (a*b, r) and T_flat has shape (a*b, p*n)
                # So W = T_flat @ pinv(D_flat^T)
                
                a_dim = n * m
                b_dim = m * p
                c_dim = p * n
                
                D = np.zeros((a_dim, b_dim, r))
                for a in range(a_dim):
                    for b in range(b_dim):
                        D[a, b, :] = U[a, :] * V[b, :]
                
                D_flat = D.reshape(a_dim * b_dim, r)
                T_flat = T.reshape(a_dim * b_dim, c_dim)
                
                # Solve: T_flat = W @ D_flat^T
                # W = T_flat @ pinv(D_flat^T)
                if np.linalg.matrix_rank(D_flat) >= r:
                    W = T_flat @ np.linalg.pinv(D_flat.T)
                
                # Similarly for U and V
                # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
                # For fixed (b,c): T[:,b,c] = U @ (V[b,:] * W[c,:])^T
                D2 = np.zeros((b_dim, c_dim, r))
                for b in range(b_dim):
                    for c in range(c_dim):
                        D2[b, c, :] = V[b, :] * W[c, :]
                D2_flat = D2.reshape(b_dim * c_dim, r)
                T2_flat = T.reshape(c_dim, a_dim, b_dim).transpose(1, 2, 0).reshape(b_dim * c_dim, a_dim)
                
                if np.linalg.matrix_rank(D2_flat) >= r:
                    U = T2_flat @ np.linalg.pinv(D2_flat.T)
                
                # For fixed (a,c): T[a,:,c] = V @ (U[a,:] * W[c,:])^T
                D3 = np.zeros((a_dim, c_dim, r))
                for a in range(a_dim):
                    for c in range(c_dim):
                        D3[a, c, :] = U[a, :] * W[c, :]
                D3_flat = D3.reshape(a_dim * c_dim, r)
                T3_flat = T.reshape(a_dim, c_dim, b_dim).transpose(0, 2, 1).reshape(a_dim * c_dim, b_dim)
                
                if np.linalg.matrix_rank(D3_flat) >= r:
                    V = T3_flat @ np.linalg.pinv(D3_flat.T)
                
                # Check residual
                T_recon = np.zeros((a_dim, b_dim, c_dim))
                for a in range(a_dim):
                    for b in range(b_dim):
                        for c in range(c_dim):
                            T_recon[a, b, c] = np.sum(U[a, :] * V[b, :] * W[c, :])
                
                residual = np.max(np.abs(T_recon - T))
                if residual < 1e-8:
                    # Found a decomposition! Convert to integer/half-integer
                    # For now, just return the float decomposition
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we got here, no decomposition found for this rank
        pass
    
    # Fallback: schoolbook decomposition
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
# EVOLVE-BLOCK-END
