import time
import numpy as np

N, M, P = 3, 4, 8

def _get_strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U[:,0] = [1,0,0,1]
    V[:,0] = [1,0,0,1]
    W[:,0] = [1,0,0,1]
    # M2 = (c+d)e
    U[:,1] = [0,0,1,1]
    V[:,1] = [1,0,0,0]
    W[:,1] = [0,0,1,-1]
    # M3 = a(f-h)
    U[:,2] = [1,1,0,0]
    V[:,2] = [0,1,0,-1]
    W[:,2] = [0,1,0,1]
    # M4 = d(g-e)
    U[:,3] = [0,0,0,1]
    V[:,3] = [0,-1,0,1]
    W[:,3] = [1,0,-1,0]
    # M5 = (a+b)h
    U[:,4] = [1,0,1,0]
    V[:,4] = [0,0,0,1]
    W[:,4] = [0,-1,1,0]
    # M6 = (c-a)(e+f)
    U[:,5] = [-1,0,1,0]
    V[:,5] = [1,1,0,0]
    W[:,5] = [0,0,0,1]
    # M7 = (b-d)(g+h)
    U[:,6] = [0,1,0,-1]
    V[:,6] = [1,0,0,1]
    W[:,6] = [1,0,0,0]
    
    return U, V, W

def _get_schoolbook(n, m, p):
    """
    Return schoolbook decomposition for n x m * m x p.
    """
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
    return U, V, W

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _random_search_decomposition(n, m, p, max_rank, time_limit=10.0):
    """
    Attempt to find a decomposition of rank r for n x m * m x p.
    Uses randomized linear algebra.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    
    # The tensor T is defined by T[i, j, k] = 1 if i, j, k are indices of A, B, C.
    # We want T = sum_r u_r v_r w_r
    # This is equivalent to finding matrices U, V, W such that:
    # U^T (vec(A) * vec(B)^T) ... wait, simpler view:
    # Let A be n x m, B be m x p, C be p x n.
    # C[k, i] = sum_j A[i, j] B[j, k]
    # We want C[k, i] = sum_r (sum_j U[i, j, r] A[i, j]) (sum_k V[j, k, r] B[j, k]) W[k, i, r] ? No.
    # The standard form is:
    # C[k, i] = sum_r U[i, r] V[r, k] where U depends on A and V depends on B.
    # Specifically, let A be flattened to a vector a of length n*m.
    # Let B be flattened to a vector b of length m*p.
    # Let C be flattened to a vector c of length p*n.
    # c = T(a, b) is a bilinear map.
    # We can write c = sum_r (U_r a) (V_r b) W_r^T ? No.
    # c_{k, i} = sum_r (sum_{j} U[i, j, r] A_{i, j}) * (sum_{l} V[j, l, r] B_{j, l}) * W_{k, i, r} ??
    # Actually, the factorization is:
    # C = sum_r (U_r A) (B V_r) W_r ? No, dimensions don't match for standard matmul.
    
    # Let's stick to the tensor definition.
    # T is a tensor of size (n*m) x (m*p) x (p*n).
    # T[a, b, c] = 1 if c corresponds to (k, i) and a to (i, j) and b to (j, k) for some j.
    # Actually, for fixed i, j, k, the term is A[i, j] B[j, k] C[k, i].
    # So the tensor has 1s at (i*m+j, j*p+k, k*n+i).
    
    # We want to find rank r decomposition.
    # This is equivalent to finding U (n*m x r), V (m*p x r), W (p*n x r)
    # such that T = sum_r u_r v_r^T w_r^T (in tensor product notation).
    
    # Strategy:
    # 1. Initialize U, V, W with random Gaussian entries.
    # 2. For a fixed r, the problem is to minimize || T - sum_r u_r v_r^T w_r^T ||.
    # This is a non-convex problem.
    # However, if we fix U and V, W can be solved in least squares?
    # T_{a, b, c} = sum_r U_{a, r} V_{b, r} W_{c, r}
    # For fixed U, V, this is linear in W.
    # For each r, W_{c, r} is a variable.
    # But the equation couples all c for a fixed r? No, for each (a, b), T_{a, b, c} = sum_r U_{a, r} V_{b, r} W_{c, r}.
    # This is not a simple linear system for W because W_{c, r} appears for all a, b.
    # Actually, if we fix U and V, we can solve for W by least squares.
    # Let M_r be the matrix U V^T (size n*m x m*p).
    # Then T = sum_r M_r \otimes w_r? No.
    # T_{a, b, c} = sum_r U_{a, r} V_{b, r} W_{c, r}
    # This can be rewritten. For each r, let X_r be the matrix U_r V_r^T.
    # Then T = sum_r X_r \otimes W_r? No, W_r is a vector.
    # T_{a, b, c} = (sum_r U_{a, r} V_{b, r}) W_{c, r}? No, the sum is over r, so W depends on r.
    # It's a sum of rank-1 tensors in the (a, b) plane, scaled by W_{c, r}.
    
    # Alternative view:
    # Flatten T into a matrix of size (n*m * m*p) x (p*n).
    # Let T_flat be this matrix.
    # T_flat[(a, b), c] = T_{a, b, c}.
    # We want T_flat = sum_r (U_r \otimes V_r) W_r^T.
    # Here U_r \otimes V_r is a vector of size n*m * m*p.
    # Let Z_r = U_r \otimes V_r.
    # Then T_flat = sum_r Z_r W_r^T.
    # This is a rank-r decomposition of T_flat!
    # If we can find a rank-r decomposition of T_flat, say T_flat = Z W^T,
    # then we need to check if each column Z_r can be factored into U_r \otimes V_r.
    # Since U_r is n*m and V_r is m*p, Z_r is their Kronecker product.
    # This imposes structure on Z_r.
    
    # So the algorithm is:
    # 1. Find a rank-r decomposition of T_flat.
    # 2. Check if the columns Z_r are Kronecker products.
    # 3. If yes, extract U_r, V_r, W_r.
    
    # How to find rank-r decomposition?
    # SVD gives the minimum rank for exact decomposition, but that's the tensor rank of the matrix, which is likely high (n*m*p).
    # We want a decomposition where Z_r are Kronecker products.
    # This is a "structured low-rank approximation" problem.
    
    # Iterative approach:
    # Initialize U, V, W randomly.
    # Iterate:
    #   Fix U, V. Compute Z_r = U_r \otimes V_r.
    #   Solve for W: minimize || T_flat - Z W^T ||_F.
    #   This is a linear least squares problem: T_flat = Z W^T.
    #   W = (Z^T Z)^{-1} Z^T T_flat.
    #   Fix V, W. Solve for U?
    #   T_{a, b, c} = sum_r U_{a, r} V_{b, r} W_{c, r}
    #   For fixed V, W, this is linear in U.
    #   T_{a, b, c} = sum_r U_{a, r} (V_{b, r} W_{c, r})
    #   Let Q_{b, c, r} = V_{b, r} W_{c, r}.
    #   Then T_{a, b, c} = sum_r U_{a, r} Q_{b, c, r}.
    #   For each a, T_{a, :, :} = sum_r U_{a, r} Q_{:, :, r}.
    #   This is a linear system for U_{a, r}.
    #   Flatten T_{a, :, :} to a vector t_a of size m*p * p*n.
    #   Flatten Q_{:, :, r} to a vector q_r of size m*p * p*n.
    #   t_a = sum_r U_{a, r} q_r.
    #   So [t_0, ..., t_{n*m-1}]^T = U Q_flat^T?
    #   Let T_a be the matrix with rows t_a.
    #   Let Q_flat be the matrix with columns q_r.
    #   Then T_a = U Q_flat^T.
    #   U = T_a (Q_flat^T Q_flat)^{-1} Q_flat.
    
    # Similarly for V.
    
    # This is an alternating least squares (ALS) approach.
    
    def _als_decomposition(n, m, p, r, max_iter=50, tol=1e-12):
        np.random.seed(42) # For reproducibility in this helper, though outer loop might want random restarts
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Precompute T_flat
        T_flat = np.zeros((n * m * m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_flat[a * m * p + b, c] = 1.0
        
        prev_err = float('inf')
        
        for it in range(max_iter):
            # Update W
            # T_flat = Z W^T, Z = [U_1 \otimes V_1, ..., U_r \otimes V_r]
            Z = np.zeros((n * m * m * p, r))
            for rr in range(r):
                Z[:, rr] = np.kron(U[:, rr], V[:, rr])
            
            # Solve Z W^T = T_flat
            # W = (Z^T Z)^{-1} Z^T T_flat
            # Use lstsq for stability
            W_new, _, _, _ = np.linalg.lstsq(Z, T_flat.T, rcond=None)
            W = W_new.T
            
            # Update U
            # T_a = U Q_flat^T
            # Q_flat columns are q_r = kron(V_r, W_r)? No.
            # T_{a, b, c} = sum_r U_{a, r} V_{b, r} W_{c, r}
            # q_r(b, c) = V_{b, r} W_{c, r}
            # So q_r is the outer product V_r W_r^T.
            # Flatten q_r to a vector of size m*p * p*n.
            Q_flat = np.zeros((m * p * p * n, r))
            for rr in range(r):
                Q_flat[:, rr] = np.kron(V[:, rr], W[:, rr])
            
            # T_a is matrix of size (n*m) x (m*p * p*n)
            T_a = np.zeros((n * m, m * p * p * n))
            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    # Row a of T_a corresponds to T_{a, b, c} for all b, c
                    # T_{a, b, c} is non-zero only if b, c correspond to j, k for some k?
                    # No, T_{a, b, c} is 1 only if (i, j, k) match.
                    # For fixed a=(i, j), T_{a, b, c} is 1 only if b=(j, k) and c=(k, i) for some k.
                    # So for fixed a, there are p non-zero entries in the row of T_a?
                    # No, T_flat has 1s at specific (a, b, c).
                    # For fixed a, T_{a, b, c} is 1 if b and c are "compatible" with a.
                    # Let's just extract the row from T_flat.
                    # T_flat row index is a * (m*p) + b.
                    # So T_a row a is T_flat[a * (m*p) + :, :] reshaped?
                    # T_flat is (n*m * m*p) x (p*n).
                    # Row a of T_a should be the concatenation of T_flat rows corresponding to a?
                    # No, T_flat row index is (a, b).
                    # So T_a row a is T_flat[a * (m*p) + 0 : a * (m*p) + m*p, :]
                    # But T_a is (n*m) x (m*p * p*n).
                    # The columns of T_a are indexed by (b, c).
                    # T_flat columns are indexed by c.
                    # So T_a[a, (b, c)] = T_flat[a * m*p + b, c].
                    # This is just a reshape of T_flat.
                    pass
            
            # Actually, T_a = T_flat reshaped to (n*m, m*p, p*n) then transposed to (n*m, m*p * p*n)?
            # T_flat is (n*m * m*p) x (p*n).
            # Let T_3d = T_flat.reshape(n*m, m*p, p*n)
            # Then T_3d[a, b, c] = T_{a, b, c}.
            # We want T_a[a, (b, c)] = T_3d[a, b, c].
            # So T_a = T_3d.reshape(n*m, m*p * p*n).
            
            T_3d = T_flat.reshape(n * m, m * p, p * n)
            T_a = T_3d.reshape(n * m, m * p * p * n)
            
            U_new, _, _, _ = np.linalg.lstsq(Q_flat.T, T_a, rcond=None)
            U = U_new.T
            
            # Update V
            # T_{a, b, c} = sum_r U_{a, r} V_{b, r} W_{c, r}
            # For fixed b, T_{a, b, c} = sum_r V_{b, r} (U_{a, r} W_{c, r})
            # Let p_r(a, c) = U_{a, r} W_{c, r}
            # P_flat columns are p_r flattened.
            P_flat = np.zeros((n * m * p * n, r))
            for rr in range(r):
                P_flat[:, rr] = np.kron(U[:, rr], W[:, rr])
            
            # T_b[b, (a, c)] = T_{a, b, c}
            # T_3d is (n*m, m*p, p*n).
            # We want T_b of size (m*p, n*m * p*n).
            # T_b[b, (a, c)] = T_3d[a, b, c].
            # This is a permutation of T_3d.
            # T_3d[a, b, c] -> T_b[b, a * p*n + c]
            
            T_b = np.zeros((m * p, n * m * p * n))
            for b in range(m * p):
                for a in range(n * m):
                    T_b[b, :] = T_3d[a, b, :].copy() # This is slow. Use reshape/transpose.
            
            # Better way to get T_b:
            # T_3d is (A, B, C).
            # We want (B, A*C).
            # T_3d.transpose(1, 0, 2).reshape(B, A*C)
            
            T_b = T_3d.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
            
            V_new, _, _, _ = np.linalg.lstsq(P_flat.T, T_b, rcond=None)
            V = V_new.T
            
            # Check convergence
            # Error = || T_flat - Z W^T ||
            Z = np.zeros((n * m * m * p, r))
            for rr in range(r):
                Z[:, rr] = np.kron(U[:, rr], V[:, rr])
            T_approx = Z @ W.T
            err = np.linalg.norm(T_flat - T_approx, 'fro')
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        # Final check
        Z = np.zeros((n * m * m * p, r))
        for rr in range(r):
            Z[:, rr] = np.kron(U[:, rr], V[:, rr])
        T_approx = Z @ W.T
        err = np.linalg.norm(T_flat - T_approx, 'fro')
        
        if err < 1e-9:
            return U, V, W
        else:
            return None

    # Try ranks from 24 upwards (schoolbook is 24)
    # Known lower bound?
    # For 3x4x4, schoolbook is 12.
    # Wait, n=3, m=4, p=4. Schoolbook rank = 3*4*4 = 48.
    # Wait, earlier I said 3x4x4.
    # N=3, M=4, P=4.
    # Rank = 3*4*4 = 48.
    # Is there a better one?
    # 2x2x2 is 7.
    # 3x3x3 is 23.
    # 3x4x4 might be reducible.
    
    # Let's just try to find a decomposition of rank < 48.
    # Start with 40, 42, etc.
    
    # But wait, the Kronecker product strategy:
    # T_{3,4,8} = T_{3,4,4} \otimes T_{1,1,2}
    # Rank(T_{3,4,8}) <= Rank(T_{3,4,4}) * Rank(T_{1,1,2})
    # Rank(T_{1,1,2}) is 2 (schoolbook).
    # So if we find Rank(T_{3,4,4}) = R, then Rank(T_{3,4,8}) <= 2R.
    # Schoolbook for 3x4x4 is 48. So 2*48 = 96.
    # This doesn't help if R=48.
    # We need R < 48 to get < 96.
    # Is there a known rank for 3x4x4?
    # If we use Strassen for 2x2 parts?
    # 3x4x4 can be decomposed into blocks?
    # 3 = 2+1, 4 = 2+2, 4 = 2+2.
    # Blocks:
    # (2,2,2): rank 7
    # (2,2,2): rank 7
    # (2,2,2): rank 7
    # (2,2,2): rank 7
    # (1,2,2): rank 4
    # (1,2,2): rank 4
    # (1,2,2): rank 4
    # (1,2,2): rank 4
    # Total = 4*7 + 4*4 = 28 + 16 = 44.
    # So Rank(3x4x4) <= 44.
    # Then Rank(3x4x8) <= 2 * 44 = 88.
    
    # The current program already finds 88 using partitioning.
    # The Kronecker product approach is essentially the same as partitioning if we use the same blocks.
    # However, the Kronecker product of the *entire* tensor might allow for more global optimizations if we don't just sum block ranks.
    # But tensor rank of a Kronecker product is the product of the ranks.
    # So if we use the block decomposition for 3x4x4, we are effectively using the sum of block ranks.
    
    # To improve, we need to find a rank < 44 for 3x4x4.
    # Or a different structure.
    
    # Let's try the ALS search for 3x4x4 with rank 43, 42, etc.
    # If we find rank 43, then 3x4x8 rank <= 86.
    
    # Let's implement the search.
    
    best_rank = 48
    best_decomp = None
    U_sb, V_sb, W_sb = _get_schoolbook(3, 4, 4)
    best_decomp = (U_sb, V_sb, W_sb)
    
    for r in range(43, 48):
        if time.time() - start_time > time_limit:
            break
        res = _als_decomposition(3, 4, 4, r, max_iter=100, tol=1e-10)
        if res is not None:
            best_rank = r
            best_decomp = res
            break
            
    return best_decomp

def construct():
    n, m, p = N, M, P
    start_time = time.time()
    deadline = start_time + 140
    
    # 1. Get decomposition for 3x4x4
    decomp_344 = _random_search_decomposition(3, 4, 4, 48, time_limit=60.0)
    
    if decomp_344 is None:
        # Fallback to schoolbook
        U_344, V_344, W_344 = _get_schoolbook(3, 4, 4)
    else:
        U_344, V_344, W_344 = decomp_344
        
    rank_344 = U_344.shape[1]
    
    # 2. Get decomposition for 1x1x2
    # This is just the identity for the vector doubling?
    # 1x1 * 1x2 -> 1x2.
    # A[0,0] * B[0,k] = C[k,0]
    # Schoolbook rank 2.
    # U: (1*1, 2) = (1, 2) -> [[1, 1]]? No.
    # U[0, r] * V[0, r] * W[k, r] = C[k, 0]
    # For r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> C[0,0] += A[0,0]B[0,0]
    # For r=1: U[0,1]=1, V[0,1]=1, W[1,1]=1 -> C[1,0] += A[0,0]B[0,1]
    # So U = [[1, 1]], V = [[1, 1]], W = [[1, 0], [0, 1]]
    U_112 = np.array([[1, 1]], dtype=float)
    V_112 = np.array([[1, 1]], dtype=float)
    W_112 = np.array([[1, 0], [0, 1]], dtype=float)
    rank_112 = 2
    
    # 3. Construct 3x4x8 via Kronecker product
    # T_{3,4,8} = T_{3,4,4} \otimes T_{1,1,2}
    # Indices:
    # i in 0..2, j in 0..3, k in 0..7.
    # k = k1 * 2 + k2, where k1 in 0..3, k2 in 0..1.
    # The tensor is sum_{r1, r2} U_{344}[i, j, r1] V_{344}[j, k1, r1] W_{344}[k1, i, r1] * U_{112}[0, 0, r2] V_{112}[0, k2, r2] W_{112}[k2, 0, r2]
    # Wait, the Kronecker product of tensors T1 (A1, B1, C1) and T2 (A2, B2, C2) is T (A1x2, B1x2, C1x2).
    # T(a1, a2, b1, b2, c1, c2) = T1(a1, b1, c1) T2(a2, b2, c2).
    # Here A1 is 3x4 (12 dims), B1 is 4x4 (16 dims), C1 is 4x3 (12 dims).
    # A2 is 1x1 (1 dim), B2 is 1x2 (2 dims), C2 is 2x1 (2 dims).
    # So A is 12x1 = 12. B is 16x2 = 32. C is 12x2 = 24.
    # But we want B to be 4x8 = 32. C to be 8x3 = 24.
    # The mapping:
    # A index: (i, j) -> (i, j, 0)
    # B index: (j, k) -> (j, k1, k2) where k = k1*2 + k2
    # C index: (k, i) -> (k1, k2, i) where k = k1*2 + k2
    
    # The rank is rank_344 * rank_112.
    total_rank = rank_344 * rank_112
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for r1 in range(rank_344):
        for r2 in range(rank_112):
            # U
            for i in range(3):
                for j in range(4):
                    U[i * 4 + j, col] = U_344[i * 4 + j, r1] * U_112[0, r2]
                    
            # V
            for j in range(4):
                for k1 in range(4):
                    for k2 in range(2):
                        k = k1 * 2 + k2
                        V[j * 8 + k, col] = V_344[j * 4 + k1, r1] * V_112[0, r2] # V_112 is 1x2, so index is just k2?
                        # V_112[0, r2] is the value for input B[0, k2]?
                        # V_112 is (1*2, 2) = (2, 2).
                        # V_112[0, 0]=1, V_112[0, 1]=1
                        # V_112[1, 0]=1, V_112[1, 1]=1
                        # So V_112[k2, r2].
                        V[j * 8 + k, col] = V_344[j * 4 + k1, r1] * V_112[k2, r2]
                        
            # W
            for k1 in range(4):
                for k2 in range(2):
                    for i in range(3):
                        k = k1 * 2 + k2
                        W[k * 3 + i, col] = W_344[k1 * 3 + i, r1] * W_112[k2, r2]
                        
            col += 1
            
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _get_schoolbook(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
