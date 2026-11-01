import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    nm, mp, pn = n*m, m*p, p*n
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                d = k*n + i
                T[a, b, d] = 1.0
    
    # Strategy: Use the block structure.
    # The 2x3 x 3x10 multiplication can be viewed as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # For each pair (i,k), we need to compute sum_j A[i,j]*B[j,k].
    # 
    # Key idea: Use a rank-5 decomposition for the 2x3x1 part,
    # and share the "k" dimension across multiple k values.
    #
    # Actually, let's use a different approach:
    # We can write the multiplication as a sum of rank-1 terms.
    # Schoolbook: 60 terms.
    # 
    # Let's try to find a decomposition with rank < 60 using alternating minimization
    # with a good initialization based on the structure.
    
    # Initialization: Start with schoolbook (rank 60), then try to merge terms.
    # Or: Use a rank-5 decomposition for each "column" of B (10 columns), giving 50 terms,
    # then try to reduce further.
    
    # Let's first construct a rank-50 decomposition (10 copies of rank-5 for 2x3x1),
    # then try to reduce it.
    
    # Rank-5 decomposition for 2x3x1:
    # We need U (6x5), V (3x5), W (2x5) such that
    # sum_c U[i*3+j, c] * V[j, c] * W[i, c] = delta_{matches}
    
    # Let me construct it explicitly.
    # The constraint is: for each (i,j), sum_c U[i*3+j,c]*V[j,c]*W[i,c] = 1
    # And for all other (a,b,d), the sum should be 0.
    
    # This is a system of equations. Let me set it up.
    # Variables: U (30 vars), V (15 vars), W (10 vars) = 55 vars for rank 5.
    # Equations: 6*3*2 = 36 entries in the tensor, but only 6 are non-zero.
    # So we have 30 zero equations and 6 one-equations = 36 equations, 55 variables.
    # Underdetermined, so solutions exist.
    
    # Let me solve it numerically.
    # Flatten the tensor and set up the system.
    
    # Actually, let me just use a known result: the rank of 2x3x1 is 5.
    # I'll construct it by solving the linear system.
    
    def _solve_rank5_2x3x1():
        """Solve for rank-5 decomposition of 2x3x1 matrix multiplication."""
        n1, m1, p1 = 2, 3, 1
        nm1, mp1, pn1 = n1*m1, m1*p1, p1*n1  # 6, 3, 2
        
        # Build target tensor
        T1 = np.zeros((nm1, mp1, pn1))
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    a = i*m1 + j
                    b = j*p1 + k
                    d = k*n1 + i
                    T1[a, b, d] = 1.0
        
        # We want U (6x5), V (3x5), W (2x5)
        # Use alternating least squares
        np.random.seed(0)
        U = np.random.randn(nm1, 5) * 0.1
        V = np.random.randn(mp1, 5) * 0.1
        W = np.random.randn(pn1, 5) * 0.1
        
        for it in range(1000):
            # Update U
            for a in range(nm1):
                # T1[a, :, :] = sum_c U[a,c] * V[:,c] * W[:,c]
                # For each (b,d): T1[a,b,d] = sum_c U[a,c] * (V[b,c]*W[d,c])
                # Let M[b*pn1+d, c] = V[b,c]*W[d,c]
                # Then T1[a].flatten() = M @ U[a]
                M = np.zeros((mp1*pn1, 5))
                for b in range(mp1):
                    for d in range(pn1):
                        for c in range(5):
                            M[b*pn1+d, c] = V[b,c] * W[d,c]
                # Solve M @ U[a] = T1[a].flatten()
                # Least squares
                U[a, :] = np.linalg.lstsq(M, T1[a, :, :].flatten(), rcond=None)[0]
            
            # Update V
            for b in range(mp1):
                M = np.zeros((nm1*pn1, 5))
                for a in range(nm1):
                    for d in range(pn1):
                        for c in range(5):
                            M[a*pn1+d, c] = U[a,c] * W[d,c]
                V[b, :] = np.linalg.lstsq(M, T1[:, b, :].flatten(), rcond=None)[0]
            
            # Update W
            for d in range(pn1):
                M = np.zeros((nm1*mp1, 5))
                for a in range(nm1):
                    for b in range(mp1):
                        for c in range(5):
                            M[a*mp1+b, c] = U[a,c] * V[b,c]
                W[d, :] = np.linalg.lstsq(M, T1[:, :, d].flatten(), rcond=None)[0]
            
            # Check convergence
            T_comp = np.einsum('ac,bc,dc->abd', U, V, W)
            err = np.linalg.norm(T_comp - T1)
            if err < 1e-10:
                break
        
        return U, V, W
    
    if time.time() < deadline:
        U5, V5, W5 = _solve_rank5_2x3x1()
        # Verify
        T5 = np.einsum('ac,bc,dc->abd', U5, V5, W5)
        T1_check = np.zeros((6, 3, 2))
        for i in range(2):
            for j in range(3):
                T1_check[i*3+j, j, i] = 1.0
        assert np.allclose(T5, T1_check, atol=1e-8), "Rank-5 decomposition failed"
    
    # Now build the full 2x3x10 decomposition.
    # For each k in 0..9, use the rank-5 decomposition for the "k-th column" of B.
    # This gives 10*5 = 50 terms.
    
    rank = 50
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for k in range(p):
        base = k * 5
        for c in range(5):
            # For this k and rank-5 component c:
            # U[i*3+j, base+c] = U5[i*3+j, c]
            # V[j*p+k, base+c] = V5[j, c]
            # W[k*n+i, base+c] = W5[i, c]
            for i in range(n):
                for j in range(m):
                    U[i*m+j, base+c] = U5[i*m1+j, c] if False else U5[i*3+j, c]
                    V[j*p+k, base+c] = V5[j, c]
                    W[k*n+i, base+c] = W5[i, c]
    
    # Verify this rank-50 decomposition
    T_comp = np.einsum('ac,bc,dc->abd', U, V, W)
    if not np.allclose(T_comp, T, atol=1e-8):
        # Fall back to schoolbook
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Now try to reduce the rank from 50 to something lower.
    # Use alternating minimization starting from this rank-50 solution,
    # but with a lower target rank.
    
    for target_rank in [48, 49]:
        if time.time() > deadline:
            break
        
        # Try to find a decomposition of rank target_rank
        # Start from the rank-50 solution and try to merge columns.
        
        # Use ALS with random restarts
        found = _try_rank(T, n, m, p, target_rank, time.time(), deadline)
        if found is not None:
            U, V, W = found
            rank = target_rank
            break
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _try_rank(T, n, m, p, target_rank, start_time, deadline):
    """Try to find a decomposition of the given rank using ALS."""
    nm, mp, pn = n*m, m*p, p*n
    
    max_restarts = 50
    max_iters = 500
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            return None
        
        np.random.seed(100 + restart)
        
        # Initialize
        U = np.random.randn(nm, target_rank) * 0.1
        V = np.random.randn(mp, target_rank) * 0.1
        W = np.random.randn(pn, target_rank) * 0.1
        
        prev_err = np.inf
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            for a in range(nm):
                M = np.zeros((mp*pn, target_rank))
                for b in range(mp):
                    for d in range(pn):
                        for c in range(target_rank):
                            M[b*pn+d, c] = V[b,c] * W[d,c]
                U[a, :] = np.linalg.lstsq(M, T[a, :, :].flatten(), rcond=None)[0]
            
            # Update V
            for b in range(mp):
                M = np.zeros((nm*pn, target_rank))
                for a in range(nm):
                    for d in range(pn):
                        for c in range(target_rank):
                            M[a*pn+d, c] = U[a,c] * W[d,c]
                V[b, :] = np.linalg.lstsq(M, T[:, b, :].flatten(), rcond=None)[0]
            
            # Update W
            for d in range(pn):
                M = np.zeros((nm*mp, target_rank))
                for a in range(nm):
                    for b in range(mp):
                        for c in range(target_rank):
                            M[a*mp+b, c] = U[a,c] * V[b,c]
                W[d, :] = np.linalg.lstsq(M, T[:, :, d].flatten(), rcond=None)[0]
            
            # Check error
            T_comp = np.einsum('ac,bc,dc->abd', U, V, W)
            err = np.linalg.norm(T_comp - T)
            
            if err < 1e-10:
                return (U, V, W)
            
            if abs(prev_err - err) < 1e-14:
                break
            prev_err = err
    
    return None

