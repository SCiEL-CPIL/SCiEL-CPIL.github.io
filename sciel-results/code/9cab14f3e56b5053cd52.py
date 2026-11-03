import time
import numpy as np

DEADLINE_S = 145.0
N, M, P = 4, 5, 5


def _validate(U, V, W, tol=1e-9):
    """Check sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = N, M, P
    R = U.shape[1]
    # Vectorized check: build the full tensor and compare
    # For each r, outer product of U[:,r], V[:,r], W[:,r]
    # Sum over r should equal the identity tensor
    # Tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We want T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # Actually the matrix multiplication tensor is:
    # sum_{j} A[i,j] B[j,k] = C[i,k]
    # So the tensor is T[i,j,k] = delta(i,j,k) where the product tensor has
    # T[i,j,k] = 1 if this triple contributes to the standard multiplication
    # The standard multiplication tensor has T[i,j,k] = 1 for all (i,j,k) in range
    # because C[i,k] = sum_j A[i,j] B[j,k]
    
    # Let's verify by checking each (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _validate_fast(U, V, W, tol=1e-9):
    """Faster validation using numpy."""
    n, m, p = N, M, P
    R = U.shape[1]
    # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
    U_reshaped = U.reshape(n, m, R)
    V_reshaped = V.reshape(m, p, R)
    W_reshaped = W.reshape(p, n, R)
    
    # For each (i,j,k), compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum or manual
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is tricky with einsum due to the cyclic index pattern
    
    # Alternative: for each r, compute the outer product contribution
    # Actually let's just do it with a loop over r but vectorize over (i,j,k)
    T = np.zeros((n, m, p))
    for r in range(R):
        u = U_reshaped[:, :, r]  # (n, m)
        v = V_reshaped[:, :, r]  # (m, p)
        w = W_reshaped[:, :, r]  # (p, n)
        # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        # This is a 3-way contraction
        # For fixed r: T_r[i,j,k] = u[i,j] * v[j,k] * w[k,i]
        # We can compute this as: for each i, u[i,:] * (v * w[:,i].T)
        # v * w[:,i].T is (m,p) * (m,) -> no, w[:,i] is (p,)
        # v[j,k] * w[k,i] summed over... no, it's not summed
        # T_r[i,j,k] = u[i,j] * v[j,k] * w[k,i]
        # Let's build it: 
        # temp[j,k] = v[j,k] * w[k,i] for each i
        # Then T_r[i,j,k] = u[i,j] * temp[j,k]
        for i in range(n):
            # temp[j,k] = v[j,k] * w[k,i]
            temp = v * w[:, i]  # (m, p) * (p,) broadcasts to (m, p)
            T[i, :, :] += u[i, :] * temp  # (m,) * (m, p) broadcasts
    
    return np.all(np.abs(T - 1.0) <= tol)


def _schoolbook(n, m, p):
    """Return (U, V, W) for the schoolbook decomposition of n x m * m x p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    return U, V, W


def _strassen_2x2x2():
    """
    Return (U, V, W) for Strassen's rank-7 decomposition of 2x2x2 matrix multiplication.
    U: (4,7), V: (4,7), W: (4,7) with W[k*2+i, r].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # m3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # m4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1
    
    # m5 = (A00+A10)B11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = 1
    
    # m6 = (A11-A10)(B00+B10)
    U[3, 5] = 1; U[2, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[0, 5] = 1
    
    # m7 = (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[2, 6] = 1
    
    return U, V, W


def _place_sub_decomposition(U_full, V_full, W_full, U_sub, V_sub, W_sub, 
                              i_off, j_off, k_off, n_sub, m_sub, p_sub, r_off):
    """
    Place a sub-decomposition into the full matrices at the given offsets.
    """
    R_sub = U_sub.shape[1]
    for r in range(R_sub):
        r_full = r_off + r
        for i in range(n_sub):
            for j in range(m_sub):
                U_full[(i+i_off)*M + (j+j_off), r_full] += U_sub[i*m_sub+j, r]
        for j in range(m_sub):
            for k in range(p_sub):
                V_full[(j+j_off)*P + (k+k_off), r_full] += V_sub[j*m_sub+k, r]
        for k in range(p_sub):
            for i in range(n_sub):
                W_full[(k+k_off)*N + (i+i_off), r_full] += W_sub[k*p_sub+i, r]


def _try_block_decomposition():
    """
    Try to build a decomposition using block structure.
    
    Split A (4x5) and B (5x5) into blocks along the inner dimension j.
    A = [A_left (4x2) | A_right (4x3)]
    B = [B_top (2x5); B_bottom (3x5)]
    
    C = A_left * B_top + A_right * B_bottom
    
    A_left * B_top: 4x2 * 2x5 -> 4x5, schoolbook rank = 4*2*5 = 40
    A_right * B_bottom: 4x3 * 3x5 -> 4x5, schoolbook rank = 4*3*5 = 60
    Total: 100. No improvement.
    
    Try: A (4x5), split rows: A_top (2x5), A_bottom (2x5)
    B (5x5), split cols: B_left (5x2), B_right (5x3)
    C (4x5)
    
    C = A_top * B_top_half + A_top * B_bottom_half + A_bottom * B_top_half + A_bottom * B_bottom_half
    where B_top_half = B[:, :2] (5x2), B_bottom_half = B[:, 2:] (5x3)
    
    This doesn't help directly.
    
    Let me try a different approach: use the fact that for small matrices,
    we can sometimes find decompositions by solving linear systems.
    
    Actually, let me try to use a known result: the 4x5x5 matrix multiplication
    tensor. The trivial rank is 100. Can we do better?
    
    One approach: if we can find a rank-r decomposition where r < 100, we win.
    
    Let me try to set up the problem as finding U, V, W with R columns such that
    sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all i,j,k.
    
    This is a system of n*m*p = 100 equations in R*(n*m + m*p + p*n) = R*(20+25+20) = 65R unknowns.
    For R=100, we have 6500 unknowns and 100 equations - very underdetermined.
    
    The schoolbook solution uses R=100 with each column having exactly one 1 in each of U, V, W.
    
    To get R<100, we need some columns to have multiple non-zero entries.
    
    Let me try R=99 and see if we can find a solution.
    
    Actually, let me try a simpler approach: start with schoolbook and try to merge
    two terms by finding a linear dependency.
    
    For the schoolbook, term (i,j,k) has:
    U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    
    Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if there exists r such that
    U[i1*m+j1, r] * V[j1*p+k1, r] * W[k1*n+i1, r] + U[i2*m+j2, r] * V[j2*p+k2, r] * W[k2*n+i2, r] = 1
    and for all other (i,j,k), the contribution is 0.
    
    This is only possible if the terms don't interfere, which is generally not the case.
    
    Let me try a numerical approach: set up the linear system and solve.
    """
    n, m, p = N, M, P
    target_rank = 99
    
    # Variables: U (n*m x R), V (m*p x R), W (p*n x R)
    # Total variables: R * (n*m + m*p + p*n) = R * 65
    # Equations: n*m*p = 100
    # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # This is a non-linear system (trilinear). Hard to solve directly.
    
    # Alternative: fix one matrix and solve for the others.
    # Fix W to be the schoolbook W (each column has one 1).
    # Then for each (i,j,k), we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # With W[k*n+i,r] = 1 if r corresponds to (i,j',k') with i'=i, k'=k... 
    # This gets complicated.
    
    # Let me just try to use scipy to solve a least-squares problem.
    # Actually, let me try a different strategy: use the 3x3 rank-19 result.
    # 3x3 matrix multiplication can be done in rank 19 (Bini et al.).
    # If we can embed this into our 4x5x5 problem, we might save.
    
    # For now, return schoolbook.
    return _schoolbook(n, m, p)


def construct():
    t0 = time.time()
    n, m, p = N, M, P
    
    # Start with schoolbook rank 100
    U, V, W = _schoolbook(n, m, p)
    
    # Try to find a better decomposition
    # Strategy: try to reduce rank by 1, 2, ... using numerical optimization
    
    best_U, best_V, best_W = U, V, W
    best_rank = n * m * p
    
    # Try ranks from best_rank-1 downward, with a time budget
    # For each rank R, try to find a solution
    
    # Actually, let's try a specific approach: 
    # Use the fact that the 4x5x5 tensor might have a rank < 100 decomposition.
    # One known technique: use "border rank" or "approximate rank" but we need exact.
    
    # Let me try to construct a rank-99 decomposition by hand.
    # Start with schoolbook and see if we can combine two terms.
    
    # The schoolbook has 100 terms. Let's see if any two terms can be replaced
    # by a single term with different U, V, W entries.
    
    # Term (i,j,k) contributes to: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # For the product to equal 1 at (i,j,k) and 0 elsewhere, we need:
    # For term r: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other (i',j',k'): U[i'*m+j', r] * V[j'*p+k', r] * W[k'*n+i', r] = 0
    
    # This is only possible if the supports are disjoint, which they are in schoolbook.
    # To merge, we need more flexibility.
    
    # Let me try a numerical approach with scipy.
    try:
        from scipy.optimize import minimize
        import random
        
        def _try_rank(R, t0, deadline):
            """Try to find a rank-R decomposition."""
            if time.time() > deadline:
                return None
            
            # Random initialization
            random.seed(42)
            np.random.seed(42)
            
            # Initialize U, V, W with small random values
            U_init = np.random.randn(n*m, R) * 0.1
            V_init = np.random.randn(m*p, R) * 0.1
            W_init = np.random.randn(p*n, R) * 0.1
            
            # Define the loss function
            def loss(params):
                # Unpack params
                idx = 0
                U_curr = params[idx:idx+n*m*R].reshape(n*m, R)
                idx += n*m*R
                V_curr = params[idx:idx+m*p*R].reshape(m*p, R)
                idx += m*p*R
                W_curr = params[idx:idx+p*n*R].reshape(p*n, R)
                idx += p*n*R
                
                # Compute the tensor
                T = np.zeros((n, m, p))
                for r in range(R):
                    u = U_curr.reshape(n, m, R)[:, :, r]
                    v = V_curr.reshape(m, p, R)[:, :, r]
                    w = W_curr.reshape(p, n, R)[:, :, r]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
                
                # Loss: sum of squared errors
                err = T - 1.0
                return np.sum(err**2)
            
            # This is too slow for direct optimization. Let me try a different approach.
            # Instead, let me use a greedy approach: start with schoolbook and try to
            # eliminate one term at a time.
            
            return None
        
        # Let me try a simpler approach: check if rank 99 is possible by solving
        # a linear system with one term removed.
        
        # Actually, let me just try to use the 3x3 rank-19 result if available.
        # For 4x5x5, we can write:
        # A (4x5) = [A1 (2x5); A2 (2x5)]
        # B (5x5) = [B1 (5x2) B2 (5x3)]
        # C = A1*B1 + A1*B2 + A2*B1 + A2*B2
        # A1*B1: 2x5 * 5x2 -> 2x2, rank 20 (schoolbook) or better?
        # 2x2x2 has rank 7 (Strassen). But 2x5*5x2 is different.
        
        # Let me just return schoolbook for now.
        
    except Exception:
        pass
    
    # Validate the result
    if time.time() < t0 + DEADLINE_S - 5.0:
        if not _validate_fast(U, V, W):
            raise ValueError("Decomposition failed validation")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
