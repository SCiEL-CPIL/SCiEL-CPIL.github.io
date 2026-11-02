"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x3():
    """Known rank-5 decomposition for 2x2 * 2x3 matrix multiplication."""
    n, m, p = 2, 2, 3
    rank = 5
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    products = [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)]),
    ]
    output_contribs = [
        [((0,0),1),((1,1),1)],
        [((1,0),1),((1,1),-1)],
        [((0,1),1),((1,1),1)],
        [((0,0),1),((1,0),1)],
        [((0,0),-1),((0,1),1)],
    ]
    
    for s in range(5):
        for (ai,aj), coeff in products[s][0]:
            U[ai*m+aj, s] = coeff
        for (bj,bk), coeff in products[s][1]:
            V[bj*p+bk, s] = coeff
        for (ci,ck), coeff in output_contribs[s]:
            W[ck*n+ci, s] = coeff
    
    return U, V, W

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Fast verification using tensor structure."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    # Check all required entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a, b, c = i*m+j, j*p+k, k*n+i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1.0) > 1e-6:
                    return False
    
    # Check zeros for invalid combinations
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i, j = a//m, a%m
                jj, k = b//p, b%p
                kk, ii = c//n, c%n
                if j != jj or k != kk or i != ii:
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val) > 1e-6:
                        return False
    return True

def _kronecker_rowpair_decomposition(n, m, p, max_iter=2000, seed=42):
    """
    Find a rank-r decomposition using the row-pairing Kronecker structure.
    
    The key constraint: U[i*m+j, r] = u_j(r) for both i (output rows share A-factor).
    This comes from the tensor product structure M_{2,2,12} = M_{1,1,2} ⊗ M_{2,1,12}.
    
    We search over rank r and use ALS to find the decomposition.
    """
    np.random.seed(seed)
    
    # The structure: U has 2*m = 4 rows, but U[0,:] = U[2,:] (row 0 and 2 share)
    # U[1,:] = U[3,:] (row 1 and 3 share)
    # So effectively we have m=2 independent A-rows, each of length rank.
    
    # For the Kronecker product structure:
    # U[i*m+j, r] should be of the form: alpha_i(r) * beta_j(r)
    # where alpha is (n, rank) and beta is (m, rank)
    # But actually, for M_{2,2,12} = M_{1,1,2} ⊗ M_{2,1,12}:
    # The A-factor is just the identity on the 2-dim space (rank 2 for M_{1,1,2})
    # So U[i*m+j, r] = delta_{i, i'} * v_j(r) where v is the A-factor of M_{2,1,12}
    
    # Let me think more carefully. M_{2,2,12}:
    # T[i, j, k] = delta_{i, k} (no, that's not right)
    # Actually T[i*m+j, j*p+k, k*n+i] = 1
    # This is the standard matrix multiplication tensor.
    
    # The Kronecker structure: if we write the 2x2x12 as (1x1x2) ⊗ (2x1x12),
    # then the tensor is T_{(i1,i2),(j1,j2),(k1,k2)} = T1_{i1,j1,k1} * T2_{i2,j2,k2}
    # where T1 is 1x1x2 (rank 2) and T2 is 2x1x12 (rank 24).
    
    # But this gives rank 48. The question is whether we can do better by
    # exploiting the specific structure.
    
    # Let me just do a direct ALS search with the constraint that U has the
    # Kronecker product structure.
    
    # Actually, let me try a different approach: just do ALS on the full problem
    # but initialize with the Kronecker structure and see if it converges to
    # a lower rank.
    
    # For now, let me try rank 24 (which is the rank of M_{2,1,12} * 2 for the
    # two output rows, but with sharing).
    
    # The rank of M_{2,1,12} is 24 (schoolbook). If we can share the A-factor
    # between the two output rows, we might get rank 24 total.
    
    # Let me set up the problem:
    # U: (4, r) with constraint U[0,:] = U[2,:] and U[1,:] = U[3,:]
    # V: (24, r)
    # W: (24, r)
    
    # The constraint means we only have 2*r free parameters in U instead of 4*r.
    
    # Let me parameterize: U[0,:] = U[2,:] = u0, U[1,:] = U[3,:] = u1
    # where u0, u1 are (r,) vectors.
    
    # Actually, I realize the Kronecker product structure is more specific:
    # For M_{2,2,12} = M_{1,1,2} ⊗ M_{2,1,12}:
    # The A-factor of M_{1,1,2} is [[1],[1]] (rank 2, both entries are 1)
    # Wait, M_{1,1,2} has tensor T[0,0,k] = 1 for k=0,1. Rank is 2.
    # A-factor: [[1],[1]], B-factor: [[1,1]], C-factor: [[1,1]]
    # So U[i*m+j, r] = A1[i, r1] * A2[j, r2] where (r1, r2) -> r is the Kronecker index.
    
    # This is getting complex. Let me just do a direct search.
    
    # Try rank 24 with the row-sharing constraint
    rank = 24
    
    # Parameterize U with the constraint:
    # U[0,:] = U[2,:] = u0 (shape (rank,))
    # U[1,:] = U[3,:] = u1 (shape (rank,))
    
    u0 = np.random.randn(rank) * 0.1
    u1 = np.random.randn(rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    U = np.zeros((n*m, rank))
    U[0] = u0
    U[1] = u1
    U[2] = u0
    U[3] = u1
    
    best_err = float('inf')
    best_u0, best_u1, best_V, best_W = u0.copy(), u1.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iter):
        # Update u0 (U[0,:] and U[2,:])
        # For each k, we need: sum_r u0[r] * V[0*p+k, r] * W[k*n+0, r] = 1
        # and: sum_r u0[r] * V[1*p+k, r] * W[k*n+1, r] = 1
        # Also: for invalid combos, the sum should be 0.
        
        # Actually, let me just do gradient descent on the full U, V, W
        # with the constraint enforced by projection.
        
        # Update V
        for b in range(m*p):
            j, k = b // p, b % p
            # V[b, r] should satisfy: for all i, sum_r U[i*m+j, r] * V[b, r] * W[k*n+i, r] = 1
            # This is a system of n=2 equations for V[b, :]
            M = np.array([U[i*m+j] * W[k*n+i] for i in range(n)])
            if np.linalg.matrix_rank(M) >= 1:
                V[b] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
        
        # Update W
        for c in range(p*n):
            k, i = c // n, c % n
            # W[c, r] should satisfy: for all j, sum_r U[i*m+j, r] * V[j*p+k, r] * W[c, r] = 1
            M = np.array([U[i*m+j] * V[j*p+k] for j in range(m)])
            if np.linalg.matrix_rank(M) >= 1:
                W[c] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
        
        # Update u0 and u1 (with constraint)
        # u0: for each k, sum_r u0[r] * V[0*p+k, r] * W[k*n+0, r] = 1
        #                    sum_r u0[r] * V[1*p+k, r] * W[k*n+1, r] = 1
        # This is a system of 2*p = 24 equations for u0 (24 unknowns)
        M_u0 = np.array([V[j*p+k] * W[k*n+i] for i in range(n) for j in range(m) for k in range(p)])
        # Wait, that's not right. Let me be more careful.
        
        # u0 appears in U[0,:] and U[2,:].
        # U[0,:] is used in: T[0*m+j, j*p+k, k*n+0] = 1 for j=0,1 and k=0..11
        #   sum_r u0[r] * V[j*p+k, r] * W[k*n+0, r] = 1
        # U[2,:] is used in: T[2*m+j, j*p+k, k*n+1] = 1 for j=0,1 and k=0..11
        #   sum_r u0[r] * V[j*p+k, r] * W[k*n+1, r] = 1
        
        M_u0 = np.zeros((2*p, rank))
        for k in range(p):
            for j in range(m):
                M_u0[k, :] = V[j*p+k] * W[k*n+0]
                M_u0[p+k, :] = V[j*p+k] * W[k*n+1]
        
        b_u0 = np.ones(2*p)
        if np.linalg.matrix_rank(M_u0) >= 1:
            u0_new = np.linalg.lstsq(M_u0, b_u0, rcond=None)[0]
            u0 = u0_new
        
        # Similarly for u1
        M_u1 = np.zeros((2*p, rank))
        for k in range(p):
            for j in range(m):
                M_u1[k, :] = V[j*p+k] * W[k*n+0]
                M_u1[p+k, :] = V[j*p+k] * W[k*n+1]
        
        # Wait, u1 is in U[1,:] and U[3,:], which correspond to i=0,j=1 and i=1,j=1
        # T[1*m+1, 1*p+k, k*n+0] = 1: sum_r u1[r] * V[1*p+k, r] * W[k*n+0, r] = 1
        # T[3, 1*p+k, k*n+1] = 1: sum_r u1[r] * V[1*p+k, r] * W[k*n+1, r] = 1
        
        M_u1 = np.zeros((2*p, rank))
        for k in range(p):
            M_u1[k, :] = V[1*p+k] * W[k*n+0]
            M_u1[p+k, :] = V[1*p+k] * W[k*n+1]
        
        b_u1 = np.ones(2*p)
        if np.linalg.matrix_rank(M_u1) >= 1:
            u1_new = np.linalg.lstsq(M_u1, b_u1, rcond=None)[0]
            u1 = u1_new
        
        # Update U
        U[0] = u0
        U[1] = u1
        U[2] = u0
        U[3] = u1
        
        # Check error
        err_sum = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a, b, c = i*m+j, j*p+k, k*n+i
                    val = np.dot(U[a], V[b] * W[c])
                    err_sum += (val - 1.0)**2
        err = np.sqrt(err_sum)
        if err < best_err:
            best_err = err
            best_u0, best_u1, best_V, best_W = u0.copy(), u1.copy(), V.copy(), W.copy()
        if err < 1e-8:
            break
    
    # Reconstruct U
    U_final = np.zeros((n*m, rank))
    U_final[0] = best_u0
    U_final[1] = best_u1
    U_final[2] = best_u0
    U_final[3] = best_u1
    
    return U_final, best_V, best_W, best_err

def _find_decomposition_direct(n, m, p, rank, max_iter=3000, seed=42):
    """Find a rank-r decomposition via ALS without structural constraints."""
    np.random.seed(seed)
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iter):
        # Update U
        for a in range(n*m):
            i, j = a // m, a % m
            M = np.array([V[j*p+k] * W[k*n+i] for k in range(p)])
            if np.linalg.matrix_rank(M) >= 1:
                U[a] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
        
        # Update V
        for b in range(m*p):
            j, k = b // p, b % p
            M = np.array([U[i*m+j] * W[k*n+i] for i in range(n)])
            if np.linalg.matrix_rank(M) >= 1:
                V[b] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
        
        # Update W
        for c in range(p*n):
            k, i = c // n, c % n
            M = np.array([U[i*m+j] * V[j*p+k] for j in range(m)])
            if np.linalg.matrix_rank(M) >= 1:
                W[c] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
        
        # Check error
        err_sum = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a, b, c = i*m+j, j*p+k, k*n+i
                    val = np.dot(U[a], V[b] * W[c])
                    err_sum += (val - 1.0)**2
        err = np.sqrt(err_sum)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < 1e-8:
            break
    
    return best_U, best_V, best_W, best_err

def _strassen_fallback(n, m, p):
    """Strassen-based rank-42 decomposition for 2x2 * 2x12."""
    num_pairs = p // 2  # 6
    rank = num_pairs * 7  # 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    products = [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)]),
    ]
    output_contribs = [
        [((0,0),1),((1,1),1)],
        [((1,0),1),((1,1),-1)],
        [((0,1),1),((1,1),1)],
        [((0,0),1),((1,0),1)],
        [((0,0),-1),((0,1),1)],
        [((1,1),1)],
        [((0,0),1)],
    ]
    
    col = 0
    for t in range(num_pairs):
        base_col = 2 * t
        for s in range(7):
            for (ai,aj), coeff in products[s][0]:
                U[ai*m+aj, col] = coeff
            for (bj,bk), coeff in products[s][1]:
                V[bj*p+base_col+bk, col] = coeff
            for (ci,ck), coeff in output_contribs[s]:
                W[(base_col+ck)*n+ci, col] = coeff
            col += 1
    
    return U, V, W

def construct():
    """Rank-minimal decomposition for 2x2 * 2x12 matrix multiplication.
    
    Strategy:
    1. Try rank-20 via 4 groups of 2x2x3 (rank 5 each)
    2. Try rank-24 via row-pairing Kronecker structure
    3. Try rank-28 via 2 groups of 2x2x6 (rank 14 each)
    4. Fallback to rank-42 (Strassen on 6 pairs)
    """
    start_time = time.time()
    deadline = start_time + 145.0
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Strategy 1: Rank-20 using 4 blocks of 2x2x3 (rank 5 each)
    U3, V3, W3 = _strassen_2x2x3()
    if _verify_decomposition_fast(U3, V3, W3, 2, 2, 3):
        rank = 20  # 4 * 5
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for t in range(4):
            base_col = 3*t
            for r in range(5):
                U[:, t*5+r] = U3[:, r]
                for j in range(2):
                    for k in range(3):
                        V[j*p + base_col + k, t*5+r] = V3[j*3+k, r]
                for k in range(3):
                    for i in range(2):
                        W[(base_col+k)*n+i, t*5+r] = W3[k*2+i, r]
        
        if _verify_decomposition_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy 2: Try rank-24 via row-pairing Kronecker structure
    if time.time() < deadline - 60:
        for seed in range(20):
            if time.time() > deadline - 60:
                break
            U24, V24, W24, err = _kronecker_rowpair_decomposition(n, m, p, max_iter=1000, seed=seed)
            if err < 1e-6:
                if _verify_decomposition_fast(U24, V24, W24, n, m, p):
                    return {"U": U24.tolist(), "V": V24.tolist(), "W": W24.tolist()}
    
    # Strategy 3: Try rank-24 via direct ALS (no structural constraint)
    if time.time() < deadline - 45:
        for seed in range(15):
            if time.time() > deadline - 45:
                break
            U24d, V24d, W24d, err = _find_decomposition_direct(n, m, p, 24, max_iter=1500, seed=seed)
            if err < 1e-6:
                if _verify_decomposition_fast(U24d, V24d, W24d, n, m, p):
                    return {"U": U24d.tolist(), "V": V24d.tolist(), "W": W24d.tolist()}
    
    # Strategy 4: Try rank-28 via 2 groups of 2x2x6 (rank 14 each)
    if time.time() < deadline - 30:
        found_14 = False
        U6 = V6 = W6 = None
        for seed in range(10):
            if time.time() > deadline - 30:
                break
            U6_t, V6_t, W6_t, err = _find_decomposition_direct(2, 2, 6, 14, max_iter=800, seed=seed)
            if err < 1e-6:
                if _verify_decomposition_fast(U6_t, V6_t, W6_t, 2, 2, 6):
                    U6, V6, W6 = U6_t, V6_t, W6_t
                    found_14 = True
                    break
        
        if found_14:
            rank = 28  # 2 * 14
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            for t in range(2):
                base_col = 6*t
                for r in range(14):
                    U[:, t*14+r] = U6[:, r]
                    for j in range(2):
                        for k in range(6):
                            V[j*p + base_col + k, t*14+r] = V6[j*6+k, r]
                    for k in range(6):
                        for i in range(2):
                            W[(base_col+k)*n+i, t*14+r] = W6[k*2+i, r]
            
            if _verify_decomposition_fast(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank-42 Strassen on 6 pairs
    U, V, W = _strassen_fallback(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
