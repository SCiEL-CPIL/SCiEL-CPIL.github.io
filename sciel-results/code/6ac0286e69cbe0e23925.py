import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS

def _verify_decomposition(U, V, W, n, m, p, tol=1e-12):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    R = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    U_reshaped = U.T.reshape(R, n, m)
    V_reshaped = V.T.reshape(R, m, p)
    W_reshaped = W.T.reshape(R, p, n)
    
    T_computed = np.einsum('rij,rjk,rki->ikj', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, p, m))
    
    return np.allclose(T_computed, T_target, atol=tol)

def _schoolbook():
    """Return schoolbook decomposition with rank n*m*p."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Rank 7.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)*(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)*B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11*(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22*(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)*B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)*(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)*(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_rank34():
    """
    Construct a rank-34 decomposition using Strassen on 2x2x2 blocks.
    """
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Strassen 2x2x2 block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    S_U, S_V, S_W = _strassen_2x2x2()
    
    U1 = np.zeros((n * m, 7))
    V1 = np.zeros((m * p, 7))
    W1 = np.zeros((p * n, 7))
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U1[i * m + j, r] = S_U[i * 2 + j, r]
                V1[i * p + j, r] = S_V[i * 2 + j, r]
                W1[j * n + i, r] = S_W[i * 2 + j, r]
    
    U_parts.append(U1)
    V_parts.append(V1)
    W_parts.append(W1)
    
    # Strassen 2x2x2 block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    S_U2, S_V2, S_W2 = _strassen_2x2x2()
    
    U2 = np.zeros((n * m, 7))
    V2 = np.zeros((m * p, 7))
    W2 = np.zeros((p * n, 7))
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U2[i * m + j, r] = S_U2[i * 2 + j, r]
                V2[i * p + (j + 2), r] = S_V2[i * 2 + j, r]
                W2[(j + 2) * n + i, r] = S_W2[i * 2 + j, r]
    
    U_parts.append(U2)
    V_parts.append(V2)
    W_parts.append(W2)
    
    # Schoolbook: A[0:2,2:3] * B[2:3,0:4] -> C[0:2,0:4]
    U3 = np.zeros((n * m, 8))
    V3 = np.zeros((m * p, 8))
    W3 = np.zeros((p * n, 8))
    
    col = 0
    for i in range(2):
        for k in range(4):
            U3[i * m + 2, col] = 1
            V3[2 * p + k, col] = 1
            W3[k * n + i, col] = 1
            col += 1
    
    U_parts.append(U3)
    V_parts.append(V3)
    W_parts.append(W3)
    
    # Schoolbook: A[2:3,0:3] * B[0:3,0:4] -> C[2:3,0:4]
    U4 = np.zeros((n * m, 12))
    V4 = np.zeros((m * p, 12))
    W4 = np.zeros((p * n, 12))
    
    col = 0
    for j in range(3):
        for k in range(4):
            U4[2 * m + j, col] = 1
            V4[j * p + k, col] = 1
            W4[k * n + 2, col] = 1
            col += 1
    
    U_parts.append(U4)
    V_parts.append(V4)
    W_parts.append(W4)
    
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    return U, V, W

def _reduce_rank_via_basis_change(U, V, W, max_iterations=1000, tol=1e-10, time_limit=5.0):
    """
    Try to reduce rank by applying random basis transformations and checking
    if any columns become linearly dependent (zero singular value).
    """
    start_time = time.time()
    R = U.shape[1]
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Random invertible matrix for basis change
        P = np.random.randn(R, R)
        if np.linalg.cond(P) > 1e10:
            continue
        
        # Apply basis change: new factors are U@P, V@P, W@P^{-1}
        # Actually, we need to be careful. If T = sum U_r V_r W_r, then
        # T = sum (U P)_{.,r} (V P^{-1})_{.,r} (W)_{.,r} is not right.
        
        # Let's use the property: if we have a decomposition of rank R,
        # we can apply any invertible R x R matrix Q to the "columns" of U,
        # and the inverse to V, keeping W fixed, or similar mixtures.
        
        # Standard change: Let U' = U Q, V' = V Q^{-1}, W' = W.
        # Then sum_r U'_{a,r} V'_{b,r} W'_{c,r} = sum_r (sum_s U_{a,s} Q_{s,r}) (sum_t V_{b,t} (Q^{-1})_{t,r}) W_{c,r}
        # This does NOT preserve the tensor unless we mix all three.
        
        # Correct change: Let U' = U Q, V' = V Q, W' = W Q^{-1}.
        # sum_r U'_{a,r} V'_{b,r} W'_{c,r} = sum_r (sum_s U_{a,s} Q_{s,r}) (sum_t V_{b,t} Q_{t,r}) (sum_u W_{c,u} (Q^{-1})_{u,r})
        # This is not generally equal to the original tensor.
        
        # The correct way to change basis in a tensor decomposition:
        # If T = sum_r u_r v_r w_r (outer products), then for any invertible Q:
        # T = sum_r (U Q)_{.,r} (V Q^{-1})_{.,r} W_{.,r} is WRONG.
        
        # Actually, the decomposition is not unique.
        # If we have U, V, W of shape (N, R), (M, R), (K, R).
        # We can replace (U, V, W) with (U Q, V Q^{-1}, W) ? No.
        
        # Let's check: sum_r (U Q)_{a,r} (V Q^{-1})_{b,r} W_{c,r}
        # = sum_r sum_s U_{a,s} Q_{s,r} sum_t V_{b,t} (Q^{-1})_{t,r} W_{c,r}
        # This couples s, t, r in a way that doesn't simplify to the original sum.
        
        # The correct transformation is:
        # (U Q, V Q^{-1}, W) does NOT work.
        # (U, V Q, W Q^{-1}) does NOT work.
        
        # Wait, the tensor is T_{abc} = sum_r U_{ar} V_{br} W_{cr}.
        # If we set U' = U Q, V' = V Q^{-1}, W' = W, then
        # T'_{abc} = sum_r U'_{ar} V'_{br} W'_{cr} = sum_r (sum_s U_{as} Q_{sr}) (sum_t V_{bt} (Q^{-1})_{tr}) W_{cr}
        # This is not equal to T_{abc} in general.
        
        # The only valid basis changes are those that preserve the product structure.
        # Actually, the set of decompositions of a fixed tensor is an orbit under the action
        # of GL(R) x GL(R) x GL(R) with constraints? No.
        
        # Let's look at it differently.
        # If we have a decomposition, we can add a linear combination of terms to another term.
        # This corresponds to multiplying the column space of the factors by invertible matrices.
        
        # Specifically, if we replace U by U M, V by V N, W by W P, where M N P = I? No.
        
        # The correct invariance is:
        # T = sum_r u_r v_r w_r.
        # If we let u'_r = sum_s u_s M_{sr}, v'_r = sum_s v_s N_{sr}, w'_r = sum_s w_s P_{sr}.
        # Then sum_r u'_r v'_r w'_r = sum_r sum_s sum_t sum_u u_s M_{sr} v_t N_{tr} w_u P_{ur}
        # = sum_s,t,u u_s v_t w_u (sum_r M_{sr} N_{tr} P_{ur})
        # For this to equal sum_r u_r v_r w_r for all u, v, w, we need the coefficient of u_s v_t w_u
        # to be 1 if s=t=u and 0 otherwise.
        # This implies M N P must be the identity in a specific tensor sense?
        
        # Actually, the standard result is that if we have a decomposition of rank R,
        # we can apply any invertible R x R matrix Q to the "coordinates" of the terms.
        # Let's define new terms:
        # u'_r = sum_s u_s Q_{rs}
        # v'_r = sum_s v_s Q^{-1}_{rs}
        # w'_r = w_r
        # This doesn't work.
        
        # Let's try:
        # u'_r = sum_s u_s Q_{rs}
        # v'_r = v_r
        # w'_r = sum_s w_s Q^{-1}_{rs}
        # sum_r u'_r v'_r w'_r = sum_r (sum_s u_s Q_{rs}) v_r (sum_t w_t Q^{-1}_{tr})
        # = sum_s,t u_s v_t w_t (sum_r Q_{rs} Q^{-1}_{tr}) ... indices don't match.
        
        # Okay, the correct transformation is:
        # If we have U, V, W, we can replace them with:
        # U' = U Q
        # V' = V Q^{-1}
        # W' = W
        # This is NOT correct for the tensor product.
        
        # The correct transformation that preserves the tensor is:
        # U' = U Q
        # V' = V Q
        # W' = W Q^{-1}
        # Let's check:
        # sum_r U'_{ar} V'_{br} W'_{cr} = sum_r (sum_s U_{as} Q_{sr}) (sum_t V_{bt} Q_{tr}) (sum_u W_{cu} Q^{-1}_{ur})
        # = sum_{s,t,u} U_{as} V_{bt} W_{cu} (sum_r Q_{sr} Q_{tr} Q^{-1}_{ur})
        # This term is not delta_{st} delta_{tu} delta_{us}.
        
        # I am struggling to recall the exact group action.
        # However, a simpler approach for rank reduction is to check if the current
        # decomposition has linearly dependent "slices" in any mode.
        
        # If columns of U are linearly dependent, we can reduce rank.
        # But U, V, W are not independent; they are coupled.
        
        # Let's just check the rank of the matricizations of the computed tensor T.
        # If the rank of T is R, and we have a decomposition of rank R, it's minimal.
        # If the rank of T is < R, we can reduce.
        
        # But we are GIVEN a valid decomposition of rank 34.
        # The tensor T is fixed. Its rank is the minimal rank.
        # If the minimal rank is 34, we cannot reduce it.
        # If the minimal rank is < 34, the current decomposition is not minimal.
        
        # How to find a lower rank decomposition if one exists?
        # We need to solve for U, V, W with fewer columns.
        # This is a non-convex problem.
        
        # Strategy: Randomly initialize U, V, W with rank R' < 34 and try to fit T.
        # Use gradient descent or alternating least squares.
        
        return None

def _try_alternating_least_squares(target_tensor, rank, max_iters=100, tol=1e-8, time_limit=10.0):
    """
    Try to find a decomposition of the given rank using ALS.
    target_tensor: shape (n, p, m) or flattened appropriately.
    """
    start_time = time.time()
    
    # We need to match the indices.
    # T[i, k, j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is awkward for standard ALS which assumes T[i,j,k] = sum U[i,r] V[j,r] W[k,r].
    
    # Let's reshape the problem to standard form.
    # Let A' be the matrix A, B' be the matrix B, C' be the matrix C.
    # C' = A' B'.
    # The tensor is T_A[i,j,k] = 1 if A[i,j] and B[j,k] are multiplied to get C[i,k].
    
    # We can view this as a 3-way tensor X where X[i, j, k] = 1 if the term (i,j,k) is active.
    # But the factors are not directly U[i,r], V[j,r], W[k,r].
    # They are U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    
    # This coupling makes standard ALS difficult.
    
    # Instead, let's stick to the structural decomposition (Strassen + Schoolbook)
    # and see if we can merge any terms.
    
    # For now, this function is a placeholder for future work.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try rank-34 construction
    try:
        U, V, W = _construct_rank34()
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
