import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _get_strassen_2x2():
    """Strassen's rank-7 algorithm for 2x2 matrix multiplication."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    # m0 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # m1 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    # m2 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    # m3 = A11(B10-B00)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1
    # m4 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    # m5 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # m6 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    return U, V, W

def _get_schoolbook_3x3():
    """Schoolbook rank-27 algorithm for 3x3 matrix multiplication."""
    U = np.zeros((9, 27), dtype=float)
    V = np.zeros((9, 27), dtype=float)
    W = np.zeros((9, 27), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = i * 9 + j * 3 + k
                U[i * 3 + j, r] = 1
                V[j * 3 + k, r] = 1
                W[k * 3 + i, r] = 1
    return U, V, W

def _compute_3x3_tensor():
    """Compute the 3x3 matrix multiplication tensor T[i,j,k] = 1."""
    T = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T[i, j, k] = 1.0
    return T

def _find_rank1_decomposition(T, max_iter=1000):
    """
    Find a rank-1 decomposition of a 3x3 tensor T.
    Returns (u, v, w, scale) such that T ≈ scale * u ⊗ v ⊗ w.
    """
    u = np.random.randn(3)
    v = np.random.randn(3)
    w = np.random.randn(3)
    
    for _ in range(max_iter):
        u_new = np.zeros(3)
        for i in range(3):
            u_new[i] = np.sum(T[i, :, :] * np.outer(v, w))
        
        v_new = np.zeros(3)
        for j in range(3):
            v_new[j] = np.sum(T[:, j, :] * np.outer(u_new, w))
        
        w_new = np.zeros(3)
        for k in range(3):
            w_new[k] = np.sum(T[:, :, k] * np.outer(u_new, v_new))
        
        norm_u = np.linalg.norm(u_new)
        norm_v = np.linalg.norm(v_new)
        norm_w = np.linalg.norm(w_new)
        
        if norm_u < 1e-15 or norm_v < 1e-15 or norm_w < 1e-15:
            return None
        
        u = u_new / norm_u
        v = v_new / norm_v
        w = w_new / norm_w
        
        T_approx = np.outer(np.outer(u, v), w)
        if np.linalg.norm(T - T_approx) < 1e-12:
            scale = np.sum(T * T_approx)
            if abs(scale) < 1e-15:
                return None
            return u, v, w, scale
    
    return None

def _find_rank_decomposition(T, target_rank, max_attempts=500, max_iter=500):
    """
    Find a rank-r decomposition of tensor T using iterative deflation.
    """
    U = np.zeros((3, target_rank))
    V = np.zeros((3, target_rank))
    W = np.zeros((3, target_rank))
    
    T_remaining = T.copy()
    
    for r in range(target_rank):
        best_result = None
        best_residual = float('inf')
        
        for attempt in range(max_attempts):
            np.random.seed(42 + r * 10000 + attempt)
            result = _find_rank1_decomposition(T_remaining, max_iter=max_iter)
            if result is None:
                continue
            
            u, v, w, scale = result
            T_approx = scale * np.outer(np.outer(u, v), w)
            residual = np.linalg.norm(T_remaining - T_approx)
            
            if residual < best_residual:
                best_residual = residual
                best_result = result
        
        if best_result is None:
            return None
        
        u, v, w, scale = best_result
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
        
        T_remaining = T_remaining - scale * np.outer(np.outer(u, v), w)
        
        if np.linalg.norm(T_remaining) < 1e-8:
            break
    
    return U, V, W

def _build_6x6_from_2x2_3x3(U2, V2, W2, U3, V3, W3):
    """
    Combine a 2x2 algorithm (U2,V2,W2 of shape (4, r2)) with a 3x3 algorithm
    (U3,V3,W3 of shape (9, r3)) to get a 6x6 algorithm of rank r2*r3.
    """
    r2 = U2.shape[1]
    r3 = U3.shape[1]
    rank = r2 * r3
    
    U6 = np.zeros((36, rank))
    V6 = np.zeros((36, rank))
    W6 = np.zeros((36, rank))
    
    for r2_idx in range(r2):
        for r3_idx in range(r3):
            r = r2_idx * r3 + r3_idx
            
            # U component
            for i2 in range(2):
                for j2 in range(2):
                    val2 = U2[i2*2+j2, r2_idx]
                    if val2 != 0:
                        for i3 in range(3):
                            for j3 in range(3):
                                val3 = U3[i3*3+j3, r3_idx]
                                if val3 != 0:
                                    U6[(i2*3+i3)*6 + (j2*3+j3), r] += val2 * val3
            
            # V component
            for j2 in range(2):
                for k2 in range(2):
                    val2 = V2[j2*2+k2, r2_idx]
                    if val2 != 0:
                        for j3 in range(3):
                            for k3 in range(3):
                                val3 = V3[j3*3+k3, r3_idx]
                                if val3 != 0:
                                    V6[(j2*3+j3)*6 + (k2*3+k3), r] += val2 * val3
            
            # W component
            for k2 in range(2):
                for i2 in range(2):
                    val2 = W2[k2*2+i2, r2_idx]
                    if val2 != 0:
                        for k3 in range(3):
                            for i3 in range(3):
                                val3 = W3[k3*3+i3, r3_idx]
                                if val3 != 0:
                                    W6[(k2*3+k3)*6 + (i2*3+i3), r] += val2 * val3
    
    return U6, V6, W6

def _round_to_half(x):
    """Round to nearest 0.5"""
    return np.round(2 * x) / 2.0

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0
    
    U2, V2, W2 = _get_strassen_2x2()
    
    # Try to find a rank 23 3x3 decomposition
    T = _compute_3x3_tensor()
    
    for target_rank in [23, 24, 27]:
        if time.time() >= deadline - 30:
            break
            
        best_U3 = None
        best_V3 = None
        best_W3 = None
        
        # Try multiple seeds
        for seed in range(50):
            if time.time() >= deadline - 20:
                break
            np.random.seed(seed)
            result = _find_rank_decomposition(T, target_rank, max_attempts=100, max_iter=300)
            if result is not None:
                U3, V3, W3 = result
                # Verify the 3x3 decomposition
                if _verify_decomposition(U3, V3, W3, 3, 3, 3):
                    best_U3 = U3
                    best_V3 = V3
                    best_W3 = W3
                    break
        
        if best_U3 is not None:
            U6, V6, W6 = _build_6x6_from_2x2_3x3(U2, V2, W2, best_U3, best_V3, best_W3)
            
            # Round to half-integers
            U6 = _round_to_half(U6)
            V6 = _round_to_half(V6)
            W6 = _round_to_half(W6)
            
            # Verify the full 6x6 decomposition with rounded values
            if _verify_decomposition(U6, V6, W6, 6, 6, 6):
                return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
    
    # Fallback to schoolbook 3x3 (rank 27)
    U3, V3, W3 = _get_schoolbook_3x3()
    U6, V6, W6 = _build_6x6_from_2x2_3x3(U2, V2, W2, U3, V3, W3)
    
    U6 = _round_to_half(U6)
    V6 = _round_to_half(V6)
    W6 = _round_to_half(W6)
    
    if _verify_decomposition(U6, V6, W6, 6, 6, 6):
        return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
    
    # Final fallback: Pure schoolbook 6x6
    U = np.zeros((36, 216))
    V = np.zeros((36, 216))
    W = np.zeros((36, 216))
    for i in range(6):
        for j in range(6):
            for k in range(6):
                r = i * 36 + j * 6 + k
                U[i*6+j, r] = 1
                V[j*6+k, r] = 1
                W[k*6+i, r] = 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
