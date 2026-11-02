import time
import numpy as np
import random

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 3 matrix by a 3 by 5 matrix.
    
    Strategy: Alternating Least Squares (ALS) with integer rounding to find 
    a decomposition of rank < 30.
    """
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify exact decomposition."""
        r = len(U[0])
        # Check all non-zero entries of the target tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    I = i * m + j
                    J = j * p + k
                    K = k * n + i
                    s = 0
                    for c in range(r):
                        s += U[I][c] * V[J][c] * W[K][c]
                    if abs(s - 1) > 1e-9:
                        return False
        # Check that other entries are 0? 
        # The tensor T is sparse. If the sum is 1 at the 30 positions, 
        # and the rank is low, it might leak.
        # We must ensure T[I,J,K] = 0 for invalid (I,J,K).
        # However, the problem statement says "exactly decompose the matrix multiplication tensor".
        # The matrix multiplication tensor is defined by the multiplication map.
        # If the map is correct, the tensor is correct.
        # The map is correct if C_ik = sum_j A_ij B_jk.
        # This is guaranteed if the 30 specific components are 1 and the "cross terms" cancel or are 0.
        # Actually, if we only check the 30 positions, we might miss errors where 
        # U,V,W produce non-zero values at other (I,J,K) which would corrupt the result.
        # We must verify the full tensor equality.
        
        # To be safe, we check all 900 entries.
        for I in range(nm):
            for J in range(mp):
                for K in range(pn):
                    s = 0
                    for c in range(r):
                        s += U[I][c] * V[J][c] * W[K][c]
                    # Determine target value
                    # I = i*m + j, J = j'*p + k, K = k'*n + i'
                    # Target is 1 if I=(i,j), J=(j,k), K=(k,i) i.e., j==j' and i==i' and k==k'
                    # Let's extract indices
                    i1 = I // m
                    j1 = I % m
                    j2 = J // p
                    k1 = J % p
                    k2 = K // n
                    i2 = K % n
                    
                    if j1 == j2 and i1 == i2 and k1 == k2:
                        target = 1
                    else:
                        target = 0
                        
                    if abs(s - target) > 1e-9:
                        return False
        return True

    def round_to_half(x):
        return np.round(x * 2) / 2.0

    def try_rank(R, max_iters=50, restarts=20):
        for _ in range(restarts):
            # Initialize with random small integers
            U = np.random.randint(-2, 3, (nm, R)).astype(float)
            V = np.random.randint(-2, 3, (mp, R)).astype(float)
            W = np.random.randint(-2, 3, (pn, R)).astype(float)
            
            # Precompute target tensor
            T = np.zeros((nm, mp, pn))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        I = i * m + j
                        J = j * p + k
                        K = k * n + i
                        T[I, J, K] = 1.0
            
            T_flat = T.flatten()
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Solve for U
                # T[I,J,K] = sum_c U[I,c] V[J,c] W[K,c]
                # For fixed V, W, this is linear in U.
                # We can reshape the problem.
                # Let M_U be a matrix of size (nm*mp*pn) x (nm*R)
                # This is too big (900 x 168). It's fine.
                
                # Construct M_U
                # Index in flat T: idx = I*mp*pn + J*pn + K
                # Index in flat U: idx_u = I*R + c
                # Coefficient for U[I,c] is V[J,c] * W[K,c]
                
                M_U = np.zeros((nm * mp * pn, nm * R))
                for I in range(nm):
                    for J in range(mp):
                        for K in range(pn):
                            idx_t = I * mp * pn + J * pn + K
                            for c in range(R):
                                idx_u = I * R + c
                                M_U[idx_t, idx_u] = V[J, c] * W[K, c]
                
                # Solve M_U * u_flat = T_flat
                try:
                    u_flat, residuals, rank_u, s = np.linalg.lstsq(M_U, T_flat, rcond=None)
                except:
                    continue
                    
                U_new = u_flat.reshape((nm, R))
                U_new = round_to_half(U_new)
                
                # Solve for V
                M_V = np.zeros((nm * mp * pn, mp * R))
                for I in range(nm):
                    for J in range(mp):
                        for K in range(pn):
                            idx_t = I * mp * pn + J * pn + K
                            for c in range(R):
                                idx_v = J * R + c
                                M_V[idx_t, idx_v] = U_new[I, c] * W[K, c]
                
                try:
                    v_flat, residuals, rank_v, s = np.linalg.lstsq(M_V, T_flat, rcond=None)
                except:
                    continue
                    
                V_new = v_flat.reshape((mp, R))
                V_new = round_to_half(V_new)
                
                # Solve for W
                M_W = np.zeros((nm * mp * pn, pn * R))
                for I in range(nm):
                    for J in range(mp):
                        for K in range(pn):
                            idx_t = I * mp * pn + J * pn + K
                            for c in range(R):
                                idx_w = K * R + c
                                M_W[idx_t, idx_w] = U_new[I, c] * V_new[J, c]
                
                try:
                    w_flat, residuals, rank_w, s = np.linalg.lstsq(M_W, T_flat, rcond=None)
                except:
                    continue
                    
                W_new = w_flat.reshape((pn, R))
                W_new = round_to_half(W_new)
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                if verify(U.tolist(), V.tolist(), W.tolist()):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    
            # End iterations
        return None

    # Try ranks lower than 30
    for R in [28, 29]:
        res = try_rank(R)
        if res:
            return res
            
    return schoolbook()
