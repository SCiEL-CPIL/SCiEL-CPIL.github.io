import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS  # 2, 2, 12
    nm, mp, pn = n * m, m * p, p * n  # 4, 24, 24
    
    # Build the matrix multiplication tensor as a matrix
    # T[a, (b, c)] = 1 if C[c] = A[a] * B[b] contributes
    T = np.zeros((nm, mp * pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b * pn + c] = 1
    
    # Try to find a low-rank exact decomposition
    # Use iterative rank-1 extraction with rounding
    
    best_rank = nm * mp  # upper bound (schoolbook)
    best_U, best_V, best_W = None, None, None
    
    # Try ranks from 40 up to 42
    for target_rank in range(40, 43):
        if time.time() > deadline - 10:
            break
        
        # Try to find rank-target_rank decomposition
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))
        residual = T.copy()
        
        success = True
        for r in range(target_rank):
            if time.time() > deadline - 5:
                success = False
                break
            
            # Find best rank-1 component of residual
            # Use SVD of residual
            U_svd, s, Vt_svd = np.linalg.svd(residual, full_matrices=False)
            
            # Take top component
            u = U_svd[:, 0]
            v = Vt_svd[0, :]
            
            # Round to integers/half-integers
            u_rounded = np.round(u * 2) / 2
            v_rounded = np.round(v * 2) / 2
            
            # Check if rounded version is close enough
            if np.linalg.norm(u - u_rounded) > 0.3 or np.linalg.norm(v - v_rounded) > 0.3:
                # Try integer rounding
                u_rounded = np.round(u)
                v_rounded = np.round(v)
                if np.linalg.norm(u - u_rounded) > 0.3 or np.linalg.norm(v - v_rounded) > 0.3:
                    # Give up on this rank
                    success = False
                    break
            
            U[:, r] = u_rounded
            V[:, r] = v_rounded
            W[:, r] = 1.0  # This won't work directly
            
            # Update residual: T -= u * v^T * w (but w is 1)
            # Actually, the tensor decomposition is T = sum_r U[:,r] * V[:,r] * W[:,r]
            # In matrix form: T = U * diag(W) * V^T ... no, that's not right
            # T[a, b*pn+c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # So residual -= U[:,r] * (V[:,r] * W[:,r].reshape(mp, pn)).flatten()
            
            vw = np.outer(V[:, r], W[:, r])
            residual -= np.outer(U[:, r], vw.flatten())
        
        if success:
            # Verify
            T_recon = np.zeros_like(T)
            for r in range(target_rank):
                vw = np.outer(V[:, r], W[:, r])
                T_recon += np.outer(U[:, r], vw.flatten())
            
            if np.allclose(T, T_recon, atol=1e-9):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
    
    # Fall back to Strassen if no better found
    if best_U is None:
        num_blocks = p // 2
        rank = num_blocks * 7
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1},        {2: 1, 3: -1}),
            ({0: 1},       {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1},       {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1},        {0: -1, 1: 1}),
            ({0: -1, 2: 1},{0: 1, 1: 1},  {3: 1}),
            ({1: 1, 3: -1},{2: 1, 3: 1},  {0: 1}),
        ]
        
        for t in range(num_blocks):
            for s in range(7):
                col = t * 7 + s
                u_c, v_c, w_c = strassen[s]
                
                for a_idx, val in u_c.items():
                    U[a_idx][col] = val
                
                for local_b, val in v_c.items():
                    if local_b == 0:
                        gb = 2 * t
                    elif local_b == 1:
                        gb = 2 * t + 1
                    elif local_b == 2:
                        gb = m * p // 2 + 2 * t
                    else:
                        gb = m * p // 2 + 2 * t + 1
                    V[gb][col] = val
                
                for local_c, val in w_c.items():
                    if local_c == 0:
                        gw = (2 * t) * n + 0
                    elif local_c == 1:
                        gw = (2 * t + 1) * n + 0
                    elif local_c == 2:
                        gw = (2 * t) * n + 1
                    else:
                        gw = (2 * t + 1) * n + 1
                    W[gw][col] = val
        
        return {"U": U, "V": V, "W": W}
    else:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
# EVOLVE-BLOCK-END
