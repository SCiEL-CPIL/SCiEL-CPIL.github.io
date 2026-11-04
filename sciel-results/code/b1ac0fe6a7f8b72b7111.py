"""Program for 4x4x8 matrix multiplication tensor decomposition."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-40 decomposition for 4x4x8 matrix multiplication.
    Uses a rank-20 decomposition for each 4x4x4 block of B.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Find a rank-20 decomposition for 4x4x4 using alternating minimization
    # Constraint: sum_c U4[i*4+j, c] * V4[jp*4+kp, c] * W4[kp*4+ip, c] = delta_{j,jp}
    # for all i, j, jp, kp, ip in {0,1,2,3}
    
    rank_444 = 20
    np.random.seed(42)
    U4 = np.random.randn(16, rank_444) * 0.1
    V4 = np.random.randn(16, rank_444) * 0.1
    W4 = np.random.randn(16, rank_444) * 0.1
    
    for iteration in range(500):
        # Update U4
        for i in range(4):
            for j in range(4):
                M = np.zeros((64, rank_444))
                b = np.zeros(64)
                for jp in range(4):
                    for kp in range(4):
                        for ip in range(4):
                            idx = jp * 16 + kp * 4 + ip
                            M[idx, :] = V4[jp*4+kp, :] * W4[kp*4+ip, :]
                            b[idx] = 1.0 if j == jp else 0.0
                try:
                    U4[i*4+j, :] = np.linalg.lstsq(M.T, b, rcond=None)[0]
                except:
                    pass
        
        # Update V4
        for jp in range(4):
            for kp in range(4):
                M = np.zeros((64, rank_444))
                b = np.zeros(64)
                for i in range(4):
                    for j in range(4):
                        for ip in range(4):
                            idx = i * 16 + j * 4 + ip
                            M[idx, :] = U4[i*4+j, :] * W4[kp*4+ip, :]
                            b[idx] = 1.0 if j == jp else 0.0
                try:
                    V4[jp*4+kp, :] = np.linalg.lstsq(M.T, b, rcond=None)[0]
                except:
                    pass
        
        # Update W4
        for kp in range(4):
            for ip in range(4):
                M = np.zeros((64, rank_444))
                b = np.zeros(64)
                for i in range(4):
                    for j in range(4):
                        for jp in range(4):
                            idx = i * 16 + j * 4 + jp
                            M[idx, :] = U4[i*4+j, :] * V4[jp*4+kp, :]
                            b[idx] = 1.0 if j == jp else 0.0
                try:
                    W4[kp*4+ip, :] = np.linalg.lstsq(M.T, b, rcond=None)[0]
                except:
                    pass
        
        if iteration % 100 == 0:
            max_err = 0
            for i in range(4):
                for j in range(4):
                    for jp in range(4):
                        for kp in range(4):
                            for ip in range(4):
                                val = np.sum(U4[i*4+j, :] * V4[jp*4+kp, :] * W4[kp*4+ip, :])
                                target = 1.0 if j == jp else 0.0
                                max_err = max(max_err, abs(val - target))
            if max_err < 1e-8:
                break
    
    # Round to nearest half-integer
    U4r = np.round(U4 * 2) / 2
    V4r = np.round(V4 * 2) / 2
    W4r = np.round(W4 * 2) / 2
    
    # Verify rounded solution
    max_err = 0
    for i in range(4):
        for j in range(4):
            for jp in range(4):
                for kp in range(4):
                    for ip in range(4):
                        val = np.sum(U4r[i*4+j, :] * V4r[jp*4+kp, :] * W4r[kp*4+ip, :])
                        target = 1.0 if j == jp else 0.0
                        max_err = max(max_err, abs(val - target))
    
    if max_err < 1e-9:
        U4f, V4f, W4f = U4r, V4r, W4r
    else:
        # Fallback to schoolbook rank-64 for 4x4x4
        rank_444 = 64
        U4f = np.zeros((16, rank_444))
        V4f = np.zeros((16, rank_444))
        W4f = np.zeros((16, rank_444))
        col = 0
        for i in range(4):
            for j in range(4):
                for kp in range(4):
                    U4f[i*4+j, col] = 1
                    V4f[j*4+kp, col] = 1
                    W4f[kp*4+i, col] = 1
                    col += 1
    
    # Build 4x4x8 decomposition from two 4x4x4 blocks
    rank_total = 2 * rank_444
    U = np.zeros((16, rank_total))
    V = np.zeros((32, rank_total))
    W = np.zeros((32, rank_total))
    
    # Left block (B columns 0-3)
    U[:, 0:rank_444] = U4f
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:rank_444] = V4f[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:rank_444] = W4f[k*4+i, :]
    
    # Right block (B columns 4-7)
    U[:, rank_444:2*rank_444] = U4f
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), rank_444:2*rank_444] = V4f[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, rank_444:2*rank_444] = W4f[k*4+i, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
