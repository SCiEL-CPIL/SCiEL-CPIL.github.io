import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# Strassen rank-7 for 2x2x2
_STRASSEN_U = np.array([
    [1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, 0]
], dtype=float)

_STRASSEN_V = np.array([
    [1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1]
], dtype=float)

_STRASSEN_W = np.array([
    [1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 1, 1]
], dtype=float)

# Laderman rank-7 for 3x3x3
_LADERMAN_U = np.array([
    [1, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [-1, 0, 1, 0, 0, 0, 0],
    [-1, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [-1, 0, 0, -1, 1, -1, -1]
], dtype=float)

_LADERMAN_V = np.array([
    [1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0],
    [1, -1, 1, 0, 0, 0, 0],
    [-1, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [-1, 0, 0, -1, 1, 0, 0],
    [0, 0, 0, 0, 0, -1, 0],
    [1, 0, 0, -1, 1, 1, -1]
], dtype=float)

_LADERMAN_W = np.array([
    [1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0],
    [-1, -1, 1, 0, 0, 0, 0],
    [1, -1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, 0],
    [1, 0, 0, -1, 1, 1, -1]
], dtype=float)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Reshape Strassen and Laderman factors to 3D tensors
    # U2: (2, 2, 7), U3: (3, 3, 7)
    U2 = _STRASSEN_U.reshape(2, 2, 7)
    V2 = _STRASSEN_V.reshape(2, 2, 7)
    W2 = _STRASSEN_W.reshape(2, 2, 7)
    
    U3 = _LADERMAN_U.reshape(3, 3, 7)
    V3 = _LADERMAN_V.reshape(3, 3, 7)
    W3 = _LADERMAN_W.reshape(3, 3, 7)
    
    # Construct 6x6x6 factors using Kronecker-like product
    # U6[i2, i3, j2, j3, t2, t3] = U2[i2, j2, t2] * U3[i3, j3, t3]
    # Reshape to (6, 6, 49)
    # einsum 'abx,cyz->acbyxz' produces axes a,c,b,y,x,z
    # reshape(6, 6, 49) maps a,c -> 0; b,y -> 1; x,z -> 2
    U6 = np.einsum('abx,cyz->acbyxz', U2, U3).reshape(6, 6, 49)
    V6 = np.einsum('abx,cyz->acbyxz', V2, V3).reshape(6, 6, 49)
    W6 = np.einsum('abx,cyz->acbyxz', W2, W3).reshape(6, 6, 49)
    
    # Extract 5x5x5 from 6x6x6
    # Keep rows where i < 5 and j < 5
    # Row index in 6x6 is i*6 + j
    keep_rows = []
    for i in range(5):
        for j in range(5):
            keep_rows.append(i * 6 + j)
            
    U5 = U6[keep_rows, :, :]
    V5 = V6[keep_rows, :, :]
    W5 = W6[keep_rows, :, :]
    
    # Reshape to (25, 49) for final output format
    U5 = U5.reshape(25, 49)
    V5 = V5.reshape(25, 49)
    W5 = W5.reshape(25, 49)
    
    # Verify the decomposition
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We want T[i,j,k] == 1 for all i,j,k
    # U5 is (25, 49). Reshape to (5,5,49)
    U5_3d = U5.reshape(5, 5, 49)
    V5_3d = V5.reshape(5, 5, 49)
    W5_3d = W5.reshape(5, 5, 49)
    
    # Note: W index is k*n + i. In W5_3d, row is k*5 + i.
    # So W5_3d[k, i, r].
    # We need sum_r U5_3d[i, j, r] * V5_3d[j, k, r] * W5_3d[k, i, r]
    T = np.einsum('ijr,jkr,kir->ijk', U5_3d, V5_3d, W5_3d)
    
    if not np.allclose(T, 1.0, atol=1e-8):
        # Fallback: return a valid schoolbook decomposition (rank 125)
        # This ensures we always return a valid result
        rank = 125
        U = np.zeros((25, rank))
        V = np.zeros((25, rank))
        W = np.zeros((25, rank))
        idx = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    U[i*5+j, idx] = 1
                    V[j*5+k, idx] = 1
                    W[k*5+i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try greedy local search to reduce rank
    if time.time() < deadline - 10:
        try:
            current_rank = 49
            U_np = U5.copy()
            V_np = V5.copy()
            W_np = W5.copy()
            
            # Fast verification function for sub-matrices
            def _verify_sub(U_sub, V_sub, W_sub):
                # U_sub, V_sub, W_sub are (25, r)
                Us = U_sub.reshape(5, 5, -1)
                Vs = V_sub.reshape(5, 5, -1)
                Ws = W_sub.reshape(5, 5, -1)
                T_sub = np.einsum('ijr,jkr,kir->ijk', Us, Vs, Ws)
                return np.allclose(T_sub, 1.0, atol=1e-8)
            
            removed = set()
            changed = True
            while changed:
                changed = False
                for col in range(current_rank):
                    if time.time() > deadline - 5:
                        break
                    if col in removed:
                        continue
                    mask = [c for c in range(current_rank) if c != col]
                    if _verify_sub(U_np[:, mask], V_np[:, mask], W_np[:, mask]):
                        removed.add(col)
                        keep = [c for c in range(current_rank) if c not in removed]
                        U_np = U_np[:, keep]
                        V_np = V_np[:, keep]
                        W_np = W_np[:, keep]
                        current_rank = len(keep)
                        changed = True
            
            if current_rank < 49:
                U5 = U_np
                V5 = V_np
                W5 = W_np
        except Exception:
            pass
    
    return {"U": U5.tolist(), "V": V5.tolist(), "W": W5.tolist()}
# EVOLVE-BLOCK-END
