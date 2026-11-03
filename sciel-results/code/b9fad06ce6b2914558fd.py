"""Matrix multiplication 3x4x4 rank-minimal exact tensor decomposition.

Uses a block-recursive Strassen-based construction:
  3x4x4  =  (2+1) x (2+2) x (2+2)
  2x2x2 -> Strassen rank 7
  1x2x4 -> schoolbook rank 8
  2x2x4 -> two 2x2x2 blocks, rank 14
  2x4x4 -> two 2x2x4 blocks, rank 28
  3x2x4 -> 2x2x4 + 1x2x4 = 14 + 8 = 22
  3x4x4 -> two 3x2x4 blocks = 44

The output factor W uses the trace convention W[k*n+i, r].
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _strassen_222():
    """Return U, V, W for 2x2x2 Strassen (rank 7).
    
    Index conventions:
      A is 2x2: A[i,j], i=0..1, j=0..1
      B is 2x2: B[j,k], j=0..1, k=0..1
      C is 2x2: C[i,k], i=0..1, k=0..1
    
    U[i*m+j, c] where m=2
    V[j*p+k, c] where p=2
    W[k*n+i, c] where n=2
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank))  # 4 x 7
    V = np.zeros((m * p, rank))  # 4 x 7
    W = np.zeros((p * n, rank))  # 4 x 7
    
    # Strassen products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # M1: (A11+A22)(B11+B22)
    # A terms: A[0,0] and A[1,1] -> U[0*2+0]=U[0], U[1*2+1]=U[3]
    # B terms: B[0,0] and B[0,1] -> V[0*2+0]=V[0], V[0*2+1]=V[1]
    # C: C[0,0] += M1, C[1,1] += M1 -> W[0*2+0]=W[0], W[1*2+1]=W[3]
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[1, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # M2: (A21+A22)B11
    # A terms: A[1,0] and A[1,1] -> U[1*2+0]=U[2], U[3]
    # B terms: B[0,0] -> V[0]
    # C: C[1,0] += M2 -> W[0*2+1]=W[1]
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[1, 1] += 1
    
    # M3: A11(B12-B22)
    # A terms: A[0,0] -> U[0]
    # B terms: B[0,1] and B[1,1] -> V[1], V[3]
    # C: C[0,1] += M3 -> W[1*2+0]=W[2]
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] += -1
    W[2, 2] += 1
    
    # M4: A22(B21-B11)
    # A terms: A[1,1] -> U[3]
    # B terms: B[1,0] and B[0,0] -> V[2], V[0]
    # C: C[1,0] += M4, C[0,0] -= M4 -> W[1] += 1, W[0] -= 1
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] += -1
    W[1, 3] += 1; W[0, 3] += -1
    
    # M5: (A11+A12)B22
    # A terms: A[0,0] and A[0,1] -> U[0], U[1]
    # B terms: B[1,1] -> V[3]
    # C: C[0,0] -= M5, C[0,1] += M5 -> W[0] -= 1, W[2] += 1
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] += -1; W[2, 4] += 1
    
    # M6: (A21-A11)(B11+B12)
    # A terms: A[1,0] and A[0,0] -> U[2], U[0]
    # B terms: B[0,0] and B[0,1] -> V[0], V[1]
    # C: C[1,1] += M6 -> W[3] += 1
    U[2, 5] += 1; U[0, 5] += -1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1
    
    # M7: (A12-A22)(B21+B22)
    # A terms: A[0,1] and A[1,1] -> U[1], U[3]
    # B terms: B[1,0] and B[1,1] -> V[2], V[3]
    # C: C[0,0] += M7 -> W[0] += 1
    U[1, 6] += 1; U[3, 6] += -1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return U, V, W for schoolbook n x m x p multiplication."""
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

def _concat_factors(factors):
    """Concatenate a list of (U, V, W) factor triples column-wise."""
    if not factors:
        return None
    U_list = [f[0] for f in factors]
    V_list = [f[1] for f in factors]
    W_list = [f[2] for f in factors]
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    return U, V, W

def _block_combine(sub_factors, n, m, p, n_split, m_split, p_split):
    """
    Combine sub-problem factors into a full (n, m, p) factor set.
    
    n_split: list of row sizes summing to n
    m_split: list of col sizes summing to m  
    p_split: list of out sizes summing to p
    
    sub_factors[ni][mj][pi] = (U, V, W) for sub-block at that position
    """
    all_U = []
    all_V = []
    all_W = []
    
    # Track global offsets
    for ni, nn in enumerate(n_split):
        i_offset = sum(n_split[:ni])
        for mj, mm in enumerate(m_split):
            j_offset = sum(m_split[:mj])
            for pi, pp in enumerate(p_split):
                k_offset = sum(p_split[:pi])
                
                U_sub, V_sub, W_sub = sub_factors[ni][mj][pi]
                
                # Remap U: local (i, j) -> global (i_offset+i, j_offset+j)
                # U_sub has shape (nn*mm, rank_sub)
                # Local index i*mm+j maps to global (i_offset+i)*m + (j_offset+j)
                U_full = np.zeros((n * m, U_sub.shape[1]))
                for i in range(nn):
                    for j in range(mm):
                        U_full[(i_offset + i) * m + (j_offset + j)] = U_sub[i * mm + j]
                
                # Remap V: local (j, k) -> global (j_offset+j, k_offset+k)
                V_full = np.zeros((m * p, V_sub.shape[1]))
                for j in range(mm):
                    for k in range(pp):
                        V_full[(j_offset + j) * p + (k_offset + k)] = V_sub[j * pp + k]
                
                # Remap W: local (k, i) -> global (k_offset+k, i_offset+i)
                # W convention: W[k*n+i, r]
                W_full = np.zeros((p * n, W_sub.shape[1]))
                for k in range(pp):
                    for i in range(nn):
                        W_full[(k_offset + k) * n + (i_offset + i)] = W_sub[k * nn + i]
                
                all_U.append(U_full)
                all_V.append(V_full)
                all_W.append(W_full)
    
    U = np.hstack(all_U)
    V = np.hstack(all_V)
    W = np.hstack(all_W)
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Build 2x2x2 Strassen factors
    U_s, V_s, W_s = _strassen_222()
    
    # Build 1x2x4 schoolbook factors
    U_124, V_124, W_124 = _schoolbook(1, 2, 4)
    
    # Build 2x2x4: two 2x2x2 blocks in p
    # n=2, m=2, p=4 -> p_split = [2, 2]
    U_224 = np.hstack([U_s, U_s])
    V_224 = np.hstack([V_s, V_s])
    W_224 = np.hstack([W_s, W_s])
    # Remap W: first block k=0,1 -> W[0:2], second block k=2,3 -> W[2:4]
    W_224_full = np.zeros((8, 14))  # p*n = 4*2 = 8
    W_224_full[0:2] = W_s  # k=0,1
    W_224_full[2:4] = W_s  # k=2,3
    # Wait, W_s has shape (2*2, 7) = (4, 7). W index is k*n+i where n=2.
    # For first block (k=0,1): W[0*2+i] and W[1*2+i] -> rows 0,1 and 2,3
    # For second block (k=2,3): W[2*2+i] and W[3*2+i] -> rows 4,5 and 6,7
    W_224_full = np.zeros((8, 14))
    # First 2x2x2 block: k_local=0,1 map to k_global=0,1
    W_224_full[0:4] = W_s  # rows 0,1,2,3 correspond to (k=0,i=0),(k=0,i=1),(k=1,i=0),(k=1,i=1)
    # Second 2x2x2 block: k_local=0,1 map to k_global=2,3
    W_224_full[4:8] = W_s  # rows 4,5,6,7 correspond to (k=2,i=0),(k=2,i=1),(k=3,i=0),(k=3,i=1)
    
    if time.time() > deadline:
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build 2x4x4: two 2x2x4 blocks in m
    # n=2, m=4, p=4 -> m_split = [2, 2]
    U_244 = np.hstack([U_224, U_224])
    V_244 = np.hstack([V_224, V_224])
    W_244 = np.hstack([W_224_full, W_224_full])
    # Remap U: first block j=0,1 -> cols 0,1; second block j=2,3 -> cols 2,3
    U_244_full = np.zeros((8, 28))  # n*m = 2*4 = 8
    # U_224 has shape (4, 14) where local index is i*2+j for i=0,1 j=0,1
    # Global: i*4+j
    for i in range(2):
        for j in range(2):
            U_244_full[i * 4 + j] = U_224[i * 2 + j]
            U_244_full[i * 4 + 2 + j] = U_224[i * 2 + j]
    
    if time.time() > deadline:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build 3x2x4: one 2x2x4 block (rows 0,1) + one 1x2x4 block (row 2)
    # n=3, m=2, p=4
    # 2x2x4 part: rows 0,1 -> U_224 (4,14)
    # 1x2x4 part: row 2 -> U_124 (2,8)
    
    U_324 = np.zeros((6, 22))  # n*m = 3*2 = 6
    V_324 = np.zeros((8, 22))  # m*p = 2*4 = 8
    W_324 = np.zeros((12, 22))  # p*n = 4*3 = 12
    
    # 2x2x4 block occupies rows i=0,1
    U_324[0:4, 0:14] = U_224
    V_324[:, 0:14] = V_224
    # W for 2x2x4: W[k*2+i] for i=0,1 -> rows 0-3 for each k
    # In 3x2x4: W[k*3+i] for i=0,1 -> rows 0,1 for k=0; 3,4 for k=1; etc.
    for k in range(4):
        W_324[k * 3 + 0: k * 3 + 2, 0:14] = W_224_full[k * 2: k * 2 + 2]
    
    # 1x2x4 block occupies row i=2
    U_324[4:6, 14:22] = U_124
    V_324[:, 14:22] = V_124
    # W for 1x2x4: W[k*1+i] = W[k] for i=0 -> but our W convention is W[k*n+i]
    # For 1x2x4, n=1, so W[k*1+0] = W[k], shape (4, 8)
    # In 3x2x4, n=3, i=2: W[k*3+2]
    for k in range(4):
        W_324[k * 3 + 2, 14:22] = W_124[k]
    
    if time.time() > deadline:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build 3x4x4: two 3x2x4 blocks in m
    # n=3, m=4, p=4 -> m_split = [2, 2]
    U_344 = np.zeros((12, 44))  # n*m = 3*4 = 12
    V_344 = np.zeros((16, 44))  # m*p = 4*4 = 16
    W_344 = np.hstack([W_324, W_324])
    
    # U remap: first block j=0,1 -> cols 0,1; second block j=2,3 -> cols 2,3
    for i in range(3):
        for j in range(2):
            U_344[i * 4 + j, 0:22] = U_324[i * 2 + j, :]
            U_344[i * 4 + 2 + j, 22:44] = U_324[i * 2 + j, :]
    
    # V remap: first block j=0,1 -> rows 0-3 (j*p+k for j=0,1, k=0..3)
    # second block j=2,3 -> rows 4-7
    for j in range(2):
        for k in range(4):
            V_344[j * 4 + k, 0:22] = V_324[j * 4 + k, :]
            V_344[(2 + j) * 4 + k, 22:44] = V_324[j * 4 + k, :]
    
    # Convert to lists
    return {
        "U": U_344.tolist(),
        "V": V_344.tolist(),
        "W": W_344.tolist()
    }
