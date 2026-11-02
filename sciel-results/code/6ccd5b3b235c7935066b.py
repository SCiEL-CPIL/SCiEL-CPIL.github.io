"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import numpy as np
import time
import random
from itertools import product

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _snap_to_half_integer(x, tol=1e-9):
    """Snap a value to the nearest half-integer if close."""
    half = round(x * 2) / 2.0
    if abs(x - half) < tol:
        return half
    return x

def _compute_error(U, V, W, n, m, p):
    """Compute max error of the decomposition."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                max_err = max(max_err, abs(s - 1.0))
    return max_err

def _strassen_222():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices:
    U: i*2 + j
    V: j*2 + k
    W: k*2 + i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    # A00: U[0,0]=1, A11: U[3,0]=1
    # B00: V[0,0]=1, B11: V[3,0]=1
    # C00 += M1, C11 += M1
    # W[0,0]=1 (C00), W[3,0]=1 (C11)
    
    # M2 = (A10+A11)B00
    # A10: U[2,1]=1, A11: U[3,1]=1
    # B00: V[0,1]=1
    # C10 += M2
    # W[2,1]=1 (C10)
    
    # M3 = A00(B01-B11)
    # A00: U[0,2]=1
    # B01: V[1,2]=1, B11: V[3,2]=-1
    # C01 += M3
    # W[1,2]=1 (C01)
    
    # M4 = A11(B10-B00)
    # A11: U[3,3]=1
    # B10: V[2,3]=1, B00: V[0,3]=-1
    # C00 += M4, C10 += M4
    # W[0,3]=1, W[2,3]=1
    
    # M5 = (A00+A01)B11
    # A00: U[0,4]=1, A01: U[1,4]=1
    # B11: V[3,4]=1
    # C00 -= M5, C01 += M5
    # W[0,4]=-1, W[1,4]=1
    
    # M6 = (A10-A00)(B00+B01)
    # A10: U[2,5]=1, A00: U[0,5]=-1
    # B00: V[0,5]=1, B01: V[1,5]=1
    # C11 += M6
    # W[3,5]=1
    
    # M7 = (A01-A11)(B10+B11)
    # A01: U[1,6]=1, A11: U[3,6]=-1
    # B10: V[2,6]=1, B11: V[3,6]=1
    # C00 += M7, C11 += M7
    # W[0,6]=1, W[3,6]=1

    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1

    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1

    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1

    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1

    return U, V, W

def _build_226():
    """
    Build 2x2x6 decomposition using 3 Strassen blocks.
    U shape (4, 21), V shape (12, 21), W shape (12, 21).
    """
    U222, V222, W222 = _strassen_222()
    U = np.zeros((4, 21))
    V = np.zeros((12, 21))
    W = np.zeros((12, 21))
    
    for block in range(3):
        offset = block * 7
        U[:, offset:offset+7] = U222
        # V rows: j in {0,1}, k in {0..5}
        # Block 0: k=0,1 -> V rows 0,1
        # Block 1: k=2,3 -> V rows 2,3
        # Block 2: k=4,5 -> V rows 4,5
        V[block*2:(block+1)*2, offset:offset+7] = V222
        # W rows: k in {0..5}, i in {0,1}
        # W index: k*2 + i
        # Block 0: k=0,1 -> W rows 0,1,2,3
        # Block 1: k=2,3 -> W rows 4,5,6,7
        # Block 2: k=4,5 -> W rows 8,9,10,11
        W[block*4:(block+1)*4, offset:offset+7] = W222
        
    return U, V, W

def _search_233(target_rank, time_limit):
    """
    Search for 2x3x3 decomposition with given rank.
    Returns (U, V, W) or None.
    """
    np.random.seed(42)
    n, m, p = 2, 3, 3
    start = time.time()
    
    for _ in range(2000):
        if time.time() > time_limit:
            break
            
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # Normalize
        for c in range(target_rank):
            nU = np.linalg.norm(U[:, c])
            nV = np.linalg.norm(V[:, c])
            nW = np.linalg.norm(W[:, c])
            if nU < 1e-10 or nV < 1e-10 or nW < 1e-10:
                continue
            scale = (nU * nV * nW) ** (-1/3)
            U[:, c] *= scale
            V[:, c] *= scale
            W[:, c] *= scale
            
        err = _compute_error(U, V, W, n, m, p)
        if err < 0.1:
            # Refine
            for _ in range(50):
                if time.time() > time_limit:
                    break
                c = np.random.randint(target_rank)
                U[:, c] += 0.1 * np.random.randn(n*m)
                V[:, c] += 0.1 * np.random.randn(m*p)
                W[:, c] += 0.1 * np.random.randn(p*n)
                err = _compute_error(U, V, W, n, m, p)
                if err < 1e-8:
                    # Snap
                    U_s = np.array([[_snap_to_half_integer(x) for x in row] for row in U])
                    V_s = np.array([[_snap_to_half_integer(x) for x in row] for row in V])
                    W_s = np.array([[_snap_to_half_integer(x) for x in row] for row in W])
                    if _verify(U_s, V_s, W_s, n, m, p):
                        return U_s, V_s, W_s
                    break
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # 1. Build 2x2x6 part
    U226, V226, W226 = _build_226()
    
    # 2. Search for 2x3x3 part
    # Try ranks 13, 14, 15, 18 (schoolbook)
    U233 = None
    V233 = None
    W233 = None
    
    for rank in [13, 14, 15]:
        if time.time() > deadline - 30:
            break
        res = _search_233(rank, deadline - 30)
        if res:
            U233, V233, W233 = res
            break
            
    if U233 is None:
        # Fall back to schoolbook for 2x3x3
        # Rank 18
        U233 = np.zeros((6, 18))
        V233 = np.zeros((9, 18))
        W233 = np.zeros((6, 18))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U233[i*3+j, col] = 1
                    V233[j*3+k, col] = 1
                    W233[k*2+i, col] = 1
                    col += 1
                    
    # 3. Extend 2x3x3 to 2x3x6
    # Repeat the block twice
    rank_233 = U233.shape[1]
    U236 = np.zeros((6, rank_233 * 2))
    V236 = np.zeros((18, rank_233 * 2))
    W236 = np.zeros((12, rank_233 * 2))
    
    for block in range(2):
        offset = block * rank_233
        U236[:, offset:offset+rank_233] = U233
        # V rows: j in {0,1,2}, k in {0..5}
        # Block 0: k=0,1,2 -> V rows 0,1,2 (j=0), 3,4,5 (j=1), 6,7,8 (j=2)
        # Wait, V index is j*p + k. p=3.
        # Block 0 (k=0,1,2):
        # j=0: V[0,1,2]
        # j=1: V[3,4,5]
        # j=2: V[6,7,8]
        # V233 has rows 0..8.
        # So V233 rows 0..8 map to V236 rows 0..8 for block 0.
        V236[block*9:(block+1)*9, offset:offset+rank_233] = V233
        
        # W rows: k in {0..5}, i in {0,1}
        # W index k*2 + i.
        # Block 0: k=0,1,2 -> W rows 0..5
        W236[block*6:(block+1)*6, offset:offset+rank_233] = W233
        
    # 4. Combine 2x3x6 and 2x2x6
    # Global U: 10 rows. Top 6 from 236, bottom 4 from 226.
    # Global V: 30 rows. Top 18 from 236, bottom 12 from 226.
    # Global W: 12 rows. Sum of 236 and 226 contributions?
    # No, we concatenate columns.
    # The tensor is T = T236 + T226.
    # T236 uses columns 0..R236-1.
    # T226 uses columns R236..R236+R226-1.
    
    R_total = U236.shape[1] + U226.shape[1]
    
    U_global = np.zeros((10, R_total))
    V_global = np.zeros((30, R_total))
    W_global = np.zeros((12, R_total))
    
    # Fill 236 part
    U_global[0:6, 0:U236.shape[1]] = U236
    V_global[0:18, 0:V236.shape[1]] = V236
    W_global[0:12, 0:W236.shape[1]] = W236
    
    # Fill 226 part
    offset = U236.shape[1]
    U_global[6:10, offset:offset+U226.shape[1]] = U226
    V_global[18:30, offset:offset+V226.shape[1]] = V226
    W_global[0:12, offset:offset+W226.shape[1]] = W226
    
    # Verify
    if _verify(U_global, V_global, W_global, n, m, p):
        return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
    else:
        # Fallback to schoolbook 60
        rank = 60
        U = [[0] * rank for _ in range(10)]
        V = [[0] * rank for _ in range(30)]
        W = [[0] * rank for _ in range(12)]
        col = 0
        for i in range(2):
            for j in range(5):
                for k in range(6):
                    U[i*5+j][col] = 1
                    V[j*6+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
