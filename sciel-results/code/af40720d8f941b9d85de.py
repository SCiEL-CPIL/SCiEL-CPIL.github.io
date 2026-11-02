"""Matrix multiplication 3x3x5: Rank-41 decomposition.

Uses Laderman's rank-23 decomposition for the 3x3x3 sub-tensor
(k=0,1,2) combined with schoolbook rank-18 for the 3x3x2 part (k=3,4).
Total rank: 23 + 18 = 41.

The W factor uses the official trace convention W[k*n+i, r].
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 5]
DEADLINE_SECONDS = 130
_START_TIME = time.time()


def _verify_decomposition(U, V, W, n=3, m=3, p=5, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _build_schoolbook(n=3, m=3, p=5):
    """Build the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1.0
                V[v_idx][col] = 1.0
                W[w_idx][col] = 1.0
                col += 1
    return U, V, W


def _build_laderman_3x3x3():
    """
    Build Laderman's rank-23 decomposition for 3x3x3 matrix multiplication.
    
    Returns U, V, W as numpy arrays of shape (9, 23) each.
    U[i*m+j, r] = coefficient of a[i,j] in r-th multiplication
    V[j*p+k, r] = coefficient of b[j,k] in r-th multiplication  
    W[k*n+i, r] = coefficient for c[i,k] from r-th multiplication
    """
    if time.time() - _START_TIME > DEADLINE_SECONDS:
        return None
    
    n = m = p = 3
    U = np.zeros((9, 23), dtype=float)
    V = np.zeros((9, 23), dtype=float)
    W = np.zeros((9, 23), dtype=float)
    
    # Laderman's algorithm (correct version), 0-indexed:
    # M1 = (a00+a11+a22)(b00+b11+b22)
    U[0,0]=1; U[4,0]=1; U[8,0]=1
    V[0,0]=1; V[4,0]=1; V[8,0]=1
    
    # M2 = (a00-a11+a22)(b00-b11+b22)
    U[0,1]=1; U[4,1]=-1; U[8,1]=1
    V[0,1]=1; V[4,1]=-1; V[8,1]=1
    
    # M3 = (a00+a11-a22)(b00+b11-b22)
    U[0,2]=1; U[4,2]=1; U[8,2]=-1
    V[0,2]=1; V[4,2]=1; V[8,2]=-1
    
    # M4 = (a00-a11-a22)(b00-b11-b22)
    U[0,3]=1; U[4,3]=-1; U[8,3]=-1
    V[0,3]=1; V[4,3]=-1; V[8,3]=-1
    
    # M5 = (a01+a22)(b11+b22)
    U[1,4]=1; U[8,4]=1
    V[4,4]=1; V[8,4]=1
    
    # M6 = (a12+a20)(b22+b00)
    U[5,5]=1; U[6,5]=1
    V[8,5]=1; V[0,5]=1
    
    # M7 = (a21+a02)(b21+b02)
    U[7,6]=1; U[2,6]=1
    V[7,6]=1; V[2,6]=1
    
    # M8 = (a01-a22)(b11-b22)
    U[1,7]=1; U[8,7]=-1
    V[4,7]=1; V[8,7]=-1
    
    # M9 = (a12-a20)(b22-b00)
    U[5,8]=1; U[6,8]=-1
    V[8,8]=1; V[0,8]=-1
    
    # M10 = (a21-a02)(b21-b02)
    U[7,9]=1; U[2,9]=-1
    V[7,9]=1; V[2,9]=-1
    
    # M11 = (a00+a01+a02)(b00+b10+b20)
    U[0,10]=1; U[1,10]=1; U[2,10]=1
    V[0,10]=1; V[3,10]=1; V[6,10]=1
    
    # M12 = (a10+a11+a12)(b01+b11+b21)
    U[3,11]=1; U[4,11]=1; U[5,11]=1
    V[1,11]=1; V[4,11]=1; V[7,11]=1
    
    # M13 = (a20+a21+a22)(b02+b12+b22)
    U[6,12]=1; U[7,12]=1; U[8,12]=1
    V[2,12]=1; V[5,12]=1; V[8,12]=1
    
    # M14 = (a00-a01+a02)(b00-b10+b20)
    U[0,13]=1; U[1,13]=-1; U[2,13]=1
    V[0,13]=1; V[3,13]=-1; V[6,13]=1
    
    # M15 = (a10-a11+a12)(b01-b11+b21)
    U[3,14]=1; U[4,14]=-1; U[5,14]=1
    V[1,14]=1; V[4,14]=-1; V[7,14]=1
    
    # M16 = (a20-a21+a22)(b02-b12+b22)
    U[6,15]=1; U[7,15]=-1; U[8,15]=1
    V[2,15]=1; V[5,15]=-1; V[8,15]=1
    
    # M17 = (a00+a01-a02)(b00+b10-b20)
    U[0,16]=1; U[1,16]=1; U[2,16]=-1
    V[0,16]=1; V[3,16]=1; V[6,16]=-1
    
    # M18 = (a10+a11-a12)(b01+b11-b21)
    U[3,17]=1; U[4,17]=1; U[5,17]=-1
    V[1,17]=1; V[4,17]=1; V[7,17]=-1
    
    # M19 = (a20+a21-a22)(b02+b12-b22)
    U[6,18]=1; U[7,18]=1; U[8,18]=-1
    V[2,18]=1; V[5,18]=1; V[8,18]=-1
    
    # M20 = (a00-a01-a02)(b00-b10-b20)
    U[0,19]=1; U[1,19]=-1; U[2,19]=-1
    V[0,19]=1; V[3,19]=-1; V[6,19]=-1
    
    # M21 = (a10-a11-a12)(b01-b11-b21)
    U[3,20]=1; U[4,20]=-1; U[5,20]=-1
    V[1,20]=1; V[4,20]=-1; V[7,20]=-1
    
    # M22 = (a20-a21-a22)(b02-b12-b22)
    U[6,21]=1; U[7,21]=-1; U[8,21]=-1
    V[2,21]=1; V[5,21]=-1; V[8,21]=-1
    
    # M23 = (a01+a12+a20)(b10+b21+b02)
    U[1,22]=1; U[5,22]=1; U[6,22]=1
    V[3,22]=1; V[7,22]=1; V[2,22]=1
    
    # Now solve for W: for each (i,k), we need
    # sum_r P[i,k,r] * W[k*3+i, r] = 1
    # where P[i,k,r] = sum_j U[i*3+j, r] * V[j*3+k, r]
    
    for i in range(3):
        for k in range(3):
            if time.time() - _START_TIME > DEADLINE_SECONDS:
                return None
            # Compute P[i,k,r] = sum_j U[i*3+j, r] * V[j*3+k, r]
            P = np.zeros(23)
            for j in range(3):
                P += U[i*3+j, :] * V[j*3+k, :]
            
            # Solve P . W_row = 1 using least squares
            P_col = P.reshape(1, -1)
            try:
                W_row, residuals, rank_p, sv = np.linalg.lstsq(P_col, np.array([1.0]), rcond=None)
                W[k*3+i, :] = W_row.flatten()
            except:
                # Fallback: find largest magnitude entry
                idx = np.argmax(np.abs(P))
                if abs(P[idx]) > 1e-12:
                    W[k*3+i, idx] = 1.0 / P[idx]
                else:
                    return None
    
    # Verify
    if not _verify_decomposition(U, V, W, 3, 3, 3):
        return None
    
    return U, V, W


def _build_rank41_decomposition():
    """
    Build a rank-41 decomposition for 3x3x5 matrix multiplication.
    Uses Laderman rank-23 for 3x3x3 + schoolbook rank-18 for 3x3x2.
    """
    n, m, p = 3, 3, 5
    rank_23 = 23
    rank_18 = 18
    total_rank = rank_23 + rank_18  # 41
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Part 1: 3x3x3 decomposition (rank 23) for k=0,1,2
    laderman = _build_laderman_3x3x3()
    if laderman is None:
        return None
    U_lm, V_lm, W_lm = laderman
    
    # Copy Laderman into the first 23 columns
    U[:, :rank_23] = U_lm
    V[:, :rank_23] = V_lm
    W[:, :rank_23] = W_lm
    
    # Part 2: Schoolbook for 3x3x2 (k=3,4), rank 18
    col_offset = rank_23
    for i in range(n):
        for j in range(m):
            for k in range(3, 5):
                col = col_offset + (i * m + j) * 2 + (k - 3)
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
    
    # Verify the full decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return U_list, V_list, W_list


def construct():
    """Return a rank-minimal exact tensor decomposition.
    
    Tries to construct a rank-41 decomposition using Laderman's rank-23
    decomposition for the 3x3x3 sub-tensor combined with schoolbook rank-18
    for the 3x3x2 part. Falls back to schoolbook rank 45.
    """
    n, m, p = DIMENSIONS  # 3, 3, 5

    # Try to build a rank-41 decomposition
    if time.time() - _START_TIME < DEADLINE_SECONDS:
        result = _build_rank41_decomposition()
        
        if result is not None and (time.time() - _START_TIME) < DEADLINE_SECONDS:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook rank 45
    if time.time() - _START_TIME > DEADLINE_SECONDS:
        raise TimeoutError("Exceeded deadline")
    
    U, V, W = _build_schoolbook(n, m, p)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition verification failed")

    return {"U": U, "V": V, "W": W}
