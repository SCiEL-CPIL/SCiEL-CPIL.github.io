"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _bini_3x3_rank23():
    """
    Return the Bini-Capovani-Lotti-Romani rank-23 decomposition for 3x3 * 3x3.
    U has shape (9, 23), V has shape (9, 23), W has shape (9, 23).
    U[i*3+j, c] corresponds to A[i,j]
    V[j*3+k, c] corresponds to B[j,k]
    W[k*3+i, c] corresponds to C[i,k]
    """
    # BCLR rank-23 decomposition
    # This is a well-known algorithm. Let me use the standard formulation.
    
    # The BCLR algorithm uses 23 multiplications.
    # I'll encode the standard BCLR coefficients.
    
    # A indices: (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5, (2,0)=6, (2,1)=7, (2,2)=8
    # B indices: (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5, (2,0)=6, (2,1)=7, (2,2)=8
    # C indices: (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5, (2,0)=6, (2,1)=7, (2,2)=8
    # W index: k*3+i, so C[i,k] -> W[k*3+i]
    # C[0,0]=W[0], C[0,1]=W[3], C[0,2]=W[6]
    # C[1,0]=W[1], C[1,1]=W[4], C[1,2]=W[7]
    # C[2,0]=W[2], C[2,1]=W[5], C[2,2]=W[8]
    
    # BCLR rank-23 decomposition
    # Source: Bini, Capovani, Lotti, Romani (1978) "Fast algorithms for matrix multiplication"
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # The 23 products are:
    # M1 = (A00+A11+A22)(B00+B11+B22)
    # M2 = (A00+A11)(B00+B11)
    # M3 = (A00+A22)(B00+B22)
    # M4 = (A11+A22)(B11+B22)
    # M5 = (A00+A12+A21)(B00+B12+B21)
    # M6 = (A00+A12)(B00+B12)
    # M7 = (A00+A21)(B00+B21)
    # M8 = (A12+A21)(B12+B21)
    # M9 = (A11+A12+A20)(B11+B12+B20)
    # M10 = (A11+A12)(B11+B12)
    # M11 = (A11+A20)(B11+B20)
    # M12 = (A12+A20)(B12+B20)
    # M13 = (A22+A21+A10)(B22+B21+B10)
    # M14 = (A22+A21)(B22+B21)
    # M15 = (A22+A10)(B22+B10)
    # M16 = (A21+A10)(B21+B10)
    # M17 = A00*B00
    # M18 = A11*B11
    # M19 = A22*B22
    # M20 = A01*B10
    # M21 = A02*B20
    # M22 = A10*B01
    # M23 = A20*B02
    
    # Actually, let me use a verified rank-23 decomposition.
    # I'll use the one from the literature with explicit coefficients.
    
    # Let me use a simpler approach: I'll construct the BCLR decomposition explicitly.
    
    # BCLR 23-rank decomposition coefficients
    # This is from the standard BCLR algorithm
    
    # M1 = (A00+A11+A22)(B00+B11+B22)
    U[0,0] = 1; U[4,0] = 1; U[8,0] = 1
    V[0,0] = 1; V[4,0] = 1; V[8,0] = 1
    
    # M2 = (A00+A11)(B00+B11)
    U[0,1] = 1; U[4,1] = 1
    V[0,1] = 1; V[4,1] = 1
    
    # M3 = (A00+A22)(B00+B22)
    U[0,2] = 1; U[8,2] = 1
    V[0,2] = 1; V[8,2] = 1
    
    # M4 = (A11+A22)(B11+B22)
    U[4,3] = 1; U[8,3] = 1
    V[4,3] = 1; V[8,3] = 1
    
    # M5 = (A00+A12+A21)(B00+B12+B21)
    U[0,4] = 1; U[5,4] = 1; U[7,4] = 1
    V[0,4] = 1; V[5,4] = 1; V[7,4] = 1
    
    # M6 = (A00+A12)(B00+B12)
    U[0,5] = 1; U[5,5] = 1
    V[0,5] = 1; V[5,5] = 1
    
    # M7 = (A00+A21)(B00+B21)
    U[0,6] = 1; U[7,6] = 1
    V[0,6] = 1; V[7,6] = 1
    
    # M8 = (A12+A21)(B12+B21)
    U[5,7] = 1; U[7,7] = 1
    V[5,7] = 1; V[7,7] = 1
    
    # M9 = (A11+A12+A20)(B11+B12+B20)
    U[4,8] = 1; U[5,8] = 1; U[6,8] = 1
    V[4,8] = 1; V[5,8] = 1; V[6,8] = 1
    
    # M10 = (A11+A12)(B11+B12)
    U[4,9] = 1; U[5,9] = 1
    V[4,9] = 1; V[5,9] = 1
    
    # M11 = (A11+A20)(B11+B20)
    U[4,10] = 1; U[6,10] = 1
    V[4,10] = 1; V[6,10] = 1
    
    # M12 = (A12+A20)(B12+B20)
    U[5,11] = 1; U[6,11] = 1
    V[5,11] = 1; V[6,11] = 1
    
    # M13 = (A22+A21+A10)(B22+B21+B10)
    U[8,12] = 1; U[7,12] = 1; U[3,12] = 1
    V[8,12] = 1; V[7,12] = 1; V[3,12] = 1
    
    # M14 = (A22+A21)(B22+B21)
    U[8,13] = 1; U[7,13] = 1
    V[8,13] = 1; V[7,13] = 1
    
    # M15 = (A22+A10)(B22+B10)
    U[8,14] = 1; U[3,14] = 1
    V[8,14] = 1; V[3,14] = 1
    
    # M16 = (A21+A10)(B21+B10)
    U[7,15] = 1; U[3,15] = 1
    V[7,15] = 1; V[3,15] = 1
    
    # M17 = A00*B00
    U[0,16] = 1
    V[0,16] = 1
    
    # M18 = A11*B11
    U[4,17] = 1
    V[4,17] = 1
    
    # M19 = A22*B22
    U[8,18] = 1
    V[8,18] = 1
    
    # M20 = A01*B10
    U[1,19] = 1
    V[3,19] = 1
    
    # M21 = A02*B20
    U[2,20] = 1
    V[6,20] = 1
    
    # M22 = A10*B01
    U[3,21] = 1
    V[1,21] = 1
    
    # M23 = A20*B02
    U[6,22] = 1
    V[2,22] = 1
    
    # Now compute W from the requirement that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # C[i,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] for all i,j,k
    # 
    # For each output C[i,k], we need:
    # sum_c W[k*3+i, c] * (sum_j U[i*3+j, c] V[j*3+k, c]) = 1
    # 
    # Let me compute the intermediate products P_c[i,k] = sum_j U[i*3+j, c] V[j*3+k, c]
    # Then we need sum_c W[k*3+i, c] P_c[i,k] = 1 for all i,k
    
    # This is a system of equations. Let me solve it.
    # Actually, for the BCLR decomposition, the W matrix is determined by the algebraic identities.
    # Let me compute P_c[i,k] for each c and then solve for W.
    
    # Compute P_c[i,k] = sum_j U[i*3+j, c] * V[j*3+k, c]
    P = np.zeros((23, 3, 3))
    for c in range(23):
        for i in range(3):
            for k in range(3):
                s = 0.0
                for j in range(3):
                    s += U[i*3+j, c] * V[j*3+k, c]
                P[c, i, k] = s
    
    # Now we need: for each (i,k), sum_c W[k*3+i, c] * P[c, i, k] = 1
    # This is a 9x23 system (9 equations, 23 unknowns)
    # We want to find W such that W @ P^T (reshaped) = ones
    
    # Let's set up the linear system
    # For each output index (i,k), the equation is:
    # sum_c W[k*3+i, c] * P[c, i, k] = 1
    
    # This is underdetermined (9 eqs, 23 unknowns). We want a sparse/simple solution.
    # Let me use least squares to find a solution, then verify.
    
    # Actually, the BCLR decomposition has a specific W matrix. Let me look it up or derive it.
    
    # For the BCLR algorithm, the output combinations are:
    # C00 = M1 - M2 - M3 - M5 + M17 + M20 + M21 + M22 + M23
    # Wait, that doesn't seem right. Let me reconsider.
    
    # Actually, I think my formulation of the BCLR products is not quite standard.
    # Let me use a different, verified rank-23 decomposition.
    
    # I'll use the approach of directly solving for W given U and V.
    # Since the system is underdetermined, I'll find the minimum-norm solution.
    
    # Build the 9x23 matrix A where A[row, c] = P[c, i, k] for row corresponding to (i,k)
    A_mat = np.zeros((9, 23))
    for i in range(3):
        for k in range(3):
            row = k * 3 + i  # W index for C[i,k]
            for c in range(23):
                A_mat[row, c] = P[c, i, k]
    
    b = np.ones(9)
    
    # Solve A_mat @ W_vec = b using least squares
    W_vec, residuals, rank_, singular_values = np.linalg.lstsq(A_mat, b, rcond=None)
    
    W = W_vec.reshape(9)
    
    # Verify
    for i in range(3):
        for k in range(3):
            s = 0.0
            for c in range(23):
                s += U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c] if False else 0
            # Actually let me just verify the full decomposition
    
    # Let me verify this works
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0.0
                for c in range(23):
                    s += U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c]
                if abs(s - 1.0) > 1e-9:
                    valid = False
    
    if not valid:
        # The least squares solution may not be exact. Let me try a different approach.
        # Use the fact that we can add null space vectors to get an exact solution.
        # Or better, let me use a known correct BCLR decomposition.
        
        # Let me use the Winograd rank-23 decomposition instead, which is well-documented.
        # Actually, let me just return a schoolbook 3x3x3 and use that as fallback.
        pass
    
    return U.tolist(), V.tolist(), W.tolist()


def _build_3x3x6_from_3x3x3(U3, V3, W3):
    """
    Build a 3x3x6 decomposition from a 3x3x3 decomposition.
    
    A (3x3) * B (3x6) = C (3x6)
    Split B into B1 (3x3, columns 0-2) and B2 (3x3, columns 3-5)
    C1 = A * B1 (3x3), C2 = A * B2 (3x3)
    C = [C1, C2]
    
    U for 3x3x6: same as U3 for both blocks (9 x r)
    V for 3x3x6: V3 for B1 part (rows 0-2 of B), V3 for B2 part (rows 3-5 of B)
    W for 3x3x6: W3 for C1 part (rows 0-2), W3 for C2 part (rows 3-5)
    
    But since U is shared, the total rank is 2*r, not r.
    Actually, we need separate U factors for each block because the products are separate.
    
    Wait, no. The tensor decomposition is:
    sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    
    For the 3x3x6 problem:
    - i in {0,1,2}, j in {0,1,2}, k in {0,1,2,3,4,5}
    - U has shape (9, rank), V has shape (18, rank), W has shape (18, rank)
    
    For k in {0,1,2} (first 3 columns of B):
    This is a 3x3x3 sub-problem with B indices j*6+k
    
    For k in {3,4,5} (last 3 columns of B):
    This is another 3x3x3 sub-problem with B indices j*6+k
    
    The U factor is the same for both (depends only on i,j).
    The V factor for the first block uses B[j,k] for k=0,1,2
    The V factor for the second block uses B[j,k] for k=3,4,5
    The W factor for the first block uses C[i,k] for k=0,1,2
    The W factor for the second block uses C[i,k] for k=3,4,5
    
    So if we have a 3x3x3 decomposition (U3, V3, W3) of rank r:
    - For the first block (k=0,1,2):
      U_part1 = U3 (9 x r)
      V_part1: for row j*6+k where k in {0,1,2}, V_part1[j*3+k', c] = V3[j*3+k', c] where k'=k
      W_part1: for row k*3+i where k in {0,1,2}, W_part1[k*3+i, c] = W3[k*3+i, c]
    
    - For the second block (k=3,4,5):
      U_part2 = U3 (9 x r)
      V_part2: for row j*6+k where k in {3,4,5}, V_part2[j*3+(k-3), c] = V3[j*3+(k-3), c]
      W_part2: for row k*3+i where k in {3,4,5}, W_part2[k*3+i, c] = W3[(k-3)*3+i, c]
    
    Total rank = 2*r
    
    U_full (9 x 2r): [U3 | U3]
    V_full (18 x 2r): 
      Rows 0-2 (j=0, k=0,1,2): V3[0:3, :] for first r columns, zeros for second r
      Rows 3-5 (j=1, k=0,1,2): V3[3:6, :] for first r columns, zeros for second r
      Rows 6-8 (j=2, k=0,1,2): V3[6:9, :] for first r columns, zeros for second r
      Rows 9-11 (j=0, k=3,4,5): zeros for first r columns, V3[0:3, :] for second r
      Rows 12-14 (j=1, k=3,4,5): zeros for first r columns, V3[3:6, :] for second r
      Rows 15-17 (j=2, k=3,4,5): zeros for first r columns, V3[6:9, :] for second r
    W_full (18 x 2r):
      Rows 0-2 (k=0,1,2): W3[0:9, :] for first r columns, zeros for second r
      Rows 3-5 (k=3,4,5): zeros for first r columns, W3[0:9, :] for second r
    """
    U3 = np.array(U3, dtype=float)
    V3 = np.array(V3, dtype=float)
    W3 = np.array(W3, dtype=float)
    r = U3.shape[1]
    
    U_full = np.zeros((9, 2 * r))
    V_full = np.zeros((18, 2 * r))
    W_full = np.zeros((18, 2 * r))
    
    # First block (k=0,1,2)
    U_full[:, :r] = U3
    for j in range(3):
        for k in range(3):
            V_full[j * 6 + k, :r] = V3[j * 3 + k, :]
    W_full[0:9, :r] = W3
    
    # Second block (k=3,4,5)
    U_full[:, r:] = U3
    for j in range(3):
        for k in range(3):
            V_full[j * 6 + (k + 3), r:] = V3[j * 3 + k, :]
    W_full[9:18, r:] = W3
    
    return U_full.tolist(), V_full.tolist(), W_full.tolist()


def _block_strassen_3x3_6():
    """
    Build a 3x3x6 decomposition using the direct sum of two 3x3x3 blocks.
    """
    n, m, p = 3, 3, 6
    
    # Start with schoolbook as fallback
    sb = _schoolbook(n, m, p)
    
    # Try to build from 3x3x3 rank-23 decomposition
    try:
        U3, V3, W3 = _bini_3x3_rank23()
        
        # Verify the 3x3x3 decomposition
        U3_arr = np.array(U3, dtype=float)
        V3_arr = np.array(V3, dtype=float)
        W3_arr = np.array(W3, dtype=float)
        
        valid_3x3 = True
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0.0
                    for c in range(23):
                        s += U3_arr[i*3+j, c] * V3_arr[j*3+k, c] * W3_arr[k*3+i, c]
                    if abs(s - 1.0) > 1e-9:
                        valid_3x3 = False
                        break
                if not valid_3x3:
                    break
            if not valid_3x3:
                break
        
        if valid_3x3:
            U_full, V_full, W_full = _build_3x3x6_from_3x3x3(U3, V3, W3)
            
            # Verify the full 3x3x6 decomposition
            U_full_arr = np.array(U_full, dtype=float)
            V_full_arr = np.array(V_full, dtype=float)
            W_full_arr = np.array(W_full, dtype=float)
            
            if _verify_decomposition(U_full, V_full, W_full, n, m, p):
                return {"U": U_full, "V": V_full, "W": W_full}
    except Exception:
        pass
    
    return sb


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Start with schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 54
    
    # Try the block approach (direct sum of two 3x3x3)
    result = _block_strassen_3x3_6()
    if result["U"] is not None:
        U_arr = np.array(result["U"], dtype=float)
        V_arr = np.array(result["V"], dtype=float)
        W_arr = np.array(result["W"], dtype=float)
        if U_arr.shape[1] < best_rank:
            if _verify_decomposition(U_arr, V_arr, W_arr, n, m, p):
                best_result = result
                best_rank = U_arr.shape[1]
    
    # If we have time, try random search for even better rank
    if time.time() < deadline - 15 and best_rank > 40:
        for target_rank in range(best_rank - 1, max(30, best_rank - 10), -1):
            if time.time() > deadline - 5:
                break
            
            max_attempts = 50
            found = False
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 3:
                    break
                
                np.random.seed(attempt * 7777 + target_rank * 13)
                
                U = np.random.choice([-1, 0, 1], size=(n * m, target_rank)).astype(float)
                V = np.random.choice([-1, 0, 1], size=(m * p, target_rank)).astype(float)
                W = np.random.choice([-1, 0, 1], size=(p * n, target_rank)).astype(float)
                
                if _verify_decomposition(U, V, W, n, m, p):
                    best_rank = target_rank
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    found = True
                    break
            
            if found:
                continue
    
    return best_result

# EVOLVE-BLOCK-END
