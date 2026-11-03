"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 6x6 matrix multiplication.
    Uses Kronecker product of Strassen (rank-7 for 2x2) and schoolbook (rank-27 for 3x3)
    to get rank 189, then attempts to reduce rank by finding linear dependencies.
    """
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = 6, 6, 6
    
    # Build Strassen decomposition for 2x2
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 0: M1 = (A00+A11)(B00+B01)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[1, 0] = 1
    W2[0, 0] = 1; W2[1, 0] = -1; W2[2, 0] = -1; W2[3, 0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # Term 2: M3 = A00(B10-B11)
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # Term 3: M4 = A11(B01-B00)
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # Term 4: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[3, 4] = -1
    
    # Term 5: M6 = (A10-A00)(B00+B10)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[2, 5] = 1
    W2[1, 5] = 1
    
    # Term 6: M7 = (A01-A11)(B01+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[1, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = -1; W2[2, 6] = 1; W2[3, 6] = 1
    
    # Build schoolbook decomposition for 3x3
    U3 = np.zeros((9, 27))
    V3 = np.zeros((9, 27))
    W3 = np.zeros((9, 27))
    r3 = 0
    for a in range(3):
        for c in range(3):
            for b in range(3):
                U3[a*3+c, r3] = 1
                V3[c*3+b, r3] = 1
                W3[b*3+a, r3] = 1
                r3 += 1
    
    # Form Kronecker product decomposition for 6x6
    rank_kron = 7 * 27  # 189
    U6 = np.zeros((36, rank_kron))
    V6 = np.zeros((36, rank_kron))
    W6 = np.zeros((36, rank_kron))
    
    for r2 in range(7):
        for r3 in range(27):
            r = r2 * 27 + r3
            # U: index (2a+i)*6 + (2c+k) = i*6 + a*12 + c*2 + k... let me redo
            # i = 2a + i', k = 2c + k'
            # U6[i*6+k, r] = U2[i'*2+k', r2] * U3[a*3+c, r3]
            for a in range(3):
                for ip in range(2):
                    for c in range(3):
                        for kp in range(2):
                            idx = (2*a + ip) * 6 + (2*c + kp)
                            val = U2[ip*2+kp, r2] * U3[a*3+c, r3]
                            if val != 0:
                                U6[idx, r] = val
            
            # V: index (2c+k)*6 + (2b+j)
            # V6[k*6+j, r] = V2[k'*2+j', r2] * V3[c*3+b, r3]
            for c in range(3):
                for kp in range(2):
                    for b in range(3):
                        for jp in range(2):
                            idx = (2*c + kp) * 6 + (2*b + jp)
                            val = V2[kp*2+jp, r2] * V3[c*3+b, r3]
                            if val != 0:
                                V6[idx, r] = val
            
            # W: index (2b+j)*6 + (2a+i)
            # W6[j*6+i, r] = W2[j'*2+i', r2] * W3[b*3+a, r3]
            for b in range(3):
                for jp in range(2):
                    for a in range(3):
                        for ip in range(2):
                            idx = (2*b + jp) * 6 + (2*a + ip)
                            val = W2[jp*2+ip, r2] * W3[b*3+a, r3]
                            if val != 0:
                                W6[idx, r] = val
    
    # Verify
    def verify(U, V, W, n, m, p, tol=1e-9):
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not verify(U6, V6, W6, 6, 6, 6):
        # Fallback to schoolbook
        rank_school = 216
        U_school = np.zeros((36, rank_school))
        V_school = np.zeros((36, rank_school))
        W_school = np.zeros((36, rank_school))
        col = 0
        for i in range(6):
            for k in range(6):
                for j in range(6):
                    U_school[i*6+k, col] = 1
                    V_school[k*6+j, col] = 1
                    W_school[j*6+i, col] = 1
                    col += 1
        return {"U": [[float(x) for x in row] for row in U_school],
                "V": [[float(x) for x in row] for row in V_school],
                "W": [[float(x) for x in row] for row in W_school]}
    
    best_rank = rank_kron
    best_U, best_V, best_W = U6.copy(), V6.copy(), W6.copy()
    
    # Try to reduce rank by finding linear dependencies
    # Strategy: check if any column can be expressed as a linear combination of others
    # For each term r, check if the rank-1 tensor T_r = u_r ⊗ v_r ⊗ w_r is in the span of others
    
    # Build the 3D tensor for each rank-1 term and check linear independence
    # This is expensive, so let's use a smarter approach:
    # For each pair of terms, check if one is a scalar multiple of the other
    
    # First, check for duplicate or proportional columns
    # Create a signature for each term based on the tensor it represents
    # T_r[i,k,j] = U[i,k,r] * V[k,j,r] * W[j,i,r]
    
    # Flatten each rank-1 tensor and check for linear dependencies
    # Use random projection to make this feasible
    
    if time.time() < deadline - 10:
        # Try to eliminate redundant terms
        # For each term, compute its "signature" via random linear functional
        rng = np.random.RandomState(42)
        
        # Number of tensor elements: 6*6*6 = 216
        # Each rank-1 term contributes a vector in R^216
        # Check if any term is in the span of others using QR with column pivoting
        
        # Build matrix where each column is a flattened rank-1 tensor
        # T_r flattened: T_r[i*36+k*6+j] = U[i*6+k, r] * V[k*6+j, r] * W[j*6+i, r]
        
        # This is a 216 x 189 matrix - manageable
        T_flat = np.zeros((216, rank_kron))
        for r in range(rank_kron):
            for i in range(6):
                for k in range(6):
                    for j in range(6):
                        T_flat[i*36 + k*6 + j, r] = U6[i*6+k, r] * V6[k*6+j, r] * W6[j*6+i, r]
        
        # Use QR with column pivoting to find the rank
        # scipy.linalg.qr with pivoting
        from scipy.linalg import qr
        Q, R, P = qr(T_flat, pivoting=True)
        
        # The rank is the number of significant diagonal entries in R
        # Find the effective rank
        diag = np.abs(np.diag(R))
        # Use a tolerance based on the largest diagonal
        if diag.max() > 0:
            tol_rank = diag.max() * 1e-10 * max(216, rank_kron)
        else:
            tol_rank = 1e-10
        
        effective_rank = 0
        for d in diag:
            if d > tol_rank:
                effective_rank += 1
            else:
                break
        
        if effective_rank < rank_kron:
            # We can reduce the rank
            # Keep only the first 'effective_rank' pivoted columns
            keep_cols = P[:effective_rank]
            new_U = U6[:, keep_cols]
            new_V = V6[:, keep_cols]
            new_W = W6[:, keep_cols]
            
            if verify(new_U, new_V, new_W, 6, 6, 6):
                best_rank = effective_rank
                best_U, best_V, best_W = new_U, new_V, new_W
        
        # Additional: try to find a basis change that makes more terms zero
        # This is harder; skip for now if time is tight
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
