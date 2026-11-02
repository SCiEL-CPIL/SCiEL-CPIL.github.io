"""Matrix multiplication 6x6x6 - rank-minimal tensor decomposition search."""
DIMENSIONS = [6, 6, 6]

import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def check_decomposition_fast(Um, Vm, Wm):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        R = Um.shape[1]
        Um_3d = Um.reshape(n, m, R)  # Um[i, j, r]
        Vm_3d = Vm.reshape(m, p, R)  # Vm[j, k, r]
        Wm_3d = Wm.reshape(p, n, R)  # Wm[k, i, r]
        T = np.einsum('ijr,jkr,kir->ijk', Um_3d, Vm_3d, Wm_3d)
        if not np.allclose(T, 1.0, atol=1e-7):
            return False
        return True

    # Build the Strassen 2x2x2 decomposition (rank 7)
    def strassen_2x2():
        """Returns U, V, W matrices for 2x2x2 Strassen decomposition (rank 7)."""
        # U: (2*2, 7) = (4, 7), indexed by (i,j) -> row i*2+j
        # V: (2*2, 7) = (4, 7), indexed by (j,k) -> row j*2+k
        # W: (2*2, 7) = (4, 7), indexed by (k,i) -> row k*2+i
        #
        # Strassen formulas:
        # P1 = (A00 + A11)(B00 + B11)
        # P2 = (A10 + A11)B00
        # P3 = A00(B01 - B11)
        # P4 = A11(B10 - B00)
        # P5 = (A00 + A01)B11
        # P6 = (A10 - A00)(B00 + B01)
        # P7 = (A01 - A11)(B10 + B11)
        #
        # C00 = P1 + P4 - P5 - P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P7
        #
        # U[i,j,r] contributes to A[i,j] coefficient
        # V[j,k,r] contributes to B[j,k] coefficient
        # W[k,i,r] contributes to C[k,i] coefficient... wait, W is indexed by (k,i)
        # Actually the tensor is T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # For 2x2: T[i,j,k] = sum_r U[i*2+j,r] * V[j*2+k,r] * W[k*2+i,r]
        # And we need T[i,j,k] = 1 for all (i,j,k)
        #
        # Let me reconsider. The multiplication tensor for 2x2x2:
        # T[i,j,k] = 1 means: (AB)[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor T[i,j,k] has T[i,j,k] = 1 for all valid indices.
        #
        # Strassen:
        # (AB)[0,0] = P1 + P4 - P5 - P7
        # (AB)[0,1] = P3 + P5
        # (AB)[1,0] = P2 + P4
        # (AB)[1,1] = P1 - P2 + P3 + P7
        #
        # So:
        # T[0,0,0] = 1: from P1,P4,P5,P7
        # T[0,1,0] = 1: from P3,P5
        # T[1,0,0] = 1: from P2,P4
        # T[1,1,0] = 1: from P1,P2,P3,P7
        # T[0,0,1] = 1: from P1,P4,P5,P7  -- wait, (AB)[0,1] not (AB)[0,0]
        #
        # Let me be more careful:
        # (AB)[i,k] = sum_j A[i,j] B[j,k]
        # T[i,j,k] = 1 means the coefficient of A[i,j]*B[j,k] in (AB)[i,k] is 1.
        #
        # (AB)[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        # (AB)[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
        # (AB)[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
        # (AB)[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
        #
        # So T[0,0,0]=1, T[0,1,0]=1, T[0,0,1]=1, T[0,1,1]=1,
        #    T[1,0,0]=1, T[1,1,0]=1, T[1,0,1]=1, T[1,1,1]=1
        #
        # Strassen P's:
        # P1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # P2 = (A10+A11)B00 = A10B00 + A11B00
        # P3 = A00(B01-B11) = A00B01 - A00B11
        # P4 = A11(B10-B00) = A11B10 - A11B00
        # P5 = (A00+A01)B11 = A00B11 + A01B11
        # P6 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01
        # P7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
        #
        # C00 = P1 + P4 - P5 - P7
        # = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 - A01B10 - A01B11 + A11B10 + A11B11
        # Hmm, this is getting complicated. Let me just use the standard mapping.
        #
        # For term r (0-indexed), U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        #
        # P1: (A00+A11)(B00+B11)
        # U[0,0]=1, U[3,3]=1  (A[0,0] and A[1,1])
        # V[0,0]=1, V[3,3]=1  (B[0,0] and B[1,1])
        # W[0,0]=1, W[1,0]=1, W[0,1]=1, W[1,1]=1  (C[0,0], C[1,0], C[0,1], C[1,1])
        # Wait, W is indexed by (k,i), so W[k*2+i, r]
        # C[0,0] -> k=0,i=0 -> W[0]
        # C[0,1] -> k=1,i=0 -> W[2]
        # C[1,0] -> k=0,i=1 -> W[1]
        # C[1,1] -> k=1,i=1 -> W[3]
        #
        # P1 contributes to all four C entries.
        # U: row 0 (i=0,j=0) = 1, row 3 (i=1,j=1) = 1
        # V: row 0 (j=0,k=0) = 1, row 3 (j=1,k=1) = 1
        # W: row 0 (k=0,i=0) = 1, row 2 (k=1,i=0) = 1, row 1 (k=0,i=1) = 1, row 3 (k=1,i=1) = 1
        #
        # P2: (A10+A11)B00
        # U: row 2 (i=1,j=0) = 1, row 3 (i=1,j=1) = 1
        # V: row 0 (j=0,k=0) = 1
        # W: row 1 (k=0,i=1) = 1  (C[1,0])
        #
        # P3: A00(B01-B11)
        # U: row 0 (i=0,j=0) = 1
        # V: row 1 (j=0,k=1) = 1, row 3 (j=1,k=1) = -1
        # W: row 2 (k=1,i=0) = 1  (C[0,1])
        #
        # P4: A11(B10-B00)
        # U: row 3 (i=1,j=1) = 1
        # V: row 2 (j=1,k=0) = 1, row 0 (j=0,k=0) = -1
        # W: row 1 (k=0,i=1) = 1  (C[1,0])
        #
        # P5: (A00+A01)B11
        # U: row 0 (i=0,j=0) = 1, row 1 (i=0,j=1) = 1
        # V: row 3 (j=1,k=1) = 1
        # W: row 2 (k=1,i=0) = 1  (C[0,1])
        #
        # P6: (A10-A00)(B00+B01)
        # U: row 2 (i=1,j=0) = 1, row 0 (i=0,j=0) = -1
        # V: row 0 (j=0,k=0) = 1, row 1 (j=0,k=1) = 1
        # W: row 1 (k=0,i=1) = 1, row 3 (k=1,i=1) = 1  (C[1,0], C[1,1])
        #
        # P7: (A01-A11)(B10+B11)
        # U: row 1 (i=0,j=1) = 1, row 3 (i=1,j=1) = -1
        # V: row 2 (j=1,k=0) = 1, row 3 (j=1,k=1) = 1
        # W: row 0 (k=0,i=0) = 1, row 2 (k=1,i=0) = 1  (C[0,0], C[0,1])
        #
        # Let me verify C00 = P1 + P4 - P5 - P7:
        # P1: C[0,0] += A00B00 + A00B11 + A11B00 + A11B11
        # P4: C[1,0] += A11B10 - A11B00  -- wait, P4 only affects C[1,0]
        # This doesn't seem right. Let me look up the correct Strassen formula.
        #
        # Standard Strassen:
        # C00 = P1 + P4 - P5 - P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P7
        #
        # So W for P1: affects C00, C11 -> W[0]=1 (C[0,0]), W[3]=1 (C[1,1])
        # W for P2: affects C10 -> W[1]=1 (C[1,0])
        # W for P3: affects C01, C11 -> W[2]=1 (C[0,1]), W[3]=1 (C[1,1])
        # W for P4: affects C00, C10 -> W[0]=1 (C[0,0]), W[1]=1 (C[1,0])
        # W for P5: affects C00, C01 -> W[0]=-1 (C[0,0]), W[2]=1 (C[0,1])
        # W for P6: affects C10, C11 -> W[1]=1 (C[1,0]), W[3]=1 (C[1,1])
        # W for P7: affects C00, C11 -> W[0]=-1 (C[0,0]), W[3]=1 (C[1,1])
        #
        # Let me redo:
        # P1: U[0]=1, U[3]=1; V[0]=1, V[3]=1; W[0]=1, W[3]=1
        # P2: U[2]=1, U[3]=1; V[0]=1; W[1]=1
        # P3: U[0]=1; V[1]=1, V[3]=-1; W[2]=1, W[3]=1
        # P4: U[3]=1; V[2]=1, V[0]=-1; W[0]=1, W[1]=1
        # P5: U[0]=1, U[1]=1; V[3]=1; W[0]=-1, W[2]=1
        # P6: U[2]=1, U[0]=-1; V[0]=1, V[1]=1; W[1]=1, W[3]=1
        # P7: U[1]=1, U[3]=-1; V[2]=1, V[3]=1; W[0]=-1, W[3]=1
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # P1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        # P2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        # P3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        # P4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        # P5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        # P6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[1, 5] = 1; W[3, 5] = 1
        # P7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1; W[3, 6] = 1
        
        return U, V, W

    # Build schoolbook 3x3x3 decomposition (rank 27)
    def schoolbook_3x3():
        """Returns U, V, W matrices for 3x3x3 schoolbook decomposition (rank 27)."""
        U = np.zeros((9, 27))
        V = np.zeros((9, 27))
        W = np.zeros((9, 27))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i * 3 + j, col] = 1.0
                    V[j * 3 + k, col] = 1.0
                    W[k * 3 + i, col] = 1.0
                    col += 1
        return U, V, W

    # Kronecker product of two decompositions
    def kron_decomp(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
        """Kronecker product of two matrix multiplication decompositions.
        U1: (n1*m1, r1), V1: (m1*p1, r1), W1: (p1*n1, r1)
        U2: (n2*m2, r2), V2: (m2*p2, r2), W2: (p2*n2, r2)
        Returns U: (n1*n2*m1*m2, r1*r2), V: (m1*m2*p1*p2, r1*r2), W: (p1*p2*n1*n2, r1*r2)
        """
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        U = np.zeros((n1 * n2 * m1 * m2, r1 * r2))
        V = np.zeros((m1 * m2 * p1 * p2, r1 * r2))
        W = np.zeros((p1 * p2 * n1 * n2, r1 * r2))
        
        for r in range(r1):
            for s in range(r2):
                col = r * r2 + s
                # U: index (i1*n2+i2, j1*m2+j2) -> (i1*n2+i2)*m1*m2 + (j1*m2+j2)
                for i1 in range(n1):
                    for j1 in range(m1):
                        u1_val = U1[i1 * m1 + j1, r]
                        if u1_val == 0:
                            continue
                        for i2 in range(n2):
                            for j2 in range(m2):
                                u2_val = U2[i2 * m2 + j2, s]
                                if u2_val == 0:
                                    continue
                                row = (i1 * n2 + i2) * m1 * m2 + (j1 * m2 + j2)
                                U[row, col] = u1_val * u2_val
                # V: index (j1*m2+j2, k1*p2+k2) -> (j1*m2+j2)*p1*p2 + (k1*p2+k2)
                for j1 in range(m1):
                    for k1 in range(p1):
                        v1_val = V1[j1 * p1 + k1, r]
                        if v1_val == 0:
                            continue
                        for j2 in range(m2):
                            for k2 in range(p2):
                                v2_val = V2[j2 * p2 + k2, s]
                                if v2_val == 0:
                                    continue
                                row = (j1 * m2 + j2) * p1 * p2 + (k1 * p2 + k2)
                                V[row, col] = v1_val * v2_val
                # W: index (k1*p2+k2, i1*n2+i2) -> (k1*p2+k2)*n1*n2 + (i1*n2+i2)
                for k1 in range(p1):
                    for i1 in range(n1):
                        w1_val = W1[k1 * n1 + i1, r]
                        if w1_val == 0:
                            continue
                        for k2 in range(p2):
                            for i2 in range(n2):
                                w2_val = W2[k2 * n2 + i2, s]
                                if w2_val == 0:
                                    continue
                                row = (k1 * p2 + k2) * n1 * n2 + (i1 * n2 + i2)
                                W[row, col] = w1_val * w2_val
        return U, V, W

    # Build Strassen 2x2 and Schoolbook 3x3
    U2, V2, W2 = strassen_2x2()
    U3, V3, W3 = schoolbook_3x3()
    
    # Kronecker: 6x6x6 = (2x3) x (2x3) x (2x3)
    # n=6=2*3, m=6=2*3, p=6=2*3
    # U: (n*m, rank) = (36, 7*27=189)
    # V: (m*p, rank) = (36, 189)
    # W: (p*n, rank) = (36, 189)
    U_kron, V_kron, W_kron = kron_decomp(U2, V2, W2, U3, V3, W3, 2, 2, 2, 3, 3, 3)
    
    # Verify the Kronecker decomposition
    if not check_decomposition_fast(U_kron, V_kron, W_kron):
        # Fallback to schoolbook
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    best_U = U_kron.copy()
    best_V = V_kron.copy()
    best_W = W_kron.copy()
    best_rank = U_kron.shape[1]
    
    # Try to reduce rank by removing columns
    def try_remove_columns(Um, Vm, Wm, col_indices_to_try):
        """Try to remove a subset of columns. Returns (success, new_U, new_V, new_W)."""
        R = Um.shape[1]
        keep = [r for r in range(R) if r not in col_indices_to_try]
        if not keep:
            return False, None, None, None
        U_new = Um[:, keep]
        V_new = Vm[:, keep]
        W_new = Wm[:, keep]
        if check_decomposition_fast(U_new, V_new, W_new):
            return True, U_new, V_new, W_new
        return False, None, None, None
    
    U_cur = U_kron.copy()
    V_cur = V_kron.copy()
    W_cur = W_kron.copy()
    current_rank = best_rank
    
    col_order = list(range(current_rank))
    random.shuffle(col_order)
    
    # Phase 1: Greedy single-column removal
    for idx in col_order:
        if elapsed() > deadline - 20:
            break
        if idx >= current_rank:
            continue
        success, U_new, V_new, W_new = try_remove_columns(U_cur, V_cur, W_cur, {idx})
        if success:
            U_cur, V_cur, W_cur = U_new, V_new, W_new
            current_rank -= 1
            if current_rank < best_rank:
                best_U = U_cur.copy()
                best_V = V_cur.copy()
                best_W = W_cur.copy()
                best_rank = current_rank
            col_order = [c for c in col_order if c != idx and c < current_rank]
            random.shuffle(col_order)
    
    # Phase 2: Try removing pairs of columns
    if elapsed() < deadline - 30:
        pairs = list(zip(col_order, col_order[1:]))
        random.shuffle(pairs)
        for a, b in pairs[:300]:
            if elapsed() > deadline - 20:
                break
            if a >= current_rank or b >= current_rank:
                continue
            if a == b:
                continue
            success, U_new, V_new, W_new = try_remove_columns(U_cur, V_cur, W_cur, {a, b})
            if success:
                U_cur, V_cur, W_cur = U_new, V_new, W_new
                current_rank -= 2
                if current_rank < best_rank:
                    best_U = U_cur.copy()
                    best_V = V_cur.copy()
                    best_W = W_cur.copy()
                    best_rank = current_rank
                col_order = [c for c in col_order if c not in {a, b} and c < current_rank]
                random.shuffle(col_order)
    
    # Final verification
    if not check_decomposition_fast(best_U, best_V, best_W):
        # Fallback
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
