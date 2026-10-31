import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120.0

    n, m, p = 4, 4, 4

    # Build the 4x4x4 matrix multiplication tensor
    # T[a_idx, b_idx, c_idx] = 1 where:
    #   a_idx = i * m + k  (A[i,k])
    #   b_idx = k * p + j  (B[k,j])
    #   c_idx = j * n + i  (C[i,j] stored as W[j*n+i])
    # C[i,j] = sum_k A[i,k] * B[k,j]
    
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a_idx = i * m + k
                b_idx = k * p + j
                c_idx = j * n + i
                target[a_idx, b_idx, c_idx] = 1.0

    def verify(U, V, W, rank):
        """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == target tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        recon = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr, optimize=True)
        return np.allclose(recon, target, atol=1e-9)

    def time_ok():
        return time.time() < deadline - 10.0

    # ============================================================
    # Method 1: Naive decomposition (rank 64)
    # ============================================================
    def naive():
        rank = 64
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((16, rank), dtype=float)
        W = np.zeros((16, rank), dtype=float)
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    r = i*16 + k*4 + j
                    U[i*4+k, r] = 1.0
                    V[k*4+j, r] = 1.0
                    W[j*4+i, r] = 1.0
        return U, V, W, rank

    # ============================================================
    # Method 2: Strassen tensor product (rank 49)
    # 4x4 = 2x2 blocks of 2x2, apply Strassen at both levels
    # ============================================================
    def strassen_2x2():
        """Return U2, V2, W2 for 2x2x2 Strassen (rank 7)."""
        U2 = np.zeros((4, 7), dtype=float)
        V2 = np.zeros((4, 7), dtype=float)
        W2 = np.zeros((4, 7), dtype=float)
        
        # For 2x2: A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]]
        # C = A*B, C[i,j] = sum_k A[i,k]*B[k,j]
        # U row = i*2+k, V row = k*2+j, W row = j*2+i (for n=p=2)
        
        # M1 = (A00+A11)*(B00+B11)
        # A00: row 0*2+0=0, A11: row 1*2+1=3
        # B00: row 0*2+0=0, B11: row 1*2+1=3
        # C00: col 0*2+0=0, C11: col 1*2+1=3
        U2[0,0] = 1; U2[3,0] = 1
        V2[0,0] = 1; V2[3,0] = 1
        W2[0,0] = 1; W2[3,0] = 1
        
        # M2 = (A10+A11)*(B00)
        # A10: row 1*2+0=2, A11: row 3
        # B00: row 0
        # C10: col 0*2+1=1, C11: col 3
        U2[2,1] = 1; U2[3,1] = 1
        V2[0,1] = 1
        W2[1,1] = 1; W2[3,1] = -1
        
        # M3 = (A00)*(B01-B11)
        # A00: row 0
        # B01: row 0*2+1=1, B11: row 3
        # C01: col 1*2+0=2, C11: col 3
        # C01 = +M3, C11 = -M3
        U2[0,2] = 1
        V2[1,2] = 1; V2[3,2] = -1
        W2[2,2] = 1; W2[3,2] = -1
        
        # M4 = (A11)*(B10-B00)
        # A11: row 3
        # B10: row 1*2+0=2, B00: row 0
        # C00: col 0, C10: col 1
        U2[3,3] = 1
        V2[2,3] = 1; V2[0,3] = -1
        W2[0,3] = 1; W2[1,3] = 1
        
        # M5 = (A00+A01)*(B11)
        # A00: row 0, A01: row 0*2+1=1
        # B11: row 3
        # C00: col 0, C01: col 2
        U2[0,4] = 1; U2[1,4] = 1
        V2[3,4] = 1
        W2[0,4] = -1; W2[2,4] = 1
        
        # M6 = (A10-A00)*(B00+B01)
        # A10: row 2, A00: row 0
        # B00: row 0, B01: row 1
        # C11: col 3
        U2[2,5] = 1; U2[0,5] = -1
        V2[0,5] = 1; V2[1,5] = 1
        W2[3,5] = 1
        
        # M7 = (A01-A11)*(B10+B11)
        # A01: row 1, A11: row 3
        # B10: row 2, B11: row 3
        # C00: col 0
        U2[1,6] = 1; U2[3,6] = -1
        V2[2,6] = 1; V2[3,6] = 1
        W2[0,6] = 1
        
        return U2, V2, W2

    def strassen_kronecker():
        """Build 4x4x4 decomposition from 2x2x2 Strassen via Kronecker product. Rank 49."""
        U2, V2, W2 = strassen_2x2()
        
        rank49 = 49
        U49 = np.zeros((16, 49), dtype=float)
        V49 = np.zeros((16, 49), dtype=float)
        W49 = np.zeros((16, 49), dtype=float)
        
        # A[i,j] where i = I*2+ii, j = J*2+jj
        # U row index = i*4 + j = (I*2+ii)*4 + (J*2+jj)
        # Block-level Strassen: U2[I*2+J, s]
        # Inner-level Strassen: U2[ii*2+jj, t]
        # Combined: U49[row, s*7+t] = U2[I*2+J, s] * U2[ii*2+jj, t]
        
        for I in range(2):
            for J in range(2):
                for ii in range(2):
                    for jj in range(2):
                        a_row = (I*2 + ii) * 4 + (J*2 + jj)
                        for s in range(7):
                            for t in range(7):
                                r = s * 7 + t
                                coeff = U2[I*2+J, s] * U2[ii*2+jj, t]
                                if coeff != 0:
                                    U49[a_row, r] = coeff
        
        # B[k,l] where k = K*2+kk, l = L*2+ll
        # V row index = k*4 + l = (K*2+kk)*4 + (L*2+ll)
        # Block-level: V2[K*2+L, s]
        # Inner-level: V2[kk*2+ll, t]
        
        for K in range(2):
            for L in range(2):
                for kk in range(2):
                    for ll in range(2):
                        b_row = (K*2 + kk) * 4 + (L*2 + ll)
                        for s in range(7):
                            for t in range(7):
                                r = s * 7 + t
                                coeff = V2[K*2+L, s] * V2[kk*2+ll, t]
                                if coeff != 0:
                                    V49[b_row, r] = coeff
        
        # C[i,j] where i = I*2+ii, j = J*2+jj
        # W row index = j*4 + i = (J*2+jj)*4 + (I*2+ii)
        # Block-level: W2[I*2+J, s]  (output C_block[I,J])
        # Inner-level: W2[ii*2+jj, t] (output C_inner[ii,jj])
        
        for I in range(2):
            for J in range(2):
                for ii in range(2):
                    for jj in range(2):
                        w_row = (J*2 + jj) * 4 + (I*2 + ii)
                        for s in range(7):
                            for t in range(7):
                                r = s * 7 + t
                                coeff = W2[I*2+J, s] * W2[ii*2+jj, t]
                                if coeff != 0:
                                    W49[w_row, r] = coeff
        
        return U49, V49, W49, 49

    # ============================================================
    # Method 3: Bini's 4x4x4 decomposition (rank 49)
    # This is a different 49-rank decomposition.
    # ============================================================
    def bini_4x4x4():
        """
        Bini, Capovani, Lotti, Romani (1979) 4x4x4 decomposition, rank 49.
        Uses a different structure than the simple Kronecker product of Strassen.
        """
        # Actually, the Kronecker product of Strassen IS the Bini et al. construction
        # for 4x4x4. Let me verify and return the same thing.
        # The key insight is that the tensor product of the 2x2x2 Strassen tensor
        # with itself gives a valid 4x4x4 decomposition of rank 7*7=49.
        U49, V49, W49, rank = strassen_kronecker()
        return U49, V49, W49, rank

    # ============================================================
    # Method 4: Try to find a lower rank via optimization
    # ============================================================
    def try_optimization(rank, max_iter=200):
        """Try to find a decomposition of given rank via iterative optimization."""
        if not time_ok():
            return None
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(16, rank) * 0.1
        V = np.random.randn(16, rank) * 0.1
        W = np.random.randn(16, rank) * 0.1
        
        # Use the naive decomposition as initialization if rank >= 64
        # Otherwise use random
        
        best_loss = float('inf')
        lr = 0.01
        
        for iteration in range(max_iter):
            if not time_ok():
                break
            
            # Compute gradient of ||sum_r U_r * V_r * W_r - target||^2
            # L = sum_{i,j,k} (sum_r U[i,r]*V[j,r]*W[k,r] - T[i,j,k])^2
            
            # For efficiency, compute the current reconstruction
            recon = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
            error = recon - target
            loss = np.sum(error**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                break
            
            # Gradient computation
            # dL/dU[i,r] = 2 * sum_{j,k} error[i,j,k] * V[j,r] * W[k,r]
            # dL/dV[j,r] = 2 * sum_{i,k} error[i,j,k] * U[i,r] * W[k,r]
            # dL/dW[k,r] = 2 * sum_{i,j} error[i,j,k] * U[i,r] * V[j,r]
            
            dU = 2 * np.einsum('ijk,jr,kr->ir', error, V, W, optimize=True)
            dV = 2 * np.einsum('ijk,ir,kr->jr', error, U, W, optimize=True)
            dW = 2 * np.einsum('ijk,ir,jr->kr', error, U, V, optimize=True)
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Adaptive learning rate
            if loss > best_loss * 1.1:
                lr *= 0.5
            else:
                lr = min(lr * 1.05, 0.05)
        
        if best_loss < 1e-8:
            # Verify with integer/half-integer rounding
            U_int = np.round(best_U * 2) / 2
            V_int = np.round(best_V * 2) / 2
            W_int = np.round(best_W * 2) / 2
            if verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), rank):
                return U_int, V_int, W_int, rank
            # Also try without rounding
            if verify(best_U.tolist(), best_V.tolist(), best_W.tolist(), rank):
                return best_U, best_V, best_W, rank
        
        return None

    # ============================================================
    # Execute methods in order of expected quality
    # ============================================================
    best_result = None
    best_rank = float('inf')

    # First: naive (rank 64) as guaranteed fallback
    U_n, V_n, W_n, r_n = naive()
    if verify(U_n.tolist(), V_n.tolist(), W_n.tolist(), r_n):
        best_result = (U_n.tolist(), V_n.tolist(), W_n.tolist(), r_n)
        best_rank = r_n

    # Second: Strassen Kronecker (rank 49)
    if time_ok() and best_rank > 49:
        U_s, V_s, W_s, r_s = strassen_kronecker()
        if verify(U_s.tolist(), V_s.tolist(), W_s.tolist(), r_s):
            best_result = (U_s.tolist(), V_s.tolist(), W_s.tolist(), r_s)
            best_rank = r_s

    # Third: Try Bini's construction (should be same as Kronecker, but verify)
    if time_ok() and best_rank > 49:
        U_b, V_b, W_b, r_b = bini_4x4x4()
        if verify(U_b.tolist(), V_b.tolist(), W_b.tolist(), r_b):
            if r_b < best_rank:
                best_result = (U_b.tolist(), V_b.tolist(), W_b.tolist(), r_b)
                best_rank = r_b

    # Fourth: Try optimization for lower rank (48, 47, ...)
    # This is very unlikely to succeed but worth trying
    if time_ok() and best_rank > 48:
        for target_rank in [48, 47, 46]:
            if not time_ok():
                break
            result = try_optimization(target_rank, max_iter=100)
            if result is not None:
                U_o, V_o, W_o, r_o = result
                if r_o < best_rank:
                    best_result = (U_o.tolist(), V_o.tolist(), W_o.tolist(), r_o)
                    best_rank = r_o
                    break

    # Return best result
    if best_result is None:
        # Ultimate fallback
        U_f, V_f, W_f, r_f = naive()
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    U_best, V_best, W_best, _ = best_result
    return {"U": U_best, "V": V_best, "W": W_best}
