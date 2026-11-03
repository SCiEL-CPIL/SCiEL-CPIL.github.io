import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 145.0
SAFETY_MARGIN = 5.0

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    
    def check_valid(U, V, W, tol=1e-9):
        """Check if the decomposition is valid."""
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_arr[i*m + j, c] * V_arr[j*p + k, c] * W_arr[k*n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def build_schoolbook():
        """Build the trivial schoolbook decomposition of rank n*m*p."""
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W

    def build_strassen_2x2_2x2():
        """
        Build Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
        Returns U (4x7), V (4x7), W (4x7) for the 2x2x2 case.
        U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # p1 = (a11 + a22)(b11 + b22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # p2 = (a21 + a22)b11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # p3 = a11(b12 - b22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # p4 = a22(b21 - b11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # p5 = (a11 + a12)b22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # p6 = (a21 - a11)(b11 + b12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # p7 = (a12 - a22)(b21 + b22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    def kronecker_decompose_4x4x4():
        """
        Build a decomposition for 4x4x4 using Kronecker product of two 2x2x2 Strassen decompositions.
        This gives rank 7*7 = 49.
        """
        U2, V2, W2 = build_strassen_2x2_2x2()
        
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                                if val != 0:
                                    U4[i*4+j][r] += val
                                  
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                                if val != 0:
                                    V4[j*4+k][r] += val
                                  
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                                if val != 0:
                                    W4[k*4+i][r] += val
        
        # Remove zero columns
        nonzero_cols = [r for r in range(49) if any(U4[i][r] != 0 or V4[j][r] != 0 or W4[k][r] != 0 for i in range(16) for j in range(16) for k in range(16))]
        
        if len(nonzero_cols) < 49:
            U4 = [[U4[i][r] for r in nonzero_cols] for i in range(16)]
            V4 = [[V4[j][r] for r in nonzero_cols] for j in range(16)]
            W4 = [[W4[k][r] for r in nonzero_cols] for k in range(16)]
        
        return U4, V4, W4

    def build_4x4x7_decomposition():
        """
        Build a decomposition for 4x4x7 by combining the 4x4x4 decomposition with additional terms.
        """
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])
        
        r_total = r4 + 48
        U = [[0] * r_total for _ in range(16)]
        V = [[0] * r_total for _ in range(28)]
        W = [[0] * r_total for _ in range(28)]
        
        for i in range(16):
            for r in range(r4):
                U[i][r] = U4[i][r]
        
        for j in range(4):
            for k in range(4):
                for r in range(r4):
                    V[j*7 + k][r] = V4[j*4 + k][r]
        
        for k in range(4):
            for i in range(4):
                for r in range(r4):
                    W[k*4 + i][r] = W4[k*4 + i][r]
        
        col_offset = r4
        for i in range(4):
            for j in range(4):
                for k in range(4, 7):
                    U[i*4 + j][col_offset] = 1
                    V[j*7 + k][col_offset] = 1
                    W[k*4 + i][col_offset] = 1
                    col_offset += 1
        
        return U, V, W

    def fft_diagonalize_kronecker():
        """
        Use FFT to diagonalize the Kronecker product structure.
        Embed the 4x4x4 multiplication tensor into a polynomial ring and use
        FFT to find a sparse representation that may reduce the effective rank.
        """
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])
        
        U4_np = np.array(U4, dtype=complex)
        V4_np = np.array(V4, dtype=complex)
        W4_np = np.array(W4, dtype=complex)
        
        # Try to find a change of basis that sparsifies the decomposition
        # by diagonalizing the Kronecker structure via FFT
        
        # The Kronecker product structure means the 49 columns are outer products
        # of 7-dimensional Strassen vectors. We can try to find a unitary transform
        # that makes most entries zero.
        
        # For each dimension, try FFT-based diagonalization
        # U4 has shape (16, 49) where 16 = 4*4 and 49 = 7*7
        
        # Reshape to see the Kronecker structure
        # U4[i*4+j, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
        # where i = i1*2+i2, j = j1*2+j2
        
        # Try applying FFT along the Kronecker dimensions
        # This is equivalent to changing basis in the 7-dimensional Strassen space
        
        # For each of the two Kronecker factors, try different unitary transforms
        # and see if we can reduce the effective rank
        
        best_U4, best_V4, best_W4 = U4, V4, W4
        best_rank = r4
        
        # Try random unitary transforms on the Strassen space
        # If we can find transforms that make many entries zero, we reduce rank
        
        rng = np.random.RandomState(42)
        
        for trial in range(200):
            if time.time() - start_time > WALL_DEADLINE - SAFETY_MARGIN:
                break
            
            # Generate random unitary matrices for the 7-dimensional Strassen space
            # Use QR decomposition of random complex Gaussian matrices
            Q1 = np.linalg.qr(rng.randn(7, 7) + 1j*rng.randn(7, 7))[0]
            Q2 = np.linalg.qr(rng.randn(7, 7) + 1j*rng.randn(7, 7))[0]
            
            # Apply the transforms: new_col = Q1 @ old_col1 * Q2 @ old_col2
            # This is equivalent to a change of basis in the Kronecker product space
            
            # Reshape U4 to (16, 7, 7) and apply transforms
            U4_reshaped = U4_np.reshape(16, 7, 7)
            V4_reshaped = V4_np.reshape(16, 7, 7)
            W4_reshaped = W4_np.reshape(16, 7, 7)
            
            # Apply Q1 on first dimension, Q2 on second dimension
            U4_new = np.einsum('ab,icd,bd->iac', Q1, U4_reshaped, Q2)
            V4_new = np.einsum('ab,icd,bd->iac', Q1, V4_reshaped, Q2)
            W4_new = np.einsum('ab,icd,bd->iac', Q1, W4_reshaped, Q2)
            
            # Reshape back
            U4_new_flat = U4_new.reshape(16, 49)
            V4_new_flat = V4_new.reshape(16, 49)
            W4_new_flat = W4_new.reshape(16, 49)
            
            # Check which columns are (approximately) zero
            col_norms = np.sqrt(np.sum(np.abs(U4_new_flat)**2, axis=0) + 
                               np.sum(np.abs(V4_new_flat)**2, axis=0) + 
                               np.sum(np.abs(W4_new_flat)**2, axis=0))
            
            # Columns with very small norm can be dropped
            threshold = 1e-10
            nonzero_mask = col_norms > threshold
            num_nonzero = np.sum(nonzero_mask)
            
            if num_nonzero < best_rank:
                best_rank = int(num_nonzero)
                best_U4 = U4_new_flat[:, nonzero_mask].real
                best_V4 = V4_new_flat[:, nonzero_mask].real
                best_W4 = W4_new_flat[:, nonzero_mask].real
        
        # Convert back to lists and verify
        if best_rank < r4:
            U4_out = best_U4.tolist()
            V4_out = best_V4.tolist()
            W4_out = best_W4.tolist()
            
            # Verify validity
            if check_valid(U4_out, V4_out, W4_out, tol=1e-6):
                return U4_out, V4_out, W4_out
        
        return U4, V4, W4

    def build_4x4x7_with_fft():
        """
        Build 4x4x7 decomposition using FFT-diagonalized 4x4x4 part.
        """
        U4, V4, W4 = fft_diagonalize_kronecker()
        r4 = len(U4[0])
        
        # Use schoolbook for the remaining 3 columns
        r_total = r4 + 48
        U = [[0] * r_total for _ in range(16)]
        V = [[0] * r_total for _ in range(28)]
        W = [[0] * r_total for _ in range(28)]
        
        for i in range(16):
            for r in range(r4):
                U[i][r] = U4[i][r]
        
        for j in range(4):
            for k in range(4):
                for r in range(r4):
                    V[j*7 + k][r] = V4[j*4 + k][r]
        
        for k in range(4):
            for i in range(4):
                for r in range(r4):
                    W[k*4 + i][r] = W4[k*4 + i][r]
        
        col_offset = r4
        for i in range(4):
            for j in range(4):
                for k in range(4, 7):
                    U[i*4 + j][col_offset] = 1
                    V[j*7 + k][col_offset] = 1
                    W[k*4 + i][col_offset] = 1
                    col_offset += 1
        
        return U, V, W

    # Start with schoolbook as fallback
    U_school, V_school, W_school = build_schoolbook()
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = 112
    
    # Try the Kronecker product approach
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_kron, V_kron, W_kron = build_4x4x7_decomposition()
            r_kron = len(U_kron[0])
            
            if check_valid(U_kron, V_kron, W_kron):
                if r_kron < best_rank:
                    best_result = {"U": U_kron, "V": V_kron, "W": W_kron}
                    best_rank = r_kron
        except Exception as e:
            pass
    
    # Try FFT-diagonalized approach
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_fft, V_fft, W_fft = build_4x4x7_with_fft()
            r_fft = len(U_fft[0])
            
            if check_valid(U_fft, V_fft, W_fft):
                if r_fft < best_rank:
                    best_result = {"U": U_fft, "V": V_fft, "W": W_fft}
                    best_rank = r_fft
        except Exception as e:
            pass
    
    # Try random search to reduce rank further
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        U, V, W = best_result["U"], best_result["V"], best_result["W"]
        r = len(U[0])
        
        # Convert to numpy for efficiency
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for attempt in range(500):
            if time.time() - start_time > WALL_DEADLINE - SAFETY_MARGIN:
                break
            
            # Try to merge two random columns
            if r <= 1:
                break
                
            c1 = random.randint(0, r-1)
            c2 = random.randint(0, r-1)
            
            if c1 == c2:
                continue
            
            # Try to find a linear combination that eliminates one column
            alpha = random.uniform(-1, 1)
            beta = random.uniform(-1, 1)
            
            U_new = U_np.copy()
            V_new = V_np.copy()
            W_new = W_np.copy()
            
            U_new[:, c1] = alpha * U_new[:, c1] + beta * U_new[:, c2]
            V_new[:, c1] = alpha * V_new[:, c1] + beta * V_new[:, c2]
            W_new[:, c1] = alpha * W_new[:, c1] + beta * W_new[:, c2]
            
            # Remove column c2
            mask = [i for i in range(r) if i != c2]
            U_new = U_new[:, mask]
            V_new = V_new[:, mask]
            W_new = W_new[:, mask]
            
            # Check validity
            r_new = len(U_new[0])
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r_new):
                            s += U_new[i*m + j, c] * V_new[j*p + k, c] * W_new[k*n + i, c]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid and r_new < best_rank:
                U_list = U_new.tolist()
                V_list = V_new.tolist()
                W_list = W_new.tolist()
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = r_new
                U_np = U_new
                V_np = V_new
                W_np = W_new
                r = r_new
    
    return best_result
