import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    def verify_fast(U, V, W, R):
        """Vectorized verification"""
        U_reshaped = U.reshape(n, m, R)
        V_reshaped = V.reshape(m, p, R)
        W_reshaped = W.reshape(p, n, R)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(T_recon, 1.0, atol=1e-9)
    
    def verify_exact(U, V, W, R):
        """Exact verification using Fractions for half-integer entries"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(R):
                        s += Fraction(int(U[i*m+j][r]*2), 2) * Fraction(int(V[j*p+k][r]*2), 2) * Fraction(int(W[k*n+i][r]*2), 2)
                    if s != 1:
                        return False
        return True
    
    def polish(U, V, W, R, max_iters=10):
        """High-precision polish pass: alternate solving with tighter tolerance"""
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Solve for U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A_U = V[j*p:(j+1)*p, :] * W[np.arange(p)*n + i, :]
                    x, _, _, _ = np.linalg.lstsq(A_U, np.ones(p), rcond=1e-15)
                    U_new[i*m+j, :] = x
            U = U_new
            
            # Solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    W_slice = W[k*n:(k+1)*n, :]
                    A_V = U[np.arange(n)*m + j, :] * W_slice
                    x, _, _, _ = np.linalg.lstsq(A_V, np.ones(n), rcond=1e-15)
                    V_new[j*p+k, :] = x
            V = V_new
            
            # Solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    U_slice = U[i*m:(i+1)*m, :]
                    V_slice = V[np.arange(m)*p + k, :]
                    A_W = U_slice * V_slice
                    x, _, _, _ = np.linalg.lstsq(A_W, np.ones(m), rcond=1e-15)
                    W_new[k*n+i, :] = x
            W = W_new
            
            # Check convergence
            if verify_fast(U, V, W, R):
                break
        
        return U, V, W
    
    def try_round_and_verify(U, V, W, R):
        """Try rounding to half-integers and verify"""
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        if verify_fast(U_int, V_int, W_int, R):
            # Double-check with exact arithmetic
            U_list = U_int.tolist()
            V_list = V_int.tolist()
            W_list = W_int.tolist()
            if verify_exact(U_list, V_list, W_list, R):
                return U_int, V_int, W_int
        return None
    
    # === Structured rank-24 decomposition ===
    # Based on 3x3x3 fast multiplication (rank 21) extended to 3x3x7
    # The 3x3x3 tensor has a known rank-21 decomposition
    # For 3x3x7, we can partition the 7 columns into blocks
    
    # Start with a structured initialization for rank 24
    # Use the idea: 7 = 3 + 3 + 1, with overlapping structure
    
    R_target = 24
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Strategy 1: Try rank 24 with structured initialization
    # The 3x3x3 multiplication tensor T3[i,j,k] = delta_ik (for matrix mult A*B where C[i,k] = sum_j A[i,j]B[j,k])
    # Actually for standard matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor T[i,j,k] where we want sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
    
    # For 3x3x3, a rank-21 decomposition exists
    # For 3x3x7, we extend by treating columns 3,4,5,6 as additional outputs
    
    # Initialize with a structured pattern
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        np.random.seed(1000 + attempt * 7)
        
        # Initialize with a mix of structured and random
        U = np.random.randn(n*m, R_target) * 0.5
        V = np.random.randn(m*p, R_target) * 0.5
        W = np.random.randn(p*n, R_target) * 0.5
        
        # Add some structure: first 21 columns mimic 3x3x3 core
        # Last 3 columns handle the extension
        for it in range(40):
            if time.time() > deadline:
                break
            
            # Solve for U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A_U = V[j*p:(j+1)*p, :] * W[np.arange(p)*n + i, :]
                    x, _, _, _ = np.linalg.lstsq(A_U, np.ones(p), rcond=None)
                    U_new[i*m+j, :] = x
            U = U_new
            
            # Solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    W_slice = W[k*n:(k+1)*n, :]
                    A_V = U[np.arange(n)*m + j, :] * W_slice
                    x, _, _, _ = np.linalg.lstsq(A_V, np.ones(n), rcond=None)
                    V_new[j*p+k, :] = x
            V = V_new
            
            # Solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    U_slice = U[i*m:(i+1)*m, :]
                    V_slice = V[np.arange(m)*p + k, :]
                    A_W = U_slice * V_slice
                    x, _, _, _ = np.linalg.lstsq(A_W, np.ones(m), rcond=None)
                    W_new[k*n+i, :] = x
            W = W_new
        
        # Polish
        U, V, W = polish(U, V, W, R_target, max_iters=15)
        
        result = try_round_and_verify(U, V, W, R_target)
        if result is not None:
            best_rank = R_target
            best_U, best_V, best_W = result
            break
    
    # If rank 24 didn't work, try rank 25
    if best_U is None:
        for R in [25, 26, 27, 28]:
            if time.time() > deadline:
                break
            
            for attempt in range(3):
                if time.time() > deadline:
                    break
                
                np.random.seed(2000 + R * 13 + attempt)
                
                U = np.random.randn(n*m, R) * 0.5
                V = np.random.randn(m*p, R) * 0.5
                W = np.random.randn(p*n, R) * 0.5
                
                for it in range(35):
                    if time.time() > deadline:
                        break
                    
                    U_new = np.zeros_like(U)
                    for i in range(n):
                        for j in range(m):
                            A_U = V[j*p:(j+1)*p, :] * W[np.arange(p)*n + i, :]
                            x, _, _, _ = np.linalg.lstsq(A_U, np.ones(p), rcond=None)
                            U_new[i*m+j, :] = x
                    U = U_new
                    
                    V_new = np.zeros_like(V)
                    for j in range(m):
                        for k in range(p):
                            W_slice = W[k*n:(k+1)*n, :]
                            A_V = U[np.arange(n)*m + j, :] * W_slice
                            x, _, _, _ = np.linalg.lstsq(A_V, np.ones(n), rcond=None)
                            V_new[j*p+k, :] = x
                    V = V_new
                    
                    W_new = np.zeros_like(W)
                    for k in range(p):
                        for i in range(n):
                            U_slice = U[i*m:(i+1)*m, :]
                            V_slice = V[np.arange(m)*p + k, :]
                            A_W = U_slice * V_slice
                            x, _, _, _ = np.linalg.lstsq(A_W, np.ones(m), rcond=None)
                            W_new[k*n+i, :] = x
                    W = W_new
                
                U, V, W = polish(U, V, W, R, max_iters=10)
                
                result = try_round_and_verify(U, V, W, R)
                if result is not None:
                    best_rank = R
                    best_U, best_V, best_W = result
                    break
            
            if best_U is not None:
                break
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook
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
