import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 140  # Leave 10s margin for safety

    # Precompute the target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify(U, V, W, R):
        """Verify exact decomposition with integer/half-integer entries."""
        # Check for integer/half-integer
        for arr in [U, V, W]:
            for val in arr:
                if not (val * 2 == int(val * 2)):
                    return False
        
        Uf = np.array(U, dtype=np.float64)
        Vf = np.array(V, dtype=np.float64)
        Wf = np.array(W, dtype=np.float64)
        
        # Exact check using Fraction is slow, so do float check first
        T_rec = np.einsum('ar,br,cr->abc', Uf, Vf, Wf)
        if not np.allclose(T_rec, T, atol=1e-10):
            return False
        
        # Final exact check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = Fraction(0)
                    for r in range(R):
                        s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                    if s != 1:
                        return False
        return True

    # Build schoolbook as fallback
    best_rank = n * m * p
    U_school = [[0] * best_rank for _ in range(n * m)]
    V_school = [[0] * best_rank for _ in range(m * p)]
    W_school = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_decomp = (U_school, V_school, W_school, best_rank)

    def try_rank(R, max_iters=5000):
        """Try to find a rank R decomposition using gradient descent."""
        if time.time() - start_time > timeout:
            return None
            
        # Initialize with small random values
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        # Gradient descent
        lr = 0.01
        for it in range(max_iters):
            if time.time() - start_time > timeout:
                return None
            
            # Compute T_rec
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            error = T_rec - T
            loss = np.sum(error**2)
            
            # Gradients
            # dL/dU = error * (V * W^T) ... wait, indices
            # T_rec_abc = sum_r U_ar V_br W_cr
            # dT_rec_abc / dU_ar = V_br W_cr
            # dL/dU_ar = sum_bc error_abc V_br W_cr = (error * (V[:,:,None] * W[:,:,None])) ...
            
            # Vectorized gradients
            # grad_U_ar = sum_bc error_abc V_br W_cr
            # grad_U = np.einsum('abc,br,cr->ar', error, V, W)
            # grad_V_br = sum_ac error_abc U_ar W_cr
            # grad_V = np.einsum('abc,ar,cr->br', error, U, W)
            # grad_W_cr = sum_ab error_abc U_ar V_br
            # grad_W = np.einsum('abc,ar,br->cr', error, U, V)
            
            grad_U = np.einsum('abc,br,cr->ar', error, V, W)
            grad_V = np.einsum('abc,ar,cr->br', error, U, W)
            grad_W = np.einsum('abc,ar,br->cr', error, U, V)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if loss < 1e-12:
                break
        
        if loss < 1e-12:
            # Check if we can round to integers/half-integers
            # If the solution is close to integers, round it
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify the rounded solution
            if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), R):
                return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), R)
                
            # If rounding failed, maybe the values are integers already?
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), R):
                return (U_int.tolist(), V_int.tolist(), W_int.tolist(), R)
                
    # Try ranks from 20 to 35
    for R in range(20, 36):
        if time.time() - start_time > timeout:
            break
            
        result = try_rank(R)
        if result is not None:
            U, V, W, rank = result
            best_decomp = (U, V, W, rank)
            break
            
    return {"U": best_decomp[0], "V": best_decomp[1], "W": best_decomp[2]}
