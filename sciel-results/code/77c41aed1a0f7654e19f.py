"""Find rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T[i,j,k] where:
    # i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 (the tensor that computes (AB)_{i,k} = sum_j A_{i,j} B_{j,k})
    # 
    # Rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # We need to find minimal r such that there exist U (nm x r), V (mp x r), W (pn x r)
    # with integer/half-integer entries satisfying the above exactly.
    
    # Lower bound: max(n*m, m*p, n*p) = max(10, 30, 12) = 30
    # Try to find decomposition starting from lower bound
    
    deadline = time.time() + 100  # Keep safety margin before 120s limit
    
    def verify(U, V, W):
        """Verify that U, V, W give exact decomposition."""
        n, m, p = DIMENSIONS
        r = U.shape[1]
        # Check: for all i,j,k: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # T[i,j,k] = 1 for all valid i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def to_fraction_matrix(M):
        """Convert numpy matrix to fraction matrix, checking for half-integers."""
        fr = np.zeros(M.shape, dtype=object)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                # Check if it's close to a half-integer
                f = Fraction(val).limit_denominator(2)
                if abs(float(f) - val) > 1e-8:
                    # Try with denominator 2
                    f2 = Fraction(round(val * 2) / 2).limit_denominator(2)
                    if abs(float(f2) - val) > 1e-8:
                        return None
                    fr[i, j] = f2
                else:
                    fr[i, j] = f
        return fr
    
    # Try ranks from lower bound upward
    for rank in range(30, 61):
        if time.time() > deadline:
            break
        
        # Try random initialization with gradient descent
        best_result = None
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with small random values
            np.random.seed(42 + attempt * 1000 + rank)
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Gradient descent
            lr = 0.01
            for step in range(2000):
                if time.time() > deadline:
                    break
                
                # Compute gradient
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # d/dU[i*m+j,c] of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
                            for c in range(rank):
                                err = 0
                                for cc in range(rank):
                                    err += U[i*m+j, cc] * V[j*p+k, cc] * W[k*n+i, cc]
                                err -= 1
                                grad_U[i*m+j, c] += 2 * err * V[j*p+k, c] * W[k*n+i, c]
                                grad_V[j*p+k, c] += 2 * err * U[i*m+j, c] * W[k*n+i, c]
                                grad_W[k*n+i, c] += 2 * err * U[i*m+j, c] * V[j*p+k, c]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if step % 100 == 0:
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for c in range(rank):
                                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                                max_err = max(max_err, abs(s - 1))
                    if max_err < 1e-6:
                        break
            
            # Check if we found a good solution
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        max_err = max(max_err, abs(s - 1))
            
            if max_err < 1e-5:
                # Try to snap to half-integers
                U_frac = to_fraction_matrix(U)
                V_frac = to_fraction_matrix(V)
                W_frac = to_fraction_matrix(W)
                if U_frac is not None and V_frac is not None and W_frac is not None:
                    # Verify exactness
                    exact_ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = Fraction(0)
                                for c in range(rank):
                                    s += U_frac[i*m+j, c] * V_frac[j*p+k, c] * W_frac[k*n+i, c]
                                if s != 1:
                                    exact_ok = False
                                    break
                        if not exact_ok:
                            break
                    if exact_ok:
                        best_result = (U_frac, V_frac, W_frac)
                        break
        
        if best_result is not None:
            U_frac, V_frac, W_frac = best_result
            U_list = [[float(U_frac[i, j]) for j in range(rank)] for i in range(n*m)]
            V_list = [[float(V_frac[i, j]) for j in range(rank)] for i in range(m*p)]
            W_list = [[float(W_frac[i, j]) for j in range(rank)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
