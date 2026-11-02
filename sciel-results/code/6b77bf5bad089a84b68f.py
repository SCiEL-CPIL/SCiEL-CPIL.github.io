import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s to be safe
    
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, R), V (m*p, R), W (p*n, R)
    # n=3, m=3, p=4
    # U: 9 x R
    # V: 12 x R
    # W: 12 x R
    
    # Construct target tensor T (dense)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def check_exact(U, V, W):
        # U, V, W are lists of lists of Fractions or floats
        # We assume they are floats here, and we check if they are close to integers/half-integers
        # and if they reproduce the tensor exactly (within tolerance)
        # Actually, we will check the tensor product first.
        pass

    def rationalize_matrix(M, tol=1e-6):
        # M is a numpy array
        # Try to find rational approximations with denominator <= 10
        # Return list of lists of Fractions or None
        rows = M.shape[0]
        cols = M.shape[1]
        result = [[None for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                val = M[i, j]
                # Try denominators 1 to 10
                found = False
                for d in range(1, 11):
                    num = round(val * d)
                    frac = Fraction(num, d)
                    if abs(float(frac) - val) < tol:
                        result[i][j] = frac
                        found = True
                        break
                if not found:
                    return None
        return result

    def is_half_integer(frac):
        # Check if frac * 2 is an integer
        return (frac * 2).denominator == 1

    def verify(U, V, W):
        # U, V, W are lists of lists of Fractions
        # Check if 2*entry is integer
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if not is_half_integer(val):
                        return False
        # Check tensor product
        # T_calc[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # We only need to check non-zero entries of T
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = Fraction(0, 1)
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 1:
                        return False
        # Check zero entries? 
        # There are many zero entries. We should check a sample or all.
        # Given the small size, we can check all.
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    # If (a,b,c) is not a valid (i,j,k) path, it should be 0
                    # Determine if (a,b,c) is valid
                    i1 = a // m
                    j1 = a % m
                    j2 = b // p
                    k2 = b % p
                    k1 = c // n
                    i2 = c % n
                    if (i1 == i2) and (j1 == j2):
                        # This is a valid path, already checked above (should be 1)
                        # But wait, if i1==i2 and j1==j2, then k1 must be k2 for it to be the same term?
                        # Actually, the tensor is 1 ONLY for the specific (i,j,k) triple.
                        # If i1==i2 and j1==j2, then the term is A[i,j] B[j,k].
                        # It contributes to C[k,i].
                        # So c must be k*n + i.
                        # Here c = k1*n + i2.
                        # So we need k1 == k2.
                        if k1 != k2:
                            # This index (a,b,c) corresponds to A[i,j] B[j,k2] contributing to C[k1,i].
                            continue
                        
        # Simpler check: just compute the full tensor and compare to T
        T_calc = np.zeros((n*m, m*p, p*n))
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    s = 0.0
                    for r in range(rank):
                        s += float(U[a][r]) * float(V[b][r]) * float(W[c][r])
                    T_calc[a, b, c] = s
        
        # Compare with T
        if not np.allclose(T_calc, T, atol=1e-9):
            return False
        return True

    def als_decompose(T, R, max_iters=50, tol=1e-8):
        n, m, p = DIMENSIONS
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        for _ in range(max_iters):
            # Update U
            for a in range(n*m):
                # Minimize sum_{b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
                # This is linear in U[a,:]
                # Let M[a,b,c] = V[b,:] * W[c,:] (element-wise product, shape R)
                # We want U[a,:] such that sum_r U[a,r] M[a,b,c][r] ~ T[a,b,c]
                # This is a least squares problem: A x = y
                # A is (m*p*p*n) x R ? No.
                # For fixed a, we have 12*12=144 equations.
                # Variables: U[a, 0..R-1]
                # Coefficients for U[a,r] in equation (b,c) is V[b,r] * W[c,r]
                
                A = np.zeros((m*p * p*n, R))
                y = np.zeros(m*p * p*n)
                idx = 0
                for b in range(m*p):
                    for c in range(p*n):
                        A[idx, :] = V[b, :] * W[c, :]
                        y[idx] = T[a, b, c]
                        idx += 1
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                U[a, :] = sol
                
            # Update V
            for b in range(m*p):
                A = np.zeros((n*m * p*n, R))
                y = np.zeros(n*m * p*n)
                idx = 0
                for a in range(n*m):
                    for c in range(p*n):
                        A[idx, :] = U[a, :] * W[c, :]
                        y[idx] = T[a, b, c]
                        idx += 1
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                V[b, :] = sol
                
            # Update W
            for c in range(p*n):
                A = np.zeros((n*m * m*p, R))
                y = np.zeros(n*m * m*p)
                idx = 0
                for a in range(n*m):
                    for b in range(m*p):
                        A[idx, :] = U[a, :] * V[b, :]
                        y[idx] = T[a, b, c]
                        idx += 1
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                W[c, :] = sol
            
            # Check convergence
            # Compute residual
            # T_calc = sum_r U[:,r,None] * V[:,r,None] * W[:,r,None]
            # This is slow if done naively.
            # But dimensions are small.
            T_calc = np.zeros((n*m, m*p, p*n))
            for r in range(R):
                T_calc += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            err = np.linalg.norm(T_calc - T)
            if err < tol:
                return U, V, W, err
                
        return U, V, W, err

    # Search for minimal R
    best_result = None
    for R in range(12, 37):
        if time.time() > deadline - 10:
            break
            
        # Try a few random restarts
        found = False
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            U, V, W, err = als_decompose(T, R, max_iters=30)
            if err < 1e-6:
                # Try to rationalize
                U_frac = rationalize_matrix(U, tol=1e-5)
                V_frac = rationalize_matrix(V, tol=1e-5)
                W_frac = rationalize_matrix(W, tol=1e-5)
                if U_frac and V_frac and W_frac:
                    if verify(U_frac, V_frac, W_frac):
                        # Convert to lists of floats/ints for output?
                        # The task says "integer/half-integer entries".
                        # The return format is {'U', 'V', 'W'}.
                        # The seed uses lists of lists of numbers.
                        # We should return numbers that are exactly representable.
                        # Fractions are good, but maybe the evaluator expects floats?
                        # "integer/half-integer entries" suggests floats like 0.5, 1.0, etc.
                        # Let's return floats derived from fractions.
                        U_out = [[float(x) for x in row] for row in U_frac]
                        V_out = [[float(x) for x in row] for row in V_frac]
                        W_out = [[float(x) for x in row] for row in W_frac]
                        return {"U": U_out, "V": V_out, "W": W_out}
        
    # Fallback to schoolbook
    return schoolbook()
