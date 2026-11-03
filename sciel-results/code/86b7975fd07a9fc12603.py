import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # n=3, m=3, p=4
    # U: 9 x R, V: 12 x R, W: 12 x R
    
    # Target tensor T (dense)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W):
        """
        U, V, W are lists of lists of Fractions
        Check if they exactly decompose the tensor
        """
        # Check if 2*entry is integer
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if not (val * 2).denominator == 1:
                        return False
        
        # Convert to float arrays for fast verification
        U_arr = np.array([[float(x) for x in row] for row in U])
        V_arr = np.array([[float(x) for x in row] for row in V])
        W_arr = np.array([[float(x) for x in row] for row in W])
        
        # Compute tensor product
        T_calc = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        # Compare with T
        if not np.allclose(T_calc, T, atol=1e-9):
            return False
        return True

    def als_decompose(T, R, max_iters=20, tol=1e-8, seed=None):
        """
        ALS decomposition with structure-aware initialization
        """
        if seed is not None:
            np.random.seed(seed)
        
        n, m, p = DIMENSIONS
        
        # Structure-aware initialization: initialize based on the permutation structure
        # Each (i,j,k) corresponds to a specific (a,b,c)
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        # Scale to help convergence
        U /= np.sqrt(R)
        V /= np.sqrt(R)
        W /= np.sqrt(R)
        
        err = float('inf')
        for it in range(max_iters):
            # Update U
            for a in range(n*m):
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
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_calc - T)
            if err < tol:
                return U, V, W, err
                
        return U, V, W, err

    def rationalize_matrix(M, tol=1e-5):
        """
        M is a numpy array
        Return list of lists of Fractions or None
        Only accepts integer or half-integer values
        """
        rows, cols = M.shape
        result = [[None for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                val = M[i, j]
                if abs(val) < tol:
                    result[i][j] = Fraction(0, 1)
                    continue
                
                # Try to find a rational with small denominator
                frac = Fraction(val).limit_denominator(1000)
                
                # Check if it's close enough
                if abs(float(frac) - val) < tol:
                    # Check if it's a half-integer (2*frac is integer)
                    if (frac * 2).denominator == 1:
                        result[i][j] = frac
                    else:
                        return None
                else:
                    return None
        return result

    def polish_solution(U, V, W, T, max_iters=50):
        """
        High-precision polish using exact rational arithmetic
        Try to find exact integer/half-integer solution near the given approximate solution
        """
        n, m, p = DIMENSIONS
        R = U.shape[1]
        
        # Convert to Fractions with high precision
        U_frac = [[Fraction(U[i, j]).limit_denominator(100000) for j in range(R)] for i in range(n*m)]
        V_frac = [[Fraction(V[i, j]).limit_denominator(100000) for j in range(R)] for i in range(m*p)]
        W_frac = [[Fraction(W[i, j]).limit_denominator(100000) for j in range(R)] for i in range(p*n)]
        
        # Try to snap to nearest half-integers
        def snap_to_half_int(frac):
            # Find nearest half-integer
            val = float(frac)
            # Nearest half-integer: round(val*2)/2
            nearest = round(val * 2) / 2.0
            return Fraction(int(round(val * 2)), 2)
        
        U_snapped = [[snap_to_half_int(f) for f in row] for row in U_frac]
        V_snapped = [[snap_to_half_int(f) for f in row] for row in V_frac]
        W_snapped = [[snap_to_half_int(f) for f in row] for row in W_frac]
        
        # Verify the snapped solution
        if verify(U_snapped, V_snapped, W_snapped):
            return U_snapped, V_snapped, W_snapped
        
        # If snapping doesn't work, try local search around the snapped values
        # This is expensive, so only do it if we have time
        if time.time() > deadline - 20:
            return None, None, None
        
        # Try small perturbations
        best_err = float('inf')
        best_sol = None
        
        # For each matrix, try perturbing each entry by small amounts
        # This is a heuristic - we'll try a limited number of perturbations
        
        for _ in range(20):  # Limited perturbations
            U_try = [[f for f in row] for row in U_snapped]
            V_try = [[f for f in row] for row in V_snapped]
            W_try = [[f for f in row] for row in W_snapped]
            
            # Randomly perturb a few entries
            for _ in range(3):
                mat_choice = np.random.choice(3)
                if mat_choice == 0:
                    i, j = np.random.randint(0, n*m), np.random.randint(0, R)
                    U_try[i][j] = U_snapped[i][j] + Fraction(np.random.choice([-1, 1]), 2)
                elif mat_choice == 1:
                    i, j = np.random.randint(0, m*p), np.random.randint(0, R)
                    V_try[i][j] = V_snapped[i][j] + Fraction(np.random.choice([-1, 1]), 2)
                else:
                    i, j = np.random.randint(0, p*n), np.random.randint(0, R)
                    W_try[i][j] = W_snapped[i][j] + Fraction(np.random.choice([-1, 1]), 2)
            
            if verify(U_try, V_try, W_try):
                return U_try, V_try, W_try
        
        return None, None, None

    # Search for minimal R
    # Theoretical lower bound: max(n*m, m*p, p*n) / something, but let's be practical
    # For matrix multiplication, the rank is often close to n*m*p but can be lower
    # Let's search from a reasonable lower bound upward
    
    best_result = None
    
    for R in range(18, 37):
        if time.time() > deadline - 15:
            break
            
        # Try several random restarts with different seeds
        found = False
        restarts = 15 if R < 25 else 8
        
        for restart in range(restarts):
            if time.time() > deadline - 10:
                break
            
            seed = restart * 1000 + R
            U, V, W, err = als_decompose(T, R, max_iters=40, tol=1e-7, seed=seed)
            
            if err < 1e-5:
                # Try to rationalize
                U_frac = rationalize_matrix(U, tol=1e-4)
                if U_frac:
                    V_frac = rationalize_matrix(V, tol=1e-4)
                    if V_frac:
                        W_frac = rationalize_matrix(W, tol=1e-4)
                        if W_frac:
                            if verify(U_frac, V_frac, W_frac):
                                # Found a valid solution!
                                U_out = [[float(x) for x in row] for row in U_frac]
                                V_out = [[float(x) for x in row] for row in V_frac]
                                W_out = [[float(x) for x in row] for row in W_frac]
                                return {"U": U_out, "V": V_out, "W": W_out}
                            else:
                                # Try polishing
                                U_pol, V_pol, W_pol = polish_solution(U, V, W, T)
                                if U_pol is not None:
                                    U_out = [[float(x) for x in row] for row in U_pol]
                                    V_out = [[float(x) for x in row] for row in V_pol]
                                    W_out = [[float(x) for x in row] for row in W_pol]
                                    return {"U": U_out, "V": V_out, "W": W_out}
        
    # Fallback to schoolbook
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
