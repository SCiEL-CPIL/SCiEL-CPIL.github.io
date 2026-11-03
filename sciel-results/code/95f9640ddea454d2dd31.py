import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
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

    def sparse_als_decompose(T, R, max_iters=200, tol=1e-8, seed=None, sparsity_k=3):
        """
        Sparse ALS decomposition with hard thresholding
        """
        if seed is not None:
            np.random.seed(seed)
        
        n, m, p = DIMENSIONS
        
        # Initialize with small random values
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
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
                
                # Hard Thresholding: keep top sparsity_k values
                if sparsity_k < R:
                    abs_sol = np.abs(sol)
                    thresh = np.partition(abs_sol, -sparsity_k)[-sparsity_k]
                    mask = abs_sol >= thresh
                    sol[~mask] = 0
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
                
                if sparsity_k < R:
                    abs_sol = np.abs(sol)
                    thresh = np.partition(abs_sol, -sparsity_k)[-sparsity_k]
                    mask = abs_sol >= thresh
                    sol[~mask] = 0
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
                
                if sparsity_k < R:
                    abs_sol = np.abs(sol)
                    thresh = np.partition(abs_sol, -sparsity_k)[-sparsity_k]
                    mask = abs_sol >= thresh
                    sol[~mask] = 0
                W[c, :] = sol
            
            # Check convergence
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_calc - T)
            if err < tol:
                return U, V, W, err
                
        return U, V, W, err

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
            val = float(frac)
            nearest = round(val * 2) / 2.0
            return Fraction(int(round(val * 2)), 2)
        
        U_snapped = [[snap_to_half_int(f) for f in row] for row in U_frac]
        V_snapped = [[snap_to_half_int(f) for f in row] for row in V_frac]
        W_snapped = [[snap_to_half_int(f) for f in row] for row in W_frac]
        
        # Verify the snapped solution
        if verify(U_snapped, V_snapped, W_snapped):
            return U_snapped, V_snapped, W_snapped
        
        # If snapping doesn't work, try local search around the snapped values
        if time.time() > deadline - 20:
            return None, None, None
        
        # Try small perturbations
        for _ in range(50):  # More perturbations
            U_try = [[f for f in row] for row in U_snapped]
            V_try = [[f for f in row] for row in V_snapped]
            W_try = [[f for f in row] for row in W_snapped]
            
            # Randomly perturb a few entries
            for _ in range(5):
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
    # Known rank for 3x3x4 is 10. Start there.
    best_result = None
    
    for R in range(10, 15):
        if time.time() > deadline - 15:
            break
            
        # Try several random restarts with different seeds
        found = False
        restarts = 20 if R == 10 else 10
        
        for restart in range(restarts):
            if time.time() > deadline - 10:
                break
            
            seed = restart * 1000 + R
            # Use sparsity to help convergence
            U, V, W, err = sparse_als_decompose(T, R, max_iters=100, tol=1e-6, seed=seed, sparsity_k=3)
            
            if err < 1e-4:
                # Try to rationalize
                U_frac = rationalize_matrix(U, tol=1e-4)
                if U_frac:
                    V_frac = rationalize_matrix(V, tol=1e-4)
                    if V_frac:
                        W_frac = rationalize_matrix(W, tol=1e-4)
                        if W_frac:
                            if verify(U_frac, V_frac, W_frac):
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
    return schoolbook()
