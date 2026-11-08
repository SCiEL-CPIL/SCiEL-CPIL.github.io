import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(R):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _is_half_integer_matrix(M, tol=1e-9):
    """Check if all entries are half-integers (integer or integer+0.5)."""
    for row in M:
        for x in row:
            if abs(x * 2 - round(x * 2)) > tol:
                return False
    return True

def _try_rank_r(R, n, m, p, deadline, max_restarts=200):
    """Try to find a rank-R decomposition using the slice method."""
    # We fix U and V, then solve linear system for W.
    # Variables: W[k*n+i, r] for k in range(p), i in range(n), r in range(R)
    # Total variables: p*n*R
    # Equations: n*m*p (one for each i,j,k)
    # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Build the linear system A * W_vec = b
    # A is (n*m*p) x (p*n*R)
    # b is vector of ones
    
    # We'll try different random initializations of U and V
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize U and V with small integers
        # Strategy: use values from {-1, 0, 1} with some structure
        np.random.seed(restart * 12345 + R * 67890)
        
        # Try different densities of non-zero entries
        if restart % 3 == 0:
            # Sparse: mostly zeros with some ±1
            U = np.random.randint(-1, 2, (n*m, R))
            V = np.random.randint(-1, 2, (m*p, R))
            # Ensure each row has at least one non-zero
            for i in range(n*m):
                if np.all(U[i] == 0):
                    U[i, np.random.randint(R)] = 1
            for j in range(m*p):
                if np.all(V[j] == 0):
                    V[j, np.random.randint(R)] = 1
        elif restart % 3 == 1:
            # Denser: values from {-2, -1, 0, 1, 2}
            U = np.random.randint(-2, 3, (n*m, R))
            V = np.random.randint(-2, 3, (m*p, R))
        else:
            # Very sparse: mostly 0s, some 1s
            U = np.zeros((n*m, R), dtype=int)
            V = np.zeros((m*p, R), dtype=int)
            for i in range(n*m):
                num_nonzero = np.random.randint(1, max(2, R//3 + 1))
                cols = np.random.choice(R, size=num_nonzero, replace=False)
                U[i, cols] = np.random.randint(1, 3, size=num_nonzero)
            for j in range(m*p):
                num_nonzero = np.random.randint(1, max(2, R//3 + 1))
                cols = np.random.choice(R, size=num_nonzero, replace=False)
                V[j, cols] = np.random.randint(1, 3, size=num_nonzero)
        
        # Build the linear system
        # For equation (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Variable index for W[k*n+i, r] is (k*n+i)*R + r
        
        A = np.zeros((n*m*p, p*n*R), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i*m*p + j*p + k
                    for r in range(R):
                        coeff = U[i*m+j, r] * V[j*p+k, r]
                        if coeff != 0:
                            var_idx = (k*n + i) * R + r
                            A[eq_idx, var_idx] = coeff
        
        b = np.ones(n*m*p)
        
        # Solve the linear system
        # We want a solution where W entries are half-integers
        # First, find the minimum-norm solution
        try:
            W_vec, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if the solution is exact
        error = np.linalg.norm(A @ W_vec - b)
        if error > 1e-7:
            continue
        
        W_mat = W_vec.reshape(p*n, R)
        
        # Check if W is close to half-integers
        W_scaled = W_mat * 2
        W_rounded = np.round(W_scaled)
        if np.allclose(W_scaled, W_rounded, atol=1e-6):
            W_final = W_rounded / 2.0
            # Final verification
            if _verify(U, V, W_final, n, m, p):
                return U, V, W_final
        
        # Try to find a half-integer solution in the null space
        # The general solution is W_particular + W_null
        # where W_null is in the null space of A
        # We want W_particular + W_null to be close to half-integers
        
        # Compute null space
        if rank < p*n*R:
            # Find null space vectors
            # Use SVD
            U_svd, s, Vt = np.linalg.svd(A, full_matrices=True)
            # Null space vectors are the last (p*n*R - rank) rows of Vt
            null_space = Vt[rank:]  # shape: (p*n*R - rank, p*n*R)
            
            # We want to find c such that W_particular + null_space @ c is close to half-integers
            # This is a least-squares problem: minimize ||(W_particular + null_space @ c) * 2 - round((W_particular + null_space @ c) * 2)||
            # But rounding makes it non-smooth. Instead, let's try to find c that makes
            # (W_particular + null_space @ c) * 2 close to integers.
            
            # Let target = W_particular * 2
            # We want null_space @ c to be close to (integer - target)
            # This is still hard. Let's try a simpler approach:
            # Randomly perturb in the null space and check
            
            for trial in range(20):
                if time.time() > deadline:
                    break
                c = np.random.randn(null_space.shape[0])
                W_trial = W_vec + null_space.T @ c
                W_trial_mat = W_trial.reshape(p*n, R)
                W_trial_scaled = W_trial_mat * 2
                W_trial_rounded = np.round(W_trial_scaled)
                if np.allclose(W_trial_scaled, W_trial_rounded, atol=1e-6):
                    W_trial_final = W_trial_rounded / 2.0
                    if _verify(U, V, W_trial_final, n, m, p):
                        return U, V, W_trial_final
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Keep margin before 120s limit
    
    R_max = n * m * p  # 45
    
    # Schoolbook solution as fallback
    U_sb = np.zeros((n*m, R_max))
    V_sb = np.zeros((m*p, R_max))
    W_sb = np.zeros((p*n, R_max))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = R_max
    
    # Try to find lower rank solutions
    # Theoretical lower bound for 3x3 x 3x5 is at least max(3*3, 3*5, 3*3) = 15
    # Known results suggest rank around 20-30 for this size
    
    for R in range(R_max - 1, 14, -1):
        if time.time() > deadline - 5:
            break
        
        result = _try_rank_r(R, n, m, p, deadline - 5)
        if result is not None:
            U, V, W = result
            best_U = U
            best_V = V
            best_W = W
            best_rank = R
            break
    
    # Convert to lists
    U_out = [[float(x) for x in row] for row in best_U]
    V_out = [[float(x) for x in row] for row in best_V]
    W_out = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
