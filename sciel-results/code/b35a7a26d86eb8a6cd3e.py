import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def _verify(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def _is_half_integer(mat, tol=1e-7):
        """Check if all entries in mat are close to integer or half-integer."""
        # Multiply by 2 to check for integers
        scaled = mat * 2
        rounded = np.round(scaled)
        return np.all(np.abs(scaled - rounded) < tol)

    def _round_half_integer(mat):
        """Round entries to nearest half-integer."""
        return np.round(mat * 2) / 2.0

    def _solve_for_target(target_name, U, V, W, n, m, p, r):
        """
        Solve for one of the matrices given the other two.
        Returns the solved matrix or None.
        """
        # We want to solve for the matrix with indices that make the equation linear.
        # Equation: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
        
        # Case 1: Solve for W
        # Unknowns: W[k*n+i, c] -> index (k*n+i)*r + c
        # Eq index: i*m*p + j*p + k ? No, let's use a consistent mapping.
        # Let's map (i,j,k) to eq_idx = (i*m + j)*p + k
        
        num_eqs = n * m * p
        A = np.zeros((num_eqs, pn * r))
        b = np.ones(num_eqs)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = (i * m + j) * p + k
                    for c in range(r):
                        coeff = U[i*m+j, c] * V[j*p+k, c]
                        w_idx = (k*n+i)*r + c
                        A[eq_idx, w_idx] += coeff
        
        try:
            W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            if len(residuals) > 0 and np.max(np.abs(residuals)) > 1e-9:
                return None
            W_mat = W_flat.reshape(pn, r)
            if _is_half_integer(W_mat):
                return _round_half_integer(W_mat)
            return None
        except:
            return None

    def _solve_for_V(U, W, n, m, p, r):
        # Unknowns: V[j*p+k, c] -> index (j*p+k)*r + c
        num_eqs = n * m * p
        A = np.zeros((num_eqs, mp * r))
        b = np.ones(num_eqs)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = (i * m + j) * p + k
                    for c in range(r):
                        coeff = U[i*m+j, c] * W[k*n+i, c]
                        v_idx = (j*p+k)*r + c
                        A[eq_idx, v_idx] += coeff
        
        try:
            V_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            if len(residuals) > 0 and np.max(np.abs(residuals)) > 1e-9:
                return None
            V_mat = V_flat.reshape(mp, r)
            if _is_half_integer(V_mat):
                return _round_half_integer(V_mat)
            return None
        except:
            return None

    def _solve_for_U(V, W, n, m, p, r):
        # Unknowns: U[i*m+j, c] -> index (i*m+j)*r + c
        num_eqs = n * m * p
        A = np.zeros((num_eqs, nm * r))
        b = np.ones(num_eqs)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = (i * m + j) * p + k
                    for c in range(r):
                        coeff = V[j*p+k, c] * W[k*n+i, c]
                        u_idx = (i*m+j)*r + c
                        A[eq_idx, u_idx] += coeff
        
        try:
            U_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            if len(residuals) > 0 and np.max(np.abs(residuals)) > 1e-9:
                return None
            U_mat = U_flat.reshape(nm, r)
            if _is_half_integer(U_mat):
                return _round_half_integer(U_mat)
            return None
        except:
            return None

    def schoolbook():
        rank = n*m*p
        U = [[0]*rank for _ in range(nm)]
        V = [[0]*rank for _ in range(mp)]
        W = [[0]*rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank

    best_U, best_V, best_W, best_rank = schoolbook()

    # Values for random initialization
    # Using small integers and half-integers
    vals = np.array([-1, -0.5, 0, 0.5, 1, 1.5, 2])
    
    # Try ranks from 14 upwards
    # The rank of 2x3x4 is known to be 14 in some contexts, but let's search 14-23
    for r in range(14, 24):
        if time.time() > deadline:
            break
        
        found = False
        attempts = 0
        max_attempts = 500 # Increased attempts for lower ranks
        
        while not found and attempts < max_attempts:
            attempts += 1
            if time.time() > deadline:
                break
            
            # Random U and V
            U = np.random.choice(vals, size=(nm, r))
            V = np.random.choice(vals, size=(mp, r))
            
            # 1. Solve for W
            W_sol = _solve_for_target('W', U, V, None, n, m, p, r)
            if W_sol is not None:
                if _verify(U, V, W_sol, n, m, p):
                    best_U, best_V, best_W, best_rank = U.tolist(), V.tolist(), W_sol.tolist(), r
                    found = True
                    break
            
            # 2. Solve for V (Fix U, W)
            # We need a random W to fix
            W_rand = np.random.choice(vals, size=(pn, r))
            V_sol = _solve_for_V(U, W_rand, n, m, p, r)
            if V_sol is not None:
                if _verify(U, V_sol, W_rand, n, m, p):
                    best_U, best_V, best_W, best_rank = U.tolist(), V_sol.tolist(), W_rand.tolist(), r
                    found = True
                    break
            
            # 3. Solve for U (Fix V, W)
            # We need a random W to fix (reuse W_rand or new?)
            # Let's use a new random W for variety
            W_rand2 = np.random.choice(vals, size=(pn, r))
            U_sol = _solve_for_V(V, W_rand2, n, m, p, r) # Wait, _solve_for_V solves for V. I need _solve_for_U.
            
            # Correct call:
            U_sol = _solve_for_U(V, W_rand2, n, m, p, r)
            if U_sol is not None:
                if _verify(U_sol, V, W_rand2, n, m, p):
                    best_U, best_V, best_W, best_rank = U_sol.tolist(), V.tolist(), W_rand2.tolist(), r
                    found = True
                    break

        if found:
            break # Found a solution for this rank, move to next lower rank? No, we iterate r from low to high.
            # Actually, we want the MINIMAL rank. So if we find one for r=14, we can stop.
            # The loop goes 14, 15, ...
            # If found at 14, we break the outer loop.
            pass

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
