"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p  # 45
    
    # Build schoolbook decomposition
    U = [[0] * base_rank for _ in range(n * m)]
    V = [[0] * base_rank for _ in range(m * p)]
    W = [[0] * base_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    def verify(U_mat, V_mat, W_mat, rank_test):
        """Verify exact decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank_test):
                        s += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_np(U_np, V_np, W_np, rank_test):
        """Vectorized verification using numpy"""
        U_r = U_np.reshape(n, m, rank_test)
        V_r = V_np.reshape(m, p, rank_test)
        W_r = W_np.reshape(p, n, rank_test)
        # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,r,k] * W[k,i,r]
        # Use einsum
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)
    
    def to_half_integer(x, tol=0.01):
        """Round to nearest half-integer if close enough"""
        h = np.round(x * 2) / 2.0
        if np.allclose(x, h, atol=tol):
            return h
        return None
    
    def solve_for_W(U_np, V_np, r):
        """
        Given U (nm x r) and V (mp x r), solve for W (pn x r).
        For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        This is a linear system in W.
        """
        # Unknowns: W[k*n+i, c] for k in range(p), i in range(n), c in range(r)
        # Equations: for each (i,j,k)
        num_eqs = n * m * p
        num_unknowns = p * n * r
        
        A = np.zeros((num_eqs, num_unknowns))
        b = np.ones(num_eqs)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq = (i * m + j) * p + k
                    for c in range(r):
                        unknown = (k * n + i) * r + c
                        A[eq, unknown] = U_np[i * m + j, c] * V_np[j * p + k, c]
        
        # Solve least squares
        try:
            W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            return None
        
        W_sol = W_flat.reshape(p * n, r)
        
        # Check if solution is close to half-integer
        W_h = to_half_integer(W_sol, tol=0.05)
        if W_h is not None:
            return W_h
        return W_sol
    
    def random_search(r, max_attempts=200, seed=None):
        """Random search for decomposition of rank r"""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Random U and V with small integers/half-integers
            vals = np.array([0, 0, 0, 1, -1, 0.5, -0.5, 2, -2])
            U_np = rng.choice(vals, size=(n * m, r)).astype(float)
            V_np = rng.choice(vals, size=(m * p, r)).astype(float)
            
            # Solve for W
            W_sol = solve_for_W(U_np, V_np, r)
            if W_sol is None:
                continue
            
            # Check if W is close to half-integer
            W_h = to_half_integer(W_sol, tol=0.05)
            if W_h is not None:
                # Verify
                if verify_np(U_np, V_np, W_h, r):
                    # Convert to lists
                    U_list = U_np.tolist()
                    V_list = V_np.tolist()
                    W_list = W_h.tolist()
                    return U_list, V_list, W_list
            else:
                # Try rounding W to half-integers and verify
                W_rounded = np.round(W_sol * 2) / 2
                if verify_np(U_np, V_np, W_rounded, r):
                    U_list = U_np.tolist()
                    V_list = V_np.tolist()
                    W_list = W_rounded.tolist()
                    return U_list, V_list, W_list
        
        return None
    
    def structured_search(r):
        """
        Structured search exploiting block structure.
        For 3x3 * 3x5, split into blocks:
        - Use Strassen-like ideas for 2x2 subblocks
        - Handle remaining terms separately
        """
        # This is a heuristic structured approach
        # Split A into [A1 (2x3); A2 (1x3)] and B into [B1 (3x2) B2 (3x3)]
        # C = [A1; A2] * [B1 B2] = [A1*B1, A1*B2; A2*B1, A2*B2]
        # A1*B1: 2x2 * 3x2 -> not square, use block decomposition
        # This gets complex; let's try a different structured approach
        
        # Try using the fact that for each output column k, we need to compute
        # C[:,k] = A * B[:,k] which is a 3x3 * 3x1 multiplication
        # The 3x3 * 3x1 tensor has rank at most 9 (schoolbook)
        # But we can potentially share structure across columns
        
        # Alternative: try random search with more structure
        # Initialize with schoolbook and perturb
        U_base = np.zeros((n * m, r))
        V_base = np.zeros((m * p, r))
        W_base = np.zeros((p * n, r))
        
        # Start from schoolbook if r >= base_rank, otherwise random
        if r >= base_rank:
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        col = i * m + j * p + k  # This doesn't map directly
                        # Actually schoolbook has one term per (i,j,k)
                        # For r > base_rank, we can use schoolbook + extra zero columns
                        pass
        
        # Just do random search for now
        return random_search(r, max_attempts=100)
    
    # Try to find lower rank decompositions
    best_rank = base_rank
    best_result = {"U": U, "V": V, "W": W}
    
    # Search from rank 44 down to a reasonable minimum
    # Known bounds: for 3x3*3x5, rank is likely around 40-45
    min_rank_to_try = 35
    
    for target_rank in range(base_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 10:
            break
        
        # Try random search
        result = random_search(target_rank, max_attempts=50)
        if result is not None:
            U_list, V_list, W_list = result
            best_rank = target_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            break
    
    # If we found a better rank, also try to improve further
    if best_rank < base_rank:
        # Try a few more attempts at the same rank or slightly lower
        for target_rank in range(best_rank - 1, max(min_rank_to_try, best_rank - 5), -1):
            if time.time() > deadline - 5:
                break
            result = random_search(target_rank, max_attempts=30)
            if result is not None:
                U_list, V_list, W_list = result
                best_rank = target_rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result
# EVOLVE-BLOCK-END
