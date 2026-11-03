"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 45
    deadline = time.time() + 140.0  # safety margin before 150s kill

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # The tensor T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For matrix multiplication: T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k)
        # Actually for matrix mult C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor is T[i,j,k] = 1 (all ones) because every (i,j,k) triple contributes
        # Wait, let me reconsider. The matrix multiplication tensor T_{ijk} = 1 for all i,j,k
        # because C_ik = sum_j A_ij B_jk, so each term A_ij * B_jk has coefficient 1.
        
        # Check: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
                    if max_err > tol:
                        return False
        return True

    def verify_fast(U, V, W, tol=1e-9):
        """Faster verification using numpy."""
        rank = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # Reshape to tensors
        U_t = Ua.reshape(n, m, rank)  # U[i,j,c]
        V_t = Va.reshape(m, p, rank)  # V[j,k,c]
        W_t = Wa.reshape(p, n, rank)  # W[k,i,c]
        
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Compute via einsum
        # First compute UV[i,j,k,c] = U[i,j,c] * V[j,k,c] ... this is expensive
        # Better: for each c, outer product of U[:, :, c], V[:, :, c], W[:, :, c]
        
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Let's compute this more efficiently
        # T = sum_c np.einsum('ic,jc,kc->ijk', U_t[:,:,c], V_t[:,:,c], W_t[:,:,c])
        
        T = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            # U_t[:,:,c] is (n,m), V_t[:,:,c] is (m,p), W_t[:,:,c] is (p,n)
            # We need T[i,j,k] += U[i,j,c] * V[j,k,c] * W[k,i,c]
            # This is not a standard outer product because of the index coupling
            # Let's do it explicitly but vectorized over i
            for i in range(n):
                # T[i,j,k] += U[i,j,c] * V[j,k,c] * W[k,i,c]
                # For fixed i: T[i,:,k] += U[i,:,c] * V[:,k,c] * W[k,i,c]
                # U_i = U_t[i,:,c] shape (m,)
                # V_c = V_t[:,:,c] shape (m,p)
                # W_ic = W_t[:,i,c] shape (p,)
                U_i = U_t[i, :, c]  # (m,)
                V_c = V_t[:, :, c]  # (m,p)
                W_ic = W_t[:, i, c]  # (p,)
                # T[i,j,k] += U_i[j] * V_c[j,k] * W_ic[k]
                # = (U_i[:,None] * V_c) * W_ic[None,:]
                T[i, :, :] += (U_i[:, None] * V_c) * W_ic[None, :]
        
        return np.allclose(T, np.ones((n, m, p)), atol=tol)

    # Start with schoolbook
    best = make_schoolbook()
    best_rank = rank_schoolbook

    # Try to find lower rank decompositions
    # Known lower bounds and upper bounds for 3x3 * 3x5:
    # Schoolbook: 45
    # We can try random search for lower rank
    
    # For random search, we need to find U, V, W with rank r such that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is a system of n*m*p = 45 equations in 3*r*rank variables (but with structure)
    # Actually U has n*m*r = 9r entries, V has m*p*r = 15r entries, W has p*n*r = 15r entries
    # Total: 39r variables, 45 equations
    
    # For r < 45/39 ≈ 1.15, we'd need more equations than variables... wait that's not right
    # The constraint is nonlinear (trilinear)
    
    # Let's try a structured approach: 
    # Use the fact that for matrix multiplication, we can sometimes use bilinear algorithms
    
    # For 3x3 * 3x5, let's try rank 40, 35, 30, 25, etc.
    
    # Random search: pick random U, V, then solve for W
    # Given U and V, the equations are linear in W:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # For fixed (i,k), sum over j,c: (sum_j U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
    # Wait, that's not quite right either because W depends on k and i separately.
    
    # Let me re-examine: W[k*n+i, c] is the variable. For each (i,k), the equation is:
    # sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # But W[k*n+i, c] appears in this sum for all j, so:
    # sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c]) = 1
    
    # So for each (i,k), we have: sum_c W[k*n+i, c] * A_{ik,c} = 1
    # where A_{ik,c} = sum_j U[i*m+j, c] * V[j*p+k, c]
    
    # This is a linear system! For each (i,k), one equation. We have n*p = 15 equations.
    # Unknowns: W has p*n*r = 15r entries.
    # So we have 15 equations in 15r unknowns. For r >= 1, this is underdetermined.
    
    # So the approach is:
    # 1. Pick random U (9 x r) and V (15 x r)
    # 2. Compute A_{ik,c} = sum_j U[i*m+j,c] * V[j*p+k,c] for all i,k,c
    # 3. Solve for W: for each (i,k), sum_c W[k*n+i,c] * A_{ik,c} = 1
    # 4. This is 15 linear equations in 15r unknowns
    # 5. Find a solution (e.g., minimum norm)
    # 6. Verify the full decomposition
    
    def try_random_rank(r, max_attempts=100):
        """Try to find a rank-r decomposition via random U,V and solving for W."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random U and V
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            
            # Compute A_{ik,c} = sum_j U[i*m+j,c] * V[j*p+k,c]
            # A is shape (n, p, r)
            A = np.zeros((n, p, r))
            for i in range(n):
                for k in range(p):
                    for c in range(r):
                        s = 0.0
                        for j in range(m):
                            s += U[i*m+j, c] * V[j*p+k, c]
                        A[i, k, c] = s
            
            # Now solve for W: W has shape (p*n, r)
            # For each (i,k): sum_c W[k*n+i, c] * A[i,k,c] = 1
            # This is 15 equations, 15r unknowns
            # Stack into a linear system: M * w = b
            # where w is the flattened W (15r entries), b is ones (15 entries)
            
            # M is 15 x 15r
            M = np.zeros((n * p, p * n * r))
            for i in range(n):
                for k in range(p):
                    eq_idx = k * n + i  # equation index
                    for c in range(r):
                        var_idx = (k * n + i) * r + c
                        M[eq_idx, var_idx] = A[i, k, c]
            
            b = np.ones(n * p)
            
            # Solve minimum norm solution
            try:
                # Use least squares for underdetermined system
                w_sol, residuals, rank_M, sv = np.linalg.lstsq(M, b, rcond=None)
            except:
                continue
            
            # Check if the solution satisfies the equations
            if not np.allclose(M @ w_sol, b, atol=1e-9):
                continue
            
            # Reshape W
            W = w_sol.reshape(p * n, r)
            
            # Convert to lists and verify
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            if verify_fast(U_list, V_list, W_list, tol=1e-9):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try decreasing ranks
    for r in range(rank_schoolbook - 1, 10, -1):
        if time.time() > deadline:
            break
        
        result = try_random_rank(r, max_attempts=50)
        if result is not None:
            best = result
            best_rank = r
            break  # Found a lower rank, stop
    
    return best
# EVOLVE-BLOCK-END
