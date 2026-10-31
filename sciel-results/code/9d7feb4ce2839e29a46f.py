"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify(U, V, W, n, m, p):
    """Check if the decomposition is exact: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _round_to_halfint(vals, tol=1e-4):
    """Round values to nearest half-integer."""
    out = []
    for v in vals:
        r = round(v * 2.0) / 2.0
        if abs(v - r) < tol:
            out.append(r)
        else:
            return None
    return out

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 16
    pn = p * n  # 16
    
    # Target: for all i,j,k in range(n) (and j in range(m)):
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # Build the constraint matrix. For each (i,j,k), we need the product to sum to 1.
    # This is a tensor equation: T[i,j,k] = 1 = sum_r U_{ij,r} V_{jk,r} W_{ik,r}
    
    # Strategy: try ranks from 49 upward
    best_result = None
    
    # Precompute index mapping
    constraints = []  # list of (i, j, k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                constraints.append((i, j, k))
    
    N_constraints = len(constraints)  # 64
    
    for target_rank in range(49, 65):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        n_trials = 5 if target_rank == 49 else 3
        for trial in range(n_trials):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + target_rank * 100 + trial)
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # Use ALS-style iterations to reduce the tensor approximation error
            # We want sum_r U_{ij,r} V_{jk,r} W_{ik,r} = 1 for all (i,j,k)
            # This is non-linear, so we use alternating minimization.
            
            for iteration in range(200):
                if time.time() > deadline - 2:
                    break
                
                # Update U: for each (i,j), we need sum_r U_{ij,r} V_{jk,r} W_{ik,r} = 1
                # Treat V and W as fixed. For each (i,j), the equation is:
                # sum_r U_{ij,r} * (V_{jk,r} * W_{ik,r}) = 1 for each k
                # This is a linear system in U_{ij,:}
                # Let A_{ij,k,r} = V_{jk,r} * W_{ik,r}, then sum_r U_{ij,r} * A_{ij,k,r} = 1
                # So A_{ij} (p x target_rank) times U_{ij} (target_rank) = 1 (p-vector)
                for i in range(n):
                    for j in range(m):
                        A_mat = np.zeros((p, target_rank))
                        for k in range(p):
                            A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve A_mat @ u = ones(p) in least squares
                        u_sol, residuals, rank_mat, sv = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                        U[i*m+j, :] = u_sol
                
                # Update V: for each (j,k), sum_r U_{ij,r} V_{jk,r} W_{ik,r} = 1 for each i
                # Let B_{jk,i,r} = U_{ij,r} * W_{ik,r}, then sum_r V_{jk,r} * B_{jk,i,r} = 1
                for j in range(m):
                    for k in range(p):
                        B_mat = np.zeros((n, target_rank))
                        for i in range(n):
                            B_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        v_sol, residuals, rank_mat, sv = np.linalg.lstsq(B_mat, np.ones(n), rcond=None)
                        V[j*p+k, :] = v_sol
                
                # Update W: for each (k,i), sum_r U_{ij,r} V_{jk,r} W_{ki... } wait W[k*n+i, r]
                # sum_r U_{ij,r} V_{jk,r} W_{k*n+i,r} = 1 for each j
                # Let C_{ki,j,r} = U_{ij,r} * V_{jk,r}, then sum_r W_{ki,r} * C_{ki,j,r} = 1
                for k in range(p):
                    for i in range(n):
                        C_mat = np.zeros((m, target_rank))
                        for j in range(m):
                            C_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        w_sol, residuals, rank_mat, sv = np.linalg.lstsq(C_mat, np.ones(m), rcond=None)
                        W[k*n+i, :] = w_sol
                
                # Check convergence
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err += (s - 1.0) ** 2
                if err < 1e-20:
                    break
                if time.time() > deadline - 2:
                    break
            
            # Verify and try to round
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                # Try to round to half-integers
                U_list = U.flatten().tolist()
                V_list = V.flatten().tolist()
                W_list = W.flatten().tolist()
                
                U_rounded = _round_to_halfint(U_list)
                V_rounded = _round_to_halfint(V_list)
                W_rounded = _round_to_halfint(W_list)
                
                if U_rounded and V_rounded and W_rounded:
                    U_r = [U_rounded[i*target_rank:(i+1)*target_rank] for i in range(nm)]
                    V_r = [V_rounded[i*target_rank:(i+1)*target_rank] for i in range(mp)]
                    W_r = [W_rounded[i*target_rank:(i+1)*target_rank] for i in range(pn)]
                    
                    if _verify(U_r, V_r, W_r, n, m, p):
                        return {"U": U_r, "V": V_r, "W": W_r}
                
                # Return floating point version (should be accepted if exact enough)
                U_out = [[float(U[i*m+j, r]) for r in range(target_rank)] for i in range(n) for j in range(m)]
                V_out = [[float(V[j*p+k, r]) for r in range(target_rank)] for j in range(m) for k in range(p)]
                W_out = [[float(W[k*n+i, r]) for r in range(target_rank)] for k in range(p) for i in range(n)]
                
                if _verify(U_out, V_out, W_out, n, m, p):
                    best_result = {"U": U_out, "V": V_out, "W": W_out}
                    return best_result
        
        # If we made it here without returning, continue to next rank
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
