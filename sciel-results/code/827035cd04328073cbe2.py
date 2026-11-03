"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # The constraint: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # And for mismatched indices, the sum = 0
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U,V,W form a valid decomposition."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # Check all valid (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if abs(val - 1) > tol:
                        return False
        # Check mismatched cases: for fixed i,k, and j != j', the contribution should be 0
        # Actually, the full constraint is that the bilinear form equals AB
        # Let's verify by checking the tensor equation more carefully
        # T[(i,j),(j',k),(k,i)] = delta_{j,j'}
        # So for j != j', we need sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k*n+i,r] = 0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        if j != jp:
                            val = sum(U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r] for r in range(rank))
                            if abs(val) > tol:
                                return False
        return True
    
    def try_rank(r, max_iters=200, n_trials=5):
        """Try to find a rank-r decomposition."""
        n, m, p = DIMENSIONS
        for trial in range(n_trials):
            if time.time() > deadline - 10:
                return None
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Iterative refinement: alternate solving for each factor
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Solve for U: for each row i*m+j, we need
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                # This is a linear system in U[i*m+j,:]
                for i in range(n):
                    for j in range(m):
                        # Build system: A_u @ U[i*m+j,:] = b
                        # where A_u[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        A_u = np.array([[V[j*p+k, r] * W[k*n+i, r] for r in range(r)] for k in range(p)])
                        b = np.ones(p)
                        # Also need constraints from mismatched j'
                        # For jp != j: sum_r U[i*m+j,r]*V[jp*p+k,r]*W[k*n+i,r] = 0
                        # These give additional equations
                        # Total equations: p + (m-1)*p = m*p
                        # Unknowns: r
                        # If r < m*p, we may not have enough freedom
                        # Let's just use the main constraints for now
                        if r >= p:
                            try:
                                U[i*m+j, :] = np.linalg.lstsq(A_u, b, rcond=None)[0]
                            except:
                                pass
                        else:
                            # Underdetermined, use pseudoinverse
                            try:
                                U[i*m+j, :] = np.linalg.lstsq(A_u, b, rcond=None)[0]
                            except:
                                pass
                
                # Solve for V similarly
                for j in range(m):
                    for k in range(p):
                        A_v = np.array([[U[i*m+j, r] * W[k*n+i, r] for r in range(r)] for i in range(n)])
                        b = np.ones(n)
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(A_v, b, rcond=None)[0]
                        except:
                            pass
                
                # Solve for W
                for k in range(p):
                    for i in range(n):
                        A_w = np.array([[U[i*m+j, r] * V[j*p+k, r] for r in range(r)] for j in range(m)])
                        b = np.ones(m)
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(A_w, b, rcond=None)[0]
                        except:
                            pass
                
                # Check if we've converged
                if check_decomposition(U, V, W, tol=1e-6):
                    # Verify with tighter tolerance
                    if check_decomposition(U, V, W, tol=1e-10):
                        return U, V, W
                    else:
                        # Try to polish
                        for _ in range(50):
                            for i in range(n):
                                for j in range(m):
                                    A_u = np.array([[V[j*p+k, r] * W[k*n+i, r] for r in range(r)] for k in range(p)])
                                    b = np.ones(p)
                                    try:
                                        U[i*m+j, :] = np.linalg.lstsq(A_u, b, rcond=None)[0]
                                    except:
                                        pass
                            for j in range(m):
                                for k in range(p):
                                    A_v = np.array([[U[i*m+j, r] * W[k*n+i, r] for r in range(r)] for i in range(n)])
                                    b = np.ones(n)
                                    try:
                                        V[j*p+k, :] = np.linalg.lstsq(A_v, b, rcond=None)[0]
                                    except:
                                        pass
                            for k in range(p):
                                for i in range(n):
                                    A_w = np.array([[U[i*m+j, r] * V[j*p+k, r] for r in range(r)] for j in range(m)])
                                    b = np.ones(m)
                                    try:
                                        W[k*n+i, :] = np.linalg.lstsq(A_w, b, rcond=None)[0]
                                    except:
                                        pass
                            if check_decomposition(U, V, W, tol=1e-10):
                                return U, V, W
                        break
            
            if time.time() > deadline - 10:
                break
        
        return None
    
    # Try ranks from 35 down to 20
    for r in range(best_rank - 1, 19, -1):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(r, max_iters=100, n_trials=3)
        if result is not None:
            U, V, W = result
            # Convert to lists with rounded values if they're close to integers/half-integers
            U_list = [[round(val, 10) for val in row] for row in U]
            V_list = [[round(val, 10) for val in row] for row in V]
            W_list = [[round(val, 10) for val in row] for row in W]
            
            # Verify the rounded version
            U_v = np.array(U_list)
            V_v = np.array(V_list)
            W_v = np.array(W_list)
            if check_decomposition(U_v, V_v, W_v, tol=1e-8):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = r
                break
            else:
                # Use unrounded
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                break
    
    return best_result
# EVOLVE-BLOCK-END
