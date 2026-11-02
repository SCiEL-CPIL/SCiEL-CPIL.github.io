"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill

    # Schoolbook decomposition of rank n*m*p
    def schoolbook():
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

    best = schoolbook()
    best_rank = n * m * p

    # Nonzero entries of the tensor
    nz = [(i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p)]

    # Theoretical lower bound
    lb = max(n*m, m*p, p*n)

    # Build the target tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for (ia, ib, ic) in nz:
        T[ia, ib, ic] = 1.0

    # Precompute Kronecker product matrices for ALS
    # For solving U: A_U[:, t] = kron(W[:,t], V[:,t]) flattened
    # Actually: T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    # For fixed a: vec(T[a,:,:]) = sum_t U[a,t] * kron(W[:,t], V[:,t])
    # So A_U has columns = kron(W[:,t], V[:,t]) for t=0..r-1
    
    # For solving V: T[a,b,c] = sum_t V[b,t] * U[a,t] * W[c,t]
    # For fixed b: vec(T[:,b,:]) = sum_t V[b,t] * kron(W[:,t], U[:,t])
    # So A_V has columns = kron(W[:,t], U[:,t])
    
    # For solving W: T[a,b,c] = sum_t W[c,t] * U[a,t] * V[b,t]
    # For fixed c: vec(T[:,:,c]) = sum_t W[c,t] * kron(V[:,t], U[:,t])
    # So A_W has columns = kron(V[:,t], U[:,t])

    def solve_als(U, V, W, r, max_iters=100, tol=1e-10):
        """Alternating least squares for tensor decomposition."""
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Solve for U: for each a, vec(T[a,:,:]) = A_U @ U[a,:]
            # A_U[:, t] = kron(W[:,t], V[:,t])
            A_U = np.zeros((m*p * p*n, r))
            for t in range(r):
                A_U[:, t] = np.kron(W[:, t], V[:, t])
            
            # For each a, solve A_U @ U[a,:] = vec(T[a,:,:])
            U_new = np.zeros((n*m, r))
            for a in range(n*m):
                target = T[a, :, :].flatten()
                # Least squares: U[a,:] = A_U^+ @ target
                U_new[a, :] = np.linalg.lstsq(A_U, target, rcond=None)[0]
            U = U_new
            
            # Solve for V: for each b, vec(T[:,b,:]) = A_V @ V[b,:]
            # A_V[:, t] = kron(W[:,t], U[:,t])
            A_V = np.zeros((n*m * p*n, r))
            for t in range(r):
                A_V[:, t] = np.kron(W[:, t], U[:, t])
            
            V_new = np.zeros((m*p, r))
            for b in range(m*p):
                target = T[:, b, :].flatten()
                V_new[b, :] = np.linalg.lstsq(A_V, target, rcond=None)[0]
            V = V_new
            
            # Solve for W: for each c, vec(T[:,:,c]) = A_W @ W[c,:]
            # A_W[:, t] = kron(V[:,t], U[:,t])
            A_W = np.zeros((n*m * m*p, r))
            for t in range(r):
                A_W[:, t] = np.kron(V[:, t], U[:, t])
            
            W_new = np.zeros((p*n, r))
            for c in range(p*n):
                target = T[:, :, c].flatten()
                W_new[c, :] = np.linalg.lstsq(A_W, target, rcond=None)[0]
            W = W_new
            
            # Check convergence: compute residual
            T_est = np.einsum('at,bt,ct->abc', U, V, W)
            residual = np.linalg.norm(T_est - T)
            if residual < tol:
                break
        
        return U, V, W, residual

    def verify_exact(U, V, W, r):
        """Verify decomposition with exact rational arithmetic."""
        # Try to convert to exact rationals
        # First check if values are close to integers or halves
        def to_fraction_array(arr):
            fracs = []
            for i in range(arr.shape[0]):
                row = []
                for j in range(arr.shape[1]):
                    val = arr[i, j]
                    # Try to find a simple fraction
                    # Check if it's close to an integer
                    int_val = round(val)
                    if abs(val - int_val) < 1e-8:
                        row.append(Fraction(int_val, 1))
                    else:
                        # Try half-integers
                        half_val = round(val * 2) / 2
                        if abs(val - half_val) < 1e-8:
                            row.append(Fraction(int(round(val * 2)), 2))
                        else:
                            # Try small denominators up to 10
                            found = False
                            for den in range(2, 11):
                                num = round(val * den)
                                if abs(val - num/den) < 1e-8:
                                    row.append(Fraction(num, den))
                                    found = True
                                    break
                            if not found:
                                return None
                fracs.append(row)
            return fracs
        
        U_frac = to_fraction_array(U)
        if U_frac is None:
            return False, None
        V_frac = to_fraction_array(V)
        if V_frac is None:
            return False, None
        W_frac = to_fraction_array(W)
        if W_frac is None:
            return False, None
        
        # Verify with exact arithmetic
        for (ia, ib, ic) in nz:
            s = sum(U_frac[ia][t] * V_frac[ib][t] * W_frac[ic][t] for t in range(r))
            if s != 1:
                return False, None
        
        # Also verify that non-nz entries are 0
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if (a, b, c) not in nz:
                        s = sum(U_frac[a][t] * V_frac[b][t] * W_frac[c][t] for t in range(r))
                        if s != 0:
                            return False, None
        
        return True, {"U": U_frac, "V": V_frac, "W": W_frac}

    np.random.seed(42)
    
    # Try ranks from lower bound up to schoolbook rank
    for r in range(lb, best_rank):
        if time.time() > deadline - 20:
            break
        
        # Try several random initializations
        for trial in range(30):
            if time.time() > deadline - 20:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Run ALS
            U, V, W, residual = solve_als(U, V, W, r, max_iters=50)
            
            if residual < 1e-8:
                # Check exactness
                valid, exact_result = verify_exact(U, V, W, r)
                if valid:
                    best = exact_result
                    best_rank = r
                    break
        
        if best_rank == r:
            break
    
    # If we found a better rank, return it; otherwise return schoolbook
    if best_rank < n * m * p:
        return best
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
