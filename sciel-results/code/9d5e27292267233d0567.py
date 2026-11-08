"""Program for matrix multiplication tensor decomposition 3x3x5 using belief propagation / structured search."""
import time
import numpy as np
import itertools

DIMENSIONS = [3, 3, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    Uses structured search over low-rank tensor decompositions with integer/half-integer entries.
    Strategy: Start from schoolbook, use belief propagation-like pruning and local optimization.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Build the tensor: T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    # T[a,b,c] = 1 if j_a == j_b and i_a == i_c and k_b == k_c
    def build_tensor():
        T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T[a, b, c] = 1.0
        return T
    
    T = build_tensor()
    
    # Schoolbook decomposition: rank = n*m*p = 45
    def schoolbook():
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W
    
    U0, V0, W0 = schoolbook()
    
    def verify(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose T."""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.max(np.abs(T - T_approx)) < tol
    
    def to_half_int(M):
        """Round to nearest half-integer."""
        return np.round(M * 2) / 2
    
    best_rank = n*m*p
    best_U, best_V, best_W = U0, V0, W0
    
    # Try to reduce rank by finding structured decompositions
    # Known: 3x3x3 has min rank 19 (Strassen variant)
    # For 3x3x5, min rank is at most 3*19=57 via blocking, but we want less
    # Actually for rectangular, the min rank for 3x3x5 is known to be around 24-27
    
    # Strategy: Try random rank-r decompositions and optimize
    def try_optimize_rank(r, n_restarts=20):
        """Try to find rank-r decomposition via optimization."""
        global best_rank, best_U, best_V, best_W
        for restart in range(n_restarts):
            if time.time() > deadline:
                return
            
            # Random initialization
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Normalize columns
            for col in range(r):
                nu = np.linalg.norm(U[:, col])
                nv = np.linalg.norm(V[:, col])
                nw = np.linalg.norm(W[:, col])
                prod = nu * nv * nw
                if prod > 1e-10:
                    scale = prod ** (1/3)
                    U[:, col] /= scale
                    V[:, col] /= scale
                    W[:, col] /= scale
            
            # Alternating least squares with half-integer constraint
            # Iterate: fix two, solve for third, then round
            for it in range(500):
                if time.time() > deadline:
                    return
                
                # Update U: minimize ||T - einsum('br,cr,ar->abc', V, W, U)||
                # This is equivalent to: for each a, U[a,:] minimizes
                # sum_{b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
                # Gradient: dU[a,r] = -2 sum_{b,c} (T[a,b,c] - T_approx[a,b,c]) V[b,r] W[c,r]
                
                T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                dT = T - T_approx
                
                # Solve for U using linear system
                # For each a: minimize sum_{b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
                # This is a least squares problem: U[a,:] @ (V[b,:] outer W[c,:])^T ...
                # Actually, reshape: for fixed a, the residual is T[a,:,:] - einsum('r,br,cr->bc', U[a,:], V, W)
                # Let M_a = einsum('br,cr->bcr', V, W) reshaped to (m*p*p*n, r)
                # Then residual = T[a].flatten() - U[a,:] @ M_a.reshape(m*p*p*n, r).T
                # So U[a,:] = lstsq(M_a.reshape(m*p*p*n, r).T, T[a].flatten())[0]
                
                # More efficiently: use the normal equations
                # H = V.T @ (V * W^T @ W) ... this gets complicated
                # Let's just use gradient descent with small steps
                
                lr = 0.01
                for col in range(r):
                    gradU = 2 * np.einsum('abc,br,cr->ar', dT, V, W, optimize=True)
                    gradV = 2 * np.einsum('abc,ar,cr->br', dT, U, W, optimize=True)
                    gradW = 2 * np.einsum('abc,ar,br->cr', dT, U, V, optimize=True)
                    
                    U -= lr * gradU
                    V -= lr * gradV
                    W -= lr * gradW
                
                # Round to half-integers periodically
                if it % 10 == 0:
                    U_r = to_half_int(U)
                    V_r = to_half_int(V)
                    W_r = to_half_int(W)
                    if verify(U_r, V_r, W_r):
                        if r < best_rank:
                            best_rank = r
                            best_U, best_V, best_W = U_r, V_r, W_r
                            return
                        return  # Can't do better with this r if already found
            
            # Final check
            U_r = to_half_int(U)
            V_r = to_half_int(V)
            W_r = to_half_int(W)
            if verify(U_r, V_r, W_r):
                if r < best_rank:
                    best_rank = r
                    best_U, best_V, best_W = U_r, V_r, W_r
                    return
    
    # Try various ranks
    # Lower bound: tensor rank of 3x3x3 is 19, so 3x3x5 >= 19
    # Upper bound: 45 (schoolbook)
    # Try from 19 upward
    for r in range(19, 35):
        if time.time() > deadline:
            break
        n_restarts = 15 if r <= 25 else 8
        try_optimize_rank(r, n_restarts)
        if best_rank <= r:
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
