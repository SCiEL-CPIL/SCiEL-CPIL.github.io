"""Matrix multiplication 2,3,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Precompute target tensor
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is exact and has integer/half-integer entries."""
        r = U.shape[1]
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        
        if np.max(np.abs(R - target)) > tol:
            return False
        
        # Check integer/half-integer vectorized
        for mat in [U, V, W]:
            if not np.all(np.abs(mat * 2 - np.round(mat * 2)) < tol):
                return False
        return True
    
    def round_to_half_integer(mat):
        """Round each entry to the nearest half-integer."""
        return np.round(mat * 2) / 2.0
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    def compute_R(U, V, W):
        """Compute the reconstructed tensor."""
        r = U.shape[1]
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        return R
    
    def tensor_error(U, V, W):
        """Compute the squared error of the tensor reconstruction."""
        R = compute_R(U, V, W)
        return np.sum((R - target) ** 2)

    def als_optimize(U, V, W, max_iters=100):
        """
        Optimize U, V, W using Alternating Least Squares.
        """
        r = U.shape[1]
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        prev_error = tensor_error(U, V, W)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Update U
            M_U = np.zeros((mp * pn, r))
            for c in range(r):
                M_U[:, c] = np.outer(V[:, c], W[:, c]).flatten()
            
            for a in range(nm):
                rhs = target[a, :, :].flatten()
                try:
                    U[a, :] = np.linalg.lstsq(M_U, rhs, rcond=None)[0]
                except:
                    U[a, :] = 0.0

            # Update V
            M_V = np.zeros((nm * pn, r))
            for c in range(r):
                M_V[:, c] = np.outer(U[:, c], W[:, c]).flatten()
            
            for b in range(mp):
                rhs = target[:, b, :].flatten()
                try:
                    V[b, :] = np.linalg.lstsq(M_V, rhs, rcond=None)[0]
                except:
                    V[b, :] = 0.0

            # Update W
            M_W = np.zeros((nm * mp, r))
            for c in range(r):
                M_W[:, c] = np.outer(U[:, c], V[:, c]).flatten()
            
            for c in range(pn):
                rhs = target[:, :, c].flatten()
                try:
                    W[c, :] = np.linalg.lstsq(M_W, rhs, rcond=None)[0]
                except:
                    W[c, :] = 0.0

            current_error = tensor_error(U, V, W)
            if abs(current_error - prev_error) < 1e-10:
                break
            prev_error = current_error
            
        return U, V, W, prev_error
    
    def try_rank(rank, num_attempts=20):
        """Try to find a valid decomposition for a given rank using ALS."""
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
            
            # Sparse random initialization
            if attempt % 2 == 0:
                U = np.random.randn(nm, rank) * 0.5
                V = np.random.randn(mp, rank) * 0.5
                W = np.random.randn(pn, rank) * 0.5
            else:
                U = np.zeros((nm, rank))
                V = np.zeros((mp, rank))
                W = np.zeros((pn, rank))
                for c in range(rank):
                    nz_u = np.random.choice(nm, size=2, replace=False)
                    nz_v = np.random.choice(mp, size=3, replace=False)
                    nz_w = np.random.choice(pn, size=2, replace=False)
                    U[nz_u, c] = np.random.choice([-1, 1], size=2) * 0.5
                    V[nz_v, c] = np.random.choice([-1, 1], size=3) * 0.5
                    W[nz_w, c] = np.random.choice([-1, 1], size=2) * 0.5
            
            U_opt, V_opt, W_opt, final_error = als_optimize(U, V, W, max_iters=50)
            
            if final_error < best_error:
                best_error = final_error
                best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
            
            U_rounded = round_to_half_integer(U_opt)
            V_rounded = round_to_half_integer(V_opt)
            W_rounded = round_to_half_integer(W_opt)
            
            if verify(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded

        return None, None, None
    
    best_result = None
    best_rank = 42
    
    # Try ranks from low to high
    for rank in range(25, 42):
        if time.time() > deadline - 25:
            break
        
        U, V, W = try_rank(rank, num_attempts=5)
        
        if U is not None:
            best_result = (U, V, W)
            best_rank = rank
            break
    
    if best_result is None:
        # Fall back to schoolbook
        U, V, W = schoolbook()
        best_result = (U, V, W)
        best_rank = 42
    
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
