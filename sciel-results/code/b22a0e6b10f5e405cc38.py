"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    # A is n x m, B is m x p, C is n x p
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # Tensor: T[(i*m+j), (j*p+k), (k*n+i)] = 1 for valid i,j,k
    
    # Build the multiplication tensor as a 3D array
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a, b, c] = 1 if i=i', j=j', k=k' else 0
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    best_U = None
    best_V = None
    best_W = None
    best_rank = rank_schoolbook
    
    # Initialize with schoolbook
    U = np.zeros((n*m, best_rank), dtype=np.float64)
    V = np.zeros((m*p, best_rank), dtype=np.float64)
    W = np.zeros((p*n, best_rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower rank using a simple search
    # Strategy: try ranks from some lower bound up to schoolbook
    # For each rank r, try to find U, V, W such that the decomposition holds
    
    # Lower bound: the rank is at least max(n*m, m*p, p*n) = max(16, 32, 32) = 32
    # But actually the tensor rank lower bound is more subtle.
    # Let's try a few candidate ranks and use a randomized approach
    
    def check_decomposition(U, V, W, T, tol=1e-6):
        """Check if U, V, W correctly decompose T."""
        # Reconstruct T from U, V, W
        # T_recon[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        r = U.shape[1]
        # Use einsum for efficiency
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=tol)
    
    def try_rank(r, max_iters=50, seed=42):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline:
            return None, None, None
        
        rng = np.random.RandomState(seed)
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None, None, None
            
            U = rng.randn(n*m, r)
            V = rng.randn(m*p, r)
            W = rng.randn(p*n, r)
            
            # Normalize to avoid overflow
            for c in range(r):
                norm = max(1.0, np.max(np.abs(U[:, c])), np.max(np.abs(V[:, c])), np.max(np.abs(W[:, c])))
                if norm > 1:
                    U[:, c] /= norm
                    V[:, c] /= norm
                    W[:, c] /= norm
            
            if check_decomposition(U, V, W, T):
                return U, V, W
        
        return None, None, None
    
    # Try a few ranks below schoolbook
    # Known: for 4x4 x 4x4, rank is at most 49 (Strassen-like), but for 4x4x4 it's actually 49? 
    # Actually for 2x2x2, rank is 7. For 4x4x4, best known is 49.
    # For 4x4 x 4x8, this is like two 4x4x4 multiplications concatenated in the output.
    # So rank might be at most 2*49 = 98? Or maybe less due to sharing.
    
    # Let's try ranks: 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, 128
    candidate_ranks = [32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, 128]
    
    for r in candidate_ranks:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            break
        
        U_try, V_try, W_try = try_rank(r, max_iters=20, seed=42 + r)
        if U_try is not None:
            best_rank = r
            best_U, best_V, best_W = U_try, V_try, W_try
            break  # Found a lower rank, stop
    
    # Convert to lists of lists with integer/half-integer entries if possible
    # The problem says entries should be integer/half-integer
    # Let's check if our solution has such entries, if not, try to round
    
    def to_half_integer(mat):
        """Try to convert matrix to half-integer values."""
        # Multiply by 2 and round
        mat2 = mat * 2
        rounded = np.round(mat2)
        # Check if rounding is accurate
        if np.allclose(mat2, rounded, atol=1e-6):
            return (rounded / 2).tolist()
        # If not, try integer
        rounded_int = np.round(mat)
        if np.allclose(mat, rounded_int, atol=1e-6):
            return rounded_int.astype(int).tolist()
        # Otherwise, return as is (shouldn't happen for valid decompositions)
        return mat.tolist()
    
    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
