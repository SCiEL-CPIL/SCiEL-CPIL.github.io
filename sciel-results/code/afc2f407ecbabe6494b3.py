"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook

    # Target rank: Theoretical rank of M_{2,2,p} is 2p+1 for p>=2?
    # For p=15, 2*15+1 = 31.
    # Let's try to find a decomposition of rank R_target.
    # If we fail, we fall back to schoolbook.
    
    def verify_decomposition(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        R = U.shape[1]
        # Reshape for faster computation
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        # Compute the tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # We want T[i,j,k] == 1 for all i,j,k
        # Use einsum for speed
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        # Check if all entries are close to 1
        return np.allclose(T, 1.0, atol=1e-9)

    def try_optimize_rank(R):
        """Try to find a rank-R decomposition using least squares optimization."""
        if time.time() > deadline:
            return None
            
        # Initialize U, V, W with random values
        # We use a structured initialization based on the schoolbook but compressed
        # For R=31, we can try to map the 60 schoolbook terms into 31 slots
        # This is hard. Instead, use random initialization with small values.
        
        # Better: Use the known Strassen-like structure for the first few k's
        # and linear structure for the rest.
        
        # Let's try a specific structure:
        # Terms 0..6: Strassen for k=0,1 (rank 7)
        # Terms 7..30: 2 terms for each k=2..14 (2*13 = 26 terms)
        # Total 7 + 26 = 33. This is > 31.
        
        # If the rank is indeed 31, we need a more efficient sharing.
        # Let's just try random initialization and least squares.
        
        # Flatten U, V, W into a single vector x
        size_u = n * m * R
        size_v = m * p * R
        size_w = p * n * R
        total_size = size_u + size_v + size_w
        
        # Random initialization
        x0 = np.random.randn(total_size) * 0.5
        x0[size_u:size_u+size_v] *= 1.0
        x0[size_u+size_v:] *= 1.0
        
        def residuals(x):
            U_flat = x[:size_u]
            V_flat = x[size_u:size_u+size_v]
            W_flat = x[size_u+size_v:]
            
            U_r = U_flat.reshape(n, m, R)
            V_r = V_flat.reshape(m, p, R)
            W_r = W_flat.reshape(p, n, R)
            
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            return (T - 1.0).flatten()
        
        try:
            # Use least squares to minimize sum (T-1)^2
            res = least_squares(residuals, x0, method='trf', max_nfev=5000, xtol=1e-12, ftol=1e-12)
        except Exception:
            return None
            
        if res.success and res.cost < 1e-8:
            U_flat = res.x[:size_u]
            V_flat = res.x[size_u:size_u+size_v]
            W_flat = res.x[size_u+size_v:]
            
            U = U_flat.reshape(n, m, R).reshape(n*m, R)
            V = V_flat.reshape(m, p, R).reshape(m*p, R)
            W = W_flat.reshape(p, n, R).reshape(p*n, R)
            
            # Snap to integers or half-integers
            # The optimal solutions for matrix multiplication tensors often have integer or half-integer entries
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if verify_decomposition(U_snap, V_snap, W_snap):
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            
            # If snapping fails, try rounding to integers
            U_snap2 = np.round(U)
            V_snap2 = np.round(V)
            W_snap2 = np.round(W)
            if verify_decomposition(U_snap2, V_snap2, W_snap2):
                return {"U": U_snap2.tolist(), "V": V_snap2.tolist(), "W": W_snap2.tolist()}
                
        return None

    # Try ranks from theoretical minimum to schoolbook
    # Theoretical lower bound: max(n*m, m*p, p*n) = 30
    # Known rank for M_{2,2,p} is 2p+1 = 31 for p>=2?
    # Let's try 30, 31, 32... up to 40
    
    for R in range(30, 41):
        if time.time() > deadline:
            break
        
        result = try_optimize_rank(R)
        if result is not None:
            return result
            
    # Fallback to schoolbook
    return best_result
# EVOLVE-BLOCK-END
