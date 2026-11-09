import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 7]

def _get_target_tensor_3x3x2():
    """
    Returns the flattened target tensor for 3x3 * 3x2 multiplication.
    U dim: 9
    V dim: 6
    W dim: 6
    """
    n, m, p = 3, 3, 2
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    return T.flatten()

def _reconstruct_tensor_3x3x2(x, rank):
    """
    Reconstructs the tensor from the flattened parameters x.
    x contains U, V, W concatenated.
    U: 9 * rank
    V: 6 * rank
    W: 6 * rank
    """
    n, m, p = 3, 3, 2
    U = x[:9*rank].reshape(9, rank)
    V = x[9*rank:9*rank+6*rank].reshape(6, rank)
    W = x[9*rank+6*rank:].reshape(6, rank)
    
    T = np.zeros((9, 6, 6))
    for r in range(rank):
        T += np.einsum('ir,jr,kr->ijk', U[:, r], V[:, r], W[:, r])
    return T.flatten()

def _residual_3x3x2(x, rank, target):
    recon = _reconstruct_tensor_3x3x2(x, rank)
    return recon - target

def _find_rank_17_decomp_3x3x2():
    """
    Attempts to find a rank-17 decomposition for 3x3 * 3x2.
    Returns (U, V, W) if found, else None.
    """
    target = _get_target_tensor_3x3x2()
    rank = 17
    num_vars = 9*rank + 6*rank + 6*rank
    
    # Multiple random restarts
    for attempt in range(50):
        # Initialize with random values
        x0 = np.random.randn(num_vars) * 0.5
        
        # Solve
        try:
            res = least_squares(
                _residual_3x3x2, 
                x0, 
                args=(rank, target),
                method='trf',
                max_nfev=1000
            )
            if res.success and res.cost < 1e-12:
                # Snap to integers/half-integers
                x_sol = res.x
                U = x_sol[:9*rank].reshape(9, rank)
                V = x_sol[9*rank:9*rank+6*rank].reshape(6, rank)
                W = x_sol[9*rank+6*rank:].reshape(6, rank)
                
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify snapped solution
                recon_snap = _reconstruct_tensor_3x3x2(np.concatenate([U_snap.flatten(), V_snap.flatten(), W_snap.flatten()]), rank)
                if np.linalg.norm(recon_snap - target) < 1e-9:
                    return U_snap, V_snap, W_snap
        except Exception:
            continue
            
    return None

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    deadline = time.time() + 140
    
    # Strategy:
    # Split B (3x7) into B1 (3x3) and B2 (3x4)?
    # Or B1 (3x2), B2 (3x2), B3 (3x3)?
    # Let's try B1 (3x2), B2 (3x2), B3 (3x3).
    # If we find rank-17 for 3x3x2, we use it twice.
    # Use schoolbook rank-27 for 3x3x3.
    # Total rank 17 + 17 + 27 = 61.
    
    # First, try to find the 3x3x2 rank-17 decomp
    decomp_3x3x2 = None
    if time.time() < deadline - 10:
        decomp_3x3x2 = _find_rank_17_decomp_3x3x2()
        
    if decomp_3x3x2 is not None:
        U1, V1, W1 = decomp_3x3x2
        # We need to map these to the full 3x3x7 context.
        # The 3x3x2 block corresponds to columns 0,1 of B.
        # U is same (9x17).
        # V is 6x17, needs to be embedded in 21x17.
        # V indices for 3x3x2: j*p + k where p=2.
        # In full problem, p=7.
        # So V_full[j*7 + k, r] = V1[j*2 + k, r] for k=0,1.
        # W is 6x17, needs to be embedded in 21x17.
        # W indices for 3x3x2: k*n + i where n=3.
        # In full problem, n=3.
        # So W_full[k*3 + i, r] = W1[k*3 + i, r] for k=0,1.
        
        rank_17 = 17
        U_block1 = np.zeros((9, rank_17))
        V_block1 = np.zeros((21, rank_17))
        W_block1 = np.zeros((21, rank_17))
        
        U_block1 = U1
        # Embed V
        for j in range(3):
            for k in range(2):
                V_block1[j*7 + k, :] = V1[j*2 + k, :]
        # Embed W
        for k in range(2):
            for i in range(3):
                W_block1[k*3 + i, :] = W1[k*3 + i, :]
                
        # Second 3x3x2 block for columns 2,3 of B
        U_block2 = np.zeros((9, rank_17))
        V_block2 = np.zeros((21, rank_17))
        W_block2 = np.zeros((21, rank_17))
        
        U_block2 = U1
        # Embed V for columns 2,3
        # In 3x3x2 local coords, k=0,1.
        # In global coords, k=2,3.
        for j in range(3):
            for k_local in range(2):
                k_global = k_local + 2
                V_block2[j*7 + k_global, :] = V1[j*2 + k_local, :]
        # Embed W for columns 2,3
        for k_local in range(2):
            k_global = k_local + 2
            for i in range(3):
                W_block2[k_global*3 + i, :] = W1[k_local*3 + i, :]
                
        # Third block: 3x3x3 for columns 4,5,6 of B
        rank_27 = 27
        U_block3 = np.zeros((9, rank_27))
        V_block3 = np.zeros((21, rank_27))
        W_block3 = np.zeros((21, rank_27))
        
        col = 0
        for i in range(3):
            for j in range(3):
                for k_local in range(3):
                    k_global = k_local + 4
                    U_block3[i*3 + j, col] = 1
                    V_block3[j*7 + k_global, col] = 1
                    W_block3[k_global*3 + i, col] = 1
                    col += 1
                    
        total_rank = rank_17 + rank_17 + rank_27
        U_final = np.hstack([U_block1, U_block2, U_block3])
        V_final = np.hstack([V_block1, V_block2, V_block3])
        W_final = np.hstack([W_block1, W_block2, W_block3])
        
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    else:
        # Fallback to schoolbook rank 63
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
                    
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
