"""
Matrix multiplication tensor decomposition for M_{2,3,10}.
Strategy: Partition 3x10 into two 3x5 blocks, find low-rank decompositions
for each 2x3x5 subproblem, then combine. Also try direct ALS with better
initialization and polishing.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140  # seconds, safety margin before 150s kill


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a dense 3D array."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _check_exact(U, V, W, T, tol=1e-10):
    """Check if U, V, W exactly decompose T."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)


def _als_decompose(T, rank, max_iters=500, seed=42, tol=1e-12):
    """Alternating least squares for tensor decomposition."""
    rng = np.random.RandomState(seed)
    n1, n2, n3 = T.shape
    
    # Initialize
    U = rng.randn(n1, rank) * 0.1
    V = rng.randn(n2, rank) * 0.1
    W = rng.randn(n3, rank) * 0.1
    
    for it in range(max_iters):
        # Solve for U
        M = np.einsum('br,cr->bcr', V, W)
        M_flat = M.transpose(2, 0, 1).reshape(rank, -1)
        T_flat = T.reshape(n1, -1)
        try:
            U = np.linalg.lstsq(M_flat.T, T_flat.T, rcond=None)[0].T
        except:
            pass
        
        # Solve for V
        N = np.einsum('ar,cr->acr', U, W)
        N_flat = N.transpose(2, 0, 1).reshape(rank, -1)
        T_flat_v = T.reshape(n2, -1)
        try:
            V = np.linalg.lstsq(N_flat.T, T_flat_v.T, rcond=None)[0].T
        except:
            pass
        
        # Solve for W
        P = np.einsum('ar,br->abr', U, V)
        P_flat = P.transpose(2, 0, 1).reshape(rank, -1)
        T_flat_w = T.reshape(n3, -1)
        try:
            W = np.linalg.lstsq(P_flat.T, T_flat_w.T, rcond=None)[0].T
        except:
            pass
        
        # Check convergence
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T))
        if err < tol:
            break
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T_recon - T))
    if err < tol:
        return U, V, W, True
    return U, V, W, False


def _polish(U, V, W, T, max_iters=200, tol=1e-14):
    """High-precision polish pass with tighter tolerance."""
    for it in range(max_iters):
        # Solve for U
        M = np.einsum('br,cr->bcr', V, W)
        M_flat = M.transpose(2, 0, 1).reshape(M.shape[2], -1)
        T_flat = T.reshape(T.shape[0], -1)
        try:
            U = np.linalg.lstsq(M_flat.T, T_flat.T, rcond=None)[0].T
        except:
            break
        
        # Solve for V
        N = np.einsum('ar,cr->acr', U, W)
        N_flat = N.transpose(2, 0, 1).reshape(N.shape[2], -1)
        T_flat_v = T.reshape(T.shape[1], -1)
        try:
            V = np.linalg.lstsq(N_flat.T, T_flat_v.T, rcond=None)[0].T
        except:
            break
        
        # Solve for W
        P = np.einsum('ar,br->abr', U, V)
        P_flat = P.transpose(2, 0, 1).reshape(P.shape[2], -1)
        T_flat_w = T.reshape(T.shape[2], -1)
        try:
            W = np.linalg.lstsq(P_flat.T, T_flat_w.T, rcond=None)[0].T
        except:
            break
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T))
        if err < tol:
            break
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T_recon - T))
    if err < tol:
        return U, V, W, True
    return U, V, W, False


def _to_half_int(arr):
    """Convert to half-integer representation and verify."""
    arr_rounded = np.round(arr * 2) / 2
    return arr_rounded


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U = np.zeros((n * m, rank_trivial))
    V = np.zeros((m * p, rank_trivial))
    W = np.zeros((p * n, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank_trivial
    
    T = _build_tensor(n, m, p)
    
    # Strategy 1: Partition 3x10 into two 3x5 blocks
    # Each block is a 2x3x5 multiplication
    # Find low-rank decompositions for each, then combine
    
    # Build the two 2x3x5 sub-tensors
    # Block 1: columns 0-4 of the 3x10 (k=0..4)
    # Block 2: columns 5-9 of the 3x10 (k=5..9)
    
    n_sub = 2
    m_sub = 3
    p_sub = 5
    
    # Sub-tensor for block 1: T1[a, b, c] where a in [0,6), b in [0,15), c in [0,10)
    # But we only use k=0..4, so b ranges over j*5+k for j=0..2, k=0..4 -> b in [0,15)
    # And c = k*n + i for k=0..4, i=0..1 -> c in [0,10)
    
    T1 = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub), dtype=np.float64)
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                a = i * m_sub + j
                b = j * p_sub + k
                c = k * n_sub + i
                T1[a, b, c] = 1.0
    
    T2 = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub), dtype=np.float64)
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                a = i * m_sub + j
                b = j * p_sub + k
                c = k * n_sub + i
                T2[a, b, c] = 1.0
    
    # Try to find low-rank decompositions for each sub-problem
    # Lower bound for 2x3x5: max(6, 15, 10) = 15
    # Try ranks from 15 upward
    
    best_sub1 = None
    best_sub2 = None
    best_sub_rank = None
    
    for r_sub in range(15, 31):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        found1 = False
        found2 = False
        
        for seed in range(10):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            U1, V1, W1, ok1 = _als_decompose(T1, r_sub, max_iters=300, seed=seed, tol=1e-12)
            if ok1:
                U1, V1, W1, ok1 = _polish(U1, V1, W1, T1, max_iters=100, tol=1e-14)
                if ok1:
                    best_sub1 = (U1, V1, W1)
                    found1 = True
                    break
        
        for seed in range(10):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            U2, V2, W2, ok2 = _als_decompose(T2, r_sub, max_iters=300, seed=seed, tol=1e-12)
            if ok2:
                U2, V2, W2, ok2 = _polish(U2, V2, W2, T2, max_iters=100, tol=1e-14)
                if ok2:
                    best_sub2 = (U2, V2, W2)
                    found2 = True
                    break
        
        if found1 and found2:
            best_sub_rank = r_sub
            break
    
    if best_sub_rank is not None and best_sub1 is not None and best_sub2 is not None:
        # Combine the two sub-decompositions
        # Total rank = 2 * best_sub_rank
        total_rank = 2 * best_sub_rank
        
        if total_rank < best_rank:
            U_combined = np.zeros((n * m, total_rank))
            V_combined = np.zeros((m * p, total_rank))
            W_combined = np.zeros((p * n, total_rank))
            
            U1, V1, W1 = best_sub1
            U2, V2, W2 = best_sub2
            
            # Block 1 uses columns 0..best_sub_rank-1
            U_combined[:, :best_sub_rank] = U1
            # For V: block 1 uses rows j*5+k for j=0..2, k=0..4
            for j in range(3):
                for k in range(5):
                    V_combined[j * 10 + k, :best_sub_rank] = V1[j * 5 + k, :]
            # For W: block 1 uses rows k*2+i for k=0..4, i=0..1
            for k in range(5):
                for i in range(2):
                    W_combined[k * 2 + i, :best_sub_rank] = W1[k * 2 + i, :]
            
            # Block 2 uses columns best_sub_rank..2*best_sub_rank-1
            U_combined[:, best_sub_rank:] = U2
            # For V: block 2 uses rows j*5+k for j=0..2, k=0..4, mapped to j*10+(5+k)
            for j in range(3):
                for k in range(5):
                    V_combined[j * 10 + (5 + k), best_sub_rank:] = V2[j * 5 + k, :]
            # For W: block 2 uses rows k*2+i for k=0..4, i=0..1, mapped to (5+k)*2+i
            for k in range(5):
                for i in range(2):
                    W_combined[(5 + k) * 2 + i, best_sub_rank:] = W2[k * 2 + i, :]
            
            # Verify
            T_recon = np.einsum('ar,br,cr->abc', U_combined, V_combined, W_combined)
            if np.allclose(T_recon, T, atol=1e-10):
                U_h = _to_half_int(U_combined)
                V_h = _to_half_int(V_combined)
                W_h = _to_half_int(W_combined)
                
                T_recon2 = np.einsum('ar,br,cr->abc', U_h, V_h, W_h)
                if np.allclose(T_recon2, T, atol=1e-10):
                    best_result = {"U": U_h.tolist(), "V": V_h.tolist(), "W": W_h.tolist()}
                    best_rank = total_rank
    
    # Strategy 2: Direct ALS on the full problem with various ranks
    # Try ranks from 30 (lower bound) upward
    for r in range(30, best_rank):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        found = False
        for seed in range(15):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            U_arr, V_arr, W_arr, success = _als_decompose(T, r, max_iters=400, seed=seed, tol=1e-12)
            if success:
                U_arr, V_arr, W_arr, success = _polish(U_arr, V_arr, W_arr, T, max_iters=100, tol=1e-14)
                if success:
                    U_h = _to_half_int(U_arr)
                    V_h = _to_half_int(V_arr)
                    W_h = _to_half_int(W_arr)
                    
                    T_recon = np.einsum('ar,br,cr->abc', U_h, V_h, W_h)
                    if np.allclose(T_recon, T, atol=1e-10):
                        best_result = {"U": U_h.tolist(), "V": V_h.tolist(), "W": W_h.tolist()}
                        best_rank = r
                        found = True
                        break
        
        if found:
            continue
    
    return best_result


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
