"""Matrix multiplication 3x3 * 3x7 rank minimization via structured algebraic search."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    # Schoolbook fallback
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
    
    best_result = None
    best_rank = n * m * p  # 63
    
    # Known decomposition ranks to try (from low to high)
    # For 3x3x3, minimal rank is 23 (over C). For 3x3x7, we can potentially do better
    # by exploiting the structure.
    # Try ranks from 25 upward
    
    # Value set for exact integer/half-integer solutions
    values = [0, 1, -1, 0.5, -0.5]
    
    # Strategy 1: Systematic search with structured initialization
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    # Key insight: For small matrices, we can use the fact that the tensor
    # has a product structure. Try to find decompositions where each term
    # contributes to a small subset of (i,j,k) tuples.
    
    target_ranks = list(range(25, 45))
    
    for rank in target_ranks:
        if time.time() > deadline - 15:
            break
        
        found = _find_decomposition(rank, n, m, p, values, deadline)
        if found is not None:
            best_rank = rank
            best_result = found
            break
    
    if best_result is not None:
        return best_result
    
    # Strategy 2: Random search with gradient-free optimization
    if time.time() < deadline - 10:
        found = _random_search(n, m, p, deadline)
        if found is not None:
            r = len(found["U"][0])
            if r < best_rank:
                best_rank = r
                best_result = found
    
    if best_result is None:
        return make_schoolbook()
    
    return best_result

def _find_decomposition(rank, n, m, p, values, deadline):
    """
    Search for rank-r decomposition using structured patterns.
    
    The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    We need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all (i,j,k).
    
    Strategy: Use the fact that for each fixed k, we need a 3x3 matrix
    multiplication. Try to share terms across different k values.
    """
    # Precompute some useful structures
    # For each pair (i, k), we need sum_j A[i,j] B[j,k] = C[i,k]
    
    # Try different structured patterns for initialization
    patterns = []
    
    # Pattern 0: Pure schoolbook-like with some sharing
    # Pattern 1: Block diagonal in k
    # Pattern 2: Random sparse
    
    # For each pattern, try multiple random seeds
    max_attempts = 30 if rank <= 30 else 15
    if time.time() > deadline - 20:
        max_attempts = 5
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
        
        rng = np.random.RandomState(42 + attempt * 7919 + rank * 13)
        
        # Initialize with structured pattern
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Fill with structured patterns based on attempt
        for c in range(rank):
            pattern_type = (c + attempt) % 5
            
            if pattern_type == 0:
                # Single (i,j,k) contribution
                i = c % n
                j = (c // n) % m
                k = (c // (n * m)) % p
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
            elif pattern_type == 1:
                # Two i's share same j,k
                i1 = c % n
                i2 = (c + 1) % n
                if i1 == i2:
                    i2 = (i2 + 1) % n
                j = (c // n) % m
                k = (c // (n * m)) % p
                U[i1 * m + j, c] = 1.0
                U[i2 * m + j, c] = 1.0
                V[j * p + k, c] = 0.5
                W[k * n + i1, c] = 1.0
                W[k * n + i2, c] = 1.0
            elif pattern_type == 2:
                # Two j's share same i,k
                i = c % n
                j1 = (c // n) % m
                j2 = (j1 + 1) % m
                k = (c // (n * m)) % p
                U[i * m + j1, c] = 0.5
                U[i * m + j2, c] = 0.5
                V[j1 * p + k, c] = 1.0
                V[j2 * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
            elif pattern_type == 3:
                # Two k's share same i,j
                i = c % n
                j = (c // n) % m
                k1 = (c // (n * m)) % p
                k2 = (k1 + 1) % p
                U[i * m + j, c] = 1.0
                V[j * p + k1, c] = 0.5
                V[j * p + k2, c] = 0.5
                W[k1 * n + i, c] = 1.0
                W[k2 * n + i, c] = 1.0
            else:
                # Random sparse pattern
                n_u = rng.randint(1, 3)
                n_v = rng.randint(1, 3)
                n_w = rng.randint(1, 3)
                
                u_indices = rng.choice(n * m, min(n_u, n * m), replace=False)
                v_indices = rng.choice(m * p, min(n_v, m * p), replace=False)
                w_indices = rng.choice(p * n, min(n_w, p * n), replace=False)
                
                for idx in u_indices:
                    U[idx, c] = rng.choice(values)
                for idx in v_indices:
                    V[idx, c] = rng.choice(values)
                for idx in w_indices:
                    W[idx, c] = rng.choice(values)
        
        # Local refinement using coordinate descent
        U, V, W = _local_refine(U, V, W, n, m, p, values, deadline)
        
        # Check if solution is valid
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        max_err = np.max(np.abs(T - 1.0))
        
        if max_err < 1e-10:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _local_refine(U, V, W, n, m, p, values, deadline):
    """
    Local refinement using coordinate descent.
    Try changing individual entries to reduce the error.
    """
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    
    # Compute initial error
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    error = np.sum((T - 1.0) ** 2)
    
    best_error = error
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Iterative improvement
    max_iter = 100
    for it in range(max_iter):
        if time.time() > deadline - 2:
            break
        
        if error < 1e-12:
            break
        
        improved = False
        
        # Try changing entries in U
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                for c in range(rank):
                    old_val = U[idx, c]
                    for new_val in values:
                        if new_val == old_val:
                            continue
                        
                        U[idx, c] = new_val
                        Ur = U.reshape(n, m, rank)
                        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
                        new_error = np.sum((T - 1.0) ** 2)
                        
                        if new_error < error - 1e-12:
                            error = new_error
                            improved = True
                            break
                        else:
                            U[idx, c] = old_val
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        # Try changing entries in V
        if not improved:
            for j in range(m):
                for k in range(p):
                    idx = j * p + k
                    for c in range(rank):
                        old_val = V[idx, c]
                        for new_val in values:
                            if new_val == old_val:
                                continue
                            
                            V[idx, c] = new_val
                            Vr = V.reshape(m, p, rank)
                            T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
                            new_error = np.sum((T - 1.0) ** 2)
                            
                            if new_error < error - 1e-12:
                                error = new_error
                                improved = True
                                break
                            else:
                                V[idx, c] = old_val
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # Try changing entries in W
        if not improved:
            for k in range(p):
                for i in range(n):
                    idx = k * n + i
                    for c in range(rank):
                        old_val = W[idx, c]
                        for new_val in values:
                            if new_val == old_val:
                                continue
                            
                            W[idx, c] = new_val
                            Wr = W.reshape(p, n, rank)
                            T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
                            new_error = np.sum((T - 1.0) ** 2)
                            
                            if new_error < error - 1e-12:
                                error = new_error
                                improved = True
                                break
                            else:
                                W[idx, c] = old_val
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
        
        # Update best
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Update reshaped views
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
    
    return best_U, best_V, best_W

def _random_search(n, m, p, deadline):
    """
    Random search with smooth objective.
    Use a soft objective to guide the search, then try to snap to exact values.
    """
    # Try different ranks
    for target_rank in range(25, 35):
        if time.time() > deadline - 10:
            break
        
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            rng = np.random.RandomState(999 + attempt * 101 + target_rank)
            
            # Initialize with small random values
            scale = 0.5
            U = rng.standard_normal((n * m, target_rank)) * scale
            V = rng.standard_normal((m * p, target_rank)) * scale
            W = rng.standard_normal((p * n, target_rank)) * scale
            
            # Add schoolbook-like components
            for c in range(min(target_rank, 9)):
                i = c % n
                j = (c // n) % m
                k = (c // (n * m)) % p
                U[i * m + j, c] += 1.0
                V[j * p + k, c] += 1.0
                W[k * n + i, c] += 1.0
            
            # Coordinate descent with continuous values, then snap
            Ur = U.reshape(n, m, target_rank)
            Vr = V.reshape(m, p, target_rank)
            Wr = W.reshape(p, n, target_rank)
            
            T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
            error = np.sum((T - 1.0) ** 2)
            
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Simple gradient-free optimization
            step_size = 0.1
            for step in range(50):
                if time.time() > deadline - 2:
                    break
                
                if error < 1e-10:
                    break
                
                # Try random perturbations
                for _ in range(5):
                    U_pert = U + rng.standard_normal((n * m, target_rank)) * step_size
                    V_pert = V + rng.standard_normal((m * p, target_rank)) * step_size
                    W_pert = W + rng.standard_normal((p * n, target_rank)) * step_size
                    
                    Ur_pert = U_pert.reshape(n, m, target_rank)
                    Vr_pert = V_pert.reshape(m, p, target_rank)
                    Wr_pert = W_pert.reshape(p, n, target_rank)
                    
                    T_pert = np.einsum('ijc,jkc,kic->ijk', Ur_pert, Vr_pert, Wr_pert)
                    new_error = np.sum((T_pert - 1.0) ** 2)
                    
                    if new_error < error:
                        U, V, W = U_pert, V_pert, W_pert
                        error = new_error
                        step_size *= 0.8
                        break
                else:
                    step_size *= 0.9
                
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Try to snap to nearest exact values
            if best_error < 1.0:
                U_snap = np.round(best_U * 2) / 2  # Snap to half-integers
                V_snap = np.round(best_V * 2) / 2
                W_snap = np.round(best_W * 2) / 2
                
                Ur_snap = U_snap.reshape(n, m, target_rank)
                Vr_snap = V_snap.reshape(m, p, target_rank)
                Wr_snap = W_snap.reshape(p, n, target_rank)
                
                T_snap = np.einsum('ijc,jkc,kic->ijk', Ur_snap, Vr_snap, Wr_snap)
                snap_error = np.sum((T_snap - 1.0) ** 2)
                
                if snap_error < 1e-10:
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    return None
# EVOLVE-BLOCK-END
