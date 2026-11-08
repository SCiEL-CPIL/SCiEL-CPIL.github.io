import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

def _build_known_66():
    n, m, p = DIMENSIONS
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    current_rank = 0

    def add_rank1(u_idx, u_vals, v_idx, v_vals, w_idx, w_vals):
        nonlocal current_rank
        if current_rank >= rank:
            return
        for idx, val in zip(u_idx, u_vals):
            U[idx][current_rank] += val
        for idx, val in zip(v_idx, v_vals):
            V[idx][current_rank] += val
        for idx, val in zip(w_idx, w_vals):
            W[idx][current_rank] += val
        current_rank += 1

    def add_1_4_2(i_off, j_off, k_off):
        for j in range(4):
            for k in range(2):
                u_idx = (i_off) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])

    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])

    def add_2_4_2(i_off, j_off, k_off):
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)

    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2(0, 0, k_off)
        add_2_4_2(1, 0, k_off)

    return {"U": U, "V": V, "W": W}


def _validate(U, V, W, n, m, p):
    """Verify exact decomposition: sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == 1 for all i,j,k"""
    Ua = np.asarray(U, dtype=object)
    Va = np.asarray(V, dtype=object)
    Wa = np.asarray(W, dtype=object)
    R = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(R):
                    s = s + Ua[i * m + j, t] * Va[j * p + k, t] * Wa[k * n + i, t]
                if s != 1:
                    return False
    return True


def _validate_float(U, V, W, n, m, p, tol=1e-8):
    """Float-based validation for speed during search"""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    R = Ua.shape[1]
    # T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
    # Reshape for einsum
    U3 = Ua.reshape(n, m, R)
    V3 = Va.reshape(m, p, R)
    W3 = Wa.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kr->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=tol)


def _try_sparse_decomposition(n, m, p, target_rank, deadline, seed=42):
    """
    Try to find a rank-`target_rank` decomposition by:
    1. Randomly choosing sparse supports for each rank-1 term
    2. Solving the resulting linear system
    3. Checking if the solution is exact (integer/half-integer)
    """
    rng = np.random.default_rng(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute index mappings
    u_indices = list(range(nm))  # i*m + j
    v_indices = list(range(mp))  # j*p + k
    w_indices = list(range(pn))  # k*n + i
    
    # For each (i,j,k), we need: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    # This is nonlinear. But if we fix the support pattern (which entries are nonzero),
    # we can set up a system.
    
    # Strategy: for each rank-1 term t, choose:
    # - A subset Su_t of U indices (size ~3-5)
    # - A subset Sv_t of V indices (size ~3-5)  
    # - A subset Sw_t of W indices (size ~3-5)
    # Then the constraint for (i,j,k) becomes:
    # sum over t where (i*m+j) in Su_t, (j*p+k) in Sv_t, (k*n+i) in Sw_t:
    #   U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    # This is still nonlinear. 
    
    # Alternative: use the fact that if we set U, V, W entries to be +/-1 or 0,
    # we can brute-force small patterns. But that's too slow.
    
    # Better approach: use the 66-rank solution as a starting point and try to
    # merge rank-1 terms by finding linear dependencies.
    
    # Let's try a different structural approach:
    # The tensor T_{i,j,k} = 1 for all i,j,k.
    # We can write T = sum_t u_t ⊗ v_t ⊗ w_t where u_t, v_t, w_t are vectors.
    # This is equivalent to finding a low-rank factorization of the 3-way tensor.
    
    # Use ALS (Alternating Least Squares) with sparsity constraints:
    # 1. Initialize U, V, W randomly with small rank
    # 2. Alternate: fix V,W, solve for U; fix U,W, solve for V; fix U,V, solve for W
    # 3. After convergence, round to half-integers and verify
    
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        U = rng.normal(size=(nm, target_rank)) * 0.5
        V = rng.normal(size=(mp, target_rank)) * 0.5
        W = rng.normal(size=(pn, target_rank)) * 0.5
        
        # ALS iterations
        for it in range(200):
            if time.time() > deadline - 5:
                break
            
            # Fix V, W, solve for U
            # T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
            # For fixed V,W: for each (i,j), U[i*m+j,:] must satisfy
            # sum_k T[i,j,k] = sum_t U[i*m+j,t] * (sum_k V[j*p+k,t] * W[k*n+i,t])
            # Let G[i,j,t] = sum_k V[j*p+k,t] * W[k*n+i,t]
            # Then: sum_t U[i*m+j,t] * G[i,j,t] = 1 for all k... wait, this doesn't work directly.
            
            # Actually, the constraint is:
            # For all i,j,k: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
            # If we fix V and W, for each (i,j), we have p equations:
            # For k=0,...,p-1: sum_t U[i*m+j,t] * (V[j*p+k,t] * W[k*n+i,t]) = 1
            # This is a linear system in U[i*m+j, 0..target_rank-1]
            # But we have p equations and target_rank unknowns. If p < target_rank, underdetermined.
            # If p >= target_rank, we can solve least squares.
            
            # Let's build the system for all (i,j) pairs:
            # Total equations: n * m * p = 72
            # Total unknowns: n * m * target_rank = 12 * target_rank
            # If target_rank <= 6, we have more unknowns than equations (underdetermined)
            # If target_rank > 6, we have more equations than unknowns (overdetermined)
            
            # For target_rank = 6: 72 equations, 72 unknowns -> square system
            # For target_rank < 6: underdetermined, can use minimum norm solution
            # For target_rank > 6: overdetermined, least squares
            
            # Build the linear system:
            # Unknowns: U_flat of size nm * target_rank
            # Equations: for each (i,j,k): sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
            
            # This is a linear system A * U_flat = b where:
            # A[(i,j,k), (i',j',t)] = delta_{i,i'} * delta_{j,j'} * V[j*p+k,t] * W[k*n+i,t]
            # b[(i,j,k)] = 1
            
            # But this A matrix has a special structure. Let's build it efficiently.
            
            # Actually, let's just do the ALS properly:
            # For fixed V,W:
            # For each (i,j), solve: min_U ||A_{ij} U_{ij} - 1||^2
            # where A_{ij} is a p x target_rank matrix with A_{ij}[k,t] = V[j*p+k,t] * W[k*n+i,t]
            
            new_U = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A_ij = np.zeros((p, target_rank))
                    for k in range(p):
                        for t in range(target_rank):
                            A_ij[k, t] = V[j * p + k, t] * W[k * n + i, t]
                    b_ij = np.ones(p)
                    # Solve least squares
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_ij, b_ij, rcond=None)
                        new_U[i * m + j, :] = sol
                    except:
                        new_U[i * m + j, :] = U[i * m + j, :]
            U = new_U
            
            # Similarly for V (fix U,W)
            new_V = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A_jk = np.zeros((n, target_rank))
                    for i in range(n):
                        for t in range(target_rank):
                            A_jk[i, t] = U[i * m + j, t] * W[k * n + i, t]
                    b_jk = np.ones(n)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_jk, b_jk, rcond=None)
                        new_V[j * p + k, :] = sol
                    except:
                        new_V[j * p + k, :] = V[j * p + k, :]
            V = new_V
            
            # Similarly for W (fix U,V)
            new_W = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A_ki = np.zeros((m, target_rank))
                    for j in range(m):
                        for t in range(target_rank):
                            A_ki[j, t] = U[i * m + j, t] * V[j * p + k, t]
                    b_ki = np.ones(m)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_ki, b_ki, rcond=None)
                        new_W[k * n + i, :] = sol
                    except:
                        new_W[k * n + i, :] = W[k * n + i, :]
            W = new_W
        
        # Check if the solution is close to half-integers
        # Round to half-integers
        Ur = np.round(2 * U) / 2.0
        Vr = np.round(2 * V) / 2.0
        Wr = np.round(2 * W) / 2.0
        
        # Check if rounding changed things significantly
        if not (np.allclose(U, Ur, atol=0.01) and 
                np.allclose(V, Vr, atol=0.01) and 
                np.allclose(W, Wr, atol=0.01)):
            continue
        
        # Verify the rounded solution
        if _validate_float(Ur.tolist(), Vr.tolist(), Wr.tolist(), n, m, p, tol=1e-8):
            # Convert to list format
            return (Ur.tolist(), Vr.tolist(), Wr.tolist())
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0  # Safety margin
    
    # Start from the known 66-rank solution
    known = _build_known_66()
    best = known
    best_rank = 66
    
    if not _validate(known["U"], known["V"], known["W"], n, m, p):
        return known
    
    # Try to find a lower rank decomposition using ALS with sparsity
    # Try ranks from 48 up to 65
    for target_rank in range(48, 66):
        if time.time() > deadline - 10:
            break
        
        result = _try_sparse_decomposition(n, m, p, target_rank, deadline)
        if result is not None:
            U, V, W = result
            if _validate(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break  # Found a valid lower rank, stop
    
    return best


if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
