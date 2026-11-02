"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x7 matrix.
    
    Strategy:
    1. Compute unfolding rank lower bounds
    2. Try constructive decompositions at ranks from lower bound upward
    3. Use the special structure of matmul tensors for valid constructions
    4. Fall back to schoolbook (rank 84) if no better found
    """
    start_time = time.time()
    deadline = start_time + 130.0

    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 28
    pn = p * n  # 21
    schoolbook_rank = n * m * p  # 84

    # Build the tensor T[a, b, c]
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]"""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol

    def verify_and_round(U, V, W, max_trials=200):
        """
        Try to find a valid half-integer decomposition.
        Returns (U_r, V_r, W_r) if valid, else None.
        """
        if time.time() > deadline:
            return None
        
        # First try direct rounding
        U_r = round_to_half_integer(U)
        V_r = round_to_half_integer(V)
        W_r = round_to_half_integer(W)
        if verify_decomposition(U_r, V_r, W_r):
            return U_r, V_r, W_r
        
        # Try systematic scalings
        rng = np.random.RandomState(42)
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            alpha = rng.uniform(0.5, 2.0)
            beta = rng.uniform(0.5, 2.0)
            gamma = 1.0 / (alpha * beta)
            
            U_s = U * alpha
            V_s = V * beta
            W_s = W * gamma
            
            U_r = round_to_half_integer(U_s)
            V_r = round_to_half_integer(V_s)
            W_r = round_to_half_integer(W_s)
            
            if verify_decomposition(U_r, V_r, W_r):
                return U_r, V_r, W_r
        
        return None

    def round_to_half_integer(x):
        """Round to nearest multiple of 0.5"""
        return np.round(2 * x) / 2.0



    # Compute lower bound from unfolding ranks
    T_unfold1 = T.reshape(nm, mp * pn)
    T_unfold2 = T.reshape(nm * mp, pn)
    T_unfold3 = T.reshape(mp, nm * pn)
    
    rank_lb1 = np.linalg.matrix_rank(T_unfold1, tol=1e-10)
    rank_lb2 = np.linalg.matrix_rank(T_unfold2, tol=1e-10)
    rank_lb3 = np.linalg.matrix_rank(T_unfold3, tol=1e-10)
    rank_lb = max(rank_lb1, rank_lb2, rank_lb3)
    rank_lb = max(1, int(rank_lb))

    best_result = None
    best_rank = schoolbook_rank

    # Try to construct valid decompositions at ranks from lower bound upward
    # Use the structure: for each (i,j,k), T[i*m+j, j*p+k, k*n+i] = 1
    
    # Strategy: try rank = rank_lb, rank_lb+1, ..., up to some limit
    # For each rank, try multiple construction methods
    
    max_rank_to_try = min(schoolbook_rank, rank_lb + 40)
    
    for rank in range(rank_lb, max_rank_to_try + 1):
        if time.time() > deadline:
            break
        
        found = False
        
        # Method 1: Structured construction using the matmul identity
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For matmul: a=i*m+j, b=j'*p+k, c=k'*n+i'
        # T = 1 iff i=i', j=j', k=k'
        
        # Try: assign each (i,j,k) triple to a unique rank component (schoolbook)
        # But try to merge components when possible
        
        # Method 2: Use ALS with good initialization
        def als_decompose(T, rank, max_iters=100, tol=1e-12, seed=None, U_init=None, V_init=None, W_init=None):
            if seed is not None:
                rng = np.random.RandomState(seed)
            else:
                rng = np.random.RandomState()
            
            if U_init is not None and V_init is not None and W_init is not None:
                U = U_init.astype(float).copy()
                V = V_init.astype(float).copy()
                W = W_init.astype(float).copy()
            else:
                U = rng.randn(nm, rank)
                V = rng.randn(mp, rank)
                W = rng.randn(pn, rank)
            
            prev_err = float('inf')
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U: minimize ||T - U*(V⊗W)||
                VW = np.kron(V, W)
                T_unfold1 = T.reshape(nm, mp * pn)
                try:
                    U_new = np.linalg.lstsq(VW.T, T_unfold1.T, rcond=None)[0].T
                except:
                    break
                U = U_new
                
                # Update V: minimize ||T - (U⊗V)*W||
                UW = np.kron(U, W)
                T_unfold2 = T.transpose(1, 2, 0).reshape(mp, nm * pn)
                try:
                    V_new = np.linalg.lstsq(UW.T, T_unfold2.T, rcond=None)[0].T
                except:
                    break
                V = V_new
                
                # Update W: minimize ||T - (U⊗V)*W||
                UV = np.kron(U, V)
                T_unfold3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
                try:
                    W_new = np.linalg.lstsq(UV.T, T_unfold3.T, rcond=None)[0].T
                except:
                    break
                W = W_new
                
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_recon - T)
                
                if abs(prev_err - err) < tol:
                    break
                prev_err = err
            
            return U, V, W, err

        # Try multiple restarts for this rank
        num_restarts = max(3, 10 - (rank - rank_lb))
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            try:
                if restart == 0:
                    # Structured init: one component per (i,j,k) but distributed
                    U_init = np.zeros((nm, rank))
                    V_init = np.zeros((mp, rank))
                    W_init = np.zeros((pn, rank))
                    idx = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                r = idx % rank
                                U_init[i*m + j, r] = 1.0
                                V_init[j*p + k, r] = 1.0
                                W_init[k*n + i, r] = 1.0
                                idx += 1
                    U, V, W, err = als_decompose(T, rank, max_iters=50, tol=1e-12, 
                                                  seed=restart*1000+rank,
                                                  U_init=U_init, V_init=V_init, W_init=W_init)
                elif restart == 1:
                    # Random init
                    U, V, W, err = als_decompose(T, rank, max_iters=50, tol=1e-12, 
                                                  seed=restart*7919+rank*104729)
                else:
                    # Sparse random init
                    rng = np.random.RandomState(restart*31337+rank)
                    U_init = rng.randn(nm, rank) * (rng.rand(nm, rank) < 0.5)
                    V_init = rng.randn(mp, rank) * (rng.rand(mp, rank) < 0.5)
                    W_init = rng.randn(pn, rank) * (rng.rand(pn, rank) < 0.5)
                    U, V, W, err = als_decompose(T, rank, max_iters=50, tol=1e-12, 
                                                  seed=restart*1597+rank,
                                                  U_init=U_init, V_init=V_init, W_init=W_init)
                
                if err < 1e-6:
                    result = verify_and_round(U, V, W, max_trials=200)
                    if result is not None:
                        U_r, V_r, W_r = result
                        best_result = (U_r, V_r, W_r, rank)
                        found = True
                        break
            except Exception:
                continue
        
        if found:
            break

    # If no better decomposition found, fall back to schoolbook
    if best_result is None:
        rank = schoolbook_rank
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        
        best_result = (U, V, W, rank)

    U, V, W, rank = best_result
    
    # Convert to lists of lists for compatibility
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
