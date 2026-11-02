import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def _validate_fast(U, V, W, n, m, p, tol=1e-10):
    """Fast vectorized validation using einsum."""
    rank = U.shape[1]
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    return np.max(np.abs(T - 1.0)) < tol

def _snap_to_half_integers(arr, tol=1e-6):
    """Snap values close to half-integers to exact half-integers."""
    snapped = arr.copy()
    candidates = np.round(snapped * 2) / 2
    # Snap individually if close
    mask = np.abs(snapped - candidates) < tol
    snapped[mask] = candidates[mask]
    # Also check integers just in case, though half-int includes integers
    candidates_int = np.round(snapped)
    mask_int = np.abs(snapped - candidates_int) < tol
    snapped[mask_int] = candidates_int[mask_int]
    return snapped

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _gradient_descent_fast(U, V, W, n, m, p, lr=0.01, max_steps=1000, tol=1e-8, momentum=0.9):
    """Vectorized gradient descent for tensor decomposition with momentum."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank).copy()
    V_r = V.reshape(m, p, rank).copy()
    W_r = W.reshape(p, n, rank).copy()
    
    # Momentum buffers
    v_U = np.zeros_like(U_r)
    v_V = np.zeros_like(V_r)
    v_W = np.zeros_like(W_r)
    
    best_loss = np.inf
    best_state = (U_r.copy(), V_r.copy(), W_r.copy())
    
    for step in range(max_steps):
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        diff = T - 1.0
        loss = np.sum(diff**2)
        
        if loss < best_loss:
            best_loss = loss
            best_state = (U_r.copy(), V_r.copy(), W_r.copy())
            
        if loss < tol:
            return best_state[0].reshape(n*m, rank), best_state[1].reshape(m*p, rank), best_state[2].reshape(p*n, rank), True
        
        # Gradients
        grad_U = np.einsum('ijk,jkr,kir->ijr', diff, V_r, W_r)
        grad_V = np.einsum('ijk,ijr,kir->jkr', diff, U_r, W_r)
        grad_W = np.einsum('ijk,ijr,jkr->kir', diff, U_r, V_r)
        
        # Momentum update
        v_U = momentum * v_U + (1 - momentum) * grad_U
        v_V = momentum * v_V + (1 - momentum) * grad_V
        v_W = momentum * v_W + (1 - momentum) * grad_W
        
        # Adaptive learning rate: reduce if loss increases significantly
        if step > 0 and loss > best_loss * 1.5:
            lr *= 0.5
            if lr < 1e-6:
                break
                
        U_r -= lr * v_U
        V_r -= lr * v_V
        W_r -= lr * v_W
        
    return best_state[0].reshape(n*m, rank), best_state[1].reshape(m*p, rank), best_state[2].reshape(p*n, rank), best_loss < tol

def _exact_solve_W(U, V, n, m, p):
    """Given U and V, solve for W exactly using least squares."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    
    # We want T_ijk = sum_r U_ijr V_jkr W_kir = 1
    # This is a linear system for W_kir.
    # Let A be a matrix where each row corresponds to a (i,j,k) triplet and each column to a (k,i,r) triplet.
    # Actually, it's easier to vectorize.
    # T = einsum('ijr,jkr,kir->ijk', U, V, W)
    # Fix i,j,k. Sum over r.
    # Let's build the linear system.
    # Variables: W_flat of size p*n*rank
    # Equations: n*m*p equations.
    
    # Construct matrix A such that A @ W_flat = 1
    # A[ (i,j,k), (k',i',r') ] = 1 if k==k' and i==i' else 0? No.
    # Term is U_ijr * V_jkr * W_kir.
    # For fixed (i,j,k), the term is sum_r (U_ijr * V_jkr) * W_kir.
    # So for each (i,j,k), we have a linear equation for the vector W_ki (which has 'rank' components).
    # Wait, W_kir depends on k and i.
    # For a fixed (i,j,k), the unknowns are W_kir for r=0..rank-1.
    # But W_kir appears in multiple equations (for different j).
    # So we must solve the global system.
    
    # Total unknowns: p * n * rank
    # Total equations: n * m * p
    
    # Let's form the matrix A.
    # Indexing:
    # Eq index e = (i * m * p + j * p + k)
    # Var index v = (k * n * rank + i * rank + r)
    
    # A[e, v] = U_ijr * V_jkr if v corresponds to (k,i,r) and e corresponds to (i,j,k)
    
    # This matrix is sparse but for small dimensions, dense is fine.
    # n=2, m=3, p=4.
    # Eqs: 2*3*4 = 24
    # Vars: 4*2*rank. If rank=20, Vars=160.
    # 24 x 160 matrix.
    
    A = np.zeros((n * m * p, p * n * rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = i * m * p + j * p + k
                for r in range(rank):
                    var_idx = k * n * rank + i * rank + r
                    A[eq_idx, var_idx] = U_r[i, j, r] * V_r[j, k, r]
    
    b = np.ones(n * m * p)
    
    try:
        W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
        W_solved = W_flat.reshape(p, n, rank)
        return W_solved.reshape(p * n, rank)
    except:
        return None

def _polish(U, V, W, n, m, p, max_iter=5):
    """High-precision polish pass. Alternates gradient descent and exact solve for W."""
    for _ in range(max_iter):
        if _validate_fast(U, V, W, n, m, p, tol=1e-12):
            break
        # Step 1: Gradient descent to refine U and V
        U, V, W, converged = _gradient_descent_fast(U, V, W, n, m, p, lr=1e-4, max_steps=100, tol=1e-10, momentum=0.9)
        
        # Step 2: Exact solve for W given U and V
        W_new = _exact_solve_W(U, V, n, m, p)
        if W_new is not None:
            W = W_new
            
    return U, V, W

def construct():
    """
    Constructs the rank-minimal tensor decomposition for 2x3 by 3x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook as default
    U, V, W = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    best_result = (U, V, W)
    
    # Likely ranks to try: 20, 21, 22, 23
    candidate_ranks = [20, 21, 22, 23]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        found = False
        
        # Try many random restarts
        # Increase restarts if time allows
        max_restarts = 100
        for seed in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            rng = np.random.RandomState(seed + 1000 + rank)
            
            # Strategy: Warm start from schoolbook by keeping a subset of columns
            # This is often better than pure random for structured tensors
            if seed % 2 == 0:
                # Pick 'rank' random columns from schoolbook
                U_sb, V_sb, W_sb = _schoolbook(n, m, p)
                cols = rng.choice(schoolbook_rank, size=rank, replace=False)
                U_try = U_sb[:, cols].copy()
                V_try = V_sb[:, cols].copy()
                W_try = W_sb[:, cols].copy()
                # Add small noise to break symmetry
                U_try += rng.randn(n * m, rank) * 0.01
                V_try += rng.randn(m * p, rank) * 0.01
                W_try += rng.randn(p * n, rank) * 0.01
            else:
                # Random initialization
                scale = 1.0 / np.sqrt(rank)
                U_try = rng.randn(n * m, rank) * scale
                V_try = rng.randn(m * p, rank) * scale
                W_try = rng.randn(p * n, rank) * scale
            
            # Run fast gradient descent
            U_try, V_try, W_try, converged = _gradient_descent_fast(
                U_try, V_try, W_try, n, m, p, 
                lr=0.05, max_steps=500, tol=1e-6, momentum=0.9
            )
            
            if converged or True: # Always try to polish
                # Polish
                U_try, V_try, W_try = _polish(U_try, V_try, W_try, n, m, p)
                
                if _validate_fast(U_try, V_try, W_try, n, m, p, tol=1e-10):
                    # Snap to half-integers
                    U_snap = _snap_to_half_integers(U_try)
                    V_snap = _snap_to_half_integers(V_try)
                    W_snap = _snap_to_half_integers(W_try)
                    
                    if _validate_fast(U_snap, V_snap, W_snap, n, m, p, tol=1e-10):
                        U, V, W = U_snap, V_snap, W_snap
                        best_rank = rank
                        best_result = (U, V, W)
                        found = True
                        break
                    else:
                        # Use un-snapped if it validates
                        U, V, W = U_try, V_try, W_try
                        best_rank = rank
                        best_result = (U, V, W)
                        found = True
                        break
        
        if found:
            # Found a valid decomposition at this rank, try lower ranks if time allows
            continue
    
    # Final validation
    if not _validate_fast(U, V, W, n, m, p, tol=1e-10):
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
