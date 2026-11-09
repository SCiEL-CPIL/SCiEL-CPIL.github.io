"""Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition search."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using NumPy."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_reshaped = U_np.reshape(n, m, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    
    # We need to check if sum_r U[i,m+j,r] * V[j,p+k,r] * W[k,n+i,r] == 1 for all i,j,k
    # This is a 3-way contraction.
    # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
    
    # We can compute this by iterating over r? No, that's what we want to avoid.
    # But rank is smallish (up to 96). n*m*p is 96.
    # 96 * 96 operations is ~10k, which is fine even in Python if vectorized per r?
    # Actually, let's just do it with np.einsum or similar.
    
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is not a standard einsum because the indices of W are permuted (k,i) vs (i,k).
    # Let's transpose W to (n, p, rank) -> W_trans[i, k, r] = W_reshaped[k, i, r]
    W_transposed = W_reshaped.transpose(1, 0, 2) # shape (n, p, rank)
    
    # Now T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W_transposed[i,k,r]
    # This is an element-wise product summed over r.
    
    # U[i,j,r] * V[j,k,r] * W_transposed[i,k,r]
    # We can compute this for all i,j,k,r at once?
    # U: (n, m, rank)
    # V: (m, p, rank)
    # Wt: (n, p, rank)
    
    # Result shape: (n, m, p)
    # T[i,j,k] = sum_r (U[i,j,r] * V[j,k,r] * Wt[i,k,r])
    
    # We can expand U to (n, m, 1, rank), V to (1, m, p, rank), Wt to (n, 1, p, rank)
    # Then multiply and sum over rank.
    
    # This might be memory heavy if rank is large, but rank <= 96, n=3, m=4, p=8.
    # 3*4*8*96 floats is small.
    
    U_exp = U_reshaped[:, :, np.newaxis, :] # (n, m, 1, rank)
    V_exp = V_reshaped[np.newaxis, :, :, :] # (1, m, p, rank)
    Wt_exp = W_transposed[:, np.newaxis, :, :] # (n, 1, p, rank)
    
    # Product: (n, m, p, rank)
    prod = U_exp * V_exp * Wt_exp
    T = np.sum(prod, axis=3)
    
    # Check if T is all 1s
    return np.allclose(T, 1.0, atol=1e-9)


def _verify(U, V, W, n, m, p):
    """Check if the factorization exactly reproduces the matrix multiplication tensor."""
    try:
        return _verify_fast(U, V, W, n, m, p)
    except Exception:
        # Fallback to slow verification if fast fails (e.g., memory)
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True


def _try_random_search(n, m, p, max_rank, n_trials, deadline):
    """Try to find a low-rank decomposition using random initialization and local search."""
    best = None
    best_rank = n * m * p  # schoolbook rank
    
    start = time.time()
    for trial in range(n_trials):
        if time.time() > deadline - 5:
            break
        
        # Try ranks from low to high
        for rank in range(max_rank, best_rank):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small integers/half-integers
            np.random.seed(trial * 1000 + rank)
            U = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(n*m, rank))
            V = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(m*p, rank))
            W = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(p*n, rank))
            
            # Check
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
            elif rank < best_rank:
                # Keep as candidate for local refinement
                pass
    
    return best, best_rank


def _structured_decomposition(n, m, p):
    """
    Try structured decompositions based on known techniques.
    For 3x4x8, we can try to decompose into smaller blocks.
    """
    # Schoolbook is 96.
    # Can we do better?
    # Let's try a simple heuristic: if p is even, split B into two p/2 blocks.
    # C = A B1 + A B2.
    # This doesn't reduce rank unless we share computations.
    
    # A known lower bound is max(nm, mp, pn) = 32.
    # It is unlikely we can reach 32.
    # Let's just return schoolbook for now, as random search is weak.
    return _schoolbook(n, m, p)


def _local_refine(U, V, W, n, m, p, max_iters, deadline):
    """
    Local search: try to reduce rank by zeroing out columns or 
    finding redundant columns that can be eliminated.
    """
    rank = len(U[0])
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    # Try to eliminate columns that are linearly dependent or redundant
    for _ in range(max_iters):
        if time.time() > deadline - 2:
            break
        
        improved = False
        for r in range(rank - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            
            # Try removing column r: check if the remaining columns still span the tensor
            # This is expensive, so only try for the last few columns
            if r < rank - 5:
                break
            
            # Check if column r is "redundant" by seeing if we can express it
            # in terms of others (approximately)
            # For now, just check if setting it to 0 still works
            U_save = [U[i][r] for i in range(n*m)]
            V_save = [V[i][r] for i in range(m*p)]
            W_save = [W[i][r] for i in range(p*n)]
            
            # Zero out column r
            for i in range(n*m):
                U[i][r] = 0
            for i in range(m*p):
                V[i][r] = 0
            for i in range(p*n):
                W[i][r] = 0
            
            if _verify(U, V, W, n, m, p):
                # Remove column r
                for i in range(n*m):
                    U[i].pop(r)
                for i in range(m*p):
                    V[i].pop(r)
                for i in range(p*n):
                    W[i].pop(r)
                rank -= 1
                improved = True
            else:
                # Restore
                for i in range(n*m):
                    U[i][r] = U_save[i]
                for i in range(m*p):
                    V[i][r] = V_save[i]
                for i in range(p*n):
                    W[i][r] = W_save[i]
            
            if improved:
                break
        
        if not improved:
            break
    
    return {"U": U, "V": V, "W": W}, rank


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Start with schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    assert _verify(best_result["U"], best_result["V"], best_result["W"], n, m, p), "Schoolbook verification failed"
    
    # Try structured decompositions
    if time.time() < deadline - 10:
        structured = _structured_decomposition(n, m, p)
        if _verify(structured["U"], structured["V"], structured["W"], n, m, p):
            s_rank = len(structured["U"][0])
            if s_rank < best_rank:
                best_result = structured
                best_rank = s_rank
    
    # Try random search with various rank targets
    if time.time() < deadline - 5:
        # Try ranks from 40 up to best_rank - 1
        max_search_rank = min(best_rank - 1, 60)
        result, found_rank = _try_random_search(n, m, p, max_search_rank, 200, deadline)
        if result is not None:
            best_result = result
            best_rank = found_rank
    
    # Local refinement on the best result found
    if time.time() < deadline - 2:
        refined, refined_rank = _local_refine(
            best_result["U"], best_result["V"], best_result["W"], 
            n, m, p, max_iters=50, deadline=deadline
        )
        if _verify(refined["U"], refined["V"], refined["W"], n, m, p):
            if refined_rank < best_rank:
                best_result = refined
                best_rank = refined_rank
    
    # Final verification
    if not _verify(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        # Fallback to schoolbook
        best_result = _schoolbook(n, m, p)
    
    return best_result
# EVOLVE-BLOCK-END
