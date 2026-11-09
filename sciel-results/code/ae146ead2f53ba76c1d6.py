"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1.0
                V[j * p + k, column] = 1.0
                W[k * n + i, column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W, "rank": rank}

def _verify(decomp, n, m, p):
    """Verify that the tensor decomposition exactly multiplies the matrices."""
    U = np.array(decomp["U"], dtype=np.float64)
    V = np.array(decomp["V"], dtype=np.float64)
    W = np.array(decomp["W"], dtype=np.float64)
    r = U.shape[1]
    
    # Check if entries are half-integers
    for mat in [U, V, W]:
        if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
            return False

    # Reshape to (n, m, r), (m, p, r), (p, n, r)
    U_t = U.reshape(n, m, r)
    V_t = V.reshape(m, p, r)
    W_t = W.reshape(p, n, r)
    
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    # Check if T is close to 1 for all i,j,k
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _compute_error(U, V, W, n, m, p):
    rank = U.shape[1]
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    return np.sum((T - 1.0) ** 2)

def _discrete_local_search(U, V, W, n, m, p, max_iters=2000, deadline=None):
    """Local search on half-integer values to minimize error."""
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    current_error = _compute_error(U, V, W, n, m, p)
    
    # Candidate values: small range of half-integers
    # We can expand this if needed, but [-2, 2] is usually sufficient for rank reduction
    candidates = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if current_error < 1e-12:
            break
        
        # Pick a random matrix, row, column
        mat_idx = random.randint(0, 2)
        if mat_idx == 0:
            row = random.randint(0, num_u - 1)
            col = random.randint(0, rank - 1)
            mat = U
        elif mat_idx == 1:
            row = random.randint(0, num_v - 1)
            col = random.randint(0, rank - 1)
            mat = V
        else:
            row = random.randint(0, num_w - 1)
            col = random.randint(0, rank - 1)
            mat = W
        
        old_val = mat[row, col]
        new_val = random.choice(candidates)
        if abs(new_val - old_val) < 1e-10:
            continue
        
        mat[row, col] = new_val
        new_error = _compute_error(U, V, W, n, m, p)
        
        if new_error < current_error - 1e-14:
            current_error = new_error
        else:
            mat[row, col] = old_val
    
    return U, V, W, current_error

def _try_rank_discrete(n, m, p, rank, deadline=None):
    """Try to find a rank-r decomposition using discrete local search."""
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # Strategy: Start from schoolbook, remove (n*m*p - rank) columns, then polish.
    # Schoolbook has rank R = n*m*p.
    R = n * m * p
    
    # If target rank is higher than R, we can't reduce, but we can just return schoolbook padded?
    # No, rank must be exact. If rank > R, we can add zero columns.
    # But we are searching for MINIMAL rank, so we expect rank <= R.
    
    if rank > R:
        # Pad schoolbook with zeros
        U = np.zeros((num_u, rank))
        V = np.zeros((num_v, rank))
        W = np.zeros((num_w, rank))
        U[:, :R] = _schoolbook()["U"]
        V[:, :R] = _schoolbook()["V"]
        W[:, :R] = _schoolbook()["W"]
        return {"U": U, "V": V, "W": W, "rank": rank}

    # Initialize by taking the first 'rank' columns of schoolbook
    sb = _schoolbook()
    U_init = sb["U"][:, :rank].copy()
    V_init = sb["V"][:, :rank].copy()
    W_init = sb["W"][:, :rank].copy()
    
    # Run local search
    U, V, W, err = _discrete_local_search(U_init, V_init, W_init, n, m, p, max_iters=1000, deadline=deadline)
    
    if err < 1e-10:
        return {"U": U, "V": V, "W": W, "rank": rank}
    
    # If it failed, try a few random restarts with different column subsets?
    # For now, just return None if it doesn't converge quickly.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    R = n * m * p
    
    # Start with schoolbook
    best_decomp = _schoolbook()
    best_rank = R
    
    # Try to reduce rank from R-1 down to a reasonable lower bound (e.g., 20)
    # Theoretical lower bound for 2x5x6 is at least max(n*m, m*p, p*n) = 30? No, that's for specific cases.
    # General lower bound is max(n*m, m*p, p*n) is not always true, but rank >= max(n*m, m*p, p*n) is false.
    # Rank >= max(n*m, m*p, p*n) is NOT a lower bound.
    # However, for matrix multiplication, rank >= max(n*m, m*p, p*n) is often cited as a heuristic lower bound for "obvious" decompositions,
    # but Strassen shows lower is possible.
    # For 2x5x6, let's try down to 25.
    
    for rank in range(R - 1, 24, -1):
        if time.time() > deadline:
            break
        
        # Try a few randomizations of the initial column selection
        found = False
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Shuffle the columns of schoolbook to get different initializations
            cols = list(range(R))
            random.shuffle(cols)
            selected_cols = cols[:rank]
            
            sb = _schoolbook()
            U_init = sb["U"][:, selected_cols].copy()
            V_init = sb["V"][:, selected_cols].copy()
            W_init = sb["W"][:, selected_cols].copy()
            
            # Run local search
            U, V, W, err = _discrete_local_search(U_init, V_init, W_init, n, m, p, max_iters=500, deadline=deadline)
            
            if err < 1e-10:
                # Verify
                decomp = {"U": U, "V": V, "W": W}
                if _verify(decomp, n, m, p):
                    best_decomp = decomp
                    best_rank = rank
                    found = True
                    break
        
        if found:
            # Continue trying to reduce further
            pass
        else:
            # If we fail to find a solution for this rank, maybe try the next lower rank?
            # Or stop? Usually if we can't reduce to R-1, we might be stuck.
            # But sometimes skipping a rank helps. Let's continue to next rank.
            pass
    
    return {"U": best_decomp["U"].tolist(), "V": best_decomp["V"].tolist(), "W": best_decomp["W"].tolist()}
# EVOLVE-BLOCK-END
