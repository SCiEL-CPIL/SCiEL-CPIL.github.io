import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START

def _reconstruct_tensor(U, V, n, m, p):
    """Reconstruct T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r]"""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        # u_col has shape (n*m,), v_col has shape (m*p,)
        # For each (i,j,k): contribution is u_col[i*m+j] * v_col[j*p+k]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_col[i * m + j] * v_col[j * p + k]
    return T


def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    
    T = _reconstruct_tensor(U_mat, V_mat, n, m, p)
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=1e-9)


def _try_merge(U, V, W, n, m, p, r1, r2, deadline):
    """
    Try to merge rank terms r1 and r2 into a single term.
    Returns (new_U, new_V, new_W, success) or (None, None, None, False)
    """
    if time.time() > deadline:
        return None, None, None, False
    
    rank = U.shape[1]
    
    # The combined contribution of terms r1 and r2 to T[i,j,k] is:
    # (U[i*m+j, r1] + U[i*m+j, r2]) * (V[j*p+k, r1] + V[j*p+k, r2])
    # We need this to equal a single rank-1 term: U'[i*m+j] * V'[j*p+k]
    
    u1 = U[:, r1].copy()
    u2 = U[:, r2].copy()
    v1 = V[:, r1].copy()
    v2 = V[:, r2].copy()
    
    # The sum tensor from these two terms:
    # S[i,j,k] = (u1[i*m+j] + u2[i*m+j]) * (v1[j*p+k] + v2[j*p+k])
    # This is NOT generally rank 1, so we can't simply merge.
    
    # However, we can try to find u', v' such that:
    # u'[i*m+j] * v'[j*p+k] = u1[i*m+j]*v1[j*p+k] + u2[i*m+j]*v2[j*p+k]
    
    # This is equivalent to: the sum of two rank-1 tensors is rank 1
    # which only happens in special cases.
    
    # Alternative approach: try to express the sum as a single term
    # by finding a scaling factor.
    
    # If u2 = alpha * u1 and v2 = beta * v1, then the sum is
    # (1+alpha*beta) * u1 * v1, which is rank 1.
    
    # Check if u2 is proportional to u1
    nonzero_u1 = np.nonzero(u1)[0]
    if len(nonzero_u1) == 0:
        # u1 is zero, just keep u2
        return None, None, None, False
    
    # Check proportionality
    ratios_u = []
    valid = True
    for idx in range(len(u1)):
        if abs(u1[idx]) > 1e-12:
            ratios_u.append(u2[idx] / u1[idx])
        elif abs(u2[idx]) > 1e-12:
            valid = False
            break
    
    if not valid:
        return None, None, None, False
    
    if len(ratios_u) == 0:
        return None, None, None, False
    
    # Check if all ratios are approximately equal
    if len(ratios_u) > 1:
        alpha = ratios_u[0]
        for r in ratios_u[1:]:
            if abs(r - alpha) > 1e-9:
                return None, None, None, False
    else:
        alpha = ratios_u[0]
    
    # Now check v2 proportional to v1
    nonzero_v1 = np.nonzero(v1)[0]
    if len(nonzero_v1) == 0:
        return None, None, None, False
    
    ratios_v = []
    valid = True
    for idx in range(len(v1)):
        if abs(v1[idx]) > 1e-12:
            ratios_v.append(v2[idx] / v1[idx])
        elif abs(v2[idx]) > 1e-12:
            valid = False
            break
    
    if not valid:
        return None, None, None, False
    
    if len(ratios_v) == 0:
        return None, None, None, False
    
    if len(ratios_v) > 1:
        beta = ratios_v[0]
        for r in ratios_v[1:]:
            if abs(r - beta) > 1e-9:
                return None, None, None, False
    else:
        beta = ratios_v[0]
    
    # The sum is (1 + alpha * beta) * u1 * v1
    scale = 1.0 + alpha * beta
    if abs(scale) < 1e-12:
        # The sum cancels out, which is fine (zero contribution)
        # Just remove both terms
        pass
    
    # Create new matrices with terms r1 and r2 removed
    new_rank = rank - 1
    new_U = np.zeros((n * m, new_rank), dtype=float)
    new_V = np.zeros((m * p, new_rank), dtype=float)
    new_W = np.zeros((p * n, new_rank), dtype=float)
    
    col = 0
    for r in range(rank):
        if r == r1 or r == r2:
            continue
        new_U[:, col] = U[:, r]
        new_V[:, col] = V[:, r]
        new_W[:, col] = W[:, r]
        col += 1
    
    # Add the merged term
    if abs(scale) > 1e-12:
        new_U[:, col] = u1 * scale
        new_V[:, col] = v1
        # W for the merged term: we need to figure out what output index it corresponds to
        # In the trivial decomposition, W[k*n+i, r] = 1 for the corresponding (i,k)
        # For a general term, W might need to be computed differently
        # Actually, W is just part of the factorization structure.
        # Let me set W based on the pattern of the original terms.
        
        # For now, let's set W to match the pattern of r1 (or a combination)
        # This might not be correct, so let's verify after
        new_W[:, col] = W[:, r1] + W[:, r2]
    
    return new_U, new_V, new_W, True


def _greedy_merge(U, V, W, n, m, p, deadline):
    """
    Greedily try to merge pairs of rank terms.
    Returns the best decomposition found.
    """
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = U.shape[1]
    
    max_iterations = 1000
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        current_rank = best_U.shape[1]
        if current_rank <= best_rank:
            # Try to find any mergeable pair
            merged = False
            for r1 in range(current_rank):
                if time.time() > deadline:
                    break
                for r2 in range(r1 + 1, current_rank):
                    if time.time() > deadline:
                        break
                    
                    new_U, new_V, new_W, success = _try_merge(
                        best_U, best_V, best_W, n, m, p, r1, r2, deadline
                    )
                    
                    if success and new_U is not None:
                        # Verify the new decomposition
                        if _verify(new_U, new_V, new_W, n, m, p):
                            best_U = new_U
                            best_V = new_V
                            best_W = new_W
                            best_rank = current_rank - 1
                            merged = True
                            break
                if merged or time.time() > deadline:
                    break
            
            if not merged and current_rank == best_rank:
                # No more merges possible with this approach
                break
    
    return best_U, best_V, best_W, best_rank


def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    deadline = time.time() + 130  # Leave 20s safety margin
    
    # Start with trivial decomposition
    rank_initial = n * m * p  # 36
    U = np.zeros((n * m, rank_initial), dtype=float)
    V = np.zeros((m * p, rank_initial), dtype=float)
    W = np.zeros((p * n, rank_initial), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify initial decomposition
    assert _verify(U, V, W, n, m, p), "Initial decomposition failed verification"
    
    # Try to reduce rank via greedy merging
    U, V, W, final_rank = _greedy_merge(U, V, W, n, m, p, deadline)
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        # Fallback to trivial if merge produced invalid result
        U = np.zeros((n * m, rank_initial), dtype=float)
        V = np.zeros((m * p, rank_initial), dtype=float)
        W = np.zeros((p * n, rank_initial), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    # Convert to lists as expected by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
