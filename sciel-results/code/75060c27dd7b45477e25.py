"""Matrix multiplication 4x4x6 rank-minimal exact decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _build_strassen():
    """Build the explicit block-Strassen rank-84 decomposition (guaranteed valid)."""
    n, m, p = DIMENSIONS

    rank = 84
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    A_blocks = {
        (0, 0): (0, 1, 0, 1),
        (0, 1): (0, 1, 2, 3),
        (1, 0): (2, 3, 0, 1),
        (1, 1): (2, 3, 2, 3),
    }
    B_blocks = {
        (0, 0): (0, 1, 0, 1),
        (0, 1): (0, 1, 2, 3),
        (0, 2): (0, 1, 4, 5),
        (1, 0): (2, 3, 0, 1),
        (1, 1): (2, 3, 2, 3),
        (1, 2): (2, 3, 4, 5),
    }
    C_blocks = {
        (0, 0): (0, 1, 0, 1),
        (0, 1): (0, 1, 2, 3),
        (0, 2): (0, 1, 4, 5),
        (1, 0): (2, 3, 0, 1),
        (1, 1): (2, 3, 2, 3),
        (1, 2): (2, 3, 4, 5),
    }

    pairs = [
        ((0, 0), (0, 0), (0, 0)),
        ((0, 1), (1, 0), (0, 0)),
        ((0, 0), (0, 1), (0, 1)),
        ((0, 1), (1, 1), (0, 1)),
        ((0, 0), (0, 2), (0, 2)),
        ((0, 1), (1, 2), (0, 2)),
        ((1, 0), (0, 0), (1, 0)),
        ((1, 1), (1, 0), (1, 0)),
        ((1, 0), (0, 1), (1, 1)),
        ((1, 1), (1, 1), (1, 1)),
        ((1, 0), (0, 2), (1, 2)),
        ((1, 1), (1, 2), (1, 2)),
    ]

    # Strassen 2x2x2 terms: local index 0=(0,0),1=(0,1),2=(1,0),3=(1,1)
    strassen_terms = [
        {'u': [(0, 1), (3, 1)], 'v': [(0, 1), (3, 1)], 'w': [(0, 1), (3, 1)]},
        {'u': [(2, 1), (3, 1)], 'v': [(0, 1)], 'w': [(2, 1), (3, -1)]},
        {'u': [(0, 1)], 'v': [(1, 1), (3, -1)], 'w': [(1, 1), (3, 1)]},
        {'u': [(3, 1)], 'v': [(2, 1), (0, -1)], 'w': [(0, 1), (2, 1)]},
        {'u': [(0, 1), (1, 1)], 'v': [(3, 1)], 'w': [(0, -1), (1, 1)]},
        {'u': [(2, 1), (0, -1)], 'v': [(0, 1), (1, 1)], 'w': [(3, 1)]},
        {'u': [(1, 1), (3, -1)], 'v': [(2, 1), (3, 1)], 'w': [(0, 1)]},
    ]

    col = 0
    for (a_idx, b_idx, c_idx) in pairs:
        a_r1, a_r2, a_c1, a_c2 = A_blocks[a_idx]
        b_r1, b_r2, b_c1, b_c2 = B_blocks[b_idx]
        c_r1, c_r2, c_c1, c_c2 = C_blocks[c_idx]

        for term in strassen_terms:
            for local_a, coeff in term['u']:
                lr, lc = divmod(local_a, 2)
                U[(a_r1 + lr) * m + (a_c1 + lc), col] += coeff
            for local_b, coeff in term['v']:
                lr, lc = divmod(local_b, 2)
                V[(b_r1 + lr) * p + (b_c1 + lc), col] += coeff
            for local_c, coeff in term['w']:
                lr, lc = divmod(local_c, 2)
                # W[k*n + i] where k=col of C, i=row of C
                W[(c_c1 + lc) * n + (c_r1 + lr), col] += coeff
            col += 1

    return U, V, W


def _verify_full(U, V, W, n, m, p):
    """Check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]))
                if abs(val - 1.0) > 1e-6:
                    return False
    return True


def _verify_random(U, V, W, n, m, p, trials=5):
    """Verify by actually multiplying random integer matrices."""
    rank = U.shape[1]
    for _ in range(trials):
        A = np.random.randint(0, 5, (n, m), dtype=float)
        B = np.random.randint(0, 5, (m, p), dtype=float)
        C = A @ B
        # C[i,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]*A[i,j]*B[j,k]
        C2 = np.zeros((n, p), dtype=float)
        for r in range(rank):
            Au = A * U[:, r].reshape(n, m)
            Bv = B * V[:, r].reshape(m, p)
            C2 += W[:, r].reshape(p, n).T * (Au @ Bv)
        if not np.allclose(C2, C, atol=1e-5):
            return False
    return True


def _snap_to_halfint(U, V, W):
    """Snap near-integer/half-integer values exactly; return None if any value is far off."""
    for M in (U, V, W):
        for r in range(M.shape[1]):
            for row in range(M.shape[0]):
                v = M[row, r]
                if v == 0:
                    continue
                # nearest half-integer
                h = np.round(2 * v) / 2
                if abs(v - h) > 1e-3:
                    return None
                M[row, r] = h
    return (U, V, W)


def _try_optimize(U0, V0, W0, target_rank, n, m, p, deadline, seed):
    """Attempt to find a rank-target decomposition via iterative refinement.
    Returns (U,V,W) or None."""
    if target_rank >= U0.shape[1]:
        return None
    rng = np.random.RandomState(seed)
    rank = target_rank

    # Initialize from a random subset of the valid rank-84 columns
    idx = rng.choice(U0.shape[1], size=rank, replace=False)
    U = U0[:, idx].copy()
    V = V0[:, idx].copy()
    W = W0[:, idx].copy()

    # Add a small random perturbation to break symmetry
    U += rng.randn(n * m, rank) * 1e-3
    V += rng.randn(m * p, rank) * 1e-3
    W += rng.randn(p * n, rank) * 1e-3

    # Precompute index lists
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    triple_indices = np.array(triples)

    def compute_errors():
        """Compute error for all triples vectorized using einsum."""
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return T.flatten() - 1.0

    def compute_smooth_loss(e, temperature=1.0):
        """Compute a smooth log-sum-exp surrogate for the max-abs error."""
        abs_e = np.abs(e)
        max_abs = np.max(abs_e)
        if max_abs < 1e-15:
            return 0.0, abs_e
        # Use scaled log-sum-exp for numerical stability with temperature control
        scale = max_abs * temperature
        if scale < 1e-15:
            scale = 1e-15
        shifted = (abs_e - max_abs) / scale
        # Clip to prevent overflow
        shifted = np.clip(shifted, -50, 50)
        log_sum_exp = np.log(np.sum(np.exp(shifted))) + max_abs
        return log_sum_exp, abs_e

    def _polish_with_fixed_U(U_fixed, V, W, max_iters=5000, lr=0.01):
        """Polish V and W while keeping U fixed. Returns (V, W, converged)."""
        for it in range(max_iters):
            if time.time() > deadline:
                return V, W, False
            e = compute_errors()
            abs_e = np.abs(e)
            max_err_idx = int(np.argmax(abs_e))
            max_err = e[max_err_idx]
            if abs(max_err) < 1e-12:
                return V, W, True
            i, j, k = triples[max_err_idx]
            # Only update V and W
            dV = U[i * m + j, :] * W[k * n + i, :]
            dW = U[i * m + j, :] * V[j * p + k, :]
            V[j * p + k, :] -= lr * max_err * dV
            W[k * n + i, :] -= lr * max_err * dW
        return V, W, False

    def _polish_with_fixed_V(U, V_fixed, W, max_iters=5000, lr=0.01):
        """Polish U and W while keeping V fixed. Returns (U, W, converged)."""
        for it in range(max_iters):
            if time.time() > deadline:
                return U, W, False
            e = compute_errors()
            abs_e = np.abs(e)
            max_err_idx = int(np.argmax(abs_e))
            max_err = e[max_err_idx]
            if abs(max_err) < 1e-12:
                return U, W, True
            i, j, k = triples[max_err_idx]
            # Only update U and W
            dU = V[j * p + k, :] * W[k * n + i, :]
            dW = U[i * m + j, :] * V[j * p + k, :]
            U[i * m + j, :] -= lr * max_err * dU
            W[k * n + i, :] -= lr * max_err * dW
        return U, W, False

    def _polish_with_fixed_W(U, V, W_fixed, max_iters=5000, lr=0.01):
        """Polish U and V while keeping W fixed. Returns (U, V, converged)."""
        for it in range(max_iters):
            if time.time() > deadline:
                return U, V, False
            e = compute_errors()
            abs_e = np.abs(e)
            max_err_idx = int(np.argmax(abs_e))
            max_err = e[max_err_idx]
            if abs(max_err) < 1e-12:
                return U, V, True
            i, j, k = triples[max_err_idx]
            # Only update U and V
            dU = V[j * p + k, :] * W[k * n + i, :]
            dV = U[i * m + j, :] * W[k * n + i, :]
            U[i * m + j, :] -= lr * max_err * dU
            V[j * p + k, :] -= lr * max_err * dV
        return U, V, False

    def _least_squares_polish(U, V, W, max_iters=200):
        """When close to solution, use least-squares to polish.
        For each triple, solve for the remaining factor entries."""
        for it in range(max_iters):
            if time.time() > deadline:
                break
            e = compute_errors()
            max_err = np.max(np.abs(e))
            if max_err < 1e-10:
                break
            
            # Find the worst triple
            max_err_idx = int(np.argmax(np.abs(e)))
            i, j, k = triples[max_err_idx]
            
            # Try to fix one factor and solve for others
            # Strategy: for each of the rank dimensions, try to adjust
            # We'll use a simple approach: adjust the factor with largest contribution
            
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            
            # Compute contributions from each rank component
            contrib_U = np.abs(V[v_idx, :] * W[w_idx, :])
            contrib_V = np.abs(U[u_idx, :] * W[w_idx, :])
            contrib_W = np.abs(U[u_idx, :] * V[v_idx, :])
            
            # Find which factor to adjust
            max_contrib_U = np.max(contrib_U)
            max_contrib_V = np.max(contrib_V)
            max_contrib_W = np.max(contrib_W)
            
            err = e[max_err_idx]
            
            # Adjust the factor with largest contribution
            if max_contrib_U >= max_contrib_V and max_contrib_U >= max_contrib_W:
                # Adjust U
                if max_contrib_U > 1e-15:
                    # Find which rank component to adjust
                    r_idx = int(np.argmax(contrib_U))
                    U[u_idx, r_idx] -= err * (V[v_idx, r_idx] * W[w_idx, r_idx]) / max_contrib_U
            elif max_contrib_V >= max_contrib_U and max_contrib_V >= max_contrib_W:
                # Adjust V
                if max_contrib_V > 1e-15:
                    r_idx = int(np.argmax(contrib_V))
                    V[v_idx, r_idx] -= err * (U[u_idx, r_idx] * W[w_idx, r_idx]) / max_contrib_V
            else:
                # Adjust W
                if max_contrib_W > 1e-15:
                    r_idx = int(np.argmax(contrib_W))
                    W[w_idx, r_idx] -= err * (U[u_idx, r_idx] * V[v_idx, r_idx]) / max_contrib_W

    # Phase 1: Coarse optimization with smooth surrogate guidance
    lr = 0.05
    prev_loss = float('inf')
    stagnation_count = 0
    for it in range(10000):
        if time.time() > deadline:
            break
        e = compute_errors()
        loss, abs_e = compute_smooth_loss(e, temperature=1.0)
        
        if loss < 1e-12:
            break
        
        # Track progress
        if loss < prev_loss * 0.999:
            lr = min(lr * 1.02, 0.1)
            stagnation_count = 0
        elif loss > prev_loss * 1.001:
            lr = max(lr * 0.98, 0.01)
            stagnation_count += 1
        else:
            stagnation_count += 1
        prev_loss = loss
        
        # If stuck, try fixing one factor
        if stagnation_count > 100:
            # Try polishing with fixed U
            V, W, conv = _polish_with_fixed_U(U, V, W, max_iters=200, lr=0.02)
            if conv:
                break
            stagnation_count = 0
            lr = max(lr * 0.5, 0.01)
        
        max_err_idx = int(np.argmax(abs_e))
        max_err = e[max_err_idx]
        
        if abs(max_err) < 1e-9:
            break
        
        i, j, k = triples[max_err_idx]
        
        # Gradients
        dU = V[j * p + k, :] * W[k * n + i, :]
        dV = U[i * m + j, :] * W[k * n + i, :]
        dW = U[i * m + j, :] * V[j * p + k, :]
        
        # Update
        U[i * m + j, :] -= lr * max_err * dU
        V[j * p + k, :] -= lr * max_err * dV
        W[k * n + i, :] -= lr * max_err * dW

    # Phase 2: Fine polish with tighter tolerance
    lr = 0.01
    prev_loss2 = float('inf')
    stagnation_count2 = 0
    for it in range(8000):
        if time.time() > deadline:
            break
        e = compute_errors()
        loss2, abs_e2 = compute_smooth_loss(e, temperature=0.5)
        
        if loss2 < 1e-14:
            break
        
        if loss2 < prev_loss2 * 0.999:
            lr = min(lr * 1.01, 0.05)
            stagnation_count2 = 0
        elif loss2 > prev_loss2 * 1.001:
            lr = max(lr * 0.99, 0.001)
            stagnation_count2 += 1
        else:
            stagnation_count2 += 1
        prev_loss2 = loss2
        
        # If stuck, try alternating fixed-factor polishing
        if stagnation_count2 > 50:
            # Alternate between fixing U, V, W
            if it % 3 == 0:
                V, W, conv = _polish_with_fixed_U(U, V, W, max_iters=100, lr=0.01)
            elif it % 3 == 1:
                U, W, conv = _polish_with_fixed_V(U, V, W, max_iters=100, lr=0.01)
            else:
                U, V, conv = _polish_with_fixed_W(U, V, W, max_iters=100, lr=0.01)
            if conv:
                break
            stagnation_count2 = 0
        
        max_err_idx = int(np.argmax(abs_e2))
        max_err = e[max_err_idx]
        if abs(max_err) < 1e-12:
            break
        i, j, k = triples[max_err_idx]
        dU = V[j * p + k, :] * W[k * n + i, :]
        dV = U[i * m + j, :] * W[k * n + i, :]
        dW = U[i * m + j, :] * V[j * p + k, :]
        U[i * m + j, :] -= lr * max_err * dU
        V[j * p + k, :] -= lr * max_err * dV
        W[k * n + i, :] -= lr * max_err * dW

    # Phase 3: High-precision final polish
    _least_squares_polish(U, V, W, max_iters=500)
    
    # Final gradient polish
    lr = 0.001
    for it in range(2000):
        if time.time() > deadline:
            break
        e = compute_errors()
        max_err_idx = int(np.argmax(np.abs(e)))
        max_err = e[max_err_idx]
        if abs(max_err) < 1e-14:
            break
        i, j, k = triples[max_err_idx]
        dU = V[j * p + k, :] * W[k * n + i, :]
        dV = U[i * m + j, :] * W[k * n + i, :]
        dW = U[i * m + j, :] * V[j * p + k, :]
        U[i * m + j, :] -= lr * max_err * dU
        V[j * p + k, :] -= lr * max_err * dV
        W[k * n + i, :] -= lr * max_err * dW

    if not _verify_full(U, V, W, n, m, p):
        return None
    snapped = _snap_to_halfint(U.copy(), V.copy(), W.copy())
    if snapped is None:
        return None
    U, V, W = snapped
    if not _verify_full(U, V, W, n, m, p):
        return None
    if not _verify_random(U, V, W, n, m, p):
        return None
    return (U, V, W)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    n, m, p = DIMENSIONS

    # Guaranteed baseline: block-Strassen rank 84
    U, V, W = _build_strassen()
    best = (U.copy(), V.copy(), W.copy())
    best_rank = 84

    # Try to find lower ranks via optimization
    # We try a range of ranks. We'll try descending ranks.
    # Start from lower ranks first to find the best possible
    for target_rank in range(60, 84):
        if time.time() > deadline:
            break
        # Try multiple seeds for each rank to increase chance of success
        # Use different seed ranges for variety
        seeds_to_try = list(range(target_rank, target_rank + 8))
        # Also try some higher seeds for diversity
        seeds_to_try.extend([100 + s for s in range(5)])
        
        for seed in seeds_to_try:
             if time.time() > deadline:
                 break
             res = _try_optimize(best[0], best[1], best[2], target_rank, n, m, p, deadline, seed)
             if res is not None:
                 best = res
                 best_rank = target_rank
                 break # Move to next lower rank
        
    U, V, W = best

    # Convert to list-of-lists as required by interface
    # Ensure values are exactly representable (half-integers)
    U_list = []
    for row in U:
        U_row = []
        for x in row:
            # Snap to nearest half-integer for exact representation
            h = round(2 * x) / 2.0
            U_row.append(float(h))
        U_list.append(U_row)
    
    V_list = []
    for row in V:
        V_row = []
        for x in row:
            h = round(2 * x) / 2.0
            V_row.append(float(h))
        V_list.append(V_row)
    
    W_list = []
    for row in W:
        W_row = []
        for x in row:
            h = round(2 * x) / 2.0
            W_row.append(float(h))
        W_list.append(W_row)

    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
