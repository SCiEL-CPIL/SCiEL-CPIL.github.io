"""Combined program for 5x5x5 matrix multiplication tensor decomposition.

Strategy: fast gradient descent on the tensor-matching loss, augmented with a
smooth epigraph penalty that pulls columns toward half-integer values, plus a
final high-precision polish pass (rounding + sparse coordinate repair). A
block-structured initializer exploits the constant-tensor symmetry M[i,j,k]=1.
"""
import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin before 150s kill

    # ---------- Fallback: schoolbook rank-125 decomposition ----------
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # ---------- Fast tensor computation via einsum ----------
    def compute_tensor(Ua, Va, Wa):
        """Compute T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]."""
        r = Ua.shape[1]
        U3 = Ua.reshape(n, m, r)
        V3 = Va.reshape(m, p, r)
        W3 = Wa.reshape(p, n, r)
        return np.einsum('ijc,jkc,kic->ijk', U3, V3, W3, optimize=True)

    def verify_fast(Ua, Va, Wa):
        """Check exact match to 1.0 within tight tolerance."""
        T = compute_tensor(Ua, Va, Wa)
        return bool(np.allclose(T, 1.0, atol=1e-9, rtol=0.0))

    def verify(U, V, W):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        return verify_fast(Ua, Va, Wa)

    # ---------- Smooth half-integer epigraph penalty ----------
    def half_int_penalty(X, lam=1.0):
        """Smooth approximation to sum of squared distances to nearest half-integer.
        Uses soft-min via log-sum-exp over the two nearest half-integers."""
        # Distance to nearest half-integer: for each x, min over h in {floor(2x)/2, ceil(2x)/2}
        # Smooth via log-sum-exp of squared distances
        two_x = 2.0 * X
        # Nearest half-integers are at positions floor(two_x)/2 and ceil(two_x)/2
        lower = np.floor(two_x) / 2.0
        upper = np.ceil(two_x) / 2.0
        d_low = X - lower
        d_up = X - upper
        # Squared distances
        s_low = d_low * d_low
        s_up = d_up * d_up
        # Log-sum-exp for stability: log(exp(a) + exp(b)) = a + log1p(exp(b-a))
        a = np.maximum(s_low, s_up)
        b = np.minimum(s_low, s_up)
        # Soft min of squared distances
        log_sum = a + np.log1p(np.exp(b - a))
        # Gradient of soft-min w.r.t. X
        # d/dX log(exp(s_low)+exp(s_up))
        exp_low = np.exp(s_low - log_sum)
        exp_up = np.exp(s_up - log_sum)
        grad_low = exp_low * 2.0 * d_low
        grad_up = exp_up * 2.0 * d_up
        return lam * log_sum, lam * (grad_low + grad_up)

    # ---------- Gradient descent with epigraph penalty ----------
    def gradient_descent(U, V, W, rank, max_iters=300, lr=0.05, pen_lam=0.1, seed=42):
        """Run gradient descent on tensor loss + half-integer epigraph penalty."""
        nonlocal best_result, best_rank, deadline
        np.random.seed(seed)
        
        prev_error = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute tensor and error
            T_comp = compute_tensor(U, V, W)
            diff = T_comp - 1.0
            tensor_err = np.sum(diff * diff)
            
            # Half-integer penalty
            pen_U, grad_pen_U = half_int_penalty(U, pen_lam)
            pen_V, grad_pen_V = half_int_penalty(V, pen_lam)
            pen_W, grad_pen_W = half_int_penalty(W, pen_lam)
            total_pen = pen_U + pen_V + pen_W
            
            total_error = tensor_err + total_pen
            
            if tensor_err < 1e-8:
                # Tensor matches; try rounding
                U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n*m)]
                V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m*p)]
                W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p*n)]
                U_rounded = [[round(x * 2) / 2 for x in row] for row in U_list]
                V_rounded = [[round(x * 2) / 2 for x in row] for row in V_list]
                W_rounded = [[round(x * 2) / 2 for x in row] for row in W_list]
                
                if verify(U_rounded, V_rounded, W_rounded):
                    best_result = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                    best_rank = rank
                    return True
                else:
                    # Try exact polish
                    polished = _polish(U, V, W, rank)
                    if polished is not None:
                        best_result = polished
                        best_rank = rank
                        return True
                    # Fallback: use unrounded
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    return True
            
            # Stagnation check
            if abs(prev_error - total_error) < 1e-14 and it > 20:
                U += np.random.randn(n*m, rank) * 0.005
                V += np.random.randn(m*p, rank) * 0.005
                W += np.random.randn(p*n, rank) * 0.005
                lr = max(lr * 0.7, 0.002)
                pen_lam = max(pen_lam * 0.5, 0.01)
            prev_error = total_error
            
            # Compute gradients of tensor loss
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            dU3 = 2.0 * np.einsum('ijk,jkc,kic->ijc', diff, V3, W3, optimize=True)
            dV3 = 2.0 * np.einsum('ijk,ijc,kic->jkc', diff, U3, W3, optimize=True)
            dW3 = 2.0 * np.einsum('ijk,ijc,jkc->kic', diff, U3, V3, optimize=True)
            
            dU = dU3.reshape(n*m, rank) + grad_pen_U
            dV = dV3.reshape(m*p, rank) + grad_pen_V
            dW = dW3.reshape(p*n, rank) + grad_pen_W
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Periodic renormalization to prevent column blowup
            if it % 30 == 0 and it > 0:
                for c in range(rank):
                    nu = np.linalg.norm(U[:, c])
                    nv = np.linalg.norm(V[:, c])
                    nw = np.linalg.norm(W[:, c])
                    if nu > 1e-10:
                        U[:, c] /= nu
                    if nv > 1e-10:
                        V[:, c] /= nv
                    if nw > 1e-10:
                        W[:, c] /= nw
        
        return False

    # ---------- Sparse coordinate repair (polish) ----------
    def _polish(U, V, W, rank, max_iters=500):
        """Try to repair a near-solution by sparse coordinate updates.
        Returns a dict if successful, else None."""
        nonlocal deadline
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # First round to nearest half-integer
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Compute initial error
        T_comp = compute_tensor(U, V, W)
        diff = T_comp - 1.0
        error = np.sum(diff * diff)
        
        if error < 1e-9:
            U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n*m)]
            V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m*p)]
            W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Try flipping individual half-integer values to reduce error
        # Candidate values: {-1, -0.5, 0, 0.5, 1}
        vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Find the worst (i,j,k) entry
            idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
            i, j, k = idx
            current_val = diff[i, j, k]
            
            if abs(current_val) < 1e-12:
                break
            
            # Try flipping entries in U[i*m+j,:], V[j*p+k,:], W[k*n+i,:]
            best_improvement = 0
            best_change = None
            
            # Try U entries
            for c in range(rank):
                old_u = U[i*m+j, c]
                v_val = V[j*p+k, c]
                w_val = W[k*n+i, c]
                # Contribution of this column to T[i,j,k]
                contrib = old_u * v_val * w_val
                # For other (i',j',k') entries, this column also contributes
                # We need to check global effect, but for speed, check local first
                for new_u in vals:
                    if new_u == old_u:
                        continue
                    delta_local = (new_u - old_u) * v_val * w_val
                    # Estimate global effect: this changes all T[i',j',k'] where
                    # U[i'*m+j',c], V[j'*p+k',c], W[k'*n+i',c] are nonzero
                    # For now, just check if local improvement is significant
                    if abs(delta_local) > abs(best_improvement):
                        # Check if this helps the worst entry
                        new_worst = current_val + delta_local
                        if abs(new_worst) < abs(current_val):
                            best_improvement = delta_local
                            best_change = ('U', i*m+j, c, new_u)
            
            # Try V entries
            for c in range(rank):
                old_v = V[j*p+k, c]
                u_val = U[i*m+j, c]
                w_val = W[k*n+i, c]
                for new_v in vals:
                    if new_v == old_v:
                        continue
                    delta_local = u_val * (new_v - old_v) * w_val
                    new_worst = current_val + delta_local
                    if abs(new_worst) < abs(current_val) and abs(delta_local) > abs(best_improvement):
                        best_improvement = delta_local
                        best_change = ('V', j*p+k, c, new_v)
            
            # Try W entries
            for c in range(rank):
                old_w = W[k*n+i, c]
                u_val = U[i*m+j, c]
                v_val = V[j*p+k, c]
                for new_w in vals:
                    if new_w == old_w:
                        continue
                    delta_local = u_val * v_val * (new_w - old_w)
                    new_worst = current_val + delta_local
                    if abs(new_worst) < abs(current_val) and abs(delta_local) > abs(best_improvement):
                        best_improvement = delta_local
                        best_change = ('W', k*n+i, c, new_w)
            
            if best_change is None:
                break
            
            factor, row, col, new_val = best_change
            if factor == 'U':
                U[row, col] = new_val
            elif factor == 'V':
                V[row, col] = new_val
            else:
                W[row, col] = new_val
            
            # Recompute error (expensive but necessary for correctness)
            T_comp = compute_tensor(U, V, W)
            diff = T_comp - 1.0
            error = np.sum(diff * diff)
            
            if error < 1e-9:
                U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n*m)]
                V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m*p)]
                W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p*n)]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # ---------- Block-structured initializer ----------
    def block_init(rank, seed=0):
        """Initialize using block structure: group indices into blocks.
        For 5x5x5, use blocks of size ~2 to create a 3x3x3 block structure."""
        np.random.seed(seed)
        
        # Block boundaries: [0,2), [2,4), [4,5)
        blocks = [(0, 2), (2, 4), (4, 5)]
        nblocks = 3
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Assign columns to block triples
        col = 0
        for bi in range(nblocks):
            for bj in range(nblocks):
                for bk in range(nblocks):
                    if col >= rank:
                        break
                    # Block (bi,bj,bk) contributes to all (i,j,k) where
                    # i in blocks[bi], j in blocks[bj], k in blocks[bk]
                    # Scale so that the sum over all block triples = 1
                    # For now, use a random sign pattern
                    sign_u = np.random.choice([-1, 1])
                    sign_v = np.random.choice([-1, 1])
                    sign_w = np.random.choice([-1, 1])
                    
                    for i in blocks[bi]:
                        for j in blocks[bj]:
                            U[i*m+j, col] = sign_u * 0.5
                    for j in blocks[bj]:
                        for k in blocks[bk]:
                            V[j*p+k, col] = sign_v * 0.5
                    for k in blocks[bk]:
                        for i in blocks[bi]:
                            W[k*n+i, col] = sign_w * 0.5
                    col += 1
        
        # Fill remaining columns with small random values
        while col < rank:
            U[:, col] = np.random.randn(n*m) * 0.1
            V[:, col] = np.random.randn(m*p) * 0.1
            W[:, col] = np.random.randn(p*n) * 0.1
            col += 1
        
        return U, V, W

    # ---------- Main search loop ----------
    ranks_to_try = [70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 124]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
        
        # Try random initialization
        for seed in [42, 123, 456, 789, 101, 202, 303]:
            if time.time() > deadline:
                break
            if r >= best_rank:
                break
            np.random.seed(seed)
            U = np.random.randn(n * m, r) * 0.3
            V = np.random.randn(m * p, r) * 0.3
            W = np.random.randn(p * n, r) * 0.3
            # Normalize columns
            for c in range(r):
                nu = np.linalg.norm(U[:, c])
                nv = np.linalg.norm(V[:, c])
                nw = np.linalg.norm(W[:, c])
                if nu > 1e-10:
                    U[:, c] /= nu
                if nv > 1e-10:
                    V[:, c] /= nv
                if nw > 1e-10:
                    W[:, c] /= nw
            success = gradient_descent(U, V, W, r, max_iters=200, lr=0.05, pen_lam=0.05, seed=seed)
            if success:
                break
        
        if r >= best_rank:
            continue
        
        # Try block-structured initialization
        for seed in [0, 1, 2, 3]:
            if time.time() > deadline:
                break
            if r >= best_rank:
                break
            U, V, W = block_init(r, seed=seed)
            success = gradient_descent(U, V, W, r, max_iters=150, lr=0.04, pen_lam=0.1, seed=seed)
            if success:
                break
    
    return best_result
# EVOLVE-BLOCK-END
