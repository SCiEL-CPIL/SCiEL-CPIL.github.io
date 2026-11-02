"""Rewritten program for matrix multiplication tensor decomposition 3x3 x 3x4."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import numpy as np
    from fractions import Fraction

    t_start = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 3, 4
    dim_U = n * m  # 9
    dim_V = m * p  # 12
    dim_W = p * n  # 12
    rank_max = n * m * p  # 36

    def elapsed():
        return time.time() - t_start

    def deadline_exceeded():
        return elapsed() > DEADLINE

    # Build the target tensor T[u,v,w]
    # T[i*m+j, j*p+k, k*n+i] = 1, else 0
    def build_tensor():
        T = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    T[u, v, w] = 1.0
        return T

    T = build_tensor()

    def schoolbook():
        U = np.zeros((dim_U, rank_max), dtype=np.float64)
        V = np.zeros((dim_V, rank_max), dtype=np.float64)
        W = np.zeros((dim_W, rank_max), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, col] = 1.0
                    V[v, col] = 1.0
                    W[w, col] = 1.0
        return U, V, W

    def verify_exact(U, V, W, r):
        """Check if U,V,W of rank r exactly decompose the tensor.
        U: (dim_U, r), V: (dim_V, r), W: (dim_W, r)
        sum_c U[u,c]*V[v,c]*W[w,c] should equal T[u,v,w]
        """
        # Use numpy for speed: compute sum over c
        # For each (u,v), compute sum_c U[u,c]*V[v,c] -> (dim_U, dim_V) matrix M
        # Then for each w, check M[:, w] ... no, it's a 3-way contraction.
        # Better: for each c, outer product U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # Sum over c and compare to T.
        
        # Efficient approach: for each c, compute A_c = U[:,c] @ V[:,c].T  (dim_U x dim_V)
        # Then the contribution to T from term c is A_c ⊗ W[:,c] reshaped
        # Actually, let's just do it directly but with numpy
        result = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
        for c in range(r):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            # uc ⊗ vc ⊗ wc
            # First: uc ⊗ vc = outer(uc, vc) shape (dim_U, dim_V)
            # Then multiply each element by wc
            # result += np.einsum('u,v,w->uvw', uc, vc, wc)
            # But einsum might be slow for small dims. Let's do:
            # temp = np.outer(uc, vc)  # (dim_U, dim_V)
            # result += temp[:, :, None] * wc[None, None, :]
            temp = np.outer(uc, vc)
            result += temp[:, :, None] * wc[None, None, :]
        
        # Check exact match
        if not np.allclose(result, T, atol=1e-10):
            return False
        # For exactness, check with fractions if needed, but for integer/half-integer entries
        # and small values, float64 should be exact for values up to ~10^6
        # Let's do a stricter check
        max_diff = np.max(np.abs(result - T))
        return max_diff < 1e-9

    def verify_exact_fraction(U, V, W, r):
        """Exact verification using fractions for small integer/half-integer entries."""
        for u in range(dim_U):
            for v in range(dim_V):
                for w in range(dim_W):
                    s = Fraction(0)
                    for c in range(r):
                        # Convert to Fraction
                        fu = Fraction(U[u, c]).limit_denominator(1000000)
                        fv = Fraction(V[v, c]).limit_denominator(1000000)
                        fw = Fraction(W[w, c]).limit_denominator(1000000)
                        s += fu * fv * fw
                    expected = 1 if T[u, v, w] > 0.5 else 0
                    if s != expected:
                        return False
        return True

    def to_half_integer_array(arr):
        """Convert float array to half-integer values (multiply by 2, round, divide by 2)."""
        scaled = np.round(arr * 2.0) / 2.0
        return scaled

    def loss_and_grad(U, V, W):
        """Compute loss ||sum_c u_c ⊗ v_c ⊗ w_c - T||^2 and gradients.
        U: (dim_U, r), V: (dim_V, r), W: (dim_W, r)
        """
        r = U.shape[1]
        # Compute the reconstructed tensor
        R = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
        for c in range(r):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            temp = np.outer(uc, vc)
            R += temp[:, :, None] * wc[None, None, :]
        
        diff = R - T
        loss = np.sum(diff * diff)
        
        # Gradients
        # dL/dU[u,c] = 2 * sum_{v,w} diff[u,v,w] * V[v,c] * W[w,c]
        # dL/dV[v,c] = 2 * sum_{u,w} diff[u,v,w] * U[u,c] * W[w,c]
        # dL/dW[w,c] = 2 * sum_{u,v} diff[u,v,w] * U[u,c] * V[v,c]
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for c in range(r):
            # dU[:,c] = 2 * diff @ (V[:,c] ⊗ W[:,c]) ... 
            # sum_{v,w} diff[u,v,w] * V[v,c] * W[w,c]
            # = (diff * (V[:,c] ⊗ W[:,c])) summed over v,w for each u
            # Let's compute: for fixed c, B = V[:,c] * W[:,c] broadcast? No.
            # sum_v sum_w diff[u,v,w] V[v,c] W[w,c]
            # = sum_v V[v,c] (sum_w diff[u,v,w] W[w,c])
            # Let's do it in two steps
            # First: for each (u,v), sum_w diff[u,v,w] W[w,c] -> (dim_U, dim_V)
            # Then: for each u, sum_v V[v,c] * that -> (dim_U,)
            
            # Efficient: diff @ (W[:,c]) along axis 2 gives (dim_U, dim_V)
            # diff shape: (dim_U, dim_V, dim_W)
            # W[:,c] shape: (dim_W,)
            # np.einsum('uvw,w->uv', diff, W[:,c]) -> (dim_U, dim_V)
            dv = np.einsum('uvw,w->uv', diff, W[:, c])
            dU[:, c] = 2.0 * np.einsum('uv,v->u', dv, V[:, c])
            
            # dV[v,c] = 2 * sum_{u,w} diff[u,v,w] U[u,c] W[w,c]
            du = np.einsum('uvw,w->uv', diff, W[:, c])  # (dim_U, dim_V), same as above
            dV[:, c] = 2.0 * np.einsum('uv,u->v', du, U[:, c])
            
            # dW[w,c] = 2 * sum_{u,v} diff[u,v,w] U[u,c] V[v,c]
            # = 2 * sum_w diff[u,v,w] * (sum_{u,v} U[u,c] V[v,c] for fixed w)
            # Let's compute: for each w, sum_{u,v} diff[u,v,w] U[u,c] V[v,c]
            # = sum_{u,v} diff[u,v,w] U[u,c] V[v,c]
            # Let M_c = U[:,c] ⊗ V[:,c] shape (dim_U, dim_V)
            # Then dW[w,c] = 2 * sum_{u,v} diff[u,v,w] M_c[u,v]
            # = 2 * (diff * M_c[None...]) summed over u,v for each w
            Mc = np.outer(U[:, c], V[:, c])  # (dim_U, dim_V)
            dW[:, c] = 2.0 * np.einsum('uvw,uv->w', diff, Mc)
        
        return loss, dU, dV, dW

    def gradient_descent(U, V, W, lr=0.01, max_iter=5000, tol=1e-12):
        """Run gradient descent to minimize loss."""
        r = U.shape[1]
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iter):
            if deadline_exceeded():
                break
            loss, dU, dV, dW = loss_and_grad(U, V, W)
            if loss < tol:
                break
            # Adaptive learning rate
            max_grad = max(np.max(np.abs(dU)), np.max(np.abs(dV)), np.max(np.abs(dW)), 1e-10)
            eff_lr = lr * 100.0 / max_grad
            U -= eff_lr * dU
            V -= eff_lr * dV
            W -= eff_lr * dW
            # Snap to half-integers periodically? No, that would break gradient flow.
            # Just do continuous optimization.
        
        return U, V, W, loss

    def try_rank(r, num_restarts=5, seed=42):
        """Try to find a rank-r decomposition."""
        if deadline_exceeded():
            return None
        
        rng = np.random.RandomState(seed)
        best_U, best_V, best_W = None, None, None
        best_loss = float('inf')
        
        for restart in range(num_restarts):
            if deadline_exceeded():
                break
            
            # Initialize: start from schoolbook and randomly perturb/merge
            # For rank r < 36, we need to merge some terms
            # Strategy: start with random initialization
            U = rng.randn(dim_U, r) * 0.5
            V = rng.randn(dim_V, r) * 0.5
            W = rng.randn(dim_W, r) * 0.5
            
            # Alternatively, start from schoolbook with random merging
            # Let's try both: random init and schoolbook-based init
            if restart == 0:
                # Schoolbook-based: take first r terms and randomly mix
                U0, V0, W0 = schoolbook()
                # Take a random subset and mix
                idx = rng.choice(rank_max, size=r, replace=False)
                U = U0[:, idx].copy()
                V = V0[:, idx].copy()
                W = W0[:, idx].copy()
                # Add small random perturbation
                U += rng.randn(dim_U, r) * 0.1
                V += rng.randn(dim_V, r) * 0.1
                W += rng.randn(dim_W, r) * 0.1
            
            # Run gradient descent
            U, V, W, loss = gradient_descent(U, V, W, lr=0.05, max_iter=3000, tol=1e-14)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Try snapping to half-integers and re-optimizing
            U2 = to_half_integer_array(U)
            V2 = to_half_integer_array(V)
            W2 = to_half_integer_array(W)
            if verify_exact(U2, V2, W2, r):
                return U2, V2, W2
            
            # Also try integer snapping
            U3 = np.round(U)
            V3 = np.round(V)
            W3 = np.round(W)
            if verify_exact(U3, V3, W3, r):
                return U3, V3, W3
        
        if best_U is not None and best_loss < 1e-8:
            # Try to snap
            U2 = to_half_integer_array(best_U)
            V2 = to_half_integer_array(best_V)
            W2 = to_half_integer_array(best_W)
            if verify_exact(U2, V2, W2, r):
                return U2, V2, W2
            # Try integer
            U3 = np.round(best_U)
            V3 = np.round(best_V)
            W3 = np.round(best_W)
            if verify_exact(U3, V3, W3, r):
                return U3, V3, W3
            # Try mixed: some integer, some half-integer
            # Just return the continuous solution snapped
            return U2, V2, W2
        
        return None

    # Start with schoolbook
    U_sb, V_sb, W_sb = schoolbook()
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_max
    
    # Try lower ranks
    for r in range(rank_max - 1, 15, -1):
        if deadline_exceeded():
            break
        result = try_rank(r, num_restarts=3, seed=r)
        if result is not None:
            U_r, V_r, W_r = result
            if verify_exact(U_r, V_r, W_r, r):
                best_result = (U_r, V_r, W_r)
                best_rank = r
                # Don't break, try even lower
            else:
                # The continuous solution didn't snap to exact, skip
                pass
    
    U_best, V_best, W_best = best_result
    
    # Convert to lists of lists for the interface
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    # Ensure entries are clean (remove tiny floating point errors)
    def clean_list(lst):
        cleaned = []
        for row in lst:
            clean_row = []
            for val in row:
                if abs(val) < 1e-10:
                    clean_row.append(0)
                else:
                    # Round to nearest half-integer
                    hv = round(val * 2) / 2
                    if abs(hv - val) < 1e-8:
                        clean_row.append(hv)
                    else:
                        clean_row.append(val)
            cleaned.append(clean_row)
        return cleaned
    
    U_list = clean_list(U_list)
    V_list = clean_list(V_list)
    W_list = clean_list(W_list)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
