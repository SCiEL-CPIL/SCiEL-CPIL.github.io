import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _build_known():
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


def _validate_fast(U, V, W, n, m, p):
    """Fast validation using NumPy einsum. Returns True if exact."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    T = np.einsum('it,jt,kt->ijk', Ua, Va, Wa)
    return np.allclose(T, 1.0, atol=1e-9)


def _validate_exact(U, V, W, n, m, p):
    """Exact validation using integer arithmetic where possible."""
    Ua = np.asarray(U)
    Va = np.asarray(V)
    Wa = np.asarray(W)
    R = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(R):
                    s += Ua[i * m + j, t] * Va[j * p + k, t] * Wa[k * n + i, t]
                if s != 1:
                    return False
    return True


def _try_rank_reduction(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to reduce rank via optimization with linear recombination of columns.
    Returns (U, V, W) with target_rank if successful, else None.
    """
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    R0 = Ua.shape[1]
    
    if R0 <= target_rank:
        return U, V, W

    T_target = np.ones((n, m, p))
    
    # Strategy 1: Try linear combinations of top-scoring columns
    U_norms = np.linalg.norm(Ua, axis=0)
    V_norms = np.linalg.norm(Va, axis=0)
    W_norms = np.linalg.norm(Wa, axis=0)
    scores = U_norms * V_norms * W_norms
    top_idx = np.argsort(scores)[::-1][:min(R0, target_rank * 2)]
    
    U_top = Ua[:, top_idx]
    V_top = Va[:, top_idx]
    W_top = Wa[:, top_idx]
    
    # Try to find matrices A, B, C such that:
    # U_new = U_top @ A, V_new = V_top @ B, W_new = W_top @ C
    # where A, B, C are (R0' x target_rank) matrices
    # This allows linear recombination
    
    # Initialize with identity-like structure
    n_top = U_top.shape[1]
    A = np.zeros((n_top, target_rank))
    B = np.zeros((n_top, target_rank))
    C = np.zeros((n_top, target_rank))
    for i in range(min(n_top, target_rank)):
        A[i, i] = 1.0
        B[i, i] = 1.0
        C[i, i] = 1.0
    
    # Gradient descent on A, B, C
    lr = 0.05
    n_iter = 300
    best_loss = np.inf
    best_ABC = (A.copy(), B.copy(), C.copy())
    
    for it in range(n_iter):
        if time.time() > deadline - 15:
            break
        
        U_curr = U_top @ A
        V_curr = V_top @ B
        W_curr = W_top @ C
        
        T_curr = np.einsum('it,jt,kt->ijk', U_curr, V_curr, W_curr)
        err = T_curr - T_target
        loss = np.sum(err ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_ABC = (A.copy(), B.copy(), C.copy())
        
        if loss < 1e-12:
            break
        
        # Compute gradients
        # dL/dA = U_top^T @ (sum over t of dL/dU_curr[:,:,t] outer U_top[:,t])
        # More carefully:
        # T_curr[i,j,k] = sum_t U_curr[i,j,t] V_curr[j,k,t] W_curr[k,i,t]
        # dL/dU_curr[i,j,t] = sum_k V_curr[j,k,t] W_curr[k,i,t] err[i,j,k]
        # dL/dA[a,b] = sum_{i,j,t} dL/dU_curr[i,j,t] * dU_curr[i,j,t]/dA[a,b]
        # U_curr[i,j,t] = sum_a U_top[i,j,a] A[a,t]
        # dU_curr[i,j,t]/dA[a,b] = U_top[i,j,a] * delta_{t,b}
        # So dL/dA[a,b] = sum_{i,j} U_top[i,j,a] * sum_k V_curr[j,k,b] W_curr[k,i,b] err[i,j,k]
        
        # Precompute for each b:
        # S_b[i,j] = sum_k V_curr[j,k,b] W_curr[k,i,b] err[i,j,k]
        # Then dL/dA[a,b] = sum_{i,j} U_top[i,j,a] S_b[i,j]
        
        dA = np.zeros((n_top, target_rank))
        dB = np.zeros((n_top, target_rank))
        dC = np.zeros((n_top, target_rank))
        
        for b in range(target_rank):
            # S_b[i,j] = sum_k V_curr[j,k,b] W_curr[k,i,b] err[i,j,k]
            Sb = np.einsum('jk,ki,ijk->ij', V_curr[:, :, b], W_curr[:, :, b], err)
            # dA[:,b] = U_top^T @ Sb (flattened)
            U_top_flat = U_top.reshape(-1, n_top)
            Sb_flat = Sb.reshape(-1)
            dA[:, b] = U_top_flat.T @ Sb_flat
            
            # Similarly for B and C
            # dB[:,b] = V_top^T @ (sum over i,k: W_curr[k,i,b] err[i,j,k] * ... )
            # Let's compute it properly
            # dV_curr[j,k,t] = sum_i W_curr[k,i,t] err[i,j,k]
            # dL/dB[a,b] = sum_{j,k} V_top[j,k,a] * sum_i W_curr[k,i,b] err[i,j,k]
            T_b = np.einsum('ki,ijk->jk', W_curr[:, :, b], err)
            V_top_flat = V_top.reshape(-1, n_top)
            T_b_flat = T_b.reshape(-1)
            dB[:, b] = V_top_flat.T @ T_b_flat
            
            # dW_curr[k,i,t] = sum_j V_curr[j,k,t] err[i,j,k]
            # dL/dC[a,b] = sum_{k,i} W_top[k,i,a] * sum_j V_curr[j,k,b] err[i,j,k]
            R_b = np.einsum('jk,ijk->ki', V_curr[:, :, b], err)
            W_top_flat = W_top.reshape(-1, n_top)
            R_b_flat = R_b.reshape(-1)
            dC[:, b] = W_top_flat.T @ R_b_flat
        
        A -= lr * dA
        B -= lr * dB
        C -= lr * dC
    
    A, B, C = best_ABC
    U_new = U_top @ A
    V_new = V_top @ B
    W_new = W_top @ C
    
    T_check = np.einsum('it,jt,kt->ijk', U_new, V_new, W_new)
    if np.allclose(T_check, T_target, atol=1e-8):
        def round_half(a):
            return np.round(2 * a) / 2.0
        Ur = round_half(U_new)
        Vr = round_half(V_new)
        Wr = round_half(W_new)
        T_final = np.einsum('it,jt,kt->ijk', Ur, Vr, Wr)
        if np.allclose(T_final, T_target, atol=1e-8):
            return Ur.tolist(), Vr.tolist(), Wr.tolist()
    
    return None


def _gradient_descent_reduce(U, V, W, n, m, p, target_rank, deadline):
    """
    Use gradient descent with proper gradient computation to find a target_rank solution.
    """
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    R0 = Ua.shape[1]
    
    # Initialize with top target_rank columns by norm
    U_norms = np.linalg.norm(Ua, axis=0)
    V_norms = np.linalg.norm(Va, axis=0)
    W_norms = np.linalg.norm(Wa, axis=0)
    scores = U_norms * V_norms * W_norms
    keep_idx = np.argsort(scores)[::-1][:target_rank]
    
    U_curr = Ua[:, keep_idx].copy()
    V_curr = Va[:, keep_idx].copy()
    W_curr = Wa[:, keep_idx].copy()
    
    T_target = np.ones((n, m, p))
    
    lr = 0.02
    n_iter = 500
    best_loss = np.inf
    best_state = (U_curr.copy(), V_curr.copy(), W_curr.copy())
    
    for it in range(n_iter):
        if time.time() > deadline - 10:
            break
        
        T_curr = np.einsum('it,jt,kt->ijk', U_curr, V_curr, W_curr)
        err = T_curr - T_target
        loss = np.sum(err ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_state = (U_curr.copy(), V_curr.copy(), W_curr.copy())
        
        if loss < 1e-14:
            break
        
        # Compute gradients using einsum for efficiency
        # dL/dU[i,j,t] = sum_k V_curr[j,k,t] W_curr[k,i,t] err[i,j,k]
        dU = np.einsum('jk,ki,ijk->ijt', V_curr, W_curr, err)
        dV = np.einsum('ki,ijk->jkt', W_curr, err)
        dW = np.einsum('jk,ijk->kit', V_curr, err)
        
        U_curr -= lr * dU
        V_curr -= lr * dV
        W_curr -= lr * dW
    
    U_curr, V_curr, W_curr = best_state
    
    T_final = np.einsum('it,jt,kt->ijk', U_curr, V_curr, W_curr)
    if np.allclose(T_final, T_target, atol=1e-8):
        def round_half(a):
            return np.round(2 * a) / 2.0
        Ur = round_half(U_curr)
        Vr = round_half(V_curr)
        Wr = round_half(W_curr)
        T_rounded = np.einsum('it,jt,kt->ijk', Ur, Vr, Wr)
        if np.allclose(T_rounded, T_target, atol=1e-8):
            return Ur.tolist(), Vr.tolist(), Wr.tolist()
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0  # 5s safety margin
    
    # Start from the known 66-rank solution
    known = _build_known()
    best = known
    
    # Validate the known solution
    if not _validate_fast(known["U"], known["V"], known["W"], n, m, p):
        return known
    
    # Try to reduce rank
    current_rank = len(known["U"][0])
    
    # First, check if the 66-rank solution has linear dependencies
    Ua = np.asarray(known["U"], dtype=float)
    Va = np.asarray(known["V"], dtype=float)
    Wa = np.asarray(known["W"], dtype=float)
    
    # For each factor, check rank
    rank_U = np.linalg.matrix_rank(Ua, tol=1e-10)
    rank_V = np.linalg.matrix_rank(Va, tol=1e-10)
    rank_W = np.linalg.matrix_rank(Wa, tol=1e-10)
    
    # If any factor has rank < 66, we might be able to reduce
    min_factor_rank = min(rank_U, rank_V, rank_W)
    
    # Try reducing from 66 down to min_factor_rank
    for target_rank in range(min(current_rank - 1, 65), min_factor_rank - 1, -1):
        if time.time() > deadline - 20:
            break
        
        # Try linear recombination approach
        result = _try_rank_reduction(
            known["U"], known["V"], known["W"],
            n, m, p, target_rank, deadline
        )
        if result is not None:
            U, V, W = result
            if _validate_fast(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                current_rank = target_rank
                continue
        
        # Try gradient descent (only for a few ranks to save time)
        if target_rank <= 62 and time.time() < deadline - 25:
            result = _gradient_descent_reduce(
                known["U"], known["V"], known["W"],
                n, m, p, target_rank, deadline
            )
            if result is not None:
                U, V, W = result
                if _validate_fast(U, V, W, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    current_rank = target_rank
    
    # Final polish: try to round the best solution more carefully
    if time.time() < deadline - 5:
        Ua = np.asarray(best["U"], dtype=float)
        Va = np.asarray(best["V"], dtype=float)
        Wa = np.asarray(best["W"], dtype=float)
        
        # Try multiple rounding strategies
        for scale in [1, 2, 4]:
            def round_scale(a, s=scale):
                return np.round(s * a) / float(s)
            
            Ur = round_scale(Ua)
            Vr = round_scale(Va)
            Wr = round_scale(Wa)
            
            T_check = np.einsum('it,jt,kt->ijk', Ur, Vr, Wr)
            if np.allclose(T_check, 1.0, atol=1e-9):
                best = {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
                break
    
    return best


if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
