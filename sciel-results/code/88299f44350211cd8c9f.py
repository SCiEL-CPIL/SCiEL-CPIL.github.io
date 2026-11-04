import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school

    # Precompute Target Tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # We want to find U, V, W such that T = sum_r U[:, r] V[:, r]^T W[:, r]^T (tensor product)
    # Equivalently, for each c, T[:, :, c] = sum_r U[:, r] V[:, r]^T W[c, r]
    # Let T_c be the matrix T[:, :, c] of shape (nm, mp).
    # T_c = U * W[c, :] @ V^T? No.
    # T_c[a, b] = sum_r U[a, r] V[b, r] W[c, r]
    
    T_slices = []
    for c in range(pn):
        T_c = np.zeros((nm, mp))
        k = c // n
        i = c % n
        for j in range(m):
            a = i * m + j
            b = j * p + k
            if 0 <= b < mp:
                T_c[a, b] = 1.0
        T_slices.append(T_c)
        
    T_slices = np.array(T_slices) # Shape (pn, nm, mp)

    start_time = time.time()
    deadline = start_time + 110.0

    # Greedy Rank-1 Projection Strategy
    # We build the decomposition term by term.
    # Residual R = T - sum_{r=1}^{r-1} u_r v_r^T w_r
    # At step r, we want to find u_r, v_r, w_r (with half-integer entries) that minimize ||R - u_r v_r^T w_r||_F^2.
    # This is a non-convex problem. However, we can fix w_r and solve for u_r, v_r? Or fix u_r, v_r and solve for w_r?
    # Actually, for a fixed w_r, the best u_r, v_r can be found via SVD of the "weighted" residual?
    # Let R be the residual tensor. We want to approximate R by a rank-1 tensor u v^T w.
    # This is equivalent to finding u, v, w that maximize <R, u v^T w> = sum_{a,b,c} R[a,b,c] u[a] v[b] w[c].
    # This is a cubic form maximization.
    # We can use a power method-like iteration:
    # 1. Initialize w randomly (or from residual).
    # 2. For fixed w, the problem reduces to finding u, v to maximize sum_{a,b} (sum_c R[a,b,c] w[c]) u[a] v[b].
    #    Let M_w[a,b] = sum_c R[a,b,c] w[c].
    #    The best rank-1 approximation of M_w is given by its top singular vectors.
    #    u, v = SVD(M_w)[:, 0] (normalized).
    # 3. For fixed u, v, the best w is proportional to M_uv where M_uv[c] = sum_{a,b} R[a,b,c] u[a] v[b].
    #    w = M_uv / ||M_uv||.
    # 4. Iterate 2-3 until convergence.
    # 5. Snap u, v, w to nearest half-integers.
    # 6. Update Residual R = R - u v^T w.
    
    # To ensure we find a solution, we can try multiple initializations for w.
    
    def snap_to_half(x):
        return np.round(x * 2) / 2.0

    def verify_decomp(U, V, W, T_slices):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Check if all entries are half-integers
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-9): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-9): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-9): return False
        
        # Check reconstruction
        # S = sum_r U[:, r] V[:, r]^T W[:, r]^T
        # S[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(S, T_slices, atol=1e-9)

    # Try ranks from 40 up to schoolbook?
    # Actually, let's try to build a low rank decomposition greedily.
    # We don't know the target rank R in advance. We can just keep adding terms until the residual is zero.
    # But we want to minimize the number of terms.
    # Greedy usually doesn't give the minimal rank, but it's a different strategy.
    # Let's try to find a decomposition of rank R by fixing R and trying to fit it?
    # No, the greedy approach builds it.
    
    # Let's try to find a decomposition of rank ~ 50-70.
    # We will run the greedy algorithm to build a list of (u, v, w) tuples.
    # We stop when the residual is effectively zero.
    
    # To improve the chance of finding a low-rank solution, we can randomize the initial w vector in the power method.
    
    def greedy_decompose(T_slices, max_terms=100, max_iter_per_term=50):
        R = T_slices.copy()
        U_list = []
        V_list = []
        W_list = []
        
        for term_idx in range(max_terms):
            if time.time() > deadline:
                break
            if np.linalg.norm(R) < 1e-9:
                break
                
            # Power method to find best rank-1 approximation
            # Initialize w
            # Random initialization from a few candidates
            best_u = None
            best_v = None
            best_w = None
            best_obj = -np.inf
            
            num_restarts = 10
            for restart in range(num_restarts):
                # Initialize w
                if restart == 0:
                    w = np.ones(pn) / np.sqrt(pn)
                else:
                    w = np.random.randn(pn)
                    w /= np.linalg.norm(w)
                
                for _ in range(max_iter_per_term):
                    # Compute M_w
                    # M_w[a, b] = sum_c R[a, b, c] w[c]
                    M_w = np.tensordot(R, w, axes=([2], [0]))
                    
                    # SVD of M_w
                    U_svd, s, Vt_svd = np.linalg.svd(M_w)
                    u = U_svd[:, 0]
                    v = Vt_svd[0, :]
                    
                    # Compute M_uv
                    # M_uv[c] = sum_{a, b} R[a, b, c] u[a] v[b]
                    M_uv = np.tensordot(R, np.outer(u, v), axes=([0, 1], [0, 1]))
                    w_new = M_uv / (np.linalg.norm(M_uv) + 1e-12)
                    
                    # Check convergence
                    if np.linalg.norm(w_new - w) < 1e-6:
                        break
                    w = w_new
                    
                # Evaluate objective
                obj = np.einsum('abc,a,b,c->', R, u, v, w)
                if obj > best_obj:
                    best_obj = obj
                    best_u = u
                    best_v = v
                    best_w = w
            
            if best_u is None:
                break
                
            # Snap to half-integers
            u_snap = snap_to_half(best_u)
            v_snap = snap_to_half(best_v)
            w_snap = snap_to_half(best_w)
            
            # Check if the snapped version is a valid improvement
            # We need to ensure that subtracting the snapped rank-1 tensor doesn't make the residual worse
            # or that it leads to an exact solution eventually.
            # For now, let's just subtract the snapped tensor if it's non-zero.
            # Actually, if we snap, the residual might not decrease monotonically.
            # Let's check if the snapped tensor is close to the optimal continuous one.
            if np.linalg.norm(u_snap) < 1e-6 or np.linalg.norm(v_snap) < 1e-6 or np.linalg.norm(w_snap) < 1e-6:
                # If snapping kills the vector, we might be stuck.
                # Try scaling? The problem allows any half-integers.
                # If the optimal u is small, snapping to 0 is bad.
                # We can try to scale the vector before snapping?
                # But the scale is fixed by the SVD normalization.
                # Maybe we should search for a scale factor alpha such that alpha * u is close to half-integers?
                # This is a hard problem.
                # Let's just skip this term if it snaps to zero, or try a different initialization.
                continue
            
            # Update Residual
            # R_new = R - u_snap v_snap^T w_snap
            rank1_tensor = np.einsum('a,b,c->abc', u_snap, v_snap, w_snap)
            R = R - rank1_tensor
            
            U_list.append(u_snap)
            V_list.append(v_snap)
            W_list.append(w_snap)
            
        if U_list:
            U = np.column_stack(U_list)
            V = np.column_stack(V_list)
            W = np.column_stack(W_list)
            return U, V, W
        return None

    # Try greedy decomposition
    try:
        U_g, V_g, W_g = greedy_decompose(T_slices)
        if U_g is not None and verify_decomp(U_g, V_g, W_g, T_slices):
            best_result = {"U": U_g.tolist(), "V": V_g.tolist(), "W": W_g.tolist()}
            best_rank = U_g.shape[1]
    except Exception:
        pass

    # If greedy failed or didn't improve, try the original random search for lower ranks
    # But we have a deadline.
    
    return best_result
