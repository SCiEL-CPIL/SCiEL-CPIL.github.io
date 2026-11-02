import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 120.0 - 15.0  # 105s safety margin

    n, m, p = DIMENSIONS

    # Build the target tensor T with shape (n, m, m, p, p, n)
    # T[i, j, j2, k, k2, i2] = 1 if j==j2 and i==i2 and k==k2 else 0
    T = np.zeros((n, m, m, p, p, n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, k, i] = 1.0

    T_flat = T.flatten()

    def compute_residual(U, V, W):
        """Compute residual for given factor matrices (shapes (n*m, R), (m*p, R), (p*n, R))"""
        R = U.shape[1]
        U_r = U.reshape(R, n, m)  # (R, n, m)
        V_r = V.reshape(R, m, p)  # (R, m, p)
        W_r = W.reshape(R, p, n)  # (R, p, n)

        recon = np.zeros_like(T)
        for r in range(R):
            u = U_r[r]  # (n, m)
            v = V_r[r]  # (m, p)
            w = W_r[r]  # (p, n)
            # outer product of u and v gives (n, m, m, p)
            uv = np.einsum('ij,jk->ijjk', u, v)
            # multiply by w: w is (p, n), we need to contract k with the 4th index and i with the 6th
            # recon[i,j,j2,k,k2,i2] += uv[i,j,j2,k] * w[k2,i2]... wait
            # Actually: sum over r of u_r[i,j] * v_r[j,k] * w_r[k,i] = T[i,j,j,k,k,i]
            # So recon[i,j,j2,k,k2,i2] = sum_r U_r[i,j] * V_r[j,k] * W_r[k2,i2]
            # But we need j==j2, k==k2, i==i2
            # So recon[i,j,j2,k,k2,i2] = sum_r u_r[i,j] * v_r[j2,k2] * w_r[k,i]... no
            # Let me reconsider: T[i, j, j2, k, k2, i2] = 1 iff j==j2 AND k==k2 AND i==i2
            # The decomposition is: T = sum_r u_r ⊗ v_r ⊗ w_r where
            # u_r[i,j], v_r[j,k], w_r[k,i]
            # So T[i,j,j2,k,k2,i2] = sum_r u_r[i,j] * v_r[j2,k2] * w_r[k,i]
            # Wait, that doesn't look right either. Let me think again.
            
            # Standard matrix multiplication tensor:
            # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
            # Tensor: T_{i,j,k} = A_{i,j} * B_{j,k} ... no
            # The multiplication tensor has 3 modes: input A (n×m), input B (m×p), output (n×p)
            # T_{i,j | j',k | i',k'} = 1 if j==j' and k==k' and i==i'
            # Wait, the output is C_{i,k} = sum_j A_{i,j} B_{j,k}
            # So the tensor decomposes as: C_{i,k} = sum_{j} sum_{r} u_{r,i,j} v_{r,j,k} w_{r,i,k}
            # Hmm, that's not quite right. Let me use the standard formulation.
            
            # The matmul tensor T ∈ R^{(n×m) × (m×p) × (n×p)}
            # T[a, b, c] where a=(i,j), b=(j',k'), c=(i',k')
            # T[a,b,c] = 1 if j==j' AND k==k' AND i==i'... no wait
            # C_{i',k'} = sum_j A_{i',j} B_{j,k'}
            # So T[(i',j), (j,k'), (i',k')] = 1
            # In our flattened notation:
            # U has shape (n*m, R) indexed by (i,j)
            # V has shape (m*p, R) indexed by (j,k)  
            # W has shape (p*n, R) indexed by (k,i) -- note the task says W[k*n+i, r]
            
            # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[(i,j),(j,k),(k,i)]
            # And T[(i,j),(j,k),(k,i)] = 1 if this is a valid matmul term, i.e., always 1 for the specific indices
            # Wait no. The tensor is: T_{(i,j),(j',k'),(i',k')} = 1 if j==j' AND i==i' AND k==k'
            # No! It's: the matmul tensor T maps (A, B) -> C where C_{i,k} = sum_j A_{i,j} B_{j,k}
            # As a bilinear form: <T, A ⊗ B> = sum_{i,k} C_{i,k} * ... 
            # Actually the tensor T satisfies: C_{i,k} = sum_{j, r} T_{(i,j), (j,k), (i,k)} * A_{i,j} * B_{j,k}
            # No. The standard is: T is a trilinear form where
            # sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = sum_{i,k} C_{i,k}
            # with T_{i,j,k} = 1 (since C_{i,k} = sum_j A_{i,j} B_{j,k})
            # But that's rank n*m*p with each basis element.
            
            # For the factorization: C_{i,k} = sum_r sum_j U_{(i,j),r} B_{j,k} V_{(j,k),r} ... 
            # No. The correct factorization is:
            # C_{i,k} = sum_r (sum_j U_{(i,j),r} A_{i,j}) * (sum_{j'} V_{(j',k),r} B_{j',k}) * W_{(k,i),r}
            # Wait, that's not standard either.
            
            # Let me use the problem's convention directly. The task says:
            # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
            # W[k*n+i, r] is the entry for output position (i,k)
            # The decomposition is:
            # C_{i,k} = sum_{j, r} A_{i,j} * U[(i,j), r] * ... 
            # Actually I think the standard tensor decomposition for matmul is:
            # sum_r U_r ⊗ V_r ⊗ W_r = T
            # where (U_r)_{i,j}, (V_r)_{j,k}, (W_r)_{i,k} and T_{(i,j),(j,k),(i,k)} = 1
            # So: sum_r U_r[i,j] * V_r[j,k] * W_r[i,k] = T[(i,j),(j,k),(i,k)]
            # And T[(i,j),(j,k),(i,k)] = 1 if the indices form a valid matmul term.
            # But which indices form valid terms? For matmul C=AB, C_{i,k} = sum_j A_{i,j} B_{j,k}
            # The tensor T has T_{(i,j),(j',k'),(i',k')} = 1 if j==j' AND i==i' AND k==k'
            # No, that's not right. Let me just use: T[(i,j),(j,k),(i,k)] = 1 for all i,j,k
            # And T[(i,j),(j',k'),(i',k')] = 0 otherwise.
            
            # So in our 6D tensor T[i, j, j2, k, k2, i2]:
            # T[i, j, j2, k, k2, i2] = 1 if j==j2 AND k==k2 AND i==i2
            # Wait, that gives: sum_r U_r[i,j] * V_r[j2,k2] * W_r[k,i2]... 
            # I'm getting confused. Let me just use the einsum directly.
            
            # The residual is: for each (i,j,j2,k,k2,i2), 
            # recon[i,j,j2,k,k2,i2] = sum_r U_r[i,j] * V_r[j2,k2] * W_r[k,i2]... 
            # No. Let me look at the original code's approach.
            # The original code used:
            # uv = np.einsum('ij,jk->ijjk', u, v)  # (n, m, m, p)
            # recon += np.einsum('ijjk,ab->ijkab', uv, w)  # (n, m, m, p, p, n)
            # So recon[i,j,j2,k,k2,i2] += uv[i,j,j2,k] * w[k2,i2]
            # And uv[i,j,j2,k] = u[i,j] * v[j2,k]
            # So recon[i,j,j2,k,k2,i2] = sum_r u_r[i,j] * v_r[j2,k] * w_r[k2,i2]
            # And T[i,j,j2,k,k2,i2] = 1 if j==j2 AND k==k2 AND i==i2
            
            # Hmm, but that means the W factor is indexed by (k2, i2) = (k', i')
            # And the task says W[k*n+i, r], so W index is (k, i)
            # So w_r[k2, i2] = W_r[k2*n + i2]
            
            # OK so the original code's convention is:
            # U_r[i,j], V_r[j,k], W_r[k,i] and the tensor is
            # T[i,j,j',k,k',i'] = sum_r U_r[i,j] * V_r[j',k] * W_r[k',i']
            # with T = 1 when j==j' AND k==k' AND i==i'
            
            # Wait, that's still weird. Let me just verify with schoolbook.
            # Schoolbook: for each (i,j,k), U[(i*m+j), idx] = 1, V[(j*p+k), idx] = 1, W[(k*n+i), idx] = 1
            # So U_r[i,j]=1 only when (i,j) is the specific pair, similarly for V and W.
            # Then recon[i,j,j2,k,k2,i2] = sum_r U_r[i,j]*V_r[j2,k]*W_r[k2,i2]
            # For the schoolbook term (i0,j0,k0): recon[i0,j0,j0,k0,k0,i0] = 1
            # And T[i0,j0,j0,k0,k0,i0] = 1. Good.
            
            # So the einsum in the original is correct. Let me keep it.
            recon += np.einsum('ijjk,ab->ijkab', uv, w)

        return recon.flatten() - T_flat

    def verify_solution(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact"""
        R = U.shape[1]
        residual = compute_residual(U, V, W)
        return np.max(np.abs(residual)) < tol

    def snap_to_half_integers(arr, tol=1e-4):
        """Try to snap values to half-integers"""
        snapped = np.round(arr * 2) / 2
        # Only return if close enough
        if np.max(np.abs(snapped - arr)) < tol:
            return snapped
        return None

    def get_schoolbook_solution():
        """Return the standard schoolbook decomposition"""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

        return U, V, W

    def try_optimize(R, n_restarts=2, warm_start=None):
        """Try to find a solution of rank R"""
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W

        best_result = None
        best_cost = np.inf

        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            if warm_start is not None and restart == 0:
                x0 = warm_start
            else:
                x0 = np.random.randn(total_size) * 0.5

            try:
                def residual_fn(params):
                    U_flat = params[:size_U]
                    V_flat = params[size_U:size_U + size_V]
                    W_flat = params[size_U + size_V:]

                    U = U_flat.reshape((n * m, R))
                    V = V_flat.reshape((m * p, R))
                    W = W_flat.reshape((p * n, R))

                    return compute_residual(U, V, W)

                res = least_squares(
                    residual_fn, x0,
                    method='lm', max_nfev=300,
                    xtol=1e-10, ftol=1e-10, gtol=1e-10
                )

                if res.cost < best_cost:
                    best_cost = res.cost
                    best_result = res

                    if best_cost < 1e-14:
                        break

            except Exception:
                continue

        if best_result is None:
            return None

        if best_result.cost < 1e-12:
            size_U = n * m * R
            size_V = m * p * R

            U_flat = best_result.x[:size_U]
            V_flat = best_result.x[size_U:size_U + size_V]
            W_flat = best_result.x[size_U + size_V:]

            U = U_flat.reshape((n * m, R))
            V = V_flat.reshape((m * p, R))
            W = W_flat.reshape((p * n, R))

            # Try snapping to half-integers
            U_snap = snap_to_half_integers(U)
            V_snap = snap_to_half_integers(V)
            W_snap = snap_to_half_integers(W)

            if U_snap is not None and V_snap is not None and W_snap is not None:
                if verify_solution(U_snap, V_snap, W_snap):
                    return U_snap, V_snap, W_snap

            if verify_solution(U, V, W):
                return U, V, W

        return None

    # Find the minimum rank
    best_solution = None

    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: at least max(n*p, m*p, n*m) = max(21, 28, 12) = 28
    # But known results suggest we can do much better
    # For 3x4x7, try from 55 up
    lower_bound = 55
    upper_bound = n * m * p  # 84

    for R in range(lower_bound, upper_bound + 1):
        if time.time() > deadline:
            break

        result = try_optimize(R, n_restarts=2)

        if result is not None:
            U, V, W = result
            best_solution = (U, V, W)
            break

    # If no better solution found, use schoolbook
    if best_solution is None:
        U, V, W = get_schoolbook_solution()
    else:
        U, V, W = best_solution

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
