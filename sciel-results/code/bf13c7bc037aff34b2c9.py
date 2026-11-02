"""Rewritten program for 3x3 x 3x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    # Shapes: U is (n*m, r), V is (m*p, r), W is (p*n, r)
    # Tensor T[i*m+j, j'*p+k, k'*n+i] = 1 if i=i', j=j', k=k'
    # Actually the tensor for matrix mult A(n,m) * B(m,p) = C(n,p):
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T[a,b,c] where a = i*m+j (A index), b = j*p+k (B index), c = k'*n+i' (C index)
    # T[i*m+j, j*p+k, k*n+i] = 1 when the indices are consistent, i.e.,
    # A[i,j] * B[j,k] contributes to C[i,k]
    # So T[i*m+j, j*p+k, k*n+i] = 1 (each term is exactly one product)

    nm = n * m  # 9
    mp = m * p  # 18
    pn = p * n  # 18

    # Build target tensor
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W (numpy arrays) reconstruct T_target."""
        r = U.shape[1]
        T_recon = np.zeros((nm, mp, pn))
        for t in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:, t], V[:, t], W[:, t])
        return np.allclose(T_recon, T_target, atol=tol, rtol=0)

    def schoolbook_decomposition():
        rank = n * m * p  # 54
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    U_sb, V_sb, W_sb, sb_rank = schoolbook_decomposition()
    best_U, best_V, best_W, best_rank = U_sb.copy(), V_sb.copy(), W_sb.copy(), sb_rank

    def try_l_bfgs(target_rank, num_trials=4, max_iter=300, maxfun=1500):
        """Try to find a rank-target_rank decomposition using L-BFGS-B."""
        if target_rank >= best_rank:
            return None
        if time.time() > deadline:
            return None

        from scipy.optimize import minimize

        for trial in range(num_trials):
            if time.time() > deadline:
                return None

            np.random.seed(trial * 7919 + 42)
            # Initialize with small random values
            U_t = np.random.randn(nm, target_rank) * 0.3
            V_t = np.random.randn(mp, target_rank) * 0.3
            W_t = np.random.randn(pn, target_rank) * 0.3

            x0 = np.concatenate([U_t.flatten(), V_t.flatten(), W_t.flatten()])

            def objective(x):
                if time.time() > deadline:
                    return 1e10
                U_f = x[:nm * target_rank].reshape(nm, target_rank)
                V_f = x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                W_f = x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)

                # Compute T_recon efficiently using einsum
                # T_recon[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
                T_recon = np.einsum('at,bt,ct->abc', U_f, V_f, W_f)
                error = T_target - T_recon
                return np.sum(error ** 2)

            def gradient(x):
                if time.time() > deadline:
                    return np.zeros_like(x)
                U_f = x[:nm * target_rank].reshape(nm, target_rank)
                V_f = x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                W_f = x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)

                T_recon = np.einsum('at,bt,ct->abc', U_f, V_f, W_f)
                error = T_target - T_recon

                # dU[a,t] = -2 * sum_{b,c} error[a,b,c] * V[b,t] * W[c,t]
                dU = -2.0 * np.einsum('abc,bt,ct->at', error, V_f, W_f)
                dV = -2.0 * np.einsum('abc,at,ct->bt', error, U_f, W_f)
                dW = -2.0 * np.einsum('abc,at,bt->ct', error, U_f, V_f)

                return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])

            try:
                result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                                  options={'maxiter': max_iter, 'gtol': 1e-12, 'maxfun': maxfun,
                                           'ftol': 1e-14})
            except Exception:
                continue

            if time.time() > deadline:
                return None

            U_final = result.x[:nm * target_rank].reshape(nm, target_rank)
            V_final = result.x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
            W_final = result.x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)

            if check_decomposition(U_final, V_final, W_final, tol=1e-7):
                return U_final, V_final, W_final, target_rank

        return None

    # Try progressively lower ranks
    # Start from 53 down to some reasonable lower bound
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # But in practice, much higher. Let's try from 50 down.
    for target_rank in range(50, 17, -1):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue

        result = try_l_bfgs(target_rank, num_trials=3, max_iter=200, maxfun=1000)
        if result is not None:
            U_t, V_t, W_t, r = result
            best_U, best_V, best_W, best_rank = U_t, V_t, W_t, r
            # Continue trying to go lower
            continue
        else:
            # If we failed at this rank, it might be too low; break or continue
            # Let's continue trying lower ranks since the optimization is stochastic
            pass

    # If no better than schoolbook was found, use schoolbook
    if best_rank >= sb_rank:
        best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, sb_rank

    # Convert to lists with integer/half-integer entries if possible
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                frac = Fraction(val).limit_denominator(1000)
                if abs(float(frac) - val) < 1e-9:
                    if frac.denominator == 1:
                        row_list.append(frac.numerator)
                    elif frac.denominator == 2:
                        row_list.append(frac.numerator / 2.0)
                    else:
                        row_list.append(float(frac))
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
