import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    # Build target tensor
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28

    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    schoolbook_rank = n * m * p  # 112

    # Strategy: Structural decomposition
    # 1. Use Strassen (rank 49) for the 4x4x4 sub-problem
    # 2. Use optimized decomposition for the 4x4x3 sub-problem
    # 3. Try to improve the 4x4x3 part using greedy peeling

    # Step 1: Build 4x4x4 Strassen decomposition
    U_2, V_2, W_2 = _strassen_2x2x2()

    # Kronecker product for 4x4x4: rank 49
    U_444 = np.zeros((16, 49))
    V_444_std = np.zeros((16, 49))
    W_444_std = np.zeros((16, 49))

    for r1 in range(7):
        for r2 in range(7):
            idx = r1 * 7 + r2
            U_444[:, idx] = np.kron(U_2[:, r1], U_2[:, r2])
            V_444_std[:, idx] = np.kron(V_2[:, r1], V_2[:, r2])
            W_444_std[:, idx] = np.kron(W_2[:, r1], W_2[:, r2])

    # Embed into full tensor dimensions
    # V index in full: j*7+k, in standard 4x4x4: j*4+k
    # W index in full: k*4+i, in standard 4x4x4: k*4+i (same)
    V_444_full = np.zeros((28, 49))
    for j in range(4):
        for k in range(4):
            V_444_full[j*7+k, :] = V_444_std[j*4+k, :]

    W_444_full = np.zeros((28, 49))
    for k in range(4):
        for i in range(4):
            W_444_full[k*4+i, :] = W_444_std[k*4+i, :]

    # Verify 4x4x4 part
    recon_444 = np.einsum('ak,bk,ck->abc', U_444, V_444_full, W_444_full)
    T_444_part = np.zeros((16, 28, 28), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i * 4 + j
                b = j * 7 + k
                c = k * 4 + i
                T_444_part[a, b, c] = 1

    if not np.allclose(recon_444, T_444_part, atol=1e-10):
        # Fallback to schoolbook
        return _schoolbook(n, m, p)

    # Step 2: Handle the right part (k=4,5,6) - 4x4x3 sub-problem
    # Schoolbook for this part: rank 4*4*3 = 48
    U_right_school = np.zeros((16, 48))
    V_right_school = np.zeros((28, 48))
    W_right_school = np.zeros((28, 48))

    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U_right_school[i*4+j, col] = 1
                V_right_school[j*7+k, col] = 1
                W_right_school[k*4+i, col] = 1
                col += 1

    # Combine: total rank = 49 + 48 = 97
    U_total = np.hstack([U_444, U_right_school])
    V_total = np.hstack([V_444_full, V_right_school])
    W_total = np.hstack([W_444_full, W_right_school])

    recon_full = np.einsum('ak,bk,ck->abc', U_total, V_total, W_total)
    if not np.allclose(recon_full, T, atol=1e-10):
        return _schoolbook(n, m, p)

    best_rank = 97
    best_result = {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }

    # Step 3: Try to improve the 4x4x3 part using greedy peeling
    # The 4x4x3 tensor has rank at most 48 (schoolbook)
    # Can we do better? Let's try.
    if time.time() < deadline - 10.0:
        T_right = np.zeros((16, 28, 28), dtype=np.float64)
        for i in range(4):
            for j in range(4):
                for k in range(4, 7):
                    a = i * 4 + j
                    b = j * 7 + k
                    c = k * 4 + i
                    T_right[a, b, c] = 1

        # Try greedy peeling with different strategies
        right_result = _greedy_peel_structured(T_right, 16, 28, 28, min(deadline - 5.0, time.time() + 80.0), 48)
        if right_result is not None:
            r_right = len(right_result["U"][0])
            U_r = np.array(right_result["U"]).T
            V_r = np.array(right_result["V"]).T
            W_r = np.array(right_result["W"]).T

            U_combined = np.hstack([U_444, U_r])
            V_combined = np.hstack([V_444_full, V_r])
            W_combined = np.hstack([W_444_full, W_r])

            recon = np.einsum('ak,bk,ck->abc', U_combined, V_combined, W_combined)
            if np.allclose(recon, T, atol=1e-10):
                total_rank = 49 + r_right
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_result = {
                        "U": U_combined.tolist(),
                        "V": V_combined.tolist(),
                        "W": W_combined.tolist()
                    }

    # Step 4: Try alternative structural decomposition
    # Split B into B_left (4x4) and B_right (4x3)
    # Use a different approach: treat the 4x4x7 as a sum of 7 4x4x1 tensors
    # Each 4x4x1 has rank 16 (schoolbook), total 7*16=112
    # But we can potentially share structure between columns

    # Try a more aggressive greedy peeling on the full tensor
    if time.time() < deadline - 5.0 and best_rank >= 97:
        full_result = _greedy_peel_full(T, dim_u, dim_v, dim_w, deadline - 2.0, best_rank)
        if full_result is not None:
            r_full = len(full_result["U"][0])
            if r_full < best_rank:
                best_rank = r_full
                best_result = full_result

    if best_rank < schoolbook_rank:
        return best_result

    return _schoolbook(n, m, p)


def _greedy_peel_structured(T, dim_u, dim_v, dim_w, deadline, max_terms):
    """Greedy rank-1 peeling optimized for the 4x4x3 sub-problem."""
    R = T.copy()
    U_list = []
    V_list = []
    W_list = []

    for term_idx in range(max_terms):
        if time.time() > deadline:
            break
        if not np.any(R):
            break

        best_score = 0
        best_u, best_v, best_w = None, None, None

        # Multiple restarts with different initializations
        n_restarts = 12
        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            if restart == 0:
                # Sign-based initialization
                cu = np.sign(np.sum(R, axis=(1, 2)))
                cv = np.sign(np.sum(R, axis=(0, 2)))
                cw = np.sign(np.sum(R, axis=(0, 1)))
            elif restart == 1:
                # Random +/-1
                cu = np.random.choice([-1, 1], size=dim_u).astype(np.float64)
                cv = np.random.choice([-1, 1], size=dim_v).astype(np.float64)
                cw = np.random.choice([-1, 1], size=dim_w).astype(np.float64)
            elif restart == 2:
                # Random with zeros
                cu = np.random.choice([-1, 0, 1], size=dim_u).astype(np.float64)
                cv = np.random.choice([-1, 0, 1], size=dim_v).astype(np.float64)
                cw = np.random.choice([-1, 0, 1], size=dim_w).astype(np.float64)
            else:
                # Random Gaussian
                cu = np.random.randn(dim_u)
                cv = np.random.randn(dim_v)
                cw = np.random.randn(dim_w)

            # Alternating minimization with more iterations
            n_iters = 20
            for it in range(n_iters):
                if time.time() > deadline:
                    break

                M_u = np.einsum('abc,b,c->a', R, cv, cw)
                new_u = np.sign(M_u)

                M_v = np.einsum('abc,a,c->b', R, new_u, cw)
                new_v = np.sign(M_v)

                M_w = np.einsum('abc,a,b->c', R, new_u, new_v)
                new_w = np.sign(M_w)

                cu, cv, cw = new_u, new_v, new_w

            score = np.einsum('abc,a,b,c', R, cu, cv, cw)
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = cu.copy(), cv.copy(), cw.copy()

        if best_score <= 0 or best_u is None:
            break

        R -= np.einsum('a,b,c->abc', best_u, best_v, best_w)
        U_list.append(best_u)
        V_list.append(best_v)
        W_list.append(best_w)

        if np.allclose(R, 0, atol=1e-10):
            break

    if np.allclose(R, 0, atol=1e-10) and len(U_list) > 0:
        rank = len(U_list)
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T

        reconstructed = np.einsum('ak,bk,ck->abc', U, V, W)
        if np.allclose(reconstructed, T, atol=1e-10):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    return None


def _greedy_peel_full(T, dim_u, dim_v, dim_w, deadline, max_terms):
    """Greedy rank-1 peeling on the full tensor with more aggressive search."""
    R = T.copy()
    U_list = []
    V_list = []
    W_list = []

    for term_idx in range(max_terms):
        if time.time() > deadline:
            break
        if not np.any(R):
            break

        best_score = 0
        best_u, best_v, best_w = None, None, None

        # Fewer restarts for full tensor due to higher dimensionality
        n_restarts = 6
        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            if restart == 0:
                cu = np.sign(np.sum(R, axis=(1, 2)))
                cv = np.sign(np.sum(R, axis=(0, 2)))
                cw = np.sign(np.sum(R, axis=(0, 1)))
            elif restart == 1:
                cu = np.random.choice([-1, 1], size=dim_u).astype(np.float64)
                cv = np.random.choice([-1, 1], size=dim_v).astype(np.float64)
                cw = np.random.choice([-1, 1], size=dim_w).astype(np.float64)
            else:
                cu = np.random.randn(dim_u)
                cv = np.random.randn(dim_v)
                cw = np.random.randn(dim_w)

            n_iters = 15
            for it in range(n_iters):
                if time.time() > deadline:
                    break

                M_u = np.einsum('abc,b,c->a', R, cv, cw)
                new_u = np.sign(M_u)

                M_v = np.einsum('abc,a,c->b', R, new_u, cw)
                new_v = np.sign(M_v)

                M_w = np.einsum('abc,a,b->c', R, new_u, new_v)
                new_w = np.sign(M_w)

                cu, cv, cw = new_u, new_v, new_w

            score = np.einsum('abc,a,b,c', R, cu, cv, cw)
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = cu.copy(), cv.copy(), cw.copy()

        if best_score <= 0 or best_u is None:
            break

        R -= np.einsum('a,b,c->abc', best_u, best_v, best_w)
        U_list.append(best_u)
        V_list.append(best_v)
        W_list.append(best_w)

        if np.allclose(R, 0, atol=1e-10):
            break

    if np.allclose(R, 0, atol=1e-10) and len(U_list) > 0:
        rank = len(U_list)
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T

        reconstructed = np.einsum('ak,bk,ck->abc', U, V, W)
        if np.allclose(reconstructed, T, atol=1e-10):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    return None


def _strassen_2x2x2():
    """Return Strassen decomposition of 2x2x2 matrix multiplication.

    Returns U, V, W each of shape (4, 7) such that
    sum_k U[a,k]*V[b,k]*W[c,k] = T[a,b,c]
    where T[i*2+j, j*2+k, k*2+i] = 1
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1

    # M3 = a11(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1

    # M4 = a22(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1

    # M5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1

    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _schoolbook(n, m, p):
    """Return schoolbook decomposition of n x m times m x p."""
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

# EVOLVE-BLOCK-END
