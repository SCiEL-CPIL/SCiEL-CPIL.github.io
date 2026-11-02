"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    deadline = time.time() + 135

    # Build the matrix multiplication tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-8):
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def to_lists(U, V, W, r):
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    # Strassen 2x2x2 decomposition (rank 7)
    def strassen_2x2_decomp():
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        U[0,0]=1.0; U[3,0]=1.0; V[0,0]=1.0; V[3,0]=1.0; W[0,0]=1.0; W[3,0]=1.0
        U[2,1]=1.0; U[3,1]=1.0; V[0,1]=1.0; W[1,1]=1.0; W[3,1]=-1.0
        U[0,2]=1.0; V[1,2]=1.0; V[3,2]=-1.0; W[2,2]=1.0; W[3,2]=1.0
        U[3,3]=1.0; V[2,3]=1.0; V[0,3]=-1.0; W[0,3]=1.0; W[1,3]=1.0
        U[0,4]=1.0; U[1,4]=1.0; V[3,4]=1.0; W[0,4]=-1.0; W[2,4]=1.0
        U[2,5]=1.0; U[0,5]=-1.0; V[0,5]=1.0; V[1,5]=1.0; W[3,5]=1.0
        U[1,6]=1.0; U[3,6]=-1.0; V[2,6]=1.0; V[3,6]=1.0; W[0,6]=1.0
        return U, V, W

    # Build rank-28 Strassen block decomposition
    def build_strassen28():
        U_s, V_s, W_s = strassen_2x2_decomp()
        full_rank = 28
        U_full = np.zeros((n * m, full_rank))
        V_full = np.zeros((m * p, full_rank))
        W_full = np.zeros((p * n, full_rank))
        blocks = [
            (lambda li,lj: li*4+lj,          lambda lj,lk: lj*4+lk,          lambda lk,li: lk*2+li),
            (lambda li,lj: li*4+(lj+2),      lambda lj,lk: (lj+2)*4+lk,      lambda lk,li: lk*2+li),
            (lambda li,lj: li*4+lj,          lambda lj,lk: lj*4+(lk+2),      lambda lk,li: (lk+2)*2+li),
            (lambda li,lj: li*4+(lj+2),      lambda lj,lk: (lj+2)*4+(lk+2),  lambda lk,li: (lk+2)*2+li),
        ]
        for bi, (ga, gb, gc) in enumerate(blocks):
            for t in range(7):
                col = bi * 7 + t
                for li in range(2):
                    for lj in range(2):
                        U_full[ga(li, lj), col] = U_s[li * 2 + lj, t]
                for lj in range(2):
                    for lk in range(2):
                        V_full[gb(lj, lk), col] = V_s[lj * 2 + lk, t]
                for lk in range(2):
                    for li in range(2):
                        W_full[gc(lk, li), col] = W_s[lk * 2 + li, t]
        return U_full, V_full, W_full

    U28, V28, W28 = build_strassen28()

    # Fallback: schoolbook rank 32
    def build_schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r)); V = np.zeros((m * p, r)); W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, r

    if not verify(U28, V28, W28, tol=1e-8):
        U, V, W, r = build_schoolbook()
        return to_lists(U, V, W, r)

    # === Subgroup decomposition strategy ===
    # Model the 2x4x4 tensor as two independent 1x4x4 row-tensors.
    # Row i of A (length m=4) times B (4x4) gives row i of C (length p=4).
    # Each 1x4x4 sub-tensor: T_i[j, b, k] = 1 iff b = j*p+k
    #   where j in [0,4), b in [0,16), k in [0,4)
    # We search for a non-block-diagonal rank-r decomposition of the full tensor
    # that shares linear forms across the two rows, potentially beating 28.

    def build_row_subtensors():
        """Return T_0 and T_1, the two 1x4x4 sub-tensors."""
        T0 = np.zeros((m, m * p, p))  # 4 x 16 x 4
        T1 = np.zeros((m, m * p, p))  # 4 x 16 x 4
        for j in range(m):
            for k in range(p):
                b = j * p + k
                T0[j, b, k] = 1.0
                T1[j, b, k] = 1.0
        return T0, T1

    def try_subgroup_search(rank, n_iters, seed, tol=1e-10):
        """
        Search for a rank-`rank` decomposition of the full 2x4x4 tensor
        using alternating minimization with a smooth L2 objective.
        """
        if time.time() > deadline:
            return None
        rng = np.random.RandomState(seed)

        # Initialize: start from a perturbed version of the rank-28 solution
        # truncated to `rank` columns, or random if rank < 28
        if rank >= 28:
            U = U28[:, :rank].copy()
            V = V28[:, :rank].copy()
            W = W28[:, :rank].copy()
        else:
            # Random initialization scaled appropriately
            U = rng.randn(n * m, rank) * 0.5
            V = rng.randn(m * p, rank) * 0.5
            W = rng.randn(p * n, rank) * 0.5

        # Alternating minimization with adaptive step size
        err = float(np.sum((np.einsum('at,bt,ct->abc', U, V, W) - T) ** 2))
        best_err = err
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        for it in range(n_iters):
            if time.time() > deadline:
                break

            recon = np.einsum('at,bt,ct->abc', U, V, W)
            F = recon - T
            err = float(np.sum(F * F))

            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            if err < tol:
                break

            # Gradients
            gU = np.einsum('abc,bt,ct->at', F, V, W)
            gV = np.einsum('abc,at,ct->bt', F, U, W)
            gW = np.einsum('abc,at,bt->ct', F, U, V)

            # Adaptive learning rate
            norm_gU = float(np.sum(gU * gU)) ** 0.5 + 1e-12
            norm_gV = float(np.sum(gV * gV)) ** 0.5 + 1e-12
            norm_gW = float(np.sum(gW * gW)) ** 0.5 + 1e-12
            max_norm = max(norm_gU, norm_gV, norm_gW)
            lr = 0.5 / (max_norm + 1e-6)

            # Try a few step sizes
            for scale in [1.0, 0.5, 0.25, 0.1]:
                U_new = U - lr * scale * gU
                V_new = V - lr * scale * gV
                W_new = W - lr * scale * gW

                recon_new = np.einsum('at,bt,ct->abc', U_new, V_new, W_new)
                err_new = float(np.sum((recon_new - T) ** 2))

                if err_new < err:
                    U, V, W = U_new, V_new, W_new
                    err = err_new
                    break
            else:
                # No improvement; try smaller step
                lr *= 0.1
                U_new = U - lr * gU
                V_new = V - lr * gV
                W_new = W - lr * gW
                recon_new = np.einsum('at,bt,ct->abc', U_new, V_new, W_new)
                err_new = float(np.sum((recon_new - T) ** 2))
                if err_new < err:
                    U, V, W = U_new, V_new, W_new
                    err = err_new
                else:
                    break  # Stuck

        # Check best solution
        if best_err > 1e-6:
            return None

        # Snap to half-integers
        Us = np.round(best_U * 2) / 2
        Vs = np.round(best_V * 2) / 2
        Ws = np.round(best_W * 2) / 2
        if verify(Us, Vs, Ws, tol=1e-8):
            return Us, Vs, Ws, rank

        # Try integer snap
        Ui = np.round(best_U)
        Vi = np.round(best_V)
        Wi = np.round(best_W)
        if verify(Ui, Vi, Wi, tol=1e-8):
            return Ui, Vi, Wi, rank

        # Try quarter-integer snap
        Uq = np.round(best_U * 4) / 4
        Vq = np.round(best_V * 4) / 4
        Wq = np.round(best_W * 4) / 4
        if verify(Uq, Vq, Wq, tol=1e-8):
            return Uq, Vq, Wq, rank

        return None

    # Try to find a rank-27 decomposition using subgroup-aware search
    best_result = None
    best_rank = 28

    # First try rank 27 with multiple seeds
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        if time.time() > deadline - 10:
            break
        res = try_subgroup_search(27, 800, seed)
        if res is not None:
            Uf, Vf, Wf, r = res
            if r < best_rank:
                best_rank = r
                best_result = (Uf, Vf, Wf, r)
                break  # Found rank 27, stop searching for 27

    # If rank 27 found, try rank 26
    if best_rank == 27 and time.time() < deadline - 15:
        for seed in [0, 1, 2, 3, 4, 5]:
            if time.time() > deadline - 5:
                break
            res = try_subgroup_search(26, 600, seed)
            if res is not None:
                Uf, Vf, Wf, r = res
                if r < best_rank:
                    best_rank = r
                    best_result = (Uf, Vf, Wf, r)
                    break

    # If rank 26 found, try rank 25
    if best_rank == 26 and time.time() < deadline - 15:
        for seed in [0, 1, 2, 3, 4]:
            if time.time() > deadline - 5:
                break
            res = try_subgroup_search(25, 500, seed)
            if res is not None:
                Uf, Vf, Wf, r = res
                if r < best_rank:
                    best_rank = r
                    best_result = (Uf, Vf, Wf, r)
                    break

    # If we found something better, use it; otherwise fall back to rank 28
    if best_result is not None and best_rank < 28:
        Uf, Vf, Wf, r = best_result
        return to_lists(Uf, Vf, Wf, r)

    # Return the verified rank-28 Strassen decomposition
    return to_lists(U28, V28, W28, 28)
# EVOLVE-BLOCK-END
