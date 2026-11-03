"""Matrix multiplication 4x5x6: search for rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 135.0  # 15s safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def time_remaining():
        return DEADLINE - elapsed()

    n, m, p = DIMENSIONS
    N = n * m  # 20
    M = m * p  # 30
    P = p * n  # 24

    def schoolbook():
        rank = n * m * p
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def to_lists(mat):
        return mat.tolist()

    def verify(result, atol=1e-6):
        U = np.array(result["U"])
        V = np.array(result["V"])
        W = np.array(result["W"])
        if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
            return False
        if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
            return False
        if U.shape[0] != N or V.shape[0] != M or W.shape[0] != P:
            return False
        T = np.einsum('ar,br,cr->abc', U, V, W)
        target = np.zeros((N, M, P))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1.0
        return np.allclose(T, target, atol=atol)

    # Build target tensor once
    target = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    # Start with schoolbook
    best = schoolbook()
    best_rank = n * m * p

    def als_decompose(target_t, rank, max_iter=50, tol=1e-10, polish_iters=10, num_restarts=3):
        """ALS decomposition with better initialization, convergence check, and polish."""
        Nt, Mt, Pt = target_t.shape
        best_err = np.inf
        best_UVW = None

        for restart in range(num_restarts):
            U = np.random.randn(Nt, rank) * 0.5
            V = np.random.randn(Mt, rank) * 0.5
            W = np.random.randn(Pt, rank) * 0.5

            prev_err = np.inf
            for iteration in range(max_iter):
                if time_remaining() < 20:
                    break

                # Update U: fix V, W
                H_U = np.einsum('br,cr->bcr', V, W).reshape(Mt * Pt, rank)
                target_flat = target_t.reshape(Nt, Mt * Pt)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(H_U, target_flat.T, rcond=None)
                    U = U_new.T  # (Nt, rank)
                except:
                    pass

                # Update V: fix U, W
                H_V = np.einsum('ar,cr->acr', U, W).reshape(Nt * Pt, rank)
                target_v = target_t.transpose(1, 0, 2).reshape(Mt, Nt * Pt)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(H_V, target_v.T, rcond=None)
                    V = V_new.T  # (Mt, rank)
                except:
                    pass

                # Update W: fix U, V
                H_W = np.einsum('ar,br->abr', U, V).reshape(Nt * Mt, rank)
                target_w = target_t.transpose(2, 0, 1).reshape(Pt, Nt * Mt)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(H_W, target_w.T, rcond=None)
                    W = W_new.T  # (Pt, rank)
                except:
                    pass

                # Check convergence
                T = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T - target_t)
                if abs(prev_err - err) < tol:
                    break
                prev_err = err

            if err < best_err:
                best_err = err
                best_UVW = (U.copy(), V.copy(), W.copy())

        # Final high-precision polish pass
        if best_UVW is not None and time_remaining() > 25:
            U, V, W = best_UVW
            for _ in range(polish_iters):
                if time_remaining() < 20:
                    break
                H_U = np.einsum('br,cr->bcr', V, W).reshape(Mt * Pt, rank)
                target_flat = target_t.reshape(Nt, Mt * Pt)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(H_U, target_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                H_V = np.einsum('ar,cr->acr', U, W).reshape(Nt * Pt, rank)
                target_v = target_t.transpose(1, 0, 2).reshape(Mt, Nt * Pt)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(H_V, target_v.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                H_W = np.einsum('ar,br->abr', U, V).reshape(Nt * Mt, rank)
                target_w = target_t.transpose(2, 0, 1).reshape(Pt, Nt * Mt)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(H_W, target_w.T, rcond=None)
                    W = W_new.T
                except:
                    pass

        return best_UVW, best_err

    def try_rank(r, num_trials=5):
        """Try to find a decomposition of rank r."""
        if time_remaining() < 25:
            return None
        for trial in range(num_trials):
            if time_remaining() < 20:
                break
            UVW, err = als_decompose(target, r, max_iter=40, num_restarts=2)
            if UVW is not None:
                U, V, W = UVW
                result = {"U": U, "V": V, "W": W}
                if verify(result):
                    return result
        return None

    # Try ranks below schoolbook, starting from 119 down
    for r in range(best_rank - 1, max(best_rank - 25, 95), -1):
        if time_remaining() < 35:
            break
        result = try_rank(r, num_trials=3)
        if result is not None:
            best = result
            best_rank = r
            print(f"Found decomposition with rank {r}")
            continue

    # Block partitioning strategy: split middle index j into 3+2
    # A is 4x5, B is 5x6, C is 4x6
    # Split j: j in 0..2 (3 values) and j in 3..4 (2 values)
    # C = A[:,:3]*B[:,:3,:] + A[:,3:5]*B[:,3:5,:]
    # Block 1: 4x3x6 multiplication (A[:,:3] * B[:,:3,:]) -> use Laderman-like or ALS
    # Block 2: 4x2x6 multiplication (A[:,3:5] * B[:,3:5,:]) -> use schoolbook or ALS

    def build_subtensor_j0_2():
        """Build the sub-tensor for j=0,1,2."""
        sub = np.zeros((N, M, P))
        for i in range(4):
            for j in range(3):  # j = 0,1,2
                for k in range(6):
                    a = i * 5 + j
                    b = j * 6 + k
                    c = k * 4 + i
                    sub[a, b, c] = 1.0
        return sub

    def build_subtensor_j3_4():
        """Build the sub-tensor for j=3,4."""
        sub = np.zeros((N, M, P))
        for i in range(4):
            for j in range(3, 5):  # j = 3,4
                for k in range(6):
                    a = i * 5 + j
                    b = j * 6 + k
                    c = k * 4 + i
                    sub[a, b, c] = 1.0
        return sub

    # Decompose sub-tensor 1 (4x3x6 part) with reduced rank
    if time_remaining() > 40:
        sub1 = build_subtensor_j0_2()
        # Schoolbook rank for 4x3x6 is 72, try to find lower
        rank1_target = 50  # Try for rank 50
        U1, V1, W1 = None, None, None
        for trial in range(5):
            if time_remaining() < 30:
                break
            UVW, err = als_decompose(sub1, rank1_target, max_iter=60, num_restarts=3)
            if UVW is not None:
                U1, V1, W1 = UVW
                T1 = np.einsum('ar,br,cr->abc', U1, V1, W1)
                err1 = np.linalg.norm(T1 - sub1)
                if err1 < 1e-6:
                    break

        # Decompose sub-tensor 2 (4x2x6 part) with reduced rank
        sub2 = build_subtensor_j3_4()
        # Schoolbook rank for 4x2x6 is 48, try to find lower
        rank2_target = 35  # Try for rank 35
        U2, V2, W2 = None, None, None
        for trial in range(5):
            if time_remaining() < 30:
                break
            UVW, err = als_decompose(sub2, rank2_target, max_iter=60, num_restarts=3)
            if UVW is not None:
                U2, V2, W2 = UVW
                T2 = np.einsum('ar,br,cr->abc', U2, V2, W2)
                err2 = np.linalg.norm(T2 - sub2)
                if err2 < 1e-6:
                    break

        # Combine: U = [U1, U2], V = [V1, V2], W = [W1, W2]
        if U1 is not None and U2 is not None:
            U_combined = np.hstack([U1, U2])
            V_combined = np.hstack([V1, V2])
            W_combined = np.hstack([W1, W2])

            result_combined = {"U": U_combined, "V": V_combined, "W": W_combined}
            if verify(result_combined):
                rank_combined = U_combined.shape[1]
                if rank_combined < best_rank:
                    best = result_combined
                    best_rank = rank_combined
                    print(f"Block partitioning found rank {rank_combined}")

    # Try another partition: j into 2+3
    if time_remaining() > 40:
        def build_subtensor_j0_1():
            """Build the sub-tensor for j=0,1."""
            sub = np.zeros((N, M, P))
            for i in range(4):
                for j in range(2):  # j = 0,1
                    for k in range(6):
                        a = i * 5 + j
                        b = j * 6 + k
                        c = k * 4 + i
                        sub[a, b, c] = 1.0
            return sub

        def build_subtensor_j2_4():
            """Build the sub-tensor for j=2,3,4."""
            sub = np.zeros((N, M, P))
            for i in range(4):
                for j in range(2, 5):  # j = 2,3,4
                    for k in range(6):
                        a = i * 5 + j
                        b = j * 6 + k
                        c = k * 4 + i
                        sub[a, b, c] = 1.0
            return sub

        sub1 = build_subtensor_j0_1()
        rank1_target = 30  # Try for rank 30
        U1, V1, W1 = None, None, None
        for trial in range(5):
            if time_remaining() < 30:
                break
            UVW, err = als_decompose(sub1, rank1_target, max_iter=60, num_restarts=3)
            if UVW is not None:
                U1, V1, W1 = UVW
                T1 = np.einsum('ar,br,cr->abc', U1, V1, W1)
                err1 = np.linalg.norm(T1 - sub1)
                if err1 < 1e-6:
                    break

        sub2 = build_subtensor_j2_4()
        rank2_target = 50  # Try for rank 50
        U2, V2, W2 = None, None, None
        for trial in range(5):
            if time_remaining() < 30:
                break
            UVW, err = als_decompose(sub2, rank2_target, max_iter=60, num_restarts=3)
            if UVW is not None:
                U2, V2, W2 = UVW
                T2 = np.einsum('ar,br,cr->abc', U2, V2, W2)
                err2 = np.linalg.norm(T2 - sub2)
                if err2 < 1e-6:
                    break

        if U1 is not None and U2 is not None:
            U_combined = np.hstack([U1, U2])
            V_combined = np.hstack([V1, V2])
            W_combined = np.hstack([W1, W2])

            result_combined = {"U": U_combined, "V": V_combined, "W": W_combined}
            if verify(result_combined):
                rank_combined = U_combined.shape[1]
                if rank_combined < best_rank:
                    best = result_combined
                    best_rank = rank_combined
                    print(f"Block partitioning 2+3 found rank {rank_combined}")

    # Final verification with tighter tolerance
    if not verify(best, atol=1e-5):
        best = schoolbook()
        best_rank = n * m * p

    return {"U": to_lists(best["U"]), "V": to_lists(best["V"]), "W": to_lists(best["W"])}
# EVOLVE-BLOCK-END
