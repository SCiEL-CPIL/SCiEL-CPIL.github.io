import time
import numpy as np

def construct():
    """
    Constructs a low-rank matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 3, 8

    def _build_rank44():
        """Build the rank-44 decomposition: 16 (schoolbook 2x1x8) + 28 (4x Strassen 2x2x2)."""
        rank = 44
        U = [[0] * rank for _ in range(n * m)]   # 6 x 44
        V = [[0] * rank for _ in range(m * p)]   # 24 x 44
        W = [[0] * rank for _ in range(p * n)]   # 16 x 44

        col = 0
        # Part 1: A[:,0] (2x1) * B[0,:] (1x8) -> 2x8, rank 16
        for i in range(2):
            for k in range(8):
                U[i * 3 + 0][col] = 1
                V[0 * 8 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1

        # Part 2: A[:,1:3] (2x2) * B[1:3,:] (2x8) -> 2x8, rank 28
        # Split B[1:3,:] into 4 blocks of 2 columns each
        for block in range(4):
            b00 = 1 * 8 + block * 2
            b01 = 1 * 8 + block * 2 + 1
            b10 = 2 * 8 + block * 2
            b11 = 2 * 8 + block * 2 + 1

            # Strassen 7 multiplications for 2x2x2
            # A local: a00=U[1], a01=U[2], a10=U[4], a11=U[5]
            # B local: b00=V[b00], b01=V[b01], b10=V[b10], b11=V[b11]
            # C local: c00=W[block*2*2+0], c01=W[block*2*2+1], c10=W[(block*2+1)*2+0], c11=W[(block*2+1)*2+1]

            # M1 = (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
            U[1][col] = 1; U[5][col] = 1
            V[b00][col] = 1; V[b11][col] = 1
            W[block * 4 + 0][col] = 1
            W[block * 4 + 3][col] = 1
            col += 1

            # M2 = (a10+a11)b00 -> c10 += M2, c11 -= M2
            U[4][col] = 1; U[5][col] = 1
            V[b00][col] = 1
            W[block * 4 + 2][col] = 1
            W[block * 4 + 3][col] = -1
            col += 1

            # M3 = a00(b01-b11) -> c01 += M3, c11 += M3
            U[1][col] = 1
            V[b01][col] = 1; V[b11][col] = -1
            W[block * 4 + 1][col] = 1
            W[block * 4 + 3][col] = 1
            col += 1

            # M4 = a11(b10-b00) -> c00 += M4, c10 += M4
            U[5][col] = 1
            V[b10][col] = 1; V[b00][col] = -1
            W[block * 4 + 0][col] = 1
            W[block * 4 + 2][col] = 1
            col += 1

            # M5 = (a00+a01)b11 -> c00 -= M5, c01 += M5
            U[1][col] = 1; U[2][col] = 1
            V[b11][col] = 1
            W[block * 4 + 0][col] = -1
            W[block * 4 + 1][col] = 1
            col += 1

            # M6 = (a10-a00)(b00+b01) -> c11 += M6
            U[4][col] = 1; U[1][col] = -1
            V[b00][col] = 1; V[b01][col] = 1
            W[block * 4 + 3][col] = 1
            col += 1

            # M7 = (a01-a11)(b10+b11) -> c00 += M7
            U[2][col] = 1; U[5][col] = -1
            V[b10][col] = 1; V[b11][col] = 1
            W[block * 4 + 0][col] = 1
            col += 1

        assert col == 44, f"Expected 44, got {col}"
        return {"U": U, "V": V, "W": W}

    def _verify(sol):
        """Verify the decomposition is correct."""
        U = np.array(sol['U']).T  # rank x 6
        V = np.array(sol['V']).T  # rank x 24
        W = np.array(sol['W']).T  # rank x 16
        rank = U.shape[0]
        T = np.einsum('ur,vr,wr->uvw', U, V, W)
        target = np.zeros((6, 24, 16))
        for i in range(2):
            for j in range(3):
                for k in range(8):
                    target[i * 3 + j, j * 8 + k, k * 2 + i] = 1.0
        return np.allclose(T, target, atol=1e-9)

    # Build and verify rank-44 solution
    sol44 = _build_rank44()
    if not _verify(sol44):
        # Fallback: pure schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    # Try to find rank-43 solution
    if time.time() < deadline - 30:
        try:
            target_rank = 43
            U44 = np.array(sol44['U']).T  # 44 x 6
            V44 = np.array(sol44['V']).T  # 44 x 24
            W44 = np.array(sol44['W']).T  # 44 x 16

            target_tensor = np.zeros((6, 24, 16))
            for i in range(2):
                for j in range(3):
                    for k in range(8):
                        target_tensor[i * 3 + j, j * 8 + k, k * 2 + i] = 1.0

            best_result = None
            best_norm = np.inf

            # Try dropping each term and re-optimizing
            for drop_idx in range(44):
                if time.time() > deadline - 15:
                    break

                U_init = np.delete(U44, drop_idx, axis=0)
                V_init = np.delete(V44, drop_idx, axis=0)
                W_init = np.delete(W44, drop_idx, axis=0)

                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

                def residuals(params, U0=U_init, V0=V_init, W0=W_init):
                    U = params[:target_rank * 6].reshape(target_rank, 6)
                    V = params[target_rank * 6:target_rank * 30].reshape(target_rank, 24)
                    W = params[target_rank * 30:].reshape(target_rank, 16)
                    T = np.einsum('ur,vr,wr->uvw', U, V, W)
                    return (T - target_tensor).flatten()

                try:
                    import scipy.optimize as opt
                    result = opt.least_squares(residuals, x0, method='trf', max_nfev=300,
                                               ftol=1e-12, xtol=1e-12, gtol=1e-12)
                    if result.success and np.linalg.norm(result.fun) < 1e-8:
                        if np.linalg.norm(result.fun) < best_norm:
                            best_norm = np.linalg.norm(result.fun)
                            best_result = result
                except Exception:
                    continue

            if best_result is not None and best_norm < 1e-8:
                U = best_result.x[:target_rank * 6].reshape(target_rank, 6).T
                V = best_result.x[target_rank * 6:target_rank * 30].reshape(target_rank, 24).T
                W = best_result.x[target_rank * 30:].reshape(target_rank, 16).T
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        except Exception:
            pass

    return sol44

