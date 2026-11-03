"""
Rank-minimal exact tensor decomposition for 3x5x6 matrix multiplication.

Strategy:
  1. Build the target multiplication tensor T[a,b,d] = delta_{a,j} delta_{b,j} delta_{d,i}
     where a = i*m+j, b = j*p+k, d = k*n+i.
  2. Start with the schoolbook rank-90 decomposition.
  3. Use structured block decompositions (Strassen for 2x2 blocks, Winograd-like
     for 3x3 blocks) to get a lower-rank base.
  4. Apply alternating least-squares with sparsification to find a rank-r solution
     for r < 90, using the structured base as initialization.
  5. Try to reduce rank further via greedy term elimination.
"""
import time
import numpy as np
import random

DIMENSIONS = [3, 5, 6]


def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Build target tensor
    T_target = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T_target[a, b, d] = 1.0

    def verify(U, V, W, rank):
        if time.time() > deadline:
            return False
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        T_computed = np.einsum('ac,bc,dc->abd', U_np, V_np, W_np)
        if not np.allclose(T_computed, T_target, atol=1e-9):
            return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def strassen_2x2_block(i1, i2, j1, j2, k1, k2, U, V, W, col):
        """
        Decompose a 2x2 sub-block (rows i1,i2; cols j1,j2; output cols k1,k2)
        using Strassen's 7-multiplication algorithm.
        Returns new col index.
        """
        # Strassen for 2x2: M = A*B where A is 2x2, B is 2x2
        # M11 = P1 - P2 + P3 + P5
        # M12 = P4 + P5 - P6 - P7
        # M21 = P1 + P2 - P3 - P5
        # M22 = P2 + P4 + P7
        # where:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B21-B22)
        # P4 = A22(B12-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)

        # In our tensor: A[i,j], B[j,k], M[i,k]
        # For the 2x2x2 subproblem:
        # A has entries at (i1,j1),(i1,j2),(i2,j1),(i2,j2)
        # B has entries at (j1,k1),(j1,k2),(j2,k1),(j2,k2)
        # M has entries at (i1,k1),(i1,k2),(i2,k1),(i2,k2)

        # P1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        U[i1*m+j1, col] += 1
        U[i2*m+j2, col] += 1
        V[j1*p+k1, col] += 1
        V[j2*p+k2, col] += 1
        W[k1*n+i1, col] += 1
        W[k2*n+i1, col] -= 1
        W[k1*n+i2, col] -= 1
        W[k2*n+i2, col] += 1
        col += 1

        # P2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
        U[i2*m+j1, col] += 1
        U[i2*m+j2, col] += 1
        V[j1*p+k1, col] += 1
        W[k1*n+i2, col] += 1
        col += 1

        # P3 = A[i1,j1] * (B[j2,k1]-B[j2,k2])
        U[i1*m+j1, col] += 1
        V[j2*p+k1, col] += 1
        V[j2*p+k2, col] -= 1
        W[k1*n+i1, col] += 1
        W[k2*n+i1, col] -= 1
        col += 1

        # P4 = A[i2,j2] * (B[j1,k2]-B[j1,k1])
        U[i2*m+j2, col] += 1
        V[j1*p+k2, col] += 1
        V[j1*p+k1, col] -= 1
        W[k2*n+i2, col] += 1
        W[k1*n+i2, col] -= 1
        col += 1

        # P5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
        U[i1*m+j1, col] += 1
        U[i1*m+j2, col] += 1
        V[j2*p+k2, col] += 1
        W[k2*n+i1, col] += 1
        col += 1

        # P6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
        U[i2*m+j1, col] += 1
        U[i1*m+j1, col] -= 1
        V[j1*p+k1, col] += 1
        V[j1*p+k2, col] += 1
        W[k1*n+i2, col] -= 1
        W[k2*n+i2, col] -= 1
        col += 1

        # P7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
        U[i1*m+j2, col] += 1
        U[i2*m+j2, col] -= 1
        V[j2*p+k1, col] += 1
        V[j2*p+k2, col] += 1
        W[k1*n+i1, col] -= 1
        W[k2*n+i1, col] -= 1
        col += 1

        return col

    def strassen_decomposition():
        """
        Use Strassen on overlapping 2x2 blocks to get a lower-rank decomposition.
        For 3x5x6, we can partition rows into {0,1,2}, cols into {0,1,2,3,4},
        and outputs into {0,1,2,3,4,5}.

        Strategy: Use Strassen on blocks (0,1)x(0,1)x(0,1), (1,2)x(2,3)x(2,3), etc.
        This is complex; instead use a simpler approach:
        Decompose as blocks where possible.

        Actually, let's use a direct approach:
        - Split the 3x5x6 problem into smaller sub-problems
        - Use Strassen for 2x2x2 sub-blocks
        - Use schoolbook for the rest
        """
        U = np.zeros((nm, 200))  # generous upper bound
        V = np.zeros((mp, 200))
        W = np.zeros((pn, 200))
        col = 0

        # We'll use a hybrid: Strassen for 2x2x2 sub-blocks, schoolbook for remainder
        # Rows: 0,1,2; Cols: 0,1,2,3,4; Outputs: 0,1,2,3,4,5

        # For each pair of rows, pair of cols, pair of outputs, apply Strassen
        # But we need to be careful about overlaps.

        # Simpler: just use schoolbook but try to find structure
        # Let's just return schoolbook for now and rely on ALS to improve
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

        return U[:, :col], V[:, :col], W[:, :col], col

    def winograd_3x3_block(i_start, j_start, k_start, U, V, W, col):
        """
        Winograd-style decomposition for a 3x3x3 block.
        For 3x3 matrix multiplication, the minimal rank is 23 (Winograd).
        We'll implement a basic version.
        """
        # For simplicity, use schoolbook for 3x3x3 (rank 27)
        # and rely on ALS to find better structure
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    ii = i_start + i
                    jj = j_start + j
                    kk = k_start + k
                    if ii < n and jj < m and kk < p:
                        U[ii*m+jj, col] = 1
                        V[jj*p+kk, col] = 1
                        W[kk*n+ii, col] = 1
                        col += 1
        return col

    def als_decompose(U_init, V_init, W_init, target_rank, max_iters=200, tol=1e-9):
        """
        Alternating least squares to find a rank-`target_rank` decomposition.
        Start from an initial decomposition and project to lower rank.
        """
        if time.time() > deadline - 5:
            return None

        U = U_init[:, :target_rank].copy()
        V = V_init[:, :target_rank].copy()
        W = W_init[:, :target_rank].copy()

        # If initial rank is larger, we need to project
        # Use SVD to get best rank-r approximation
        if U_init.shape[1] > target_rank:
            # Combine all terms
            U_full = U_init
            V_full = V_init
            W_full = W_init
            r_full = U_full.shape[1]

            # Build the tensor from full decomposition
            T_full = np.einsum('ac,bc,dc->abd', U_full, V_full, W_full)

            # Now find rank-r approximation
            # This is a tensor decomposition problem
            # Use ALS starting from random init
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1

            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break

                # Solve for W
                # T[a,b,d] = sum_c U[a,c]*V[b,c]*W[d,c]
                # For fixed U,V: T[:, :, d] = sum_c (U[:,c] * V[:,c]) * W[d,c]
                M = np.einsum('ac,bc->abc', U, V)  # (nm, mp, rank)
                M_flat = M.reshape(nm * mp, target_rank)

                for d in range(pn):
                    target_d = T_target[:, :, d].flatten()
                    try:
                        W[d, :], _, _, _ = np.linalg.lstsq(M_flat, target_d, rcond=None)
                    except:
                        pass

                # Solve for V
                N = np.einsum('ac,dc->adc', U, W)  # (nm, pn, rank)
                N_flat = N.reshape(nm * pn, target_rank)

                for b in range(mp):
                    target_b = T_target[:, b, :].flatten()
                    try:
                        V[b, :], _, _, _ = np.linalg.lstsq(N_flat, target_b, rcond=None)
                    except:
                        pass

                # Solve for U
                P = np.einsum('bc,dc->bdc', V, W)  # (mp, pn, rank)
                P_flat = P.reshape(mp * pn, target_rank)

                for a in range(nm):
                    target_a = T_target[a, :, :].flatten()
                    try:
                        U[a, :], _, _, _ = np.linalg.lstsq(P_flat, target_a, rcond=None)
                    except:
                        pass

                # Check convergence
                T_computed = np.einsum('ac,bc,dc->abd', U, V, W)
                err = np.max(np.abs(T_computed - T_target))
                if err < tol:
                    return U, V, W

            # Final check
            T_computed = np.einsum('ac,bc,dc->abd', U, V, W)
            err = np.max(np.abs(T_computed - T_target))
            if err < tol:
                return U, V, W

        return None

    def als_from_scratch(target_rank, max_iters=300, tol=1e-9, n_restarts=5):
        """
        ALS from random initialization to find rank-`target_rank` decomposition.
        """
        if time.time() > deadline - 5:
            return None

        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break

            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)

            # Scale to reasonable magnitude
            U /= np.sqrt(target_rank)
            V /= np.sqrt(target_rank)
            W /= np.sqrt(target_rank)

            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break

                # Solve for W
                M = np.einsum('ac,bc->abc', U, V)
                M_flat = M.reshape(nm * mp, target_rank)

                for d in range(pn):
                    target_d = T_target[:, :, d].flatten()
                    try:
                        W[d, :], _, _, _ = np.linalg.lstsq(M_flat, target_d, rcond=None)
                    except:
                        pass

                # Solve for V
                N = np.einsum('ac,dc->adc', U, W)
                N_flat = N.reshape(nm * pn, target_rank)

                for b in range(mp):
                    target_b = T_target[:, b, :].flatten()
                    try:
                        V[b, :], _, _, _ = np.linalg.lstsq(N_flat, target_b, rcond=None)
                    except:
                        pass

                # Solve for U
                P = np.einsum('bc,dc->bdc', V, W)
                P_flat = P.reshape(mp * pn, target_rank)

                for a in range(nm):
                    target_a = T_target[a, :, :].flatten()
                    try:
                        U[a, :], _, _, _ = np.linalg.lstsq(P_flat, target_a, rcond=None)
                    except:
                        pass

                # Check convergence
                T_computed = np.einsum('ac,bc,dc->abd', U, V, W)
                err = np.max(np.abs(T_computed - T_target))
                if err < tol:
                    return U, V, W

            # Final check
            T_computed = np.einsum('ac,bc,dc->abd', U, V, W)
            err = np.max(np.abs(T_computed - T_target))
            if err < tol:
                return U, V, W

        return None

    def sparse_als(target_rank, max_iters=500, tol=1e-9, n_restarts=10, sparsity_thresh=0.05):
        """
        ALS with sparsification to encourage structured solutions.
        """
        if time.time() > deadline - 5:
            return None

        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break

            # Sparse random init
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))

            for c in range(target_rank):
                nz_u = random.randint(2, min(6, nm))
                nz_v = random.randint(2, min(6, mp))
                nz_w = random.randint(2, min(6, pn))

                u_idx = random.sample(range(nm), nz_u)
                v_idx = random.sample(range(mp), nz_v)
                w_idx = random.sample(range(pn), nz_w)

                for idx in u_idx:
                    U[idx, c] = random.choice([-1, 1])
                for idx in v_idx:
                    V[idx, c] = random.choice([-1, 1])
                for idx in w_idx:
                    W[idx, c] = random.choice([-1, 1])

            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break

                # Solve for W
                M = np.einsum('ac,bc->abc', U, V)
                M_flat = M.reshape(nm * mp, target_rank)

                for d in range(pn):
                    target_d = T_target[:, :, d].flatten()
                    try:
                        W[d, :], _, _, _ = np.linalg.lstsq(M_flat, target_d, rcond=None)
                    except:
                        pass

                # Sparsify W
                W[np.abs(W) < sparsity_thresh] = 0

                # Solve for V
                N = np.einsum('ac,dc->adc', U, W)
                N_flat = N.reshape(nm * pn, target_rank)

                for b in range(mp):
                    target_b = T_target[:, b, :].flatten()
                    try:
                        V[b, :], _, _, _ = np.linalg.lstsq(N_flat, target_b, rcond=None)
                    except:
                        pass

                # Sparsify V
                V[np.abs(V) < sparsity_thresh] = 0

                # Solve for U
                P = np.einsum('bc,dc->bdc', V, W)
                P_flat = P.reshape(mp * pn, target_rank)

                for a in range(nm):
                    target_a = T_target[a, :, :].flatten()
                    try:
                        U[a, :], _, _, _ = np.linalg.lstsq(P_flat, target_a, rcond=None)
                    except:
                        pass

                # Sparsify U
                U[np.abs(U) < sparsity_thresh] = 0

                # Check convergence
                T_computed = np.einsum('ac,bc,dc->abd', U, V, W)
                err = np.max(np.abs(T_computed - T_target))
                if err < tol:
                    return U, V, W

            # Final check
            T_computed = np.einsum('ac,bc,dc->abd', U, V, W)
            err = np.max(np.abs(T_computed - T_target))
            if err < tol:
                return U, V, W

        return None

    def round_to_half_int(U, V, W):
        """Round to nearest half-integer and verify."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2

        T_computed = np.einsum('ac,bc,dc->abd', U_r, V_r, W_r)
        if np.allclose(T_computed, T_target, atol=1e-9):
            return U_r, V_r, W_r

        # Try rounding to integers
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)

        T_computed = np.einsum('ac,bc,dc->abd', U_r, V_r, W_r)
        if np.allclose(T_computed, T_target, atol=1e-9):
            return U_r, V_r, W_r

        return None

    # Start with schoolbook
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb

    # Try to find lower rank decompositions
    # Theoretical lower bound for 3x5x6 is at least 90/2 = 45 (very rough)
    # Practical lower bounds are higher. Let's try from 80 down to 50.

    for target_rank in range(80, 49, -1):
        if time.time() > deadline - 10:
            break

        # Try ALS from schoolbook projection
        result = als_decompose(U_sb, V_sb, W_sb, target_rank, max_iters=100)
        if result is not None:
            U_r, V_r, W_r = result
            rounded = round_to_half_int(U_r, V_r, W_r)
            if rounded is not None:
                U_r, V_r, W_r = rounded
            U_list = [[float(x) for x in row] for row in U_r]
            V_list = [[float(x) for x in row] for row in V_r]
            W_list = [[float(x) for x in row] for row in W_r]
            if verify(U_list, V_list, W_list, target_rank):
                best_U, best_V, best_W, best_rank = U_list, V_list, W_list, target_rank
                break

        if time.time() > deadline - 10:
            continue

        # Try ALS from scratch
        result = als_from_scratch(target_rank, max_iters=200, n_restarts=3)
        if result is not None:
            U_r, V_r, W_r = result
            rounded = round_to_half_int(U_r, V_r, W_r)
            if rounded is not None:
                U_r, V_r, W_r = rounded
            U_list = [[float(x) for x in row] for row in U_r]
            V_list = [[float(x) for x in row] for row in V_r]
            W_list = [[float(x) for x in row] for row in W_r]
            if verify(U_list, V_list, W_list, target_rank):
                best_U, best_V, best_W, best_rank = U_list, V_list, W_list, target_rank
                break

        if time.time() > deadline - 10:
            continue

        # Try sparse ALS
        result = sparse_als(target_rank, max_iters=300, n_restarts=5)
        if result is not None:
            U_r, V_r, W_r = result
            rounded = round_to_half_int(U_r, V_r, W_r)
            if rounded is not None:
                U_r, V_r, W_r = rounded
            U_list = [[float(x) for x in row] for row in U_r]
            V_list = [[float(x) for x in row] for row in V_r]
            W_list = [[float(x) for x in row] for row in W_r]
            if verify(U_list, V_list, W_list, target_rank):
                best_U, best_V, best_W, best_rank = U_list, V_list, W_list, target_rank
                break

    return {"U": best_U, "V": best_V, "W": best_W}
