"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    # n=2, m=2, p=6
    U_dim = n * m   # 4
    V_dim = m * p   # 12
    W_dim = p * n   # 12

    def build_target():
        """Build the target multiplication tensor."""
        target = np.zeros((U_dim, V_dim, W_dim))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    target[u_idx, v_idx, w_idx] = 1.0
        return target

    def strassen_3blocks():
        """Rank 21 decomposition using Strassen's algorithm for 3 blocks."""
        rank = 21
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        # Strassen coefficients
        U_coeffs = [
            {0: 1, 3: 1},       # M1: A00 + A11
            {2: 1, 3: 1},       # M2: A10 + A11
            {0: 1},             # M3: A00
            {3: 1},             # M4: A11
            {0: 1, 1: 1},       # M5: A00 + A01
            {2: 1, 0: -1},      # M6: A10 - A00
            {1: 1, 3: -1}       # M7: A01 - A11
        ]
        V_coeffs = [
            [(0, 0, 1), (1, 1, 1)],      # M1: B00 + B11
            [(0, 0, 1)],                 # M2: B00
            [(0, 1, 1), (1, 1, -1)],     # M3: B01 - B11
            [(1, 0, 1), (0, 0, -1)],     # M4: B10 - B00
            [(1, 1, 1)],                 # M5: B11
            [(0, 0, 1), (0, 1, 1)],      # M6: B00 + B01
            [(1, 0, 1), (1, 1, 1)]       # M7: B10 + B11
        ]
        W_map = [
            [(0, 0, 1), (1, 1, 1)],   # M1 -> C00, C11
            [(1, 0, 1), (1, 1, -1)],  # M2 -> C10, C11
            [(0, 1, 1), (1, 1, 1)],   # M3 -> C01, C11
            [(0, 0, 1), (1, 0, 1)],   # M4 -> C00, C10
            [(0, 0, -1), (0, 1, 1)],  # M5 -> C00, C01
            [(1, 1, 1)],              # M6 -> C11
            [(0, 0, 1)]               # M7 -> C00
        ]

        for r in range(3):
            for k in range(7):
                col = r * 7 + k
                for idx, val in U_coeffs[k].items():
                    U[idx, col] = val
                for j_local, k_local, val in V_coeffs[k]:
                    global_k = 2 * r + k_local
                    v_idx = j_local * p + global_k
                    V[v_idx, col] = val
                for i, j_local, val in W_map[k]:
                    global_k = 2 * r + j_local
                    w_idx = global_k * n + i
                    W[w_idx, col] = val

        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, target, tol=1e-9):
        """Verify exact decomposition."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(T - target) < tol

    def try_rank_20_fast():
        """
        Try to find rank-20 by checking if any Strassen term is linearly
        dependent on others. Uses efficient tensor rank-1 testing via
        matrix unfoldings (SVD-based).
        """
        base = strassen_3blocks()
        U_base = base["U"].copy()
        V_base = base["V"].copy()
        W_base = base["W"].copy()
        target = build_target()
        rank21 = 21

        for t_remove in range(rank21):
            if time.time() > deadline:
                break

            # Build remaining terms
            keep = [r for r in range(rank21) if r != t_remove]
            U_rem = U_base[:, keep]
            V_rem = V_base[:, keep]
            W_rem = W_base[:, keep]

            # Compute residual
            T_rec = np.einsum('ar,br,cr->abc', U_rem, V_rem, W_rem)
            residual = target - T_rec

            # Quick check: is residual zero?
            if np.linalg.norm(residual) < 1e-10:
                # Already rank-20!
                return {"U": U_rem, "V": V_rem, "W": W_rem}

            # Check if residual is a rank-1 tensor using matrix unfoldings
            # Mode-0 unfolding: (U_dim) x (V_dim * W_dim)
            R0 = residual.reshape(U_dim, V_dim * W_dim)
            s0 = np.linalg.svd(R0, compute_uv=False)
            if len(s0) < 1 or s0[0] < 1e-10:
                continue
            if len(s0) >= 2 and s0[1] > 1e-8 * s0[0]:
                continue  # Not rank-1

            # Mode-1 unfolding: (V_dim) x (U_dim * W_dim)
            R1 = residual.transpose(1, 0, 2).reshape(V_dim, U_dim * W_dim)
            s1 = np.linalg.svd(R1, compute_uv=False)
            if len(s1) >= 2 and s1[1] > 1e-8 * s1[0]:
                continue

            # Mode-2 unfolding: (W_dim) x (U_dim * V_dim)
            R2 = residual.transpose(2, 0, 1).reshape(W_dim, U_dim * V_dim)
            s2 = np.linalg.svd(R2, compute_uv=False)
            if len(s2) >= 2 and s2[1] > 1e-8 * s2[0]:
                continue

            # Residual is rank-1. Extract vectors.
            # From R0: R0 = u_vec @ (v_vec ⊗ w_vec)^T
            U_svd, s, Vt = np.linalg.svd(R0, full_matrices=False)
            u_vec = U_svd[:, 0] * np.sqrt(s[0])

            # Extract v_vec and w_vec
            # R0[u, v*W_dim + w] = u_vec[u] * v_vec[v] * w_vec[w]
            u_max = np.argmax(np.abs(u_vec))
            if abs(u_vec[u_max]) < 1e-10:
                continue

            vw_flat = R0[u_max, :] / u_vec[u_max]  # length V_dim * W_dim
            vw_matrix = vw_flat.reshape(V_dim, W_dim)

            # Check if vw_matrix is rank-1
            s_vw = np.linalg.svd(vw_matrix, compute_uv=False)
            if len(s_vw) >= 2 and s_vw[1] > 1e-8 * (s_vw[0] if s_vw[0] > 0 else 1):
                continue

            if s_vw[0] < 1e-10:
                continue

            # Extract v_vec and w_vec from vw_matrix
            U_vw, s_vw, Vt_vw = np.linalg.svd(vw_matrix, full_matrices=False)
            v_vec = U_vw[:, 0] * np.sqrt(s_vw[0])
            w_vec = Vt_vw[0, :] * np.sqrt(s_vw[0])

            # Verify extraction
            if np.linalg.norm(np.outer(v_vec, w_vec) - vw_matrix) > 1e-6:
                continue

            # Build new rank-20 decomposition
            U_new = np.zeros((U_dim, rank21 - 1))
            V_new = np.zeros((V_dim, rank21 - 1))
            W_new = np.zeros((W_dim, rank21 - 1))

            for r, orig_r in enumerate(keep):
                U_new[:, r] = U_base[:, orig_r]
                V_new[:, r] = V_base[:, orig_r]
                W_new[:, r] = W_base[:, orig_r]

            # Add new term
            U_new[:, rank21 - 2] = u_vec
            V_new[:, rank21 - 2] = v_vec
            W_new[:, rank21 - 2] = w_vec

            # Verify
            if verify_decomposition(U_new, V_new, W_new, target, tol=1e-6):
                # Snap to half-integers
                U_snap = np.round(U_new * 2) / 2
                V_snap = np.round(V_new * 2) / 2
                W_snap = np.round(W_new * 2) / 2

                if verify_decomposition(U_snap, V_snap, W_snap, target, tol=1e-6):
                    return {"U": U_snap, "V": V_snap, "W": W_snap}

        return None

    def try_rank_20_structured():
        """
        Structured search: try merging Strassen terms across blocks.
        The idea: if two terms from different blocks have compatible
        U and W patterns, they might be mergeable.
        """
        if time.time() > deadline:
            return None

        base = strassen_3blocks()
        U_base = base["U"].copy()
        V_base = base["V"].copy()
        W_base = base["W"].copy()
        target = build_target()
        rank21 = 21

        # Try removing pairs of terms that might be redundant
        # Focus on terms that share the same U pattern (same Strassen step)
        # but different blocks
        for s in range(7):
            if time.time() > deadline:
                break
            # Terms with same Strassen step s: indices 7*s, 7*s+7, 7*s+14
            # Try removing one and seeing if residual is rank-1
            for block_to_remove in range(3):
                t_remove = 7 * block_to_remove + s
                keep = [r for r in range(rank21) if r != t_remove]
                U_rem = U_base[:, keep]
                V_rem = V_base[:, keep]
                W_rem = W_base[:, keep]

                T_rec = np.einsum('ar,br,cr->abc', U_rem, V_rem, W_rem)
                residual = target - T_rec

                if np.linalg.norm(residual) < 1e-10:
                    return {"U": U_rem, "V": V_rem, "W": W_rem}

                # Quick rank-1 check
                R0 = residual.reshape(U_dim, V_dim * W_dim)
                s0 = np.linalg.svd(R0, compute_uv=False)
                if len(s0) < 2 or s0[1] > 1e-8 * (s0[0] if s0[0] > 0 else 1):
                    continue

        return None

    # First try the fast rank-20 search
    result_20 = try_rank_20_fast()
    if result_20 is not None:
        return {
            "U": result_20["U"].tolist(),
            "V": result_20["V"].tolist(),
            "W": result_20["W"].tolist()
        }

    # Try structured search
    result_20_struct = try_rank_20_structured()
    if result_20_struct is not None:
        return {
            "U": result_20_struct["U"].tolist(),
            "V": result_20_struct["V"].tolist(),
            "W": result_20_struct["W"].tolist()
        }

    # Fallback to rank 21
    base = strassen_3blocks()
    return {
        "U": base["U"].tolist(),
        "V": base["V"].tolist(),
        "W": base["W"].tolist()
    }

# EVOLVE-BLOCK-END
