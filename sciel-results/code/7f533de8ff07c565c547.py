import numpy as np
import time
from itertools import combinations

def construct():
    n, m, p = 6, 6, 6
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s

    # Target tensor T[u, v, w] where u=i*m+k, v=k*p+j, w=j*n+i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                T[i * m + k, k * p + j, j * n + i] = 1.0

    # Block-Strassen with block size 3: 2x2 blocks of 3x3
    bs = 3
    nb = n // bs  # 2

    # Strassen 2x2 definitions (7 multiplications)
    # Blocks indexed 0:00, 1:01, 2:10, 3:11
    strassen_defs = [
        # M0: (A00 + A11)(B00 + B11)
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        # M1: (A10 + A11)B00
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        # M2: A00(B01 - B11)
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        # M3: A11(B10 - B00)
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        # M4: (A00 + A01)B11
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        # M5: (A10 - A00)(B00 + B01)
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        # M6: (A01 - A11)(B10 + B11)
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]

    # Reconstruction coefficients: C_block = sum gamma_s * M_s
    recon_coeffs = [
        # C00 = M0 + M3 - M4 + M6
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        # C01 = M2 + M4
        [(2, 1.0), (4, 1.0)],
        # C10 = M1 + M3
        [(1, 1.0), (3, 1.0)],
        # C11 = M0 - M1 + M2 + M5
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]

    rank_bs = nb * nb * nb * 7  # 2*2*2*7 = 56? No: for each (i_b, j_b, k_b) block triple, 7 Strassen terms, each expanded over bs*bs*bs inner indices
    # Actually the rank is: for each of the 2x2x2 = 8 block positions (i_b, j_b, k_b), and for each of the 7 Strassen steps,
    # we have bs*bs*bs = 27 inner scalar multiplications. So rank = 8 * 7 * 27 = 1512? That's too high.
    # Wait, let me re-read the inspiration. The inspiration uses bs=3 and iterates u,v,j in range(bs) for each Strassen step,
    # but the block triple (i_b, j_b, k_b) is NOT iterated separately - it's handled by the block indexing within the Strassen defs.
    # Let me re-examine: the Strassen defs operate on 2x2 blocks. For 6x6 with bs=3, we have 2x2 blocks.
    # So there's only ONE 2x2 block structure (no outer iteration over block triples).
    # The rank is 7 (Strassen) * bs*bs*bs (inner) = 7 * 27 = 189.
    
    rank_bs = 7 * bs * bs * bs  # 7 * 27 = 189

    U = np.zeros((n * m, rank_bs), dtype=float)
    V = np.zeros((m * p, rank_bs), dtype=float)
    W = np.zeros((p * n, rank_bs), dtype=float)

    r_idx = 0
    for s in range(7):
        s_def = strassen_defs[s]
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    # U: A_s[u, v] contributions
                    for block_idx, coeff_a in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        i_glob = br * bs + u
                        k_glob = bc * bs + v
                        U[i_glob * m + k_glob, r_idx] += coeff_a
                    # V: B_s[v, j] contributions
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        V[k_glob * p + j_glob, r_idx] += coeff_b
                    # W: reconstruction
                    for c_block_idx in range(4):
                        gamma = 0.0
                        for (s_idx, g) in recon_coeffs[c_block_idx]:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            cr = c_block_idx // 2
                            cc = c_block_idx % 2
                            i_glob = cr * bs + u
                            j_glob = cc * bs + j
                            W[j_glob * n + i_glob, r_idx] += gamma
                    r_idx += 1

    assert r_idx == rank_bs

    # Verify base decomposition
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-9):
        # Fallback: trivial decomposition
        rank_trivial = n * m * p
        U_t = np.zeros((n * m, rank_trivial), dtype=float)
        V_t = np.zeros((m * p, rank_trivial), dtype=float)
        W_t = np.zeros((p * n, rank_trivial), dtype=float)
        idx = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_t[i * m + k, idx] = 1.0
                    V_t[k * p + j, idx] = 1.0
                    W_t[j * n + i, idx] = 1.0
                    idx += 1
        return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}

    # Rank reduction: find and eliminate linearly dependent terms
    # Two terms can be merged if they are proportional (same scalar c for U, V, W)
    # More generally, a term can be removed if it's a linear combination of others
    # For efficiency, first do pairwise proportional merge, then try small-group elimination

    if time.time() < deadline - 30:
        new_U = U.copy()
        new_V = V.copy()
        new_W = W.copy()
        removed = [False] * rank_bs

        # Pass 1: Pairwise proportional merge
        for r in range(rank_bs):
            if removed[r]:
                continue
            for r2 in range(r + 1, rank_bs):
                if removed[r2]:
                    continue
                # Check if term r is a scalar multiple of term r2
                nz_u = np.nonzero(U[:, r])[0]
                if len(nz_u) == 0:
                    removed[r] = True
                    break
                if U[nz_u[0], r2] == 0:
                    continue
                c = U[nz_u[0], r] / U[nz_u[0], r2]
                if not np.allclose(U[:, r], c * U[:, r2], atol=1e-10):
                    continue
                nz_v = np.nonzero(V[:, r])[0]
                if len(nz_v) == 0:
                    removed[r] = True
                    break
                if V[nz_v[0], r2] == 0:
                    continue
                c_v = V[nz_v[0], r] / V[nz_v[0], r2]
                if abs(c_v - c) > 1e-10:
                    continue
                if not np.allclose(V[:, r], c * V[:, r2], atol=1e-10):
                    continue
                nz_w = np.nonzero(W[:, r])[0]
                if len(nz_w) == 0:
                    removed[r] = True
                    break
                if W[nz_w[0], r2] == 0:
                    continue
                c_w = W[nz_w[0], r] / W[nz_w[0], r2]
                if abs(c_w - c) > 1e-10:
                    continue
                if not np.allclose(W[:, r], c * W[:, r2], atol=1e-10):
                    continue
                # Merge: add term r to term r2
                new_U[:, r2] += c * new_U[:, r]
                new_V[:, r2] += c * new_V[:, r]
                new_W[:, r2] += c * new_W[:, r]
                removed[r] = True
                break

        # Pass 2: Try to eliminate terms that are linear combinations of 2-3 others
        # This is more expensive, so only do it if we have time
        if time.time() < deadline - 15:
            remaining = [i for i in range(rank_bs) if not removed[i]]
            # For each remaining term, check if it's a combo of up to 2 others
            for r in list(remaining):
                if removed[r]:
                    continue
                # Check pairs
                found = False
                for combo in combinations(remaining, 2):
                    if r in combo:
                        continue
                    # Solve U[:,r] = a*U[:,c1] + b*U[:,c2]
                    # Use least squares on the U vectors
                    try:
                        A_u = np.column_stack([U[:, combo[0]], U[:, combo[1]]])
                        coeff_u, _, _, _ = np.linalg.lstsq(A_u, U[:, r], rcond=None)
                        if np.allclose(A_u @ coeff_u, U[:, r], atol=1e-10):
                            a, b = coeff_u
                            # Verify with V
                            A_v = np.column_stack([V[:, combo[0]], V[:, combo[1]]])
                            if np.allclose(A_v @ coeff_u, V[:, r], atol=1e-10):
                                # Verify with W
                                A_w = np.column_stack([W[:, combo[0]], W[:, combo[1]]])
                                if np.allclose(A_w @ coeff_u, W[:, r], atol=1e-10):
                                    # Remove term r
                                    new_U[:, combo[0]] += a * new_U[:, r]
                                    new_V[:, combo[0]] += a * new_V[:, r]
                                    new_W[:, combo[0]] += a * new_W[:, r]
                                    new_U[:, combo[1]] += b * new_U[:, r]
                                    new_V[:, combo[1]] += b * new_V[:, r]
                                    new_W[:, combo[1]] += b * new_W[:, r]
                                    removed[r] = True
                                    found = True
                                    break
                    except Exception:
                        pass
                if found:
                    continue
                # Check triples if still time
                if time.time() < deadline - 5:
                    for combo in combinations(remaining, 3):
                        if r in combo:
                            continue
                        try:
                            A_u = np.column_stack([U[:, combo[0]], U[:, combo[1]], U[:, combo[2]]])
                            coeff_u, _, _, _ = np.linalg.lstsq(A_u, U[:, r], rcond=None)
                            if np.allclose(A_u @ coeff_u, U[:, r], atol=1e-10):
                                a, b, c = coeff_u
                                A_v = np.column_stack([V[:, combo[0]], V[:, combo[1]], V[:, combo[2]]])
                                if np.allclose(A_v @ coeff_u, V[:, r], atol=1e-10):
                                    A_w = np.column_stack([W[:, combo[0]], W[:, combo[1]], W[:, combo[2]]])
                                    if np.allclose(A_w @ coeff_u, W[:, r], atol=1e-10):
                                        new_U[:, combo[0]] += a * new_U[:, r]
                                        new_V[:, combo[0]] += a * new_V[:, r]
                                        new_W[:, combo[0]] += a * new_W[:, r]
                                        new_U[:, combo[1]] += b * new_U[:, r]
                                        new_V[:, combo[1]] += b * new_V[:, r]
                                        new_W[:, combo[1]] += b * new_W[:, r]
                                        new_U[:, combo[2]] += c * new_U[:, r]
                                        new_V[:, combo[2]] += c * new_V[:, r]
                                        new_W[:, combo[2]] += c * new_W[:, r]
                                        removed[r] = True
                                        break
                        except Exception:
                            pass

        # Build reduced matrices
        kept = [i for i in range(rank_bs) if not removed[i]]
        if len(kept) < rank_bs:
            U_red = new_U[:, kept]
            V_red = new_V[:, kept]
            W_red = new_W[:, kept]
            # Verify
            T_red = np.einsum('ir,jr,kr->ijk', U_red, V_red, W_red)
            if np.allclose(T_red, T, atol=1e-9):
                return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
