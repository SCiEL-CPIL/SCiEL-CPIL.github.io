import numpy as np
import random
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 140  # safety margin before 150s kill

    # Build the target tensor T
    # T[u, v, w] where u in [0, nm), v in [0, mp), w in [0, pn)
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    # T[u, v, w] = 1 if j_u == j_v and k_v == k_w and i_w == i_u else 0
    T = np.zeros((nm, mp, pn), dtype=np.int32)
    # Precompute non-zero indices for each w
    T_nonzero = {w: [] for w in range(pn)}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
                T_nonzero[w].append((u, v))

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_full(U, V, W, rank):
        """Full tensor verification."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)

        for w in range(pn):
            M = np.zeros((nm, mp), dtype=np.int64)
            for c in range(rank):
                if W_np[w, c] != 0:
                    M += U_np[:, c] * V_np[:, c] * W_np[w, c]
            if not np.array_equal(M, T[:, :, w]):
                return False
        return True

    # Try the "peeling" construction: 2x3x7 = 2x3x1 + 2x3x6
    # 2x3x6 = 2x3x1 + 2x3x5
    # 2x3x5 has rank 5 (Winograd-like)
    # Total: 2 + 2 + 5 = 9? No, that's too low.
    # Actually, the peeling is not that simple. Let's stick with search.
    
    best = schoolbook()
    best_rank = n * m * p

    # Precompute indices for faster access
    w_indices = []
    for w in range(pn):
        w_indices.append(T_nonzero[w])

    # Try ranks from best_rank - 1 down to 30
    for target_rank in range(best_rank - 1, 29, -1):
        if time.time() > deadline - 10:
            break

        max_attempts = 5000
        found = False

        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break

            # Generate random U, V with small integers
            # Use {-1, 0, 1} for simplicity, sometimes add 2
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(mp)]

            U_np = np.array(U, dtype=np.float64)
            V_np = np.array(V, dtype=np.float64)

            W = [[0] * target_rank for _ in range(pn)]
            valid = True

            for w in range(pn):
                indices = w_indices[w]
                # Build A_w (3 x target_rank)
                A_w = np.zeros((m, target_rank), dtype=np.float64)
                for j, (u, v) in enumerate(indices):
                    A_w[j, :] = U_np[u, :] * V_np[v, :]
                
                b_w = np.ones(m, dtype=np.float64)
                
                # Solve A_w W_w = b_w
                try:
                    W_w, residuals, rank_w, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Snap to nearest half-integer
                W_w_snapped = np.round(W_w * 2) / 2.0
                
                # Verify snapped solution
                residual = A_w.dot(W_w_snapped) - b_w
                if np.linalg.norm(residual) > 1e-6:
                    valid = False
                    break
                
                for c in range(target_rank):
                    W[w][c] = W_w_snapped[c]
            
            if valid:
                if verify_full(U, V, W, target_rank):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            continue

    return best
# EVOLVE-BLOCK-END
