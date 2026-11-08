"""Matrix multiplication 2x4x5 rank-minimal decomposition."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    _deadline = _start + 118

    n, m, p = DIMENSIONS  # n=2, m=4, p=5

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        num_U = len(U)
        num_V = len(V)
        num_W = len(W)
        rank = len(U[0])
        # Convert to numpy arrays for faster dot products if rank is large
        # But rank is small (36), so pure python might be faster due to overhead
        for a in range(num_U):
            i, j = divmod(a, m)
            for b in range(num_V):
                j2, k = divmod(b, p)
                if j != j2:
                    continue
                c = k * n + i
                target = 1
                computed = 0
                for r in range(rank):
                    computed += U[a][r] * V[b][r] * W[c][r]
                if abs(computed - target) > 1e-9:
                    return False
        return True

    def build_strassen_36():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def strassen_block(y, z, w):
            nonlocal col
            y00, y01, y10, y11 = y
            z00, z01, z10, z11 = z

            # S1 = (A00+A11)(B00+B11)
            U[u_idx(*y00)][col] = 1; U[u_idx(*y11)][col] = 1
            V[v_idx(*z00)][col] = 1; V[v_idx(*z11)][col] = 1
            W[w_idx(*w[0])][col] += 1; W[w_idx(*w[3])][col] += 1
            col += 1

            # S2 = (A10+A11)B00
            U[u_idx(*y10)][col] = 1; U[u_idx(*y11)][col] = 1
            V[v_idx(*z00)][col] = 1
            W[w_idx(*w[2])][col] += 1; W[w_idx(*w[3])][col] -= 1
            col += 1

            # S3 = A00(B01-B11)
            U[u_idx(*y00)][col] = 1
            V[v_idx(*z01)][col] = 1; V[v_idx(*z11)][col] = -1
            W[w_idx(*w[1])][col] += 1; W[w_idx(*w[3])][col] += 1
            col += 1

            # S4 = A11(B10-B00)
            U[u_idx(*y11)][col] = 1
            V[v_idx(*z10)][col] = 1; V[v_idx(*z00)][col] = -1
            W[w_idx(*w[0])][col] += 1; W[w_idx(*w[2])][col] += 1
            col += 1

            # S5 = (A00+A01)B11
            U[u_idx(*y00)][col] = 1; U[u_idx(*y01)][col] = 1
            V[v_idx(*z11)][col] = 1
            W[w_idx(*w[0])][col] -= 1; W[w_idx(*w[1])][col] += 1
            col += 1

            # S6 = (A10-A00)(B00+B01)
            U[u_idx(*y10)][col] = 1; U[u_idx(*y00)][col] = -1
            V[v_idx(*z00)][col] = 1; V[v_idx(*z01)][col] = 1
            W[w_idx(*w[3])][col] += 1
            col += 1

            # S7 = (A01-A11)(B10+B11)
            U[u_idx(*y01)][col] = 1; U[u_idx(*y11)][col] = -1
            V[v_idx(*z10)][col] = 1; V[v_idx(*z11)][col] = 1
            W[w_idx(*w[0])][col] += 1
            col += 1

        # Block 1: C[:, 0:2] from A[:,0:2]*B[0:2,0:2]
        strassen_block(
            [(0,0),(0,1),(1,0),(1,1)],
            [(0,0),(0,1),(1,0),(1,1)],
            [(0,0),(1,0),(0,1),(1,1)]
        )
        # Block 2: C[:, 0:2] from A[:,2:4]*B[2:4,0:2]
        strassen_block(
            [(0,2),(0,3),(1,2),(1,3)],
            [(2,0),(2,1),(3,0),(3,1)],
            [(0,0),(1,0),(0,1),(1,1)]
        )
        # Block 3: C[:, 2:4] from A[:,0:2]*B[0:2,2:4]
        strassen_block(
            [(0,0),(0,1),(1,0),(1,1)],
            [(0,2),(0,3),(1,2),(1,3)],
            [(2,0),(3,0),(2,1),(3,1)]
        )
        # Block 4: C[:, 2:4] from A[:,2:4]*B[2:4,2:4]
        strassen_block(
            [(0,2),(0,3),(1,2),(1,3)],
            [(2,2),(2,3),(3,2),(3,3)],
            [(2,0),(3,0),(2,1),(3,1)]
        )

        # Naive 2x4x1 blocks for C[:, 4]
        # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4] + A[0,2]*B[2,4] + A[0,3]*B[3,4]
        U[u_idx(0,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(0,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(0,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(0,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
        # C[1,4]
        U[u_idx(1,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1
        U[u_idx(1,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1
        U[u_idx(1,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1
        U[u_idx(1,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1

        return U, V, W, rank

    U0, V0, W0, rank0 = build_strassen_36()
    assert verify(U0, V0, W0), "Baseline verification failed"

    best_U, best_V, best_W, best_rank = U0, V0, W0, rank0

    def try_remove_columns(U, V, W, cols_to_remove):
        rank = len(U[0])
        if len(cols_to_remove) >= rank:
            return None
        keep = [r for r in range(rank) if r not in cols_to_remove]
        if not keep:
            return None
        new_rank = len(keep)
        U_new = [[U[i][r] for r in keep] for i in range(len(U))]
        V_new = [[V[i][r] for r in keep] for i in range(len(V))]
        W_new = [[W[i][r] for r in keep] for i in range(len(W))]
        if verify(U_new, V_new, W_new):
            return U_new, V_new, W_new, new_rank
        return None

    current_U, current_V, current_W, current_rank = U0, V0, W0, rank0

    # Try removing single columns repeatedly
    removed = True
    while removed and time.time() < _deadline - 40:
        removed = False
        for c in range(current_rank):
            if time.time() > _deadline - 40:
                break
            result = try_remove_columns(current_U, current_V, current_W, {c})
            if result:
                current_U, current_V, current_W, current_rank = result
                removed = True
                break

    if current_rank < best_rank:
        best_U, best_V, best_W, best_rank = current_U, current_V, current_W, current_rank

    # Try removing pairs of columns
    if time.time() < _deadline - 20:
        for c1 in range(current_rank):
            if time.time() > _deadline - 20:
                break
            for c2 in range(c1 + 1, current_rank):
                if time.time() > _deadline - 20:
                    break
                result = try_remove_columns(current_U, current_V, current_W, {c1, c2})
                if result:
                    current_U, current_V, current_W, current_rank = result
                    if current_rank < best_rank:
                        best_U, best_V, best_W, best_rank = current_U, current_V, current_W, current_rank
                    break
            if current_rank < best_rank:
                break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
