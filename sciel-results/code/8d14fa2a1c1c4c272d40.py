"""Matrix multiplication 5,5,5 tensor decomposition via block Strassen construction."""
import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 5, 5, 5
    t0 = time.time()
    deadline = t0 + 130.0

    def get_strassen_factors():
        """Return list of (U_vec, V_vec, W_vec) for Strassen 2x2x2.
        U_vec: [a11,a12,a21,a22], V_vec: [b11,b12,b21,b22], W_vec: [c11,c12,c21,c22]
        """
        return [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),   # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # M7
        ]

    def block_rank(a, b, c):
        """Rank of a×b×c block multiplication."""
        if a == 2 and b == 2 and c == 2:
            return 7
        return a * b * c

    def compute_partition_rank(partition):
        """Compute total rank for a given partition of 5."""
        total = 0
        for a in partition:
            for b in partition:
                for c in partition:
                    total += block_rank(a, b, c)
        return total

    def build_decomposition(partition, strassen_factors):
        """Build U, V, W for a given partition."""
        k = len(partition)
        offsets = [0]
        for s in partition:
            offsets.append(offsets[-1] + s)

        total_rank = compute_partition_rank(partition)
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]

        col = 0
        for i in range(k):
            for j in range(k):
                for kk in range(k):
                    ai = partition[i]
                    bj = partition[j]
                    ck = partition[kk]

                    if ai == 2 and bj == 2 and ck == 2:
                        for (uf, vf, wf) in strassen_factors:
                            local_U = [(offsets[i]+di)*m + (offsets[j]+dj) for di in range(2) for dj in range(2)]
                            local_V = [(offsets[j]+di)*p + (offsets[kk]+dj) for di in range(2) for dj in range(2)]
                            local_W = [(offsets[kk]+di)*n + (offsets[i]+dj) for di in range(2) for dj in range(2)]
                            for idx in range(4):
                                if uf[idx] != 0:
                                    U[local_U[idx]][col] += uf[idx]
                                if vf[idx] != 0:
                                    V[local_V[idx]][col] += vf[idx]
                                if wf[idx] != 0:
                                    W[local_W[idx]][col] += wf[idx]
                            col += 1
                    else:
                        for li in range(ai):
                            for lj in range(bj):
                                for lk in range(ck):
                                    U_idx = (offsets[i]+li)*m + (offsets[j]+lj)
                                    V_idx = (offsets[j]+lj)*p + (offsets[kk]+lk)
                                    W_idx = (offsets[kk]+lk)*n + (offsets[i]+li)
                                    U[U_idx][col] = 1.0
                                    V[V_idx][col] = 1.0
                                    W[W_idx][col] = 1.0
                                    col += 1

        assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
        return U, V, W, total_rank

    def verify_float(U, V, W):
        """Quick float verification."""
        r = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        U_r = Ua.reshape(n, m, r)
        V_r = Va.reshape(m, p, r)
        W_r = Wa.reshape(p, n, r)
        T_comp = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T_comp, np.ones((n, m, p)), atol=1e-10)

    # Generate all integer partitions of 5
    def get_partitions(total, max_part=None):
        if max_part is None:
            max_part = total
        if total == 0:
            yield []
            return
        for first in range(min(total, max_part), 0, -1):
            for rest in get_partitions(total - first, first):
                yield [first] + rest

    strassen_factors = get_strassen_factors()

    # Schoolbook fallback
    U_school = [[0.0] * 125 for _ in range(25)]
    V_school = [[0.0] * 125 for _ in range(25)]
    W_school = [[0.0] * 125 for _ in range(25)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1.0
                V_school[j*p+k][col] = 1.0
                W_school[k*n+i][col] = 1.0
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = 125

    # Try partitions sorted by rank
    partitions = list(get_partitions(5))
    partitions_with_rank = [(p, compute_partition_rank(p)) for p in partitions]
    partitions_with_rank.sort(key=lambda x: x[1])

    for part, rank in partitions_with_rank:
        if time.time() > deadline:
            break
        if rank >= best_rank:
            break

        U, V, W, actual_rank = build_decomposition(part, strassen_factors)

        if verify_float(U, V, W):
            best_result = {"U": U, "V": V, "W": W}
            best_rank = actual_rank
            break

    return best_result
# EVOLVE-BLOCK-END
