import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 5, 5, 5

    def get_strassen_factors():
        factors = []
        factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return factors

    strassen_factors = get_strassen_factors()

    def build_solution(block_sizes):
        """Build solution for given block partition."""
        offsets = [0]
        for s in block_sizes:
            offsets.append(offsets[-1] + s)
        nb = len(block_sizes)

        total_rank = 0
        for i in range(nb):
            for j in range(nb):
                for k in range(nb):
                    ni, mj, pk = block_sizes[i], block_sizes[j], block_sizes[k]
                    if ni == 2 and mj == 2 and pk == 2:
                        total_rank += 7
                    else:
                        total_rank += ni * mj * pk

        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))

        col = 0
        for i in range(nb):
            for j in range(nb):
                for k in range(nb):
                    ni, mj, pk = block_sizes[i], block_sizes[j], block_sizes[k]
                    if ni == 2 and mj == 2 and pk == 2:
                        for (uf, vf, wf) in strassen_factors:
                            lU = [(offsets[i]+0)*m+(offsets[j]+0), (offsets[i]+0)*m+(offsets[j]+1),
                                   (offsets[i]+1)*m+(offsets[j]+0), (offsets[i]+1)*m+(offsets[j]+1)]
                            lV = [(offsets[j]+0)*p+(offsets[k]+0), (offsets[j]+0)*p+(offsets[k]+1),
                                   (offsets[j]+1)*p+(offsets[k]+0), (offsets[j]+1)*p+(offsets[k]+1)]
                            lW = [(offsets[k]+0)*n+(offsets[i]+0), (offsets[k]+1)*n+(offsets[i]+0),
                                   (offsets[k]+0)*n+(offsets[i]+1), (offsets[k]+1)*n+(offsets[i]+1)]
                            for idx in range(4):
                                if uf[idx]: U[lU[idx], col] += uf[idx]
                                if vf[idx]: V[lV[idx], col] += vf[idx]
                                if wf[idx]: W[lW[idx], col] += wf[idx]
                            col += 1
                    else:
                        for li in range(ni):
                            for lj in range(mj):
                                for lk in range(pk):
                                    U[(offsets[i]+li)*m+(offsets[j]+lj), col] = 1.0
                                    V[(offsets[j]+lj)*p+(offsets[k]+lk), col] = 1.0
                                    W[(offsets[k]+lk)*n+(offsets[i]+li), col] = 1.0
                                    col += 1

        return U, V, W, total_rank

    # Try multiple partitions
    partitions = [
        [2, 2, 1],
        [2, 1, 2],
        [1, 2, 2],
        [3, 2],
        [2, 3],
        [4, 1],
        [1, 4],
    ]

    best = None
    for bs in partitions:
        if sum(bs) != 5:
            continue
        U, V, W, rank = build_solution(bs)
        if best is None or rank < best[3]:
            best = (U, V, W, rank)

    if best is None:
        # Fallback to schoolbook
        U = np.zeros((25, 125))
        V = np.zeros((25, 125))
        W = np.zeros((25, 125))
        col = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    U[i*5+j, col] = 1.0
                    V[j*5+k, col] = 1.0
                    W[k*5+i, col] = 1.0
                    col += 1
        best = (U, V, W, 125)

    U, V, W, _ = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
