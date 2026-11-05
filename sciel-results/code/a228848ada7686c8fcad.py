import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = 5, 5, 5
    
    # Block sizes for partitioning 5 = 2 + 2 + 1
    block_sizes = [2, 2, 1]
    
    # Block offsets
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    # Strassen decomposition for 2x2x2
    # Returns list of (U_factors, V_factors, W_factors)
    # Each factor is a list of 4 values corresponding to indices (00, 01, 10, 11)
    def get_strassen_factors():
        # Standard Strassen factors for 2x2x2
        # Order: U (A part), V (B part), W (C part)
        # Indices: 00, 01, 10, 11
        factors = []
        # m1 = (A00+A11)(B00+B11) -> C00 += m1, C01 -= m1, C10 -= m1, C11 += m1
        factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, -1, -1, 1]))
        # m2 = (A10+A11)B00 -> C10 += m2, C11 += m2
        factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]))
        # m3 = A00(B10-B11) -> C00 += m3, C10 += m3
        factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [1, 0, 1, 0]))
        # m4 = A11(B01-B00) -> C11 += m4, C01 += m4
        factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [0, 1, 0, 1]))
        # m5 = (A00+A01)B11 -> C00 += m5, C01 += m5
        factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]))
        # m6 = (A10-A00)(B00+B01) -> C10 += m6, C11 -= m6
        factors.append(([0, -1, 1, 0], [1, 1, 0, 0], [0, 0, 1, -1]))
        # m7 = (A01-A11)(B10+B11) -> C01 += m7, C11 += m7
        factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 1]))
        return factors

    strassen_factors = get_strassen_factors()
    
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    for (uf, vf, wf) in strassen_factors:
                        # Map local indices (0..3) to global indices
                        # Local order: 00, 01, 10, 11
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
                            (offsets[k] + 0) * n + (offsets[i] + 1),
                            (offsets[k] + 1) * n + (offsets[i] + 0),
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
                    # Schoolbook for remaining blocks
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
