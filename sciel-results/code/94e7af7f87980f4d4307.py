import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 5, 5, 5
    
    # Block sizes: 5 = 2 + 2 + 1
    block_sizes = [2, 2, 1]
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    # Strassen factors for 2x2x2: each factor is (u_flat, v_flat, w_flat)
    # where flat index = row*2 + col for 2x2 matrices
    strassen_factors = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Calculate total rank
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni, mj, pk = block_sizes[i], block_sizes[j], block_sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    # Allocate
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni, mj, pk = block_sizes[i], block_sizes[j], block_sizes[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    for (uf, vf, wf) in strassen_factors:
                        for idx in range(4):
                            if uf[idx] != 0:
                                lr, lc = idx // 2, idx % 2
                                U[(offsets[i] + lr) * m + (offsets[j] + lc), col] += uf[idx]
                            if vf[idx] != 0:
                                lr, lc = idx // 2, idx % 2
                                V[(offsets[j] + lr) * p + (offsets[k] + lc), col] += vf[idx]
                            if wf[idx] != 0:
                                lr, lc = idx // 2, idx % 2
                                W[(offsets[k] + lc) * n + (offsets[i] + lr), col] += wf[idx]
                        col += 1
                else:
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U[(offsets[i] + li) * m + (offsets[j] + lj), col] = 1.0
                                V[(offsets[j] + lj) * p + (offsets[k] + lk), col] = 1.0
                                W[(offsets[k] + lk) * n + (offsets[i] + li), col] = 1.0
                                col += 1
    
    if time.time() > deadline - 15.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Attempt rank reduction via exact linear dependency check
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(int(round(x)), 1) for x in row] for row in U]
    V_frac = [[Fraction(int(round(x)), 1) for x in row] for row in V]
    W_frac = [[Fraction(int(round(x)), 1) for x in row] for row in W]
    
    # Check for zero columns
    zero_cols = []
    for c in range(total_rank):
        if all(U_frac[r][c] == 0 for r in range(n*m)) and \
           all(V_frac[r][c] == 0 for r in range(m*p)) and \
           all(W_frac[r][c] == 0 for r in range(p*n)):
            zero_cols.append(c)
    
    if zero_cols:
        keep = [c for c in range(total_rank) if c not in zero_cols]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        total_rank = len(keep)
    
    if time.time() > deadline - 10.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Check for identical columns (same U, V, W vectors)
    # Use a dictionary-based approach for efficiency
    col_sigs = {}
    dup_to_remove = set()
    for c in range(total_rank):
        sig = (
            tuple(U_frac[r][c] for r in range(n*m)),
            tuple(V_frac[r][c] for r in range(m*p)),
            tuple(W_frac[r][c] for r in range(p*n)),
        )
        if sig in col_sigs:
            dup_to_remove.add(c)
        else:
            col_sigs[sig] = c
    
    if dup_to_remove:
        keep = [c for c in range(total_rank) if c not in dup_to_remove]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        total_rank = len(keep)
    
    if time.time() > deadline - 5.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
