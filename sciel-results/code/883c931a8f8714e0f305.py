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
    
    # Additional refinement: try to find linear dependencies among columns
    # using the tensor structure. For each pair of columns (a, b), check if
    # column b can be expressed as a linear combination of a subset of other columns.
    # This is expensive, so we limit it to a heuristic approach.
    
    # A more effective approach: try different block partitions
    # 5 = 1 + 2 + 2 is the same as 2 + 2 + 1 by symmetry
    # 5 = 3 + 1 + 1: one 3x3x3 block (rank 27 or better) + 6 mixed blocks
    # 5 = 2 + 1 + 2: same as 2+2+1
    # 5 = 1 + 1 + 3: same as 3+1+1
    
    # Let's try 5 = 3 + 1 + 1 partition
    # 3x3x3 can use a known decomposition. The best known for 3x3x3 is rank 23 (Landsberg)
    # But we don't have that. Schoolbook is 27.
    # Actually, for 3x3x3, there's no known decomposition better than 27 that's simple.
    # Let's stick with 2+2+1.
    
    # Try to find if any column is a linear combination of others using SVD-like approach
    # on the concatenated tensor
    if time.time() < deadline - 3.0:
        # Build the full tensor T of shape (n*m, m*p, p*n) implicitly
        # T[a,b,c] = sum over rank of U[a,r] * V[b,r] * W[c,r]
        # We want to check if removing column r still preserves T
        
        # For efficiency, only try removing one column at a time and check
        # if the resulting decomposition still equals T
        # This is O(rank * n*m*m*p*p*n) which is too expensive for rank~117
        
        # Instead, use a simpler check: for each column r, see if
        # U[:,r] is in the span of {U[:,c] for c != r} AND
        # V[:,r] is in the span of {V[:,c] for c != r} AND  
        # W[:,r] is in the span of {W[:,c] for c != r}
        # This is necessary but not sufficient for the tensor equality.
        
        # Even simpler: check if column r is a scalar multiple of another column
        # in all three matrices simultaneously
        removed = []
        for r in range(total_rank):
            if r in removed:
                continue
            for s in range(r + 1, total_rank):
                if s in removed:
                    continue
                # Check if column s is a scalar multiple of column r in U, V, W
                # Find the ratio from the first nonzero entry in U
                ratio = None
                for a in range(n * m):
                    if U_frac[a][r] != 0:
                        if U_frac[a][s] != 0:
                            ratio = U_frac[a][s] / U_frac[a][r]
                        else:
                            ratio = Fraction(0, 1)
                        break
                    elif U_frac[a][s] != 0:
                        ratio = None
                        break
                if ratio is None:
                    continue
                # Verify ratio holds for all entries in U, V, W
                valid = True
                for a in range(n * m):
                    if U_frac[a][s] != U_frac[a][r] * ratio:
                        valid = False
                        break
                if not valid:
                    continue
                for b in range(m * p):
                    if V_frac[b][s] != V_frac[b][r] * ratio:
                        valid = False
                        break
                if not valid:
                    continue
                for c in range(p * n):
                    if W_frac[c][s] != W_frac[c][r] * ratio:
                        valid = False
                        break
                if valid:
                    removed.append(s)
        
        if removed:
            keep = [c for c in range(total_rank) if c not in removed]
            U = U[:, keep]
            V = V[:, keep]
            W = W[:, keep]
            total_rank = len(keep)
            # Rebuild frac matrices
            U_frac = [[Fraction(int(round(x)), 1) for x in row] for row in U]
            V_frac = [[Fraction(int(round(x)), 1) for x in row] for row in V]
            W_frac = [[Fraction(int(round(x)), 1) for x in row] for row in W]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
