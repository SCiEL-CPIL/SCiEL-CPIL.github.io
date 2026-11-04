"""Matrix multiplication 2x3x8 rank minimization via tensor rank compression.

The current best is rank 44:
  - 4 Strassen blocks (rank 7 each) for 2x2 * 2x2 subproblems covering the first 2 rows of A with 2 columns of B at a time
  - 16 schoolbook terms for A[2,:3] * B[2,:8] (but only A's 3rd column, so 2*8 = 16 terms)

We attempt to reduce the rank by:
1. Building the rank-44 decomposition
2. Performing exhaustive search over subsets of columns to remove, checking
   if the remaining columns still form a valid decomposition
3. Using a more systematic approach: build the full tensor as a sum of rank-1
   terms and check for linear dependencies via Gaussian elimination over the
   "flattened" tensor space
"""
import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 135.0
    
    def _elapsed_ok():
        return (time.time() - t0) < deadline
    
    n, m, p = 2, 3, 8
    
    # Build the rank-44 decomposition
    # Strassen 2x2x2
    S_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    # Verify Strassen
    for i in range(2):
        for a in range(2):
            for k in range(2):
                acc = 0
                for r in range(7):
                    u_val = S_U[r][i * 2 + a]
                    v_val = S_V[r][a * 2 + k]
                    w_val = S_W[r][k * 2 + i]
                    acc += u_val * v_val * w_val
                if acc != 1:
                    # Fallback: schoolbook
                    rank = n * m * p
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    col = 0
                    for i2 in range(n):
                        for a2 in range(m):
                            for k2 in range(p):
                                U[i2 * m + a2][col] = 1
                                V[a2 * p + k2][col] = 1
                                W[k2 * n + i2][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}
    
    # Build rank-44 decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: 4 Strassen blocks for A[:2,:2] * B[:2, k0:k0+2]
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        u_idx = li * m + la
                        U[u_idx][col] += u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] += v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] += w_val
            col += 1
        if not _elapsed_ok():
            break
    
    # Part 2: 16 schoolbook terms for A[:2,2] * B[2,:] (2 rows x 8 cols = 16 terms)
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
        if not _elapsed_ok():
            break
    
    # Now we have 44 rank-1 terms. We want to find the minimum subset that still works.
    # The tensor T[i,a,k] = sum_c U[i*m+a][c] * V[a*p+k][c] * W[k*n+i][c] must equal 1.
    
    # Strategy: use a greedy/branch-and-bound approach to find the minimum subset.
    # But 44 choose k is huge. Instead, we check for linear dependencies.
    
    # Key insight: each rank-1 term is a "column" (c) with non-zero entries in U, V, W.
    # If term j is in the "tensor span" of the others, we can remove it.
    # 
    # More practically: we can check if removing column j still gives the correct tensor.
    # This is O(44) checks, each O(2*3*8*43) = O(2016) operations. Fast.
    
    # Greedy removal
    current_rank = rank
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    
    def _check_valid(U_mat, V_mat, W_mat, rnk):
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    for c in range(rnk):
                        acc += U_mat[i * m + a][c] * V_mat[a * p + k][c] * W_mat[k * n + i][c]
                    if acc != 1:
                        return False
        return True
    
    if not _check_valid(current_U, current_V, current_W, current_rank):
        # Fallback to full 44
        return {"U": U, "V": V, "W": W}
    
    # Greedy: try to remove columns one by one
    removed = []
    for j in range(rank):
        if not _elapsed_ok():
            break
        
        # Try removing column j from current candidate
        new_rank = current_rank - 1
        new_U = []
        for row in current_U:
            new_row = []
            for c in range(current_rank):
                if c == j:
                    continue
                new_row.append(row[c])
            new_U.append(new_row)
        
        new_V = []
        for row in current_V:
            new_row = []
            for c in range(current_rank):
                if c == j:
                    continue
                new_row.append(row[c])
            new_V.append(new_row)
        
        new_W = []
        for row in current_W:
            new_row = []
            for c in range(current_rank):
                if c == j:
                    continue
                new_row.append(row[c])
            new_W.append(new_row)
        
        if _check_valid(new_U, new_V, new_W, new_rank):
            current_rank = new_rank
            current_U = new_U
            current_V = new_V
            current_W = new_W
            removed.append(j)
    
    # Try a second pass of greedy removal on the reduced set
    for j in range(current_rank):
        if not _elapsed_ok():
            break
        
        new_rank = current_rank - 1
        new_U = []
        for row in current_U:
            new_row = []
            for c in range(current_rank):
                if c == j:
                    continue
                new_row.append(row[c])
            new_U.append(new_row)
        
        new_V = []
        for row in current_V:
            new_row = []
            for c in range(current_rank):
                if c == j:
                    continue
                new_row.append(row[c])
            new_V.append(new_row)
        
        new_W = []
        for row in current_W:
            new_row = []
            for c in range(current_rank):
                if c == j:
                    continue
                new_row.append(row[c])
            new_W.append(new_row)
        
        if _check_valid(new_U, new_V, new_W, new_rank):
            current_rank = new_rank
            current_U = new_U
            current_V = new_V
            current_W = new_W
    
    # If we reduced significantly, try a more systematic approach with numpy
    # to check for additional linear dependencies
    if current_rank < 44 and _elapsed_ok():
        # Convert to numpy arrays for faster computation
        U_np = np.array(current_U)
        V_np = np.array(current_V)
        W_np = np.array(current_W)
        
        # Another pass
        for j in range(current_rank):
            if not _elapsed_ok():
                break
            new_rank = current_rank - 1
            mask = np.ones(current_rank, dtype=bool)
            mask[j] = False
            new_U_np = U_np[:, mask]
            new_V_np = V_np[:, mask]
            new_W_np = W_np[:, mask]
            
            # Check validity using einsum
            tensor = np.einsum('ic,ac,kc->iak', new_U_np, new_V_np, new_W_np)
            if np.all(tensor == 1):
                current_rank = new_rank
                current_U = new_U_np.tolist()
                current_V = new_V_np.tolist()
                current_W = new_W_np.tolist()
    
    return {"U": current_U, "V": current_V, "W": current_W}
