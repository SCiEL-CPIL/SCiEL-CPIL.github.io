import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 4, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20

    def verify(U, V, W, n, m, p, tol=1e-9):
        # Vectorized verification
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        rank = U_np.shape[1]
        
        # For each (i,j,k), check sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :]
                    if abs(np.sum(prod) - 1.0) > tol:
                        return False
        return True

    def standard_term(U, V, W, i, j, k, col):
        U[i*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+i][col] = 1

    def strassen_block(U, V, W, i1, i2, j1, j2, k1, k2, col):
        # M1 = (a11+a22)(b11+b22)
        U[i1*m+j1][col] += 1; U[i2*m+j2][col] += 1
        V[j1*p+k1][col] += 1; V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += 1; W[k2*n+i2][col] += 1
        col += 1
        # M2 = (a21+a22)b11
        U[i2*m+j1][col] += 1; U[i2*m+j2][col] += 1
        V[j1*p+k1][col] += 1
        W[k1*n+i2][col] += 1; W[k2*n+i2][col] += -1
        col += 1
        # M3 = a11(b12-b22)
        U[i1*m+j1][col] += 1
        V[j1*p+k2][col] += 1; V[j2*p+k2][col] += -1
        W[k2*n+i1][col] += 1; W[k2*n+i2][col] += 1
        col += 1
        # M4 = a22(b21-b11)
        U[i2*m+j2][col] += 1
        V[j2*p+k1][col] += 1; V[j1*p+k1][col] += -1
        W[k1*n+i1][col] += 1; W[k1*n+i2][col] += 1
        col += 1
        # M5 = (a11+a12)b22
        U[i1*m+j1][col] += 1; U[i1*m+j2][col] += 1
        V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += -1; W[k2*n+i1][col] += 1
        col += 1
        # M6 = (a21-a11)(b11+b12)
        U[i2*m+j1][col] += 1; U[i1*m+j1][col] += -1
        V[j1*p+k1][col] += 1; V[j1*p+k2][col] += 1
        W[k2*n+i2][col] += 1
        col += 1
        # M7 = (a12-a22)(b21+b22)
        U[i1*m+j2][col] += 1; U[i2*m+j2][col] += -1
        V[j2*p+k1][col] += 1; V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += 1
        col += 1
        return col

    def build_decomposition():
        U = [[0] * 92 for _ in range(nm)]
        V = [[0] * 92 for _ in range(mp)]
        W = [[0] * 92 for _ in range(pn)]
        
        blocks = []
        for i_base in [0, 2]:
            for j_base in [0, 2]:
                for k_base in [0, 2]:
                    blocks.append((i_base, i_base+1, j_base, j_base+1, k_base, k_base+1))
        
        skip_set = set()
        for i1, i2, j1, j2, k1, k2 in blocks:
            for i in [i1, i2]:
                for j in [j1, j2]:
                    for k in [k1, k2]:
                        skip_set.add((i, j, k))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in skip_set:
                        standard_term(U, V, W, i, j, k, col)
                        col += 1
        
        for i1, i2, j1, j2, k1, k2 in blocks:
            col = strassen_block(U, V, W, i1, i2, j1, j2, k1, k2, col)
        
        return U, V, W, col

    def try_alternative_blocks():
        """Try different block configurations that might give lower rank."""
        best_rank = 92
        best_UVW = None
        
        # Alternative: use Strassen on different 2x2x2 sub-blocks
        # The 4x5x5 tensor has indices i:0-3, j:0-4, k:0-4
        # We can place 2x2x2 blocks at various positions
        
        # Try: shift the blocks to cover different regions
        for i_offset in [0, 1]:
            for j_offset in [0, 1]:
                for k_offset in [0, 1]:
                    blocks = []
                    for i_base in [i_offset, i_offset+2]:
                        if i_base + 1 > 3:
                            continue
                        for j_base in [j_offset, j_offset+2]:
                            if j_base + 1 > 4:
                                continue
                            for k_base in [k_offset, k_offset+2]:
                                if k_base + 1 > 4:
                                    continue
                                blocks.append((i_base, i_base+1, j_base, j_base+1, k_base, k_base+1))
                    
                    if len(blocks) < 8:
                        continue
                    
                    skip_set = set()
                    for i1, i2, j1, j2, k1, k2 in blocks:
                        for i in [i1, i2]:
                            for j in [j1, j2]:
                                for k in [k1, k2]:
                                    skip_set.add((i, j, k))
                    
                    n_standard = 100 - len(skip_set)
                    n_strassen = len(blocks) * 7
                    total_rank = n_standard + n_strassen
                    
                    if total_rank < best_rank:
                        U = [[0] * total_rank for _ in range(nm)]
                        V = [[0] * total_rank for _ in range(mp)]
                        W = [[0] * total_rank for _ in range(pn)]
                        
                        col = 0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    if (i, j, k) not in skip_set:
                                        standard_term(U, V, W, i, j, k, col)
                                        col += 1
                        
                        for i1, i2, j1, j2, k1, k2 in blocks:
                            col = strassen_block(U, V, W, i1, i2, j1, j2, k1, k2, col)
                        
                        if verify(U, V, W, n, m, p):
                            best_rank = total_rank
                            best_UVW = (U, V, W)
        
        return best_rank, best_UVW

    # Build the base decomposition
    U, V, W, rank = build_decomposition()
    
    if not verify(U, V, W, n, m, p):
        # Fallback to standard
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    standard_term(U, V, W, i, j, k, col)
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try alternative block configurations
    if time.time() < deadline - 30:
        alt_rank, alt_UVW = try_alternative_blocks()
        if alt_UVW is not None and alt_rank < rank:
            U, V, W = alt_UVW
            rank = alt_rank
    
    # Check for zero or linearly dependent columns
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    current_rank = rank
    
    # Remove zero columns
    keep = []
    for r in range(current_rank):
        if not (np.all(U_np[:, r] == 0) and np.all(V_np[:, r] == 0) and np.all(W_np[:, r] == 0)):
            keep.append(r)
    
    if len(keep) < current_rank:
        U_np = U_np[:, keep]
        V_np = V_np[:, keep]
        W_np = W_np[:, keep]
        current_rank = len(keep)
    
    # Try to find linearly dependent columns using SVD on the combined structure
    # A column r is redundant if it can be expressed as a linear combination of others
    # in the tensor decomposition sense. This is complex, so we'll just check for
    # exact duplicates or simple dependencies.
    
    def to_list(mat):
        return [[int(round(mat[i][r])) if abs(mat[i][r] - round(mat[i][r])) < 1e-9 else float(mat[i][r])
                 for r in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    U_final = to_list(U_np)
    V_final = to_list(V_np)
    W_final = to_list(W_np)
    
    return {"U": U_final, "V": V_final, "W": W_final}
