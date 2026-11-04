import time
import numpy as np

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    def build_decomposition(tiling):
        """
        tiling: list of block specs. Each spec is either
          ('strassen', j_base, k_base)  -> 2x2 Strassen block at A cols j_base,j_base+1; B rows k_base,k_base+1
          ('naive_2112', j_base, k_base) -> A col j_base (2x1) * B row k_base (1x2) -> 4 naive terms
        k_base must be even and k_base+1 < p.
        j_base must be in 0..m-2 for strassen, 0..m-1 for naive.
        All j_base and k_base pairs must cover the full 5x6 grid exactly once in terms of (j,k) cells.
        """
        rank_cap = 100
        U = [[0] * rank_cap for _ in range(n * m)]
        V = [[0] * rank_cap for _ in range(m * p)]
        W = [[0] * rank_cap for _ in range(p * n)]
        
        col = 0
        
        def add_strassen_block(i_base, j_base, k_base):
            nonlocal col
            # m0: (A00+A11)(B00+B11)
            U[i_base * m + j_base][col] += 1
            U[(i_base+1) * m + (j_base+1)][col] += 1
            V[j_base * p + k_base][col] += 1
            V[(j_base+1) * p + (k_base+1)][col] += 1
            W[k_base * n + i_base][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] += 1
            col += 1
            
            # m1: (A10+A11)B00
            U[(i_base+1) * m + j_base][col] += 1
            U[(i_base+1) * m + (j_base+1)][col] += 1
            V[j_base * p + k_base][col] += 1
            W[k_base * n + (i_base+1)][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] -= 1
            col += 1
            
            # m2: A00(B01-B11)
            U[i_base * m + j_base][col] += 1
            V[j_base * p + (k_base+1)][col] += 1
            V[(j_base+1) * p + (k_base+1)][col] -= 1
            W[(k_base+1) * n + i_base][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] += 1
            col += 1
            
            # m3: A11(B10-B00)
            U[(i_base+1) * m + (j_base+1)][col] += 1
            V[(j_base+1) * p + k_base][col] += 1
            V[j_base * p + k_base][col] -= 1
            W[k_base * n + i_base][col] += 1
            W[k_base * n + (i_base+1)][col] += 1
            col += 1
            
            # m4: (A00+A01)B11
            U[i_base * m + j_base][col] += 1
            U[i_base * m + (j_base+1)][col] += 1
            V[(j_base+1) * p + (k_base+1)][col] += 1
            W[k_base * n + i_base][col] -= 1
            W[(k_base+1) * n + i_base][col] += 1
            col += 1
            
            # m5: (A10-A00)(B00+B01)
            U[(i_base+1) * m + j_base][col] += 1
            U[i_base * m + j_base][col] -= 1
            V[j_base * p + k_base][col] += 1
            V[j_base * p + (k_base+1)][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] += 1
            col += 1
            
            # m6: (A01-A11)(B10+B11)
            U[i_base * m + (j_base+1)][col] += 1
            U[(i_base+1) * m + (j_base+1)][col] -= 1
            V[(j_base+1) * p + k_base][col] += 1
            V[(j_base+1) * p + (k_base+1)][col] += 1
            W[k_base * n + i_base][col] += 1
            col += 1
        
        def add_naive_2112_block(i_base, j_base, k_base):
            nonlocal col
            U[i_base * m + j_base][col] += 1
            V[j_base * p + k_base][col] += 1
            W[k_base * n + i_base][col] += 1
            col += 1
            
            U[i_base * m + j_base][col] += 1
            V[j_base * p + (k_base+1)][col] += 1
            W[(k_base+1) * n + i_base][col] += 1
            col += 1
            
            U[(i_base+1) * m + j_base][col] += 1
            V[j_base * p + k_base][col] += 1
            W[k_base * n + (i_base+1)][col] += 1
            col += 1
            
            U[(i_base+1) * m + j_base][col] += 1
            V[j_base * p + (k_base+1)][col] += 1
            W[(k_base+1) * n + (i_base+1)][col] += 1
            col += 1
        
        for spec in tiling:
            if spec[0] == 'strassen':
                add_strassen_block(0, spec[1], spec[2])
            elif spec[0] == 'naive_2112':
                add_naive_2112_block(0, spec[1], spec[2])
        
        return U[:n*m], V[:m*p], W[:p*n], col
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def nullspace_reduce(U, V, W, col):
        """
        Given factor matrices with `col` columns, compute the coefficient matrix M
        (size n*m*p x col) where M[tensor_idx, r] = U[i,j,r]*V[j,k,r]*W[k,i,r].
        Find a basis for the column space of M (rank r), then express each original
        column as a combination of the basis. Return new factor matrices with r columns.
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        T_size = n * m * p
        M = np.zeros((T_size, col))
        for r in range(col):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        M[idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Find rank and column basis
        # Use SVD for numerical stability
        try:
            U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        
        tol = 1e-10
        r = int(np.sum(s_vals > tol))
        if r == 0:
            return None
        if r >= col:
            return None  # no reduction possible
        
        # Basis for column space: first r left singular vectors
        basis = U_svd[:, :r]  # T_size x r
        
        # Express each column of M as basis @ c_r
        # c_r = basis^T @ M[:, r]  (least squares)
        coeffs = np.zeros((r, col))
        for r_idx in range(col):
            coeffs[:, r_idx] = basis.T @ M[:, r_idx]
        
        # New factor matrices: each column is a linear combination of original columns
        U_new = U @ coeffs  # (n*m) x r
        V_new = V @ coeffs  # (m*p) x r
        W_new = W @ coeffs  # (p*n) x r
        
        # Verify
        if not verify(U_new, V_new, W_new, r):
            # Try a slightly looser tolerance or fall back
            return None
        
        return U_new, V_new, W_new, r
    
    def to_clean_list(arr):
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                half = round(val * 2) / 2
                if abs(val - half) < 1e-8:
                    clean_row.append(half if half != int(half) else int(half))
                else:
                    clean_row.append(val)
            result.append(clean_row)
        return result
    
    # Define several tilings to try.
    # The 5x6 grid (j in 0..4, k in 0..5) must be tiled by:
    # - Strassen blocks: cover 2x2 cells at (j_base, j_base+1) x (k_base, k_base+1), k_base even
    # - Naive 2112 blocks: cover 1x2 cells at j_base x (k_base, k_base+1), k_base even
    # 
    # Strategy: pair up j's into 2x2 Strassen blocks where possible, use naive for leftover columns.
    # j has 5 columns: 0,1,2,3,4. k has 6 rows: 0..5, paired as (0,1),(2,3),(4,5).
    # 
    # Tiling A: Strassen at j=0,1; j=2,3; naive at j=4. (This is the current approach)
    # Tiling B: Strassen at j=1,2; j=3,4; naive at j=0.
    # Tiling C: Strassen at j=0,1; j=3,4; naive at j=2.
    # Tiling D: Strassen at j=0,1; naive at j=2; Strassen at j=3,4. (same as C)
    # Tiling E: Strassen at j=1,2; naive at j=3; naive at j=4. (only one strassen block)
    # Tiling F: Strassen at j=0,1; Strassen at j=2,3; naive at j=4. (same as A)
    # 
    # Let's try A, B, C, and a few others.
    
    tilings = [
        # Tiling A: Strassen(0,0), Strassen(0,2), Strassen(0,4), Strassen(2,0), Strassen(2,2), Strassen(2,4), Naive(4,0), Naive(4,2), Naive(4,4)
        [('strassen', 0, 0), ('strassen', 0, 2), ('strassen', 0, 4),
         ('strassen', 2, 0), ('strassen', 2, 2), ('strassen', 2, 4),
         ('naive_2112', 4, 0), ('naive_2112', 4, 2), ('naive_2112', 4, 4)],
        
        # Tiling B: Strassen at j=1,2 and j=3,4; naive at j=0
        [('strassen', 1, 0), ('strassen', 1, 2), ('strassen', 1, 4),
         ('strassen', 3, 0), ('strassen', 3, 2), ('strassen', 3, 4),
         ('naive_2112', 0, 0), ('naive_2112', 0, 2), ('naive_2112', 0, 4)],
        
        # Tiling C: Strassen at j=0,1 and j=3,4; naive at j=2
        [('strassen', 0, 0), ('strassen', 0, 2), ('strassen', 0, 4),
         ('strassen', 3, 0), ('strassen', 3, 2), ('strassen', 3, 4),
         ('naive_2112', 2, 0), ('naive_2112', 2, 2), ('naive_2112', 2, 4)],
        
        # Tiling D: Strassen at j=1,2 and j=3,4; naive at j=0 (same as B)
        # Tiling E: One Strassen block at j=0,1; naive at j=2,3,4
        [('strassen', 0, 0), ('strassen', 0, 2), ('strassen', 0, 4),
         ('naive_2112', 2, 0), ('naive_2112', 2, 2), ('naive_2112', 2, 4),
         ('naive_2112', 3, 0), ('naive_2112', 3, 2), ('naive_2112', 3, 4),
         ('naive_2112', 4, 0), ('naive_2112', 4, 2), ('naive_2112', 4, 4)],
        
        # Tiling F: Strassen at j=0,1; naive at j=2; Strassen at j=3,4
        [('strassen', 0, 0), ('strassen', 0, 2), ('strassen', 0, 4),
         ('naive_2112', 2, 0), ('naive_2112', 2, 2), ('naive_2112', 2, 4),
         ('strassen', 3, 0), ('strassen', 3, 2), ('strassen', 3, 4)],
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for tiling in tilings:
        if time.time() > START_TIME + TIME_LIMIT - 10.0:
            break
        
        U, V, W, col = build_decomposition(tiling)
        
        # Try null-space reduction
        reduced = nullspace_reduce(U, V, W, col)
        
        if reduced is not None:
            U_new, V_new, W_new, r = reduced
            if r < best_rank:
                best_rank = r
                best_result = (U_new, V_new, W_new, r)
        else:
            # Fall back to original
            if col < best_rank:
                best_rank = col
                best_result = (np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float), col)
    
    if best_result is None:
        # Last resort: pure naive
        U = [[0] * (n*m*p) for _ in range(n * m)]
        V = [[0] * (n*m*p) for _ in range(m * p)]
        W = [[0] * (n*m*p) for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": to_clean_list(np.array(U, dtype=float)), "V": to_clean_list(np.array(V, dtype=float)), "W": to_clean_list(np.array(W, dtype=float))}
    
    U_new, V_new, W_new, r = best_result
    return {"U": to_clean_list(U_new), "V": to_clean_list(V_new), "W": to_clean_list(W_new)}
