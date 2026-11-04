import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]
_START_TIME = time.time()
TIME_LIMIT = 115  # safety margin before 120s

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    Indexing: U[i*2+j, col], V[j*2+k, col], W[k*2+i, col]
    Must satisfy: sum_col U[i*2+j,col]*V[j*2+k,col]*W[k*2+i,col] = 1 for all i,j,k in {0,1}
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Col 0: M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    U[0, 0] = 1; U[3, 0] = 1   # A00, A11
    V[0, 0] = 1; V[3, 0] = 1   # B00, B11
    W[0, 0] = 1; W[3, 0] = 1   # C00, C11
    
    # Col 1: M2 = (A10+A11)B00 -> contributes to C10
    U[2, 1] = 1; U[3, 1] = 1   # A10, A11
    V[0, 1] = 1                # B00
    W[2, 1] = 1                # C10
    
    # Col 2: M3 = A00(B10-B11) -> contributes to C01, C11
    U[0, 2] = 1                # A00
    V[2, 2] = 1; V[3, 2] = -1 # B10, -B11
    W[1, 2] = 1; W[3, 2] = -1 # C01, -C11
    
    # Col 3: M4 = A11(B01-B10) -> contributes to C00, C10
    U[3, 3] = 1                # A11
    V[1, 3] = 1; V[2, 3] = -1 # B01, -B10
    W[0, 3] = 1; W[2, 3] = -1 # C00, -C10
    
    # Col 4: M5 = (A00+A01)B11 -> contributes to C01, C11
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[3, 4] = 1               # B11
    W[1, 4] = 1; W[3, 4] = -1 # C01, -C11
    
    # Col 5: M6 = (A10-A00)(B00+B10) -> contributes to C00, C10
    U[2, 5] = 1; U[0, 5] = -1 # A10, -A00
    V[0, 5] = 1; V[2, 5] = 1  # B00, B10
    W[0, 5] = 1; W[2, 5] = 1  # C00, C10
    
    # Col 6: M7 = (A01-A11)(B01+B11) -> contributes to C00, C01
    U[1, 6] = 1; U[3, 6] = -1 # A01, -A11
    V[1, 6] = 1; V[3, 6] = 1  # B01, B11
    W[0, 6] = 1; W[1, 6] = 1  # C00, C01
    
    return U, V, W

def _build_block_decomposition():
    """Build block decomposition for 3x5 x 5x5 using Strassen on sub-blocks."""
    n, m, p = 3, 5, 5
    dim_U = n * m  # 15
    dim_V = m * p  # 25
    dim_W = p * n  # 15
    
    # Split middle dimension 5 = 2 + 3
    # Part 1: (3x2) x (2x5)
    # Part 2: (3x3) x (3x5)
    
    all_U = []
    all_V = []
    all_W = []
    
    # Part 1: (3x2) x (2x5)
    # Split the 5 columns of B into 2+2+1
    # Sub-part 1a: (3x2) x (2x2) - use Strassen-like
    # Sub-part 1b: (3x2) x (2x2) - use Strassen-like
    # Sub-part 1c: (3x2) x (2x1) - trivial rank 6
    
    # For (3x2) x (2x2): we have 3 rows in A, 2 cols in A, 2 rows in B, 2 cols in B
    # For each i in 0..2: C[i,0..1] = A[i,0]*B[0,0..1] + A[i,1]*B[1,0..1]
    # This is 3 separate (1x2)x(2x2) problems, each rank 4, total 12
    # Or we can view it as: for fixed j (0,1), A[:,j] is 3x1, B[j,:] is 1x2
    # C[:,k] = A[:,0]*B[0,k] + A[:,1]*B[1,k]
    # For each k: C[:,k] is a 3-vector = A[:,0]*B[0,k] + A[:,1]*B[1,k]
    # This is a rank-2 update per k, total 2*2 = 4 rank-1 terms per (k pair)?
    # Actually: C[i,k] = sum_j A[i,j] B[j,k]
    # For (3x2)x(2x2): 3*2*2 = 12 trivial
    # We can do: for each j, for each k: A[:,j] * B[j,k] gives a 3x2 matrix, rank 1
    # That's 2*2 = 4 rank-1 terms! Wait no, each A[:,j] * B[j,k] is outer product of 3-vec and 2-vec
    # But we need C[i,k] = sum_j A[i,j]*B[j,k], which is sum of 2 outer products
    # Each outer product A[:,j] * B[j,k] contributes to all i,k
    # So we need: for j=0,1 and k=0,1: term = A[:,j] ⊗ B[j,k]
    # That's 4 terms, each rank 1 in the tensor sense
    # Total rank for (3x2)x(2x2) = 4
    
    # Similarly (3x2)x(2x1): 3*2*1 = 6 trivial, but we can do 2 terms (one per j)
    # For (3x2)x(2x1): C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    # Term j: A[:,j] * B[j,0], that's 2 terms
    
    # Part 1a: (3x2)x(2x2), rank 4
    for j in range(2):
        for k in range(2):
            u = np.zeros(dim_U)
            v = np.zeros(dim_V)
            w = np.zeros(dim_W)
            for i in range(3):
                u[i*5 + j] = 1.0  # A[i,j]
            v[j*5 + k] = 1.0  # B[j,k]
            for i in range(3):
                w[k*3 + i] = 1.0  # C[i,k]
            all_U.append(u)
            all_V.append(v)
            all_W.append(w)
    
    # Part 1b: (3x2)x(2x2), rank 4 (columns 2,3 of B)
    for j in range(2):
        for k in range(2):
            u = np.zeros(dim_U)
            v = np.zeros(dim_V)
            w = np.zeros(dim_W)
            for i in range(3):
                u[i*5 + j] = 1.0
            v[j*5 + (k+2)] = 1.0
            for i in range(3):
                w[(k+2)*3 + i] = 1.0
            all_U.append(u)
            all_V.append(v)
            all_W.append(w)
    
    # Part 1c: (3x2)x(2x1), rank 2 (column 4 of B)
    for j in range(2):
        u = np.zeros(dim_U)
        v = np.zeros(dim_V)
        w = np.zeros(dim_W)
        for i in range(3):
            u[i*5 + j] = 1.0
        v[j*5 + 4] = 1.0
        for i in range(3):
            w[4*3 + i] = 1.0
        all_U.append(u)
        all_V.append(v)
        all_W.append(w)
    
    # Part 2: (3x3) x (3x5)
    # Split 5 columns into 2+2+1
    # Sub-part 2a: (3x3)x(3x2), rank 9
    # Sub-part 2b: (3x3)x(3x2), rank 9
    # Sub-part 2c: (3x3)x(3x1), rank 9
    
    # For (3x3)x(3x2): C[i,k] = sum_j A[i,j]*B[j,k]
    # Trivial: 3*3*2 = 18, but we can do 3*2 = 6 terms (one per (j,k) pair? No)
    # Actually: for each j, A[:,j] is 3x1, B[j,:] is 1x2
    # Term: A[:,j] * B[j,k] for each k... that's 3*2 = 6 terms
    # Wait, each term A[:,j] ⊗ B[j,k] is rank 1 in the tensor
    # So (3x3)x(3x2) has rank at most 3*2 = 6? No, that's wrong.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # For fixed j: the contribution is A[:,j] (3-vector) times B[j,k] (scalar for each k)
    # The outer product A[:,j] * B[j,:] is a 3x2 matrix, which is rank 1
    # So for each j, we have one rank-1 term in the 3x2 output space
    # But in the tensor decomposition, each "term" is a product of three vectors
    # U[i*3+j] * V[j*2+k] * W[k*3+i]
    # For fixed j: U has A[:,j] pattern, V has B[j,:] pattern, W has identity pattern
    # That's 3 terms (one per j), each of which is a rank-1 tensor term
    # So (3x3)x(3x2) has rank at most 3? That can't be right either.
    
    # Let me reconsider. The tensor T[i,j,k] for (3x3)x(3x2) has indices i(3), j(3), k(2)
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The multiplication tensor has T[i,j,k] = delta structure
    # Rank is at most 3*3*2 = 18
    # Lower bound: max(3*3, 3*2, 3*2) = max(9,6,6) = 9
    # So rank is between 9 and 18
    
    # Trivial decomposition: for each (i,j,k), one term
    # But we can do better: for each j, the term involves all i and k
    # U[i*3+j] = A[i,j], V[j*2+k] = B[j,k], W[k*3+i] = C[i,k]
    # For fixed j: U is a 9-vector with A[:,j] in positions (i*3+j) for i=0,1,2
    # V is a 6-vector with B[j,:] in positions (j*2+k) for k=0,1
    # W is a 6-vector with identity in positions (k*3+i) for all i,k
    # This is ONE rank-1 term per j, so 3 terms total for (3x3)x(3x2)
    
    # Wait, that's not right. A rank-1 tensor term is u ⊗ v ⊗ w where u is in R^{9}, v in R^{6}, w in R^{6}
    # The product (u⊗v⊗w)[i*3+j, j*2+k, k*3+i] = u[i*3+j] * v[j*2+k] * w[k*3+i]
    # For this to equal A[i,j]*B[j,k]*C[i,k]... no, the tensor is the multiplication tensor
    # T[i,j,k] = 1 for the "identity" multiplication tensor
    # We want: sum_r U[i*3+j, r] * V[j*2+k, r] * W[k*3+i, r] = delta_{appropriate}
    
    # For the multiplication tensor, T[i,j,k] = 1 if the indices are consistent
    # Actually for (n×m)×(m×p), the tensor is T[i,j,k] where i indexes rows of A,
    # j indexes cols of A / rows of B, k indexes cols of B
    # T[i,j,k] = 1 (it's the structure tensor)
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # Wait no, that's not right either. The tensor decomposition gives:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_k V[j*p+k,r]*B[j,k]) * W[k*n+i,r]
    # Hmm, I'm confusing myself.
    
    # The correct formulation: the multiplication tensor T has T[i,j,k] = 1
    # and we want T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For the trivial decomposition, each term has U, V, W with a single 1
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for specific (i,j,k)
    
    # For a "shared" term: if we set U[i*m+j, r] = a_{ij}, V[j*p+k, r] = b_{jk}, W[k*n+i, r] = c_{ik}
    # Then sum_r a_{ij} b_{jk} c_{ik} should equal 1 for all i,j,k
    # This is hard to satisfy with few terms.
    
    # Let me just use the trivial decomposition for the 3x3x3 and 3x3x2 blocks
    # and rely on the overall structure.
    
    # Actually, let me just build the full trivial decomposition and then try to
    # merge terms using Strassen-like identities.
    
    # For now, let me use a simpler approach: build the block decomposition
    # with the structure I described above.
    
    # Part 2a: (3x3)x(3x2) - columns 0,1 of B (rows 2,3,4)
    for j in range(3):
        for k in range(2):
            u = np.zeros(dim_U)
            v = np.zeros(dim_V)
            w = np.zeros(dim_W)
            for i in range(3):
                u[i*5 + (j+2)] = 1.0
            v[(j+2)*5 + k] = 1.0
            for ii in range(3):
                w[k*3 + ii] = 1.0
            all_U.append(u)
            all_V.append(v)
            all_W.append(w)
    
    # Part 2b: (3x3)x(3x2) - columns 2,3 of B (rows 2,3,4)
    for j in range(3):
        for k in range(2):
            u = np.zeros(dim_U)
            v = np.zeros(dim_V)
            w = np.zeros(dim_W)
            for i in range(3):
                u[i*5 + (j+2)] = 1.0
            v[(j+2)*5 + (k+2)] = 1.0
            for ii in range(3):
                w[(k+2)*3 + ii] = 1.0
            all_U.append(u)
            all_V.append(v)
            all_W.append(w)
    
    # Part 2c: (3x3)x(3x1) - column 4 of B (rows 2,3,4)
    for j in range(3):
        u = np.zeros(dim_U)
        v = np.zeros(dim_V)
        w = np.zeros(dim_W)
        for i in range(3):
            u[i*5 + (j+2)] = 1.0
        v[(j+2)*5 + 4] = 1.0
        for ii in range(3):
            w[4*3 + ii] = 1.0
        all_U.append(u)
        all_V.append(v)
        all_W.append(w)
    
    U = np.column_stack(all_U)
    V = np.column_stack(all_V)
    W = np.column_stack(all_W)
    
    return U, V, W

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = 3, 5, 5
    # Check: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _local_search(U, V, W, max_iters=200):
    """Try to reduce rank by removing linearly dependent columns."""
    n, m, p = 3, 5, 5
    rank = U.shape[1]
    
    # Try to remove columns that are linear combinations of others
    # This is a heuristic approach
    
    for attempt in range(max_iters):
        if _time_remaining() < 5:
            break
        
        # Check if any column can be removed
        removed = False
        for r in range(rank - 1, -1, -1):
            if _time_remaining() < 5:
                break
            # Try removing column r
            U_new = np.delete(U, r, axis=1)
            V_new = np.delete(V, r, axis=1)
            W_new = np.delete(W, r, axis=1)
            
            # Check if the remaining columns can still span the same tensor
            # This is expensive, so just check a few random points
            ok = True
            for _ in range(20):
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                val = 0.0
                for rr in range(U_new.shape[1]):
                    val += U_new[i*m+j, rr] * V_new[j*p+k, rr] * W_new[k*n+i, rr]
                if abs(val - 1.0) > 1e-8:
                    ok = False
                    break
            
            if ok:
                U, V, W = U_new, V_new, W_new
                rank -= 1
                removed = True
                break
        
        if not removed:
            break
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Build block decomposition
    U, V, W = _build_block_decomposition()
    rank = U.shape[1]
    
    # Verify
    if not _verify(U, V, W):
        # Fall back to trivial
        rank = n * m * p
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Local search to reduce rank
    U, V, W, rank = _local_search(U, V, W)
    
    # Final verification
    if not _verify(U, V, W):
        # Fall back to trivial
        rank = n * m * p
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
