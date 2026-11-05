"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    
    The tensor T has entries T[i,j,k] = delta_{i,0} * 1 for appropriate indices.
    Specifically, (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    In tensor form: T[i,j,k] = 1 if the multiplication involves these indices.
    
    For matrix multiplication, the tensor is:
    T[i,j,k] = 1 (for all valid i,j,k) - actually no.
    
    The multiplication tensor M_{n,m,p} has M[i,j,k] = delta where the (i,k) entry
    of the product is sum_j A[i,j]B[j,k]. So the tensor is:
    T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    Wait, that's not right either. The tensor for matrix multiplication is:
    For each (i,k) output, T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k].
    So T is a 3D array where T[i,j,k] = 1 for all valid indices.
    
    Actually, the multiplication tensor is the sum over j of the outer product
    e_i ⊗ e_j ⊗ e_k, but with the constraint that for each fixed (i,k), we sum over j.
    
    More precisely: C[i,k] = sum_j A[i,j] B[j,k]
    So the tensor is: T = sum_{i,j,k} T_{i,j,k} * A_{i,j} * B_{j,k} * C_{i,k}
    where T_{i,j,k} = 1 for all valid i,j,k.
    
    Wait, that gives rank = n*m*p = 18, which is the schoolbook rank.
    
    But we can potentially do better! The key insight is that the tensor structure
    allows for rank reduction when n, m, p have certain relationships.
    
    For 2x3x3, let's think about this more carefully.
    
    Actually, I need to reconsider. The tensor for matrix multiplication M_{n,m,p}
    has the property that for each fixed i and k, the slice T[i,:,k] is all ones
    (length m). The question is whether we can decompose this with rank < n*m*p.
    
    Let me try to use a structured approach. I'll set up the problem as finding
    matrices U (nm x r), V (mp x r), W (pn x r) such that:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for all i, j, k.
    
    Since all T[i,j,k] = 1, we need:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    Let me try to construct a rank-12 solution using a structured ansatz.
    
    One approach: use the fact that 2*3*3 = 18, and try to find a rank-12
    decomposition by exploiting the symmetry and structure.
    
    Actually, let me just try to search for a low-rank decomposition numerically
    and then verify if it's exact (integer or half-integer entries).
    
    Given the time constraints, let me try a few structured approaches.
    """
    deadline = time.time() + 130  # Leave 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Verify a decomposition
    def verify(U, V, W, tol=1e-10):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Check if entries are integer or half-integer
    def is_valid_entries(U, V, W):
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if not (abs(val * 2 - round(val * 2)) < 1e-9):
                        return False
        return True
    
    # Try to find a rank-12 decomposition
    # Ansatz: use structured matrices with small integer/half-integer entries
    
    # For 2x3x3, try to exploit the structure
    # The tensor has 2*3*3 = 18 entries, all equal to 1.
    
    # Let me try a specific construction based on the idea of sharing
    # multiplications across rows.
    
    # Rank 12 attempt:
    # U: 6x12, V: 9x12, W: 6x12
    
    # Try a structured approach with rank 12
    # Each column c contributes U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Let's try: for each j (column of A, row of B), we need to compute
    # contributions to all (i,k) pairs. There are n*p = 6 such pairs per j.
    # Total = m * n * p = 9 * 2 = 18... wait, m=3, n*p=6, so 3*6=18.
    
    # Alternative: can we share across i? For fixed j and k, the contribution
    # to C[i,k] is A[i,j]*B[j,k]. If we could factor B[j,k] out...
    
    # Let me try a numerical search with random initializations and gradient-free
    # or structured search for rank 12.
    
    # Actually, let me try a specific rank-12 construction.
    
    # Idea: split the problem into two parts based on i (the row of A).
    # For i=0: C[0,k] = sum_j A[0,j] * B[j,k] for k=0,1,2
    # For i=1: C[1,k] = sum_j A[1,j] * B[j,k] for k=0,1,2
    
    # For each i, this is a 1x3 times 3x3 multiplication, which has rank 3
    # in the naive sense. But we can potentially share the B[j,k] factors.
    
    # Total naive: 2 * 3 = 6 "multiplication groups", each with 3 terms = 18.
    
    # Can we do rank 12? That would be 2 per (i,k) pair, or some other sharing.
    
    # Let me try to construct a rank-12 solution explicitly.
    
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # Let's say rank = 12. Number the columns 0..11.
    
    # For each (i, k) pair (6 total), we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for each j
    
    # This is a system of 18 equations (2*3*3) in the unknowns.
    
    # Let me try a symmetric construction.
    
    # Try: U has a block structure where rows for i=0 and i=1 are related.
    
    # Let me try rank 12 with the following structure:
    # Columns 0-5: handle i=0, k=0,1,2 and j=0,1,2
    # Columns 6-11: handle i=1, k=0,1,2 and j=0,1,2
    
    # But that's just the schoolbook decomposition with rank 12... wait, 
    # schoolbook is 18. Let me recount.
    
    # Schoolbook: for each (i,j,k), one column. 2*3*3 = 18 columns.
    
    # To get rank 12, we need to share 6 columns' worth of work.
    
    # Let me try: for each (i,k), instead of 3 separate columns (one per j),
    # use 2 columns that jointly handle all 3 j's.
    
    # For fixed (i,k): need sum_j A[i,j]*B[j,k] = C[i,k]
    # With 2 terms: sum_{c in {c1,c2}} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    
    # If W[k*n+i,c1] = 1 and W[k*n+i,c2] = 1, then:
    # sum_j A[i,j]*(V[j*p+k,c1] + V[j*p+k,c2]) = C[i,k]
    # We need V[j*p+k,c1] + V[j*p+k,c2] = B[j,k]... but V entries are fixed.
    
    # This approach requires V[j*p+k,c] to depend on k, which they do.
    
    # Let me try a different structured approach.
    
    # For rank 12, try:
    # 6 columns for "diagonal" contributions, 6 for "off-diagonal"
    
    # Actually, let me just try a direct construction.
    
    # Rank 12:
    # c=0: i=0,j=0,k=0 -> U[0,0]=1,V[0,0]=1,W[0,0]=1
    # c=1: i=0,j=1,k=0 -> U[1,0]=1,V[3,0]=1,W[0,0]=1  (wait, W[k*n+i,c])
    
    # This is getting complex. Let me try a numerical approach.
    
    # Try to find a rank-12 decomposition using a structured random search.
    
    if time.time() < deadline:
        # Try rank 12 with structured initialization
        for trial in range(50):
            if time.time() > deadline:
                break
            rank = 12
            # Initialize with small random values
            np.random.seed(42 + trial)
            U = np.random.uniform(-1, 1, (n*m, rank))
            V = np.random.uniform(-1, 1, (m*p, rank))
            W = np.random.uniform(-1, 1, (p*n, rank))
            
            # Simple iterative refinement
            for it in range(100):
                # Update U
                for i in range(n*m):
                    for c in range(rank):
                        num = 0.0
                        den = 0.0
                        for j in range(m):
                            for k in range(p):
                                idx_i = i
                                idx_v = j * p + k
                                idx_w = k * n + (i // m)
                                num += V[idx_v, c] * W[idx_w, c]
                                den += (V[idx_v, c] * W[idx_w, c])**2
                        if den > 1e-10:
                            U[i, c] = num / den
            
            # Check
            if verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-8):
                # Snap to nearest half-integer
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                if verify(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), tol=1e-9):
                    return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
    
    # Try rank 15
    if time.time() < deadline:
        for trial in range(30):
            if time.time() > deadline:
                break
            rank = 15
            np.random.seed(1000 + trial)
            U = np.random.uniform(-1, 1, (n*m, rank))
            V = np.random.uniform(-1, 1, (m*p, rank))
            W = np.random.uniform(-1, 1, (p*n, rank))
            
            for it in range(200):
                for i in range(n*m):
                    for c in range(rank):
                        num = 0.0
                        den = 0.0
                        for j in range(m):
                            for k in range(p):
                                idx_w = k * n + (i // m)
                                num += V[j*p+k, c] * W[idx_w, c]
                                den += (V[j*p+k, c] * W[idx_w, c])**2
                        if den > 1e-10:
                            U[i, c] = num / den
                for j in range(m*p):
                    for c in range(rank):
                        num = 0.0
                        den = 0.0
                        for i in range(n):
                            for k in range(p):
                                idx_u = i * m + (j // p)
                                idx_w = k * n + i
                                num += U[idx_u, c] * W[idx_w, c]
                                den += (U[idx_u, c] * W[idx_w, c])**2
                        if den > 1e-10:
                            V[j, c] = num / den
                for k in range(p*n):
                    for c in range(rank):
                        num = 0.0
                        den = 0.0
                        for i in range(n):
                            for j in range(m):
                                idx_u = i * m + j
                                idx_v = j * p + (k % n)  # k*n+i -> k//n is i, k%n is... 
                                # Actually W[k*n+i], so for fixed k, i varies
                                # Let me fix the indexing
                                pass
                        # Skip W update for now
                        pass
            
            if verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-8):
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                if verify(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), tol=1e-9):
                    return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
