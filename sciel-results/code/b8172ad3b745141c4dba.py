"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START

def _build_decomp(U_list, V_list, W_list):
    """Convert list-of-lists to numpy arrays and return dict."""
    U = np.array(U_list, dtype=float)
    V = np.array(V_list, dtype=float)
    W = np.array(W_list, dtype=float)
    return {"U": U, "V": V, "W": W}

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The multiplication tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    We check that this equals the standard multiplication tensor:
    T_std[i, j, k] = 1 if j is the "middle" index connecting A and B, else 0.
    Actually, for matrix multiplication C = A * B:
    C[i,k] = sum_j A[i,j] * B[j,k]
    The tensor is T[i,j,k] = 1 (it's a 3-way tensor where the "product" is sum over j of A[i,j]*B[j,k])
    
    In the decomposition format:
    U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    The reconstructed tensor entry for (i, j, k) is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    This should equal 1 for all valid (i, j, k) in the multiplication tensor.
    Wait, that's not quite right. Let me re-read the convention.
    
    The multiplication tensor for multiplying an n x m matrix by a m x p matrix:
    T_{i,j,k} = 1 (this is the structure tensor)
    C[i,k] = sum_{j} A[i,j] * B[j,k]
    
    The tensor decomposition says:
    T_{i,j,k} = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{i,k}^{(r)}  ... no that's not right either.
    
    Let me think again. The standard tensor decomposition of matrix multiplication:
    The multiplication tensor T has entries T[i, j, k] = 1 (for the "1" structure).
    Actually, the tensor is defined such that:
    (A * B)[i, k] = sum_j A[i, j] * B[j, k]
    
    The decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r where
    U_r has shape (n, m), V_r has shape (m, p), W_r has shape (p, n) ... no.
    
    Looking at the schoolbook example:
    U[i*m + j, c] = 1, V[j*p + k, c] = 1, W[k*n + i, c] = 1
    
    So for a fixed (i, j, k), the contribution is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    In schoolbook, each (i,j,k) has its own column, so for column c corresponding to (i',j',k'):
    U[i*m+j, c] = 1 iff i=i' and j=j'
    V[j*p+k, c] = 1 iff j=j' and k=k'
    W[k*n+i, c] = 1 iff k=k' and i=i'
    
    So the product is 1 iff i=i', j=j', k=k'. Thus the reconstructed tensor has
    T[i,j,k] = 1 for each (i,j,k) independently.
    
    But wait, that gives T[i,j,k] = 1 for ALL (i,j,k), which is just the all-ones tensor.
    That can't be right for matrix multiplication.
    
    Actually, I think I'm confusing myself. Let me re-read the task:
    "Matrix multiplication 2,2,12"
    "multiplying a 2 by 2 matrix by a 2 by 12 matrix"
    
    So A is 2x2 (n=2, m=2), B is 2x12 (m=2, p=12). C is 2x12 (n=2, p=12).
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The multiplication tensor T is of order 3 with dimensions (n*m, m*p, p*n) = (4, 24, 24).
    T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    
    The entry T[a,b,c] = 1 if and only if the indices are consistent:
    a = i*m+j, b = j'*p+k, c = k'*n+i' where j=j' and k=k' and i=i'.
    Wait, no. The tensor entry is 1 when the three indices share the same "path" through the multiplication.
    
    Actually, the standard way: the multiplication tensor T ∈ R^{(n×m) × (m×p) × (p×n)}
    T_{(i,j), (j,k), (k,i)} = 1 (using consistent indices)
    and 0 otherwise.
    
    So T has a 1 at position ((i,j), (j,k), (k,i)) for each valid (i,j,k) combination.
    For n=2, m=2, p=12: there are 2*2*12 = 48 ones in the tensor.
    
    The decomposition: T = sum_r U^{(r)} ⊗ V^{(r)} ⊗ W^{(r)}
    where U^{(r)} ∈ R^{n*m}, V^{(r)} ∈ R^{m*p}, W^{(r)} ∈ R^{p*n}
    
    So T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    And we need this to equal 1 when a=(i,j), b=(j,k), c=(k,i) for consistent (i,j,k),
    and 0 otherwise.
    
    Let me verify with schoolbook:
    For column c corresponding to (i',j',k'):
    U[a, c] = 1 iff a = i'*m + j'
    V[b, c] = 1 iff b = j'*p + k'
    W[c, c] = 1 iff c = k'*n + i'
    
    So T[a, b, c] = 1 iff a = i'*m+j' AND b = j'*p+k' AND c = k'*n+i'
    which means the three indices are all consistent with the same (i',j',k').
    
    So T has a 1 at ((i,j),(j,k),(k,i)) for each (i,j,k), and 0 elsewhere. Correct!
    """
    rank = U.shape[1]
    T = np.zeros((U.shape[0], V.shape[0], W.shape[0]))
    for r in range(rank):
        T += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
    
    # Build expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_expected[a, b, c] = 1
    
    err = np.max(np.abs(T - T_expected))
    return err < tol

def _schoolbook_decomp(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_block_decomp(n, m, p):
    """
    Try to find a decomposition by partitioning p into groups.
    For each group of size g, we need a decomposition for n x m by m x g.
    If we can find rank r_g for each group, total rank is sum of r_g.
    
    For n=2, m=2, g=3: schoolbook gives 12. Can we do better?
    For n=2, m=2, g=2: schoolbook gives 8. Known minimum is 7 (Strassen).
    For n=2, m=2, g=1: schoolbook gives 4. Minimum is 4.
    
    Let's try to construct a rank-7 decomposition for 2x2 * 2x2 using Strassen.
    """
    n, m, p = 2, 2, 12
    # Try partitioning into 3 groups of 4
    group_sizes = [4, 4, 4]
    
    def strassen_2x2_2x2():
        """Return rank-7 decomposition for 2x2 * 2x2."""
        # Strassen's 7-multiplication algorithm
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
        
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # Indexing: a=i*m+j, for 2x2: a=0 (i=0,j=0), b=1 (i=0,j=1), c=2 (i=1,j=0), d=3 (i=1,j=1)
        # B: e=0 (j=0,k=0), f=1 (j=0,k=1), g=2 (j=1,k=0), h=3 (j=1,k=1)
        # C: 0=(k=0,i=0), 1=(k=0,i=1), 2=(k=1,i=0), 3=(k=1,i=1)
        
        # M1 = (a+d)(e+h)
        # a contributes to M1, d contributes to M1
        # e contributes to M1, h contributes to M1
        # Result goes to C[0] and C[3]
        U[0, 0] = 1; U[3, 0] = 1  # a, d
        V[0, 0] = 1; V[3, 0] = 1  # e, h
        W[0, 0] = 1; W[3, 0] = 1  # C[0], C[3]
        
        # M2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1  # c, d
        V[0, 1] = 1  # e
        W[1, 1] = 1; W[3, 1] = 1  # C[1], C[3]
        
        # M3 = a(f-h)
        U[0, 2] = 1  # a
        V[1, 2] = 1; V[3, 2] = -1  # f, -h
        W[1, 2] = 1; W[3, 2] = 1  # C[1], C[3]
        
        # M4 = d(g-e)
        U[3, 3] = 1  # d
        V[2, 3] = 1; V[0, 3] = -1  # g, -e
        W[0, 3] = 1; W[1, 3] = 1  # C[0], C[1]
        
        # M5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1  # a, b
        V[3, 4] = 1  # h
        W[0, 4] = 1; W[3, 4] = 1  # C[0], C[3]
        
        # M6 = (c-a)(e+f)
        U[2, 5] = 1; U[0, 5] = -1  # c, -a
        V[0, 5] = 1; V[1, 5] = 1  # e, f
        W[1, 5] = 1  # C[1]
        
        # M7 = (b-d)(g+h)
        U[1, 6] = 1; U[3, 6] = -1  # b, -d
        V[2, 6] = 1; V[3, 6] = 1  # g, h
        W[0, 6] = 1  # C[0]
        
        return U, V, W
        
    # For 2x2 * 2x4, we can use two Strassen blocks
    def strassen_2x2_2x4():
        """Rank-14 decomposition for 2x2 * 2x4 using two Strassen blocks."""
        rank = 14
        U = np.zeros((4, rank))
        V = np.zeros((8, rank))
        W = np.zeros((8, rank))
        
        U1, V1, W1 = strassen_2x2_2x2()
        
        # First block: columns 0,1 of B
        U[:, 0:7] += U1
        V[0:4, 0:7] += V1
        W[0:4, 0:7] += W1
        
        # Second block: columns 2,3 of B
        U[:, 7:14] += U1
        V[4:8, 7:14] += V1
        W[4:8, 7:14] += W1
        
        return U, V, W
        
    # Build full decomposition from 3 blocks of 2x2*2x4
    total_rank = 3 * 14
    U = np.zeros((4, total_rank))
    V = np.zeros((24, total_rank))
    W = np.zeros((24, total_rank))
    
    U1, V1, W1 = strassen_2x2_2x4()
    
    for block in range(3):
        offset = block * 14
        U[:, offset:offset+14] += U1
        V[block*8:(block+1)*8, offset:offset+14] += V1
        W[block*8:(block+1)*8, offset:offset+14] += W1
    
    decomp = _build_decomp(U.tolist(), V.tolist(), W.tolist())
    if _verify_decomp(U, V, W, n, m, p):
        return decomp
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x12 matrix.
    
    Strategy: Use Strassen-like block decomposition to reduce rank below schoolbook (48).
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin before 120s limit
    
    # Try block decomposition with Strassen
    result = _try_block_decomp(n, m, p)
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook_decomp(n, m, p)

# EVOLVE-BLOCK-END
