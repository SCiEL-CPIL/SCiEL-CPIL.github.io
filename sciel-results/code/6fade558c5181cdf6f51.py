import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    deadline = time.time() + 110
    n, m, p = 2, 3, 8

    def verify(U, V, W, n, m, p):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-10)

    def build_strassen_44():
        """
        Rank-44 decomposition.
        Split m=3 into 1+2.
        Part 1: 2x1 * 1x8 -> 16 schoolbook mults.
        Part 2: 2x2 * 2x8 -> 4 blocks of 2x2 * 2x2 (Strassen, rank 7 each) -> 28 mults.
        Total: 16 + 28 = 44.
        """
        rank = 44
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        col = 0
        
        # Part 1: A[:,0] (2x1) * B[0,:] (1x8)
        for i in range(n):
            for k in range(p):
                U[i*m + 0][col] = 1
                V[0*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
        
        # Part 2: A[:,1:3] (2x2) * B[1:3,:] (2x8)
        for block in range(4):
            b00 = 1 * p + block * 2
            b01 = 1 * p + block * 2 + 1
            b10 = 2 * p + block * 2
            b11 = 2 * p + block * 2 + 1
            
            c00 = (block * 2) * n + 0
            c01 = (block * 2 + 1) * n + 0
            c10 = (block * 2) * n + 1
            c11 = (block * 2 + 1) * n + 1
            
            # Strassen step 1: (a00+a11)(b00+b11)
            U[1][col] = 1
            U[5][col] = 1
            V[b00][col] = 1
            V[b11][col] = 1
            W[c00][col] = 1
            W[c11][col] = 1
            col += 1
            
            # Strassen step 2: (a10+a11)b00
            U[4][col] = 1
            U[5][col] = 1
            V[b00][col] = 1
            W[c10][col] = 1
            W[c11][col] = -1
            col += 1
            
            # Strassen step 3: a00(b01-b11)
            U[1][col] = 1
            V[b01][col] = 1
            V[b11][col] = -1
            W[c01][col] = 1
            W[c11][col] = 1
            col += 1
            
            # Strassen step 4: a11(b10-b00)
            U[5][col] = 1
            V[b10][col] = 1
            V[b00][col] = -1
            W[c00][col] = 1
            W[c10][col] = 1
            col += 1
            
            # Strassen step 5: (a00+a01)b11
            U[1][col] = 1
            U[2][col] = 1
            V[b11][col] = 1
            W[c00][col] = -1
            W[c01][col] = 1
            col += 1
            
            # Strassen step 6: (a10-a00)(b00+b01)
            U[4][col] = 1
            U[1][col] = -1
            V[b00][col] = 1
            V[b01][col] = 1
            W[c11][col] = 1
            col += 1
            
            # Strassen step 7: (a01-a11)(b10+b11)
            U[2][col] = 1
            U[5][col] = -1
            V[b10][col] = 1
            V[b11][col] = 1
            W[c00][col] = 1
            col += 1
        
        return U, V, W

    def try_rank_43():
        """
        Attempt rank-43 by using a 2x2x4 block of rank 13 instead of 14.
        Split: A[2x3] x B[3x8]
        = A[2x2] x B[2x8]  (using two 2x2x4 blocks, each potentially rank 13)
        + A[2x1] x B[1x8]  (16 schoolbook)
        
        If we can do 2x2x4 in rank 13, total = 13*2 + 16 = 42.
        If we can do it in rank 14 (standard), total = 14*2 + 16 = 44.
        
        Actually, let's try: use one 2x2x4 in rank 13 and one in rank 14: 13+14+16=43.
        Or both in rank 13: 13+13+16=42.
        
        The 2x2x4 problem: C[2x4] = A[2x2] * B[2x4].
        Standard: two 2x2x2 Strassen = rank 14.
        
        Can we do 2x2x4 in rank 13? This is an open problem in general,
        but let's try a specific construction.
        
        Alternative approach: try to numerically find if rank 43 works
        by starting from rank 44 and trying to zero out one coefficient.
        """
        # Start with rank-44 decomposition
        U44, V44, W44 = build_strassen_44()
        U44 = np.array(U44, dtype=float)
        V44 = np.array(V44, dtype=float)
        W44 = np.array(W44, dtype=float)
        
        # Try to eliminate one term by expressing it as combination of others
        # This is a linear algebra problem. The tensor T = sum_c u_c ⊗ v_c ⊗ w_c
        # If we can write T = sum_{c=1}^{43} u'_c ⊗ v'_c ⊗ w'_c, rank is 43.
        
        # A simpler approach: try a different structural decomposition.
        # Use the fact that M_{2,3,8} can be viewed differently.
        
        # Try: split p=8 into 4+4, and for each 2x3x4, use a rank-13 scheme.
        # 2x3x4: C[2x4] = A[2x3] * B[3x4]
        # Split A as [2x2 | 2x1], B as [2x4; 1x4]
        # 2x2x4 part + 2x1x4 part
        # If 2x2x4 has rank 13 and 2x1x4 has rank 8, total = 21 per half, 42 total.
        # If 2x2x4 has rank 14 and 2x1x4 has rank 8, total = 22 per half, 44 total.
        
        # Let me try to construct a 2x2x4 rank-13 decomposition.
        # This is equivalent to asking if the 2x2x4 tensor has rank 13.
        # Known: rank(2,2,4) is between 12 and 14.
        
        # For now, let's try a numerical approach: attempt to find rank-43
        # by perturbing the rank-44 solution.
        
        # Build the tensor as a matrix for rank minimization
        # T[i,j,k] = 1 for all i,j,k
        # We want T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
        
        # Use a simple iterative approach: start with 44 terms, try to reduce
        # one term by solving a least-squares problem.
        
        # Actually, let's just return the known rank-44 solution.
        # A rank-43 construction would require deeper insight.
        
        return None

    # Try rank-44 construction first (known to work)
    U, V, W = build_strassen_44()
    if verify(U, V, W, n, m, p):
        # Try to improve to rank 43
        if time.time() < deadline - 30:
            result = try_rank_43()
            if result is not None:
                U, V, W = result
                if verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
        
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook (should not happen)
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
