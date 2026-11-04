"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 115.0  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition."""
        U = [
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 1],
        ]
        V = [
            [1, 0, 0, 0, 0, 1, 1],
            [0, 1, -1, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, -1, 1, 0, 0],
        ]
        W = [
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, -1, 0, 0],
        ]
        return U, V, W

    def verify_fast(U, V, W, n, m, p, tol=1e-9):
        """Faster verification using einsum."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

    # Start with schoolbook baseline
    best_result = schoolbook()
    best_rank = n * m * p

    # Strategy: For i in {0,1}, decompose 2x4x5 as two 2x2x5 sub-problems.
    # Each 2x2x5 uses Strassen on k-pairs {0,1} and {2,3} (7+7=14) plus schoolbook for k=4 (2 terms) = 16 terms.
    # Two such sub-problems (j in {0,1} and j in {2,3}) give 32 terms.
    # For i=2: schoolbook 1x4x5 = 20 terms.
    # Total: 32 + 20 = 52 terms.

    U_cols = []
    V_cols = []
    W_cols = []

    # For i in {0,1}, j in {0,1}: 2x2x5 problem
    # k in {0,1}: Strassen
    U_s, V_s, W_s = strassen_2x2x2()
    for c in range(7):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        for i in range(2):
            for j in range(2):
                U_col[i*m + j] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_col[j*p + k] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_col[k*n + i] = W_s[k*2+i][c]
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # k in {2,3}: Strassen
    U_s, V_s, W_s = strassen_2x2x2()
    for c in range(7):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        for i in range(2):
            for j in range(2):
                U_col[i*m + j] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_col[j*p + (2+k)] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_col[(2+k)*n + i] = W_s[k*2+i][c]
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # k in {4}: schoolbook
    for i in range(2):
        for j in range(2):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            U_col[i*m + j] = 1.0
            V_col[j*p + 4] = 1.0
            W_col[4*n + i] = 1.0
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    # For i in {0,1}, j in {2,3}: 2x2x5 problem
    # k in {0,1}: Strassen
    U_s, V_s, W_s = strassen_2x2x2()
    for c in range(7):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        for i in range(2):
            for j in range(2):
                U_col[i*m + (2+j)] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_col[(2+j)*p + k] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_col[k*n + i] = W_s[k*2+i][c]
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # k in {2,3}: Strassen
    U_s, V_s, W_s = strassen_2x2x2()
    for c in range(7):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        for i in range(2):
            for j in range(2):
                U_col[i*m + (2+j)] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_col[(2+j)*p + (2+k)] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_col[(2+k)*n + i] = W_s[k*2+i][c]
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # k in {4}: schoolbook
    for i in range(2):
        for j in range(2):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            U_col[i*m + (2+j)] = 1.0
            V_col[(2+j)*p + 4] = 1.0
            W_col[4*n + i] = 1.0
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    # For i=2: schoolbook 1x4x5
    for j in range(m):
        for k in range(p):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            U_col[2*m + j] = 1.0
            V_col[j*p + k] = 1.0
            W_col[k*n + 2] = 1.0
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    # Convert to row-major format
    rank = len(U_cols)
    U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
    V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
    W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]

    if verify_fast(U_mat, V_mat, W_mat, n, m, p):
        if rank < best_rank:
            best_result = {"U": U_mat, "V": V_mat, "W": W_mat}
            best_rank = rank

    # Now try to reduce rank further by exploiting linear dependencies
    # and attempting smaller structured decompositions
    
    def try_decomposition(n, m, p, block_sizes, use_strassen=True):
        """
        Try a structured decomposition based on block sizes.
        block_sizes: list of (ni, mj, pk) tuples for each block.
        """
        U_cols = []
        V_cols = []
        W_cols = []
        
        for (ni, mj, pk) in block_sizes:
            if use_strassen and ni == 2 and mj == 2 and pk == 2:
                U_s, V_s, W_s = strassen_2x2x2()
                for c in range(7):
                    U_col = [0.0] * (n * m)
                    V_col = [0.0] * (m * p)
                    W_col = [0.0] * (p * n)
                    # Need to know which block this is - this approach is getting complex
                    # Let's use a different strategy
                    pass
            else:
                # Schoolbook for this block
                for i in range(ni):
                    for j in range(mj):
                        for k in range(pk):
                            U_col = [0.0] * (n * m)
                            V_col = [0.0] * (m * p)
                            W_col = [0.0] * (p * n)
                            # This is too generic, let's skip this approach
                            pass
        
        return None  # This approach is too complex, let's try something else

    # Alternative: Try to find a better decomposition by considering the structure more carefully
    # For 3x4x5, we can think of it as:
    # - 3 rows of A (each 1x4)
    # - 4 columns of B (each 4x5)
    # 
    # A better approach might be to use the fact that 4 = 2+2 and 5 = 2+2+1
    
    # Let's try a different block structure:
    # Split j into {0,1} and {2,3}, split k into {0,1}, {2,3}, {4}
    # For each i:
    #   For j in {0,1}: 1x2x5 -> can use Strassen-like for k pairs
    #   For j in {2,3}: 1x2x5 -> similar
    # 
    # Actually, let's try to be more systematic. The current 52-rank solution is:
    # i in {0,1}, j in {0,1}: 2x2x5 = 16 terms (14 Strassen + 2 schoolbook)
    # i in {0,1}, j in {2,3}: 2x2x5 = 16 terms
    # i=2, all j,k: 1x4x5 = 20 terms
    # Total: 52
    
    # Can we do better? Let's think about 1x2x5:
    # A is 1x2, B is 2x5, C is 1x5
    # C[k] = A[0]*B[0,k] + A[1]*B[1,k]
    # This is just 10 scalar multiplications (schoolbook) or...
    # Actually for 1x2x5, the tensor rank is at most 5 (one per output)
    # because C[k] = sum_j A[j]*B[j,k], which is 5 independent products if we fix the pattern
    
    # Wait, let me reconsider. For 1xm x pxn, the rank is mn (schoolbook) in general.
    # But we might be able to do better with special structure.
    
    # Let's try a completely different approach: use the fact that we can share
    # computations across different i values.
    
    # Actually, the current approach is already quite good. Let's try to verify
    # if we can get below 52 by using a smarter decomposition.
    
    # Try: Split i into {0,1} and {2}
    # For i in {0,1}: 2x4x5
    #   Split j into {0,1,2,3}, k into {0,1,2,3,4}
    #   Use Strassen on 2x2 blocks of j,k:
    #   j in {0,1}, k in {0,1}: 7 terms
    #   j in {0,1}, k in {2,3}: 7 terms  
    #   j in {2,3}, k in {0,1}: 7 terms
    #   j in {2,3}, k in {2,3}: 7 terms
    #   k=4: 2x4 = 8 terms (schoolbook)
    #   Total for i in {0,1}: 7+7+7+7+8 = 36 terms
    # For i=2: 1x4x5 = 20 terms
    # Total: 56 terms - worse than 52
    
    # The current 52-term solution is better. Let's see if we can improve it.
    
    # What if we use a 2x2x5 decomposition that's better than 16 terms?
    # 2x2x5: A is 2x2, B is 2x5, C is 2x5
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # For fixed i, this is 1x2x5 which has rank at most 5
    # For 2x2x5, can we do better than 16?
    # 
    # Actually, let's just accept the 52-term solution and return it.
    
    return best_result
# EVOLVE-BLOCK-END
