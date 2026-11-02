"""Block-structured low-rank decomposition for 4x5x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill

    # Strassen's 7-term decomposition for 2x2x2 matrix multiplication
    # T[i,j,k; j',k'] = delta(j,j')
    # We need: sum_c U_c[i,j] * V_c[j',k] * W_c[k,i] = 1 if j==j' else 0
    # (using row-major: U indexed by (i,j), V by (j',k), W by (k,i))
    #
    # Standard Strassen for 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[C00,C01],[C10,C11]]
    #
    # The 7 products and their contributions:
    # p1=(a+d)(e+h): contributes to C00, C11
    # p2=(c+d)e: contributes to C10, -C11
    # p3=a(f-h): contributes to C01
    # p4=d(g-e): contributes to -C00, C01, C10
    # p5=(a+b)g: contributes to C00, C01
    # p6=(c-a)(f+h): contributes to -C10, C11
    # p7=(b-d)(h+f): contributes to -C00, C01, C10
    #
    # For each product, U picks a linear combo of A entries, V picks a linear combo of B entries,
    # and W distributes the product to C entries.
    #
    # U vectors (for A = [[a,b],[c,d]], indexed [a,b,c,d]):
    # p1: [1,0,0,1] (a+d)
    # p2: [0,0,1,1] (c+d)
    # p3: [1,0,0,0] (a)
    # p4: [0,0,0,1] (d)
    # p5: [1,1,0,0] (a+b)
    # p6: [-1,0,1,0] (c-a)
    # p7: [0,1,0,-1] (b-d)
    #
    # V vectors (for B = [[e,f],[g,h]], indexed [e,f,g,h]):
    # p1: [1,0,0,1] (e+h)
    # p2: [1,0,0,0] (e)
    # p3: [0,1,0,-1] (f-h)
    # p4: [-1,0,1,0] (g-e)
    # p5: [0,0,1,0] (g)
    # p6: [0,1,0,1] (f+h)
    # p7: [0,1,0,1] (h+f) -- same as p6! Wait, let me recheck.
    # Actually p7=(b-d)(h+f): V is h+f = [0,1,0,1]
    #
    # W vectors (for C = [[C00,C01],[C10,C11]], indexed [C00,C01,C10,C11] = [k*n+i for k=0,1; i=0,1]
    # which is [0*2+0, 0*2+1, 1*2+0, 1*2+1] = [C00, C01, C10, C11]):
    # p1: C00 + C11 => [1,0,0,1]
    # p2: C10 - C11 => [0,0,1,-1]
    # p3: C01 => [0,1,0,0]
    # p4: -C00 + C01 + C10 => [-1,1,1,0]
    # p5: C00 + C01 => [1,1,0,0]
    # p6: -C10 + C11 => [0,0,-1,1]
    # p7: -C00 + C01 + C10 => [-1,1,1,0]
    
    strassen_U = [
        [1,0,0,1],  # p1
        [0,0,1,1],  # p2
        [1,0,0,0],  # p3
        [0,0,0,1],  # p4
        [1,1,0,0],  # p5
        [-1,0,1,0], # p6
        [0,1,0,-1], # p7
    ]
    strassen_V = [
        [1,0,0,1],  # p1
        [1,0,0,0],  # p2
        [0,1,0,-1], # p3
        [-1,0,1,0], # p4
        [0,0,1,0],  # p5
        [0,1,0,1],  # p6
        [0,1,0,1],  # p7
    ]
    strassen_W = [
        [1,0,0,1],  # p1
        [0,0,1,-1], # p2
        [0,1,0,0],  # p3
        [-1,1,1,0], # p4
        [1,1,0,0],  # p5
        [0,0,-1,1], # p6
        [-1,1,1,0], # p7
    ]
    
    # Verify Strassen correctness
    def verify_strassen():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for jp in range(2):
                        s = 0
                        for c in range(7):
                            s += strassen_U[c][i*2+j] * strassen_V[c][jp*2+k] * strassen_W[c][k*2+i]
                        expected = 1 if j == jp else 0
                        if abs(s - expected) > 1e-10:
                            return False
        return True
    
    strassen_ok = verify_strassen()
    
    # Build the full 4x5x6 decomposition using block structure
    # i in {0,1,2,3}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5}
    # 
    # Partition into blocks:
    # i-blocks: [0,1], [2,3]  (two blocks of size 2)
    # j-blocks: [0,1], [2,3], [4]  (two blocks of size 2, one of size 1)
    # k-blocks: [0,1], [2,3], [4,5]  (three blocks of size 2)
    #
    # For each combination of (i_block, j_block, k_block):
    # - If all sizes are 2: use Strassen (7 terms)
    # - Otherwise: use schoolbook (i_size * j_size * k_size terms)
    
    terms_U = []  # Each is a list of length n*m=20
    terms_V = []  # Each is a list of length m*p=30
    terms_W = []  # Each is a list of length p*n=24
    
    i_blocks = [(0, 2), (2, 2)]  # (offset, size)
    j_blocks = [(0, 2), (2, 2), (4, 1)]
    k_blocks = [(0, 2), (2, 2), (4, 2)]
    
    for i_off, i_sz in i_blocks:
        for j_off, j_sz in j_blocks:
            for k_off, k_sz in k_blocks:
                if i_sz == 2 and j_sz == 2 and k_sz == 2 and strassen_ok:
                    # Use Strassen for 2x2x2 block
                    for c in range(7):
                        ufull = [0] * (n * m)
                        vfull = [0] * (m * p)
                        wfull = [0] * (p * n)
                        
                        # Fill U: strassen_U[c] has 4 entries for 2x2 A block
                        for ii in range(2):
                            for jj in range(2):
                                idx = (i_off + ii) * m + (j_off + jj)
                                ufull[idx] = strassen_U[c][ii * 2 + jj]
                        
                        # Fill V: strassen_V[c] has 4 entries for 2x2 B block
                        for jj in range(2):
                            for kk in range(2):
                                idx = (j_off + jj) * p + (k_off + kk)
                                vfull[idx] = strassen_V[c][jj * 2 + kk]
                        
                        # Fill W: strassen_W[c] has 4 entries for 2x2 C block
                        # W indexed by [k*n+i], so for k in {k_off, k_off+1}, i in {i_off, i_off+1}
                        for kk in range(2):
                            for ii in range(2):
                                idx = (k_off + kk) * n + (i_off + ii)
                                wfull[idx] = strassen_W[c][kk * 2 + ii]
                        
                        terms_U.append(ufull)
                        terms_V.append(vfull)
                        terms_W.append(wfull)
                else:
                    # Schoolbook for this block
                    for ii in range(i_sz):
                        for jj in range(j_sz):
                            for kk in range(k_sz):
                                ufull = [0] * (n * m)
                                vfull = [0] * (m * p)
                                wfull = [0] * (p * n)
                                ufull[(i_off + ii) * m + (j_off + jj)] = 1
                                vfull[(j_off + jj) * p + (k_off + kk)] = 1
                                wfull[(k_off + kk) * n + (i_off + ii)] = 1
                                terms_U.append(ufull)
                                terms_V.append(vfull)
                                terms_W.append(wfull)
    
    rank = len(terms_U)
    
    # Build U, V, W matrices as lists of lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for a in range(n * m):
            if terms_U[c][a] != 0:
                U[a][c] = terms_U[c][a]
        for b in range(m * p):
            if terms_V[c][b] != 0:
                V[b][c] = terms_V[c][b]
        for d in range(p * n):
            if terms_W[c][d] != 0:
                W[d][c] = terms_W[c][d]
    
    # Verify the decomposition
    def verify(U, V, W, rank):
        import random
        random.seed(42)
        for trial in range(3):
            A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
            B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
            C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
            for i in range(n):
                for k in range(p):
                    acc = 0
                    for j in range(m):
                        for c in range(rank):
                            u = U[i * m + j][c]
                            if u != 0:
                                acc += u * V[j * p + k][c] * W[k * n + i][c]
                    if acc != C[i][k]:
                        return False
        return True
    
    if not verify(U, V, W, rank):
        # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
