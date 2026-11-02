"""Block-structured low-rank decomposition for 4x5x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Strassen's 7-term decomposition for 2x2x2 matrix multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[C00,C01],[C10,C11]]
    # p1=(a+d)(e+h), p2=(c+d)e, p3=a(f-h), p4=d(g-e), p5=(a+b)g, p6=(c-a)(e+f), p7=(b-d)(h+f)
    # C00=p1-p4+p5-p7, C01=p3+p4+p5+p7, C10=p2+p4-p6+p7, C11=p1-p2+p6-p7
    # 
    # Each term c has: U_vec (4 entries for A), V_vec (4 entries for B), W_vec (4 entries for C)
    # U_vec[i*2+j] = coefficient of A[i][j]
    # V_vec[j*2+k] = coefficient of B[j][k]  
    # W_vec[k*2+i] = coefficient of C[i][k] (trace convention: W[k*n+i])
    
    strassen_terms = []
    # p1=(a+d)(e+h): A: a+d -> (1,0,0,1); B: e+h -> (1,0,0,1); C: C00+C11 -> W: k*n+i, C00->0*2+0=0, C11->1*2+1=3 => (1,0,0,1)
    strassen_terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # p2=(c+d)e: A: c+d -> (0,0,1,1); B: e -> (1,0,0,0); C: C10-C11 -> C10->0*2+1=1, C11->1*2+1=3 => (0,1,0,-1)
    strassen_terms.append(([0,0,1,1], [1,0,0,0], [0,1,0,-1]))
    # p3=a(f-h): A: a -> (1,0,0,0); B: f-h -> (0,1,0,-1); C: C01 -> 0*2+1=1 => (0,1,0,0)
    strassen_terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    # p4=d(g-e): A: d -> (0,0,0,1); B: g-e -> (-1,0,1,0); C: -C00+C01+C10 -> C00->0, C01->1, C10->1 => (0,1,1,0) but with -C00: W[0]=-1, W[1]=1, W[2]=1 => (-1,1,1,0)
    strassen_terms.append(([0,0,0,1], [-1,0,1,0], [-1,1,1,0]))
    # p5=(a+b)g: A: a+b -> (1,1,0,0); B: g -> (0,0,1,0); C: C00+C01 -> (1,1,0,0)
    strassen_terms.append(([1,1,0,0], [0,0,1,0], [1,1,0,0]))
    # p6=(c-a)(e+f): A: c-a -> (-1,0,1,0); B: e+f -> (1,1,0,0); C: -C10+C11 -> C10->1, C11->3 => (0,0,-1,1)
    strassen_terms.append(([-1,0,1,0], [1,1,0,0], [0,0,-1,1]))
    # p7=(b-d)(h+f): A: b-d -> (0,1,0,-1); B: h+f -> (0,1,0,1); C: -C00+C01+C10 -> (-1,1,1,0)
    strassen_terms.append(([0,1,0,-1], [0,1,0,1], [-1,1,1,0]))
    
    # Verify Strassen correctness on a simple case
    def verify_strassen():
        # Test with A=[[1,2],[3,4]], B=[[5,6],[7,8]], C=[[19,22],[43,50]]
        A = [[1,2],[3,4]]
        B = [[5,6],[7,8]]
        C = [[19,22],[43,50]]
        # Compute via Strassen
        C_comp = [[0,0],[0,0]]
        for u, v, w in strassen_terms:
            prod = 0
            for i in range(2):
                for j in range(2):
                    prod += u[i*2+j] * A[i][j]
            prod2 = 0
            for j in range(2):
                for k in range(2):
                    prod2 += v[j*2+k] * B[j][k]
            # Actually this isn't right. The rank-1 term means:
            # contribution to C[i][k] = sum over the term: U[i,j]*V[j,k]*W[k,i]
            # For a single rank-1 term, the "product" is U·A * V·B, and it contributes to C via W
            # Let me re-derive: term c contributes to C[i][k] the value:
            # (sum_{j} U_c[i*m+j] * A[i][j]) * (sum_{j'} V_c[j'*p+k] * B[j'][k]) * ... 
            # No wait. The tensor decomposition is: T[a,b,d] = sum_c U[a,c]*V[b,c]*W[d,c]
            # where a=i*m+j (input A index), b=j'*p+k (input B index), d=k'*n+i (output C index)
            # So for the 2x2x2 block: a = i*2+j, b = j'*2+k, d = k'*2+i
            # T[i*2+j, j'*2+k, k'*2+i] = 1 if j==j' (the shared index)
            # The rank-1 term c: U_c[i*2+j] * V_c[j'*2+k] * W_c[k'*2+i]
            # Sum over c should give 1 when j==j'
            
            # For verification: compute C[i][k] = sum_{j} A[i][j]*B[j][k]
            # = sum_{j,j'} A[i][j]*B[j'][k] * delta(j,j')
            # = sum_{j,j'} sum_c U_c[i*2+j]*V_c[j'*2+k]*W_c[k'*2+i] * A[i][j]*B[j'][k]
            # = sum_c (sum_j U_c[i*2+j]*A[i][j]) * (sum_{j'} V_c[j'*2+k]*B[j'][k]) * W_c[k'*2+i]
            # Wait that's not right either. Let me think again.
            # 
            # C[i][k] = sum_j A[i][j]*B[j][k]
            # The tensor T has T[a,b,d]=1 when a=i*2+j, b=j*2+k, d=k*2+i (same j)
            # So sum_c U_c[a]*V_c[b]*W_c[d] = 1 when a=i*2+j, b=j*2+k, d=k*2+i
            # 
            # C[i][k] = sum_j A[i][j]*B[j][k] = sum_{a,b,d} T[a,b,d]*A_flat[a]*B_flat[b]
            # where A_flat[i*2+j]=A[i][j], B_flat[j*2+k]=B[j][k]
            # = sum_c (sum_a U_c[a]*A_flat[a]) * (sum_b V_c[b]*B_flat[b]) * W_c[d]... no
            # = sum_c sum_a sum_b sum_d U_c[a]*V_c[b]*W_c[d]*T[a,b,d]*A_flat[a]*B_flat[b]
            # = sum_c (sum_a U_c[a]*A_flat[a]) * (sum_b V_c[b]*B_flat[b]) * ... 
            # This doesn't factor that way. Let me just do direct verification.
            pass
        
        # Direct verification: for each (i,j,k), check sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] == 1
        # And for non-matching (i,j',k) with j!=j', it should be 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for jp in range(2):
                        s = 0
                        for c in range(7):
                            u, v, w = strassen_terms[c]
                            s += u[i*2+j] * v[jp*2+k] * w[k*2+i]
                        expected = 1 if j == jp else 0
                        if abs(s - expected) > 1e-10:
                            return False
        return True
    
    # Don't run verification in production, but we trust the Strassen terms
    
    # Build the full 4x5x6 decomposition
    # Partition: i in {0,1,2,3}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5}
    # i-blocks: [0,1], [2,3]
    # j-blocks: [0,1], [2,3], [4]
    # k-blocks: [0,1], [2,3], [4,5]
    
    terms_U = []  # Each is a list of length n*m=20
    terms_V = []  # Each is a list of length m*p=30
    terms_W = []  # Each is a list of length p*n=24
    
    def add_term(u_block, v_block, w_block, i_offset, j_offset, k_offset, i_size, j_size, k_size):
        """Add a block term. u_block is for i_size x j_size A block, etc."""
        ufull = [0] * (n * m)
        vfull = [0] * (m * p)
        wfull = [0] * (p * n)
        
        if i_size == 2 and j_size == 2 and k_size == 2:
            # Use Strassen
            for c in range(7):
                u, v, w = strassen_terms[c]
                for ii in range(2):
                    for jj in range(2):
                        idx = (i_offset + ii) * m + (j_offset + jj)
                        ufull[idx] += u[ii * 2 + jj]
                for jj in range(2):
                    for kk in range(2):
                        idx = (j_offset + jj) * p + (k_offset + kk)
                        vfull[idx] += v[jj * 2 + kk]
                for kk in range(2):
                    for ii in range(2):
                        idx = (k_offset + kk) * n + (i_offset + ii)
                        wfull[idx] += w[kk * 2 + ii]
                terms_U.append(ufull[:])
                terms_V.append(vfull[:])
                terms_W.append(wfull[:])
                ufull = [0] * (n * m)
                vfull = [0] * (m * p)
                wfull = [0] * (p * n)
        else:
            # Schoolbook for this block
            for ii in range(i_size):
                for jj in range(j_size):
                    for kk in range(k_size):
                        ufull2 = [0] * (n * m)
                        vfull2 = [0] * (m * p)
                        wfull2 = [0] * (p * n)
                        ufull2[(i_offset + ii) * m + (j_offset + jj)] = 1
                        vfull2[(j_offset + jj) * p + (k_offset + kk)] = 1
                        wfull2[(k_offset + kk) * n + (i_offset + ii)] = 1
                        terms_U.append(ufull2)
                        terms_V.append(vfull2)
                        terms_W.append(wfull2)
    
    # Add all blocks
    i_blocks = [(0, 2), (2, 2)]  # (offset, size)
    j_blocks = [(0, 2), (2, 2), (4, 1)]
    k_blocks = [(0, 2), (2, 2), (4, 2)]
    
    for i_off, i_sz in i_blocks:
        for j_off, j_sz in j_blocks:
            for k_off, k_sz in k_blocks:
                if i_sz == 2 and j_sz == 2 and k_sz == 2:
                    add_term(None, None, None, i_off, j_off, k_off, 2, 2, 2)
                else:
                    add_term(None, None, None, i_off, j_off, k_off, i_sz, j_sz, k_sz)
    
    rank = len(terms_U)
    
    # Build U, V, W matrices
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
        n, m, p = DIMENSIONS
        import random
        random.seed(42)
        for trial in range(5):
            A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
            B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
            C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
            for i in range(n):
                for k in range(p):
                    acc = 0
                    for j in range(m):
                        for c in range(rank):
                            u = U[i * m + j][c]
                            if u == 0:
                                continue
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
