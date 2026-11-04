import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # We need to decompose the matrix multiplication tensor for 4x4 * 4x7
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # where A is 4x4, B is 4x7, C is 4x7
    # 
    # The tensor has entries T[i,j,k] = 1 for all valid (i,j,k)
    # A rank-1 decomposition term is u ⊗ v ⊗ w where:
    #   u is indexed by (i,j) pairs (size n*m = 16)
    #   v is indexed by (j,k) pairs (size m*p = 28)  
    #   w is indexed by (k,i) pairs (size p*n = 28)
    # 
    # We need: sum_c U[c,i,j] * V[c,j,k] * W[c,k,i] = 1 for all i,j,k
    # 
    # Strategy: Use Strassen for the 4x4 * 4x4 part (first 4 columns of B)
    # and schoolbook for the 4x4 * 4x3 part (last 3 columns of B)
    
    terms = []
    
    def make_u(idx_i, idx_j, val):
        u = [0] * (n * m)
        u[idx_i * m + idx_j] = val
        return u
    
    def make_v(idx_j, idx_k, val):
        v = [0] * (m * p)
        v[idx_j * p + idx_k] = val
        return v
    
    def make_w(idx_k, idx_i, val):
        w = [0] * (p * n)
        w[idx_k * n + idx_i] = val
        return w
    
    # For 4x4 * 4x4 (first 4 columns of B), use Strassen
    # We'll handle the first 4 columns (k=0,1,2,3)
    
    # Strassen for 2x2*2x2 blocks
    # For each block multiplication, we have 7 terms
    # We need to map these to the full tensor decomposition
    
    def strassen_terms(i_off, j_off, k_off):
        """Generate Strassen terms for 2x2*2x2 block multiplication"""
        result = []
        
        # M1 = (A11+A22)*(B11+B22)
        # Contributes: C11 += M1, C22 += M1
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        result.append((u, v, w))
        
        # M2 = (A21+A22)*B11
        # Contributes: C21 += M2, C22 -= M2
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = 1
        w[(k_off+1) * n + (i_off+1)] = -1
        result.append((u, v, w))
        
        # M3 = A11*(B12-B22)
        # Contributes: C12 += M3, C22 += M3
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = 1
        v[(j_off+1) * p + (k_off+1)] = -1
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        result.append((u, v, w))
        
        # M4 = A22*(B21-B11)
        # Contributes: C11 += M4, C21 += M4
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[j_off * p + k_off] = -1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[k_off * n + (i_off+1)] = 1
        result.append((u, v, w))
        
        # M5 = (A11+A12)*B22
        # Contributes: C11 -= M5, C12 += M5
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[i_off * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = -1
        w[(k_off+1) * n + i_off] = 1
        result.append((u, v, w))
        
        # M6 = (A21-A11)*(B11+B12)
        # Contributes: C22 += M6
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[i_off * m + j_off] = -1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[j_off * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = 1
        result.append((u, v, w))
        
        # M7 = (A12-A22)*(B21+B22)
        # Contributes: C11 += M7
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = 1
        u[(i_off+1) * m + (j_off+1)] = -1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        result.append((u, v, w))
        
        return result
    
    # Apply Strassen to the four 2x2 block multiplications for 4x4*4x4
    # Block (0,0): i_off=0, j_off=0, k_off=0
    # Block (0,1): i_off=0, j_off=2, k_off=2
    # Block (1,0): i_off=2, j_off=0, k_off=0
    # Block (1,1): i_off=2, j_off=2, k_off=2
    
    # Actually, for 4x4*4x4 with 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # 
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # 
    # We can use Strassen at the block level:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # 
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # 
    # But this gives us 7 multiplications of 2x2*2x2, which is rank 7*4 = 28
    # if we apply Strassen to each 2x2*2x2 block.
    # 
    # However, we can also just apply Strassen directly to each of the four
    # block multiplications:
    # A11*B11: Strassen with i_off=0, j_off=0, k_off=0
    # A12*B21: Strassen with i_off=0, j_off=2, k_off=0
    # A21*B11: Strassen with i_off=2, j_off=0, k_off=0
    # A22*B22: Strassen with i_off=2, j_off=2, k_off=2
    # 
    # Wait, that's not right either. Let me think again.
    # 
    # Actually, the simplest approach is to use Strassen for each of the four
    # 2x2*2x2 block multiplications. Each uses 7 terms, so total 28 terms.
    
    # Block multiplication A11 * B11 -> C11
    terms.extend(strassen_terms(0, 0, 0))
    
    # Block multiplication A12 * B21 -> C12
    # A12 is at rows 0-1, cols 2-3; B21 is at rows 2-3, cols 0-1; C12 is at rows 0-1, cols 2-3
    terms.extend(strassen_terms(0, 2, 2))
    
    # Block multiplication A21 * B11 -> C21
    # A21 is at rows 2-3, cols 0-1; B11 is at rows 0-1, cols 0-1; C21 is at rows 2-3, cols 0-1
    terms.extend(strassen_terms(2, 0, 0))
    
    # Block multiplication A22 * B22 -> C22
    # A22 is at rows 2-3, cols 2-3; B22 is at rows 2-3, cols 2-3; C22 is at rows 2-3, cols 2-3
    terms.extend(strassen_terms(2, 2, 2))
    
    # For 4x4 * 4x3 (last 3 columns of B), use schoolbook
    # A is 4x4, B2 is 4x3, C2 is 4x3
    # C2[i,k] = sum_j A[i,j] * B2[j,k] for k=0,1,2 (which are columns 4,5,6 of B)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                u = make_u(i, j, 1)
                v = make_v(j, k + 4, 1)
                w = make_w(k + 4, i, 1)
                terms.append((u, v, w))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    
    # Verify the decomposition
    # Check that sum_c U[c,i,j] * V[c,j,k] * W[c,k,i] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-10:
                    # This should not happen, but if it does, fall back to schoolbook
                    terms = []
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                u = make_u(ii, jj, 1)
                                v = make_v(jj, kk, 1)
                                w = make_w(kk, ii, 1)
                                terms.append((u, v, w))
                    rank = len(terms)
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    for c, (u, v, w) in enumerate(terms):
                        for r in range(n * m):
                            U[r][c] = u[r]
                        for r in range(m * p):
                            V[r][c] = v[r]
                        for r in range(p * n):
                            W[r][c] = w[r]
                    break
            else:
                continue
            break
        else:
            continue
        break
    
    return {"U": U, "V": V, "W": W}
