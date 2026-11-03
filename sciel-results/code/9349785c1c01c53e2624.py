"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    nm, mp, pn = n*m, m*p, p*n

    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    # Verify decomposition
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0
                    rhs = 0.0
                    for c in range(U.shape[1]):
                        rhs += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(lhs - rhs) > 1e-6:
                        return False
        return True

    # We'll build the decomposition using:
    # 1. One Bini 3x3x3 block (rank 23) for I={0,1,2}, J={0,1,2}, K={0,1,2}
    # 2. Strassen 2x2x2 blocks (rank 7 each) for disjoint regions
    # 3. Schoolbook for the rest
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Add Bini 3x3x3 block
    # The Bini decomposition of the 3x3x3 matrix multiplication tensor has rank 23.
    # I'll use a known set of 23 terms.
    # Each term is a triple (u, v, w) where u is in R^9, v in R^9, w in R^9.
    # The product sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = delta_{i,k} (for the correct i,j,k)
    # Actually, the tensor is T[i,j,k,l] = delta_{i,l} * delta_{j,k} for (A*B)[i,k] = sum_j A[i,j]B[j,k]
    # Wait, the tensor for matrix multiplication is: T[(i,j),(j,k),(k,i)] = 1
    # So for the 3x3x3 sub-block, we need terms that satisfy:
    # sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1 if (i,j,k) is in the sub-block, else 0
    # Actually, the tensor is supported on (i,j) in A, (j,k) in B, (k,i) in C.
    # So for the 3x3x3 block with I={0,1,2}, J={0,1,2}, K={0,1,2}, we need:
    # sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1 for all i,j,k in {0,1,2}
    
    # I'll use a hardcoded Bini decomposition.
    # Reference: Bini et al., 1978, "Decomposition of the 3x3x3 tensor"
    # The 23 terms are as follows (each is (u, v, w) as sparse representations):
    
    bini_terms = []
    
    # I'll use a known Bini decomposition. Let me construct it carefully.
    # Actually, let me use a simpler approach: I'll use the known Bini 23-term decomposition.
    # Here's one valid set of 23 terms for the 3x3x3 MM tensor:
    
    # Term 1: (1,0,0; 0,1,0; 0,0,1) -> but this is just one schoolbook term
    # I need the actual Bini terms. Let me use a reference implementation.
    
    # Bini 23 terms (u, v, w) where each is a 9-dim vector indexed as (i,j) for u, (j,k) for v, (k,i) for w
    # I'll use a known set. Here's a valid Bini decomposition:
    
    # Let me use the following 23 terms (from literature):
    terms_bini = [
        # (u_coeffs, v_coeffs, w_coeffs) where each is a dict mapping (i,j) or (j,k) or (k,i) to coefficient
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1, (0,1): 1, (1,0): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(1,0): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        # These are just 7 Strassen terms for a 2x2x2 block, not Bini.
        # I need the actual Bini 23 terms.
    ]
    
    # Let me use a different approach. I'll construct the Bini terms programmatically
    # by using a known rank-23 decomposition.
    
    # Actually, let me just use 6 Strassen blocks + schoolbook, which gives rank 99.
    # The Bini approach is complex to implement correctly from memory.
    # Let me stick with the Strassen + schoolbook approach but optimize the block placement.
    
    U_cols.clear()
    V_cols.clear()
    W_cols.clear()
    
    # Define 6 disjoint Strassen 2x2x2 blocks
    blocks = [
        ({0, 1}, {0, 1}, {0, 1}),
        ({1, 2}, {0, 1}, {2, 3}),
        ({0, 2}, {0, 1}, {4, 5}),
        ({0, 1}, {2, 3}, {0, 1}),
        ({1, 2}, {2, 3}, {2, 3}),
        ({0, 2}, {2, 3}, {4, 5}),
    ]
    
    def add_strassen_block(I, J, K):
        i0, i1 = sorted(I)
        j0, j1 = sorted(J)
        k0, k1 = sorted(K)
        
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        u1 = np.zeros(nm)
        v1 = np.zeros(mp)
        w1 = np.zeros(pn)
        u1[i0*m + j0] = 1.0
        u1[i1*m + j1] = 1.0
        v1[j0*p + k0] = 1.0
        v1[j1*p + k1] = 1.0
        w1[k0*n + i0] = 1.0
        w1[k1*n + i1] = 1.0
        terms.append((u1, v1, w1))
        
        # M2 = (A10+A11)B00
        u2 = np.zeros(nm)
        v2 = np.zeros(mp)
        w2 = np.zeros(pn)
        u2[i1*m + j0] = 1.0
        u2[i1*m + j1] = 1.0
        v2[j0*p + k0] = 1.0
        w2[k0*n + i1] = 1.0
        w2[k1*n + i1] = -1.0
        terms.append((u2, v2, w2))
        
        # M3 = A00(B01-B11)
        u3 = np.zeros(nm)
        v3 = np.zeros(mp)
        w3 = np.zeros(pn)
        u3[i0*m + j0] = 1.0
        v3[j0*p + k1] = 1.0
        v3[j1*p + k1] = -1.0
        w3[k1*n + i0] = 1.0
        w3[k1*n + i1] = 1.0
        terms.append((u3, v3, w3))
        
        # M4 = A11(B10-B00)
        u4 = np.zeros(nm)
        v4 = np.zeros(mp)
        w4 = np.zeros(pn)
        u4[i1*m + j1] = 1.0
        v4[j1*p + k0] = 1.0
        v4[j0*p + k0] = -1.0
        w4[k0*n + i0] = 1.0
        w4[k0*n + i1] = 1.0
        terms.append((u4, v4, w4))
        
        # M5 = (A00+A01)B11
        u5 = np.zeros(nm)
        v5 = np.zeros(mp)
        w5 = np.zeros(pn)
        u5[i0*m + j0] = 1.0
        u5[i0*m + j1] = 1.0
        v5[j1*p + k1] = 1.0
        w5[k0*n + i0] = -1.0
        w5[k1*n + i0] = 1.0
        terms.append((u5, v5, w5))
        
        # M6 = (A10-A00)(B00+B01)
        u6 = np.zeros(nm)
        v6 = np.zeros(mp)
        w6 = np.zeros(pn)
        u6[i1*m + j0] = 1.0
        u6[i0*m + j0] = -1.0
        v6[j0*p + k0] = 1.0
        v6[j0*p + k1] = 1.0
        w6[k1*n + i1] = 1.0
        terms.append((u6, v6, w6))
        
        # M7 = (A01-A11)(B10+B11)
        u7 = np.zeros(nm)
        v7 = np.zeros(mp)
        w7 = np.zeros(pn)
        u7[i0*m + j1] = 1.0
        u7[i1*m + j1] = -1.0
        v7[j1*p + k0] = 1.0
        v7[j1*p + k1] = 1.0
        w7[k0*n + i0] = 1.0
        terms.append((u7, v7, w7))
        
        for u, v, w in terms:
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)

    # Add Strassen blocks
    for I, J, K in blocks:
        add_strassen_block(I, J, K)
        
    # Add remaining schoolbook terms
    covered = set()
    for I, J, K in blocks:
        for i in I:
            for j in J:
                for k in K:
                    covered.add((i, j, k))
                    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u = np.zeros(nm)
                    v = np.zeros(mp)
                    w = np.zeros(pn)
                    u[i*m + j] = 1.0
                    v[j*p + k] = 1.0
                    w[k*n + i] = 1.0
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)
                    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    
    if verify(U_mat, V_mat, W_mat):
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
