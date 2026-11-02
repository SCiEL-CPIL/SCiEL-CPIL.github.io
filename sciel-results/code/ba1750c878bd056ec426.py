"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
import itertools
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    # 2x5x6 matrix multiplication
    # We want to find a rank-minimal decomposition
    # The trivial rank is n*m*p = 60
    # Strassen-like approaches can reduce this
    
    # Let's try a systematic approach:
    # 1. Start with the trivial decomposition (rank 60)
    # 2. Try to merge terms using algebraic identities
    # 3. Use known results for small matrix multiplication
    
    # For 2x5x6, let's think about block structure
    # A is 2x5, B is 5x6, C is 2x6
    
    # The tensor is T[i,j,k] = delta_{i,k} for the multiplication
    # where i in {0,1}, j in {0..4}, k in {0..5}
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # Let's try a hybrid approach:
    # Split the 5 columns of A into groups
    # Group 1: columns 0,1 (use Strassen 2x2x2)
    # Group 2: columns 2,3 (use Strassen 2x2x2)  
    # Column 4: direct
    
    # For each 2x2 block of B (rows j0,j1 and cols k0,k1),
    # we can use 7 multiplications instead of 4
    
    # Let's construct the decomposition:
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal rank
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        rank += 1
    
    # Helper to create sparse vectors
    def make_vec(length, entries):
        vec = [0] * length
        for idx, val in entries:
            vec[idx] += val
        return vec
    
    nm = n * m  # 10
    mp = m * p  # 30
    pn = p * n  # 12
    
    # Process 2x2 blocks of the middle dimension
    # j pairs: (0,1), (2,3)
    # k pairs: (0,1), (2,3), (4,5)
    
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # Strassen's 7 products for 2x2x2
            # C_00 = A_00*B_00 + A_01*B_10
            # C_01 = A_00*B_01 + A_01*B_11
            # C_10 = A_10*B_00 + A_11*B_10
            # C_11 = A_10*B_01 + A_11*B_11
            
            # Where A_ij is element (i,j) of the 2x2 A block
            # and B_ij is element (i,j) of the 2x2 B block
            
            # Strassen:
            # M1 = (A_00 + A_11) * (B_00 + B_11)
            # M2 = (A_10 + A_11) * B_00
            # M3 = A_00 * (B_01 - B_11)
            # M4 = A_11 * (B_10 - B_00)
            # M5 = (A_00 + A_01) * B_11
            # M6 = (A_10 - A_00) * (B_00 + B_01)
            # M7 = (A_01 - A_11) * (B_10 + B_11)
            
            # C_00 = M1 + M4 - M5 + M7
            # C_01 = M3 + M5
            # C_10 = M2 + M4
            # C_11 = M1 - M2 + M3 + M6
            
            # In our indexing:
            # A[i, j] where i in {0,1}, j in {j0, j1}
            # B[j, k] where j in {j0, j1}, k in {k0, k1}
            # C[i, k] where i in {0,1}, k in {k0, k1}
            
            # M1 = (A[0,j0] + A[1,j1]) * (B[j0,k0] + B[j1,k1])
            u1 = make_vec(nm, [(0*m+j0, 1), (1*m+j1, 1)])
            v1 = make_vec(mp, [(j0*p+k0, 1), (j1*p+k1, 1)])
            w1 = make_vec(pn, [(k0*n+0, 1), (k1*n+1, 1)])
            add_term(u1, v1, w1)
            
            # M2 = (A[1,j0] + A[1,j1]) * B[j0,k0]
            u2 = make_vec(nm, [(1*m+j0, 1), (1*m+j1, 1)])
            v2 = make_vec(mp, [(j0*p+k0, 1)])
            w2 = make_vec(pn, [(k0*n+1, 1), (k1*n+1, -1)])
            add_term(u2, v2, w2)
            
            # M3 = A[0,j0] * (B[j0,k1] - B[j1,k1])
            u3 = make_vec(nm, [(0*m+j0, 1)])
            v3 = make_vec(mp, [(j0*p+k1, 1), (j1*p+k1, -1)])
            w3 = make_vec(pn, [(k1*n+0, 1), (k1*n+1, 1)])
            add_term(u3, v3, w3)
            
            # M4 = A[1,j1] * (B[j1,k0] - B[j0,k0])
            u4 = make_vec(nm, [(1*m+j1, 1)])
            v4 = make_vec(mp, [(j1*p+k0, 1), (j0*p+k0, -1)])
            w4 = make_vec(pn, [(k0*n+0, 1), (k0*n+1, 1)])
            add_term(u4, v4, w4)
            
            # M5 = (A[0,j0] + A[0,j1]) * B[j1,k1]
            u5 = make_vec(nm, [(0*m+j0, 1), (0*m+j1, 1)])
            v5 = make_vec(mp, [(j1*p+k1, 1)])
            w5 = make_vec(pn, [(k0*n+0, -1), (k1*n+0, 1)])
            add_term(u5, v5, w5)
            
            # M6 = (A[1,j0] - A[0,j0]) * (B[j0,k0] + B[j0,k1])
            u6 = make_vec(nm, [(1*m+j0, 1), (0*m+j0, -1)])
            v6 = make_vec(mp, [(j0*p+k0, 1), (j0*p+k1, 1)])
            w6 = make_vec(pn, [(k1*n+1, 1)])
            add_term(u6, v6, w6)
            
            # M7 = (A[0,j1] - A[1,j1]) * (B[j1,k0] + B[j1,k1])
            u7 = make_vec(nm, [(0*m+j1, 1), (1*m+j1, -1)])
            v7 = make_vec(mp, [(j1*p+k0, 1), (j1*p+k1, 1)])
            w7 = make_vec(pn, [(k0*n+0, 1)])
            add_term(u7, v7, w7)
    
    # Handle remaining column j=4
    for k in range(p):
        # C[0,k] += A[0,4] * B[4,k]
        u_a = make_vec(nm, [(0*m+4, 1)])
        v_a = make_vec(mp, [(4*p+k, 1)])
        w_a = make_vec(pn, [(k*n+0, 1)])
        add_term(u_a, v_a, w_a)
        
        # C[1,k] += A[1,4] * B[4,k]
        u_b = make_vec(nm, [(1*m+4, 1)])
        v_b = make_vec(mp, [(4*p+k, 1)])
        w_b = make_vec(pn, [(k*n+1, 1)])
        add_term(u_b, v_b, w_b)
    
    # Now try to reduce the rank by finding linear dependencies
    # or by using additional algebraic identities
    
    # Let's verify the decomposition first
    def verify(U, V, W, n, m, p):
        # Check that sum_r U[i,j,r] * V[j,k,r] * W[k,i',r] = delta_{i,i'} * delta_{j,k}
        # Actually, the tensor for matrix multiplication is:
        # T[i,j,k,i'] = delta_{i,i'} if we consider C[i',k] = sum_j A[i,j] B[j,k]
        # Wait, let me re-derive.
        
        # C[i', k] = sum_j A[i, j] B[j, k]
        # The tensor T has indices (i, j, k) for the input and (i', k) for output
        # But in the standard tensor decomposition, we have:
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # where a = (i,j) for A, b = (j,k) for B, c = (i',k) for C
        
        # Actually, the convention is:
        # T[i,j,k,i'] = delta_{i,i'} * delta_{j,k} ... no
        
        # Let me think again. The matrix multiplication tensor is:
        # For each (i, j, k), the contribution to C is A[i,j] * B[j,k]
        # So the tensor T has entries T[i,j,k] which when contracted gives C
        # But in the bilinear form, we have:
        # C[i', k'] = sum_{i,j,k} T[i',k',i,j,k] A[i,j] B[k,k']
        # This is getting confusing. Let me use the standard definition.
        
        # The matrix multiplication tensor mu_{n,m,p} is a 3-tensor
        # with indices (i, j, k) where i in {0..n-1}, j in {0..m-1}, k in {0..p-1}
        # But actually it's a 6-index tensor or equivalently a 3-tensor with
        # composite indices.
        
        # Standard: mu_{(i,j),(j,k),(i',k')} = delta_{i,i'} * delta_{j,k} ... 
        # No, that's not right either.
        
        # Let me use: C[i', k'] = sum_j A[i, j] B[j, k']
        # So the tensor is T[i,j,k,i',k'] = delta_{i,i'} if j==k... no
        
        # The correct tensor is:
        # T[a,b,c] where a=(i,j), b=(j,k), c=(i',k')
        # T[a,b,c] = 1 if the indices are consistent, i.e., the j in a matches
        # the j in b, and i' in c matches i in a, and k' in c matches k in b.
        # Otherwise 0.
        
        # So T[(i,j),(j',k),(i',k')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
        
        # In our decomposition:
        # T[(i,j),(j',k),(i',k')] = sum_r U[(i,j),r] * V[(j',k),r] * W[(i',k'),r]
        
        # Let's verify this.
        
        rank_val = len(U)
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for i2 in range(n):
                            for k2 in range(p):
                                expected = 1 if (j == j2 and i == i2 and k == k2) else 0
                                actual = 0
                                for r in range(rank_val):
                                    actual += U[i*m+j][r] * V[j2*p+k][r] * W[i2*p+k2][r]
                                if abs(actual - expected) > 1e-9:
                                    return False
        return True
    
    # Verify
    is_valid = verify(U, V, W, n, m, p)
    
    if not is_valid:
        # Fallback to trivial decomposition
        rank = n * m * p
        U = []
        V = []
        W = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = make_vec(nm, [(i*m+j, 1)])
                    v = make_vec(mp, [(j*p+k, 1)])
                    w = make_vec(pn, [(k*n+i, 1)])
                    U.append(u)
                    V.append(v)
                    W.append(w)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
