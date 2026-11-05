DIMENSIONS = [4, 4, 7]

def construct():
    import time
    import numpy as np
    from itertools import product
    
    start_time = time.time()
    deadline = 150.0
    
    n, m, p = 4, 4, 7
    
    # Try to find a low-rank decomposition using cyclic convolution embedding
    # The idea: embed 4x4x7 multiplication into Z_N x Z_N x Z_N where
    # multiplication becomes pointwise in Fourier domain
    
    best_rank = None
    best_decomp = None
    
    # Schoolbook baseline: n*m*p = 112 terms
    def schoolbook():
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0] * (n * m)
                    u[i * m + j] = 1
                    v = [0] * (m * p)
                    v[j * p + k] = 1
                    w = [0] * (p * n)
                    w[k * n + i] = 1
                    terms.append((u, v, w))
        return terms
    
    # Try cyclic convolution approach for various group sizes
    # For 4x4x7, we need to find a group where the multiplication tensor
    # has low rank in the Fourier basis
    
    # Approach: use the fact that matrix multiplication A*B where A is 4x4, B is 4x7
    # can be viewed as a bilinear map. We'll try to find a low-rank decomposition
    # by exploiting structure.
    
    # First, let's try a block-based approach with Strassen-like ideas
    # Split into 2x2 blocks and use Strassen (rank 7) for each 2x2x2 block
    
    def strassen_2x2_block(i_offset, j_offset, k_offset):
        """Return Strassen terms for a 2x2x2 block at given offsets"""
        terms = []
        
        # Strassen's 7 products for 2x2 matrix multiplication
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # For 2x2x2 block, indices:
        # i in [i_offset, i_offset+1], j in [j_offset, j_offset+1], k in [k_offset, k_offset+1]
        
        def make_term(u_coeffs, v_coeffs, w_coeffs):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            for (idx, val) in u_coeffs:
                u[idx] = val
            for (idx, val) in v_coeffs:
                v[idx] = val
            for (idx, val) in w_coeffs:
                w[idx] = val
            return (u, v, w)
        
        i0, i1 = i_offset, i_offset + 1
        j0, j1 = j_offset, j_offset + 1
        k0, k1 = k_offset, k_offset + 1
        
        # M1 = (A00+A11)(B00+B11)
        u1 = [(i0*m+j0, 1), (i1*m+j1, 1)]
        v1 = [(j0*p+k0, 1), (j1*p+k1, 1)]
        w1 = [(k0*n+i0, 1), (k1*n+i1, 1)]
        terms.append(make_term(u1, v1, w1))
        
        # M2 = (A10+A11)B00
        u2 = [(i1*m+j0, 1), (i1*m+j1, 1)]
        v2 = [(j0*p+k0, 1)]
        w2 = [(k0*n+i1, 1)]
        terms.append(make_term(u2, v2, w2))
        
        # M3 = A00(B10-B11)
        u3 = [(i0*m+j0, 1)]
        v3 = [(j1*p+k0, 1), (j1*p+k1, -1)]
        w3 = [(k0*n+i0, 1)]
        terms.append(make_term(u3, v3, w3))
        
        # M4 = A11(B01-B10)
        u4 = [(i1*m+j1, 1)]
        v4 = [(j0*p+k1, 1), (j0*p+k0, -1)]
        w4 = [(k1*n+i1, 1)]
        terms.append(make_term(u4, v4, w4))
        
        # M5 = (A00+A01)B11
        u5 = [(i0*m+j0, 1), (i0*m+j1, 1)]
        v5 = [(j1*p+k1, 1)]
        w5 = [(k1*n+i0, 1)]
        terms.append(make_term(u5, v5, w5))
        
        # M6 = (A10-A00)(B00+B10)
        u6 = [(i1*m+j0, 1), (i0*m+j0, -1)]
        v6 = [(j0*p+k0, 1), (j1*p+k0, 1)]
        w6 = [(k0*n+i1, 1)]
        terms.append(make_term(u6, v6, w6))
        
        # M7 = (A01-A11)(B01+B11)
        u7 = [(i0*m+j1, 1), (i1*m+j1, -1)]
        v7 = [(j0*p+k1, 1), (j1*p+k1, 1)]
        w7 = [(k1*n+i0, 1)]
        terms.append(make_term(u7, v7, w7))
        
        return terms
    
    # For 4x4x7, we have:
    # - 4x4x4 part: can use 4 blocks of 2x2x2, each with Strassen (7 terms) = 28 terms
    # - 4x4x3 part: 12 terms (schoolbook)
    # Total: 40 terms
    
    # But we can do better by overlapping the Strassen decompositions
    # Let's try a more careful construction
    
    terms = []
    
    # Handle 4x4x4 part using Strassen on 2x2 blocks
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0 in [0, 2]:
                terms.extend(strassen_2x2_block(i0, j0, k0))
    
    # Handle 4x4x3 part (columns 4, 5, 6)
    for k in range(4, 7):
        for i in range(4):
            for j in range(4):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + k] = 1
                w = [0] * (p * n)
                w[k * n + i] = 1
                terms.append((u, v, w))
    
    rank = len(terms)
    
    # Verify the decomposition
    def verify_decomposition(terms):
        if not terms:
            return False
        rank = len(terms)
        U = np.array([t[0] for t in terms]).T  # (n*m, rank)
        V = np.array([t[1] for t in terms]).T  # (m*p, rank)
        W = np.array([t[2] for t in terms]).T  # (p*n, rank)
        
        # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # For matrix multiplication, T[i,j,k] should be 1 if this is the (i,j,k) entry
        # Actually, the tensor for matrix multiplication has T[i,j,k] = 1 for all i,j,k
        # because C[i,k] = sum_j A[i,j] * B[j,k]
        
        # Let's check: for each (i,j,k), the sum should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1) > 1e-10:
                        return False
        return True
    
    if verify_decomposition(terms):
        best_rank = rank
        best_decomp = terms
    
    # Try to find even lower rank using random search with time limit
    if time.time() < deadline - 10:
        import random
        random.seed(42)
        
        # Try random linear combinations to reduce rank
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            # Create random perturbation of the current decomposition
            new_terms = []
            for t in terms:
                u, v, w = t
                # Randomly scale
                scale = random.uniform(0.5, 2.0)
                u_new = [x * scale for x in u]
                v_new = [x / scale for x in v]
                w_new = w[:]
                new_terms.append((u_new, v_new, w_new))
            
            # Check if we can merge some terms
            # This is a heuristic - just check if the decomposition is still valid
            if verify_decomposition(new_terms):
                pass  # Same rank, no improvement
    
    # If we have a valid decomposition, format it
    if best_decomp is not None:
        rank = len(best_decomp)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for c, (u, v, w) in enumerate(best_decomp):
            for r in range(n * m):
                U[r][c] = u[r]
            for r in range(m * p):
                V[r][c] = v[r]
            for r in range(p * n):
                W[r][c] = w[r]
        
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        terms = schoolbook()
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
        
        return {"U": U, "V": V, "W": W}
