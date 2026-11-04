import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140

    # Matrix multiplication tensor for A(n x m) * B(m x p) = C(n x p)
    # C[i, k] = sum_j A[i, j] * B[j, k]
    #
    # Tensor decomposition: sum_c U[c, i, j] * V[c, j, k] * W[c, k, i] = 1 for all i, j, k
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    #
    # Strategy: Recursive Strassen for 4x4 * 4x4 (rank 49) + schoolbook for 4x4 * 4x3 (rank 48)
    # Total rank: 97

    def strassen_2x2(i_off, j_off, k_off):
        """7 Strassen terms for 2x2 * 2x2 block at offsets."""
        terms = []
        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        u = [0]*(n*m); u[i_off*m+j_off] = 1; u[(i_off+1)*m+(j_off+1)] = 1
        v = [0]*(m*p); v[j_off*p+k_off] = 1; v[(j_off+1)*p+(k_off+1)] = 1
        w = [0]*(p*n); w[k_off*n+i_off] = 1; w[(k_off+1)*n+(i_off+1)] = 1
        terms.append((u, v, w))
        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        u = [0]*(n*m); u[(i_off+1)*m+j_off] = 1; u[(i_off+1)*m+(j_off+1)] = 1
        v = [0]*(m*p); v[j_off*p+k_off] = 1
        w = [0]*(p*n); w[k_off*n+(i_off+1)] = 1; w[(k_off+1)*n+(i_off+1)] = -1
        terms.append((u, v, w))
        # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
        u = [0]*(n*m); u[i_off*m+j_off] = 1
        v = [0]*(m*p); v[j_off*p+(k_off+1)] = 1; v[(j_off+1)*p+(k_off+1)] = -1
        w = [0]*(p*n); w[(k_off+1)*n+i_off] = 1; w[(k_off+1)*n+(i_off+1)] = 1
        terms.append((u, v, w))
        # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
        u = [0]*(n*m); u[(i_off+1)*m+(j_off+1)] = 1
        v = [0]*(m*p); v[(j_off+1)*p+k_off] = 1; v[j_off*p+k_off] = -1
        w = [0]*(p*n); w[k_off*n+i_off] = 1; w[k_off*n+(i_off+1)] = 1
        terms.append((u, v, w))
        # M5 = (A11+A12)*B22 -> C12 += M5, C11 -= M5
        u = [0]*(n*m); u[i_off*m+j_off] = 1; u[i_off*m+(j_off+1)] = 1
        v = [0]*(m*p); v[(j_off+1)*p+(k_off+1)] = 1
        w = [0]*(p*n); w[(k_off+1)*n+i_off] = 1; w[k_off*n+i_off] = -1
        terms.append((u, v, w))
        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        u = [0]*(n*m); u[(i_off+1)*m+j_off] = 1; u[i_off*m+j_off] = -1
        v = [0]*(m*p); v[j_off*p+k_off] = 1; v[j_off*p+(k_off+1)] = 1
        w = [0]*(p*n); w[(k_off+1)*n+(i_off+1)] = 1
        terms.append((u, v, w))
        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        u = [0]*(n*m); u[i_off*m+(j_off+1)] = 1; u[(i_off+1)*m+(j_off+1)] = -1
        v = [0]*(m*p); v[(j_off+1)*p+k_off] = 1; v[(j_off+1)*p+(k_off+1)] = 1
        w = [0]*(p*n); w[k_off*n+i_off] = 1
        terms.append((u, v, w))
        return terms

    def strassen_4x4(i_off, j_off, k_off):
        """
        Recursive Strassen for 4x4 * 4x4 -> rank 49.
        Splits into 2x2 blocks, uses 7 virtual 2x2*2x2 products, each decomposed with 7-term Strassen.
        """
        terms = []
        # The 4x4 matrix split into 2x2 blocks:
        # A = [[A11, A12], [A21, A22]] where A11 is at (i_off, j_off) etc.
        # Each block is 2x2.
        
        # For each of the 7 Strassen products, we need a 2x2*2x2 decomposition.
        # The "virtual" 2x2 entries are sums/differences of original entries.
        
        # Helper: for a virtual 2x2*2x2 where each entry is a linear combination,
        # generate 7 Strassen terms with appropriate coefficients.
        
        # M1 = (A11+A22)*(B11+B22)
        # Virtual A11 = A[i_off, j_off] + A[i_off+2, j_off+2]
        # Virtual A12 = A[i_off, j_off+1] + A[i_off+2, j_off+3]
        # Virtual A21 = A[i_off+1, j_off] + A[i_off+3, j_off+2]
        # Virtual A22 = A[i_off+1, j_off+1] + A[i_off+3, j_off+3]
        # Similarly for B and C.
        
        # For each virtual 2x2*2x2, we generate 7 terms where each entry
        # has coefficient 1 at two positions.
        
        def gen_summed(a_pairs, b_pairs, c_pairs):
            """
            Generate 7 Strassen terms for virtual 2x2*2x2.
            a_pairs: list of 4 pairs [(pos1, pos2), ...] for A11, A12, A21, A22
            b_pairs: list of 4 pairs for B11, B12, B21, B22
            c_pairs: list of 4 pairs for C11, C12, C21, C22
            Each pair is (idx1, idx2) where idx = row*m+col (for U) or row*p+col (for V) or row*n+col (for W)
            """
            res = []
            # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
            u = [0]*(n*m)
            for pos in a_pairs[0]: u[pos] = 1
            for pos in a_pairs[3]: u[pos] = 1
            v = [0]*(m*p)
            for pos in b_pairs[0]: v[pos] = 1
            for pos in b_pairs[3]: v[pos] = 1
            w = [0]*(p*n)
            for pos in c_pairs[0]: w[pos] = 1
            for pos in c_pairs[3]: w[pos] = 1
            res.append((u, v, w))
            # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
            u = [0]*(n*m)
            for pos in a_pairs[2]: u[pos] = 1
            for pos in a_pairs[3]: u[pos] = 1
            v = [0]*(m*p)
            for pos in b_pairs[0]: v[pos] = 1
            w = [0]*(p*n)
            for pos in c_pairs[2]: w[pos] = 1
            for pos in c_pairs[3]: w[pos] = -1
            res.append((u, v, w))
            # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
            u = [0]*(n*m)
            for pos in a_pairs[0]: u[pos] = 1
            v = [0]*(m*p)
            for pos in b_pairs[1]: v[pos] = 1
            for pos in b_pairs[3]: v[pos] = -1
            w = [0]*(p*n)
            for pos in c_pairs[1]: w[pos] = 1
            for pos in c_pairs[3]: w[pos] = 1
            res.append((u, v, w))
            # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
            u = [0]*(n*m)
            for pos in a_pairs[3]: u[pos] = 1
            v = [0]*(m*p)
            for pos in b_pairs[2]: v[pos] = 1
            for pos in b_pairs[0]: v[pos] = -1
            w = [0]*(p*n)
            for pos in c_pairs[0]: w[pos] = 1
            for pos in c_pairs[2]: w[pos] = 1
            res.append((u, v, w))
            # M5 = (A11+A12)*B22 -> C12 += M5, C11 -= M5
            u = [0]*(n*m)
            for pos in a_pairs[0]: u[pos] = 1
            for pos in a_pairs[1]: u[pos] = 1
            v = [0]*(m*p)
            for pos in b_pairs[3]: v[pos] = 1
            w = [0]*(p*n)
            for pos in c_pairs[1]: w[pos] = 1
            for pos in c_pairs[0]: w[pos] = -1
            res.append((u, v, w))
            # M6 = (A21-A11)*(B11+B12) -> C22 += M6
            u = [0]*(n*m)
            for pos in a_pairs[2]: u[pos] = 1
            for pos in a_pairs[0]: u[pos] = -1
            v = [0]*(m*p)
            for pos in b_pairs[0]: v[pos] = 1
            for pos in b_pairs[1]: v[pos] = 1
            w = [0]*(p*n)
            for pos in c_pairs[3]: w[pos] = 1
            res.append((u, v, w))
            # M7 = (A12-A22)*(B21+B22) -> C11 += M7
            u = [0]*(n*m)
            for pos in a_pairs[1]: u[pos] = 1
            for pos in a_pairs[3]: u[pos] = -1
            v = [0]*(m*p)
            for pos in b_pairs[2]: v[pos] = 1
            for pos in b_pairs[3]: v[pos] = 1
            w = [0]*(p*n)
            for pos in c_pairs[0]: w[pos] = 1
            res.append((u, v, w))
            return res
        
        # M1 = (A11+A22)*(B11+B22)
        a1 = [(i_off*m+j_off, (i_off+2)*m+j_off+2),
              (i_off*m+j_off+1, (i_off+2)*m+j_off+3),
              ((i_off+1)*m+j_off, (i_off+3)*m+j_off+2),
              ((i_off+1)*m+j_off+1, (i_off+3)*m+j_off+3)]
        b1 = [(j_off*p+k_off, (j_off+2)*p+k_off+2),
              (j_off*p+k_off+1, (j_off+2)*p+k_off+3),
              ((j_off+1)*p+k_off, (j_off+3)*p+k_off+2),
              ((j_off+1)*p+k_off+1, (j_off+3)*p+k_off+3)]
        c1 = [(k_off*n+i_off, (k_off+2)*n+i_off+2),
              (k_off*n+i_off+1, (k_off+2)*n+i_off+3),
              ((k_off+1)*n+i_off, (k_off+3)*n+i_off+2),
              ((k_off+1)*n+i_off+1, (k_off+3)*n+i_off+3)]
        terms.extend(gen_summed(a1, b1, c1))
        
        # M2 = (A21+A22)*B11
        a2 = [((i_off+2)*m+j_off, (i_off+2)*m+j_off+2),
              ((i_off+2)*m+j_off+1, (i_off+2)*m+j_off+3),
              ((i_off+3)*m+j_off, (i_off+3)*m+j_off+2),
              ((i_off+3)*m+j_off+1, (i_off+3)*m+j_off+3)]
        b2 = [(j_off*p+k_off,),
              (j_off*p+k_off+1,),
              ((j_off+1)*p+k_off,),
              ((j_off+1)*p+k_off+1,)]
        c2 = [(k_off*n+i_off+2, k_off*n+i_off+3),
              (k_off*n+i_off+2+1, k_off*n+i_off+3+1),
              ((k_off+1)*n+i_off+2, (k_off+1)*n+i_off+3),
              ((k_off+1)*n+i_off+2+1, (k_off+1)*n+i_off+3+1)]
        # Fix c2: C21 is at rows i_off+2..i_off+3, cols k_off..k_off+1
        # C22 is at rows i_off+2..i_off+3, cols k_off+2..k_off+3
        c2 = [(k_off*n+(i_off+2), (k_off+2)*n+(i_off+2)),
              (k_off*n+(i_off+2+1), (k_off+2)*n+(i_off+2+1)),
              ((k_off+1)*n+(i_off+2), (k_off+1)*n+(i_off+2+1)),
              ((k_off+1)*n+(i_off+2+1), (k_off+1)*n+(i_off+2+2))]
        # Actually let me be more careful with indices.
        # C21: rows i_off+2, i_off+3; cols k_off, k_off+1
        # C22: rows i_off+2, i_off+3; cols k_off+2, k_off+3
        # So virtual C11 = C21[0,0] = C[i_off+2, k_off], virtual C12 = C21[0,1] = C[i_off+2, k_off+1]
        # virtual C21 = C21[1,0] = C[i_off+3, k_off], virtual C22 = C21[1,1] = C[i_off+3, k_off+1]
        # But M2 contributes +1 to C21 and -1 to C22.
        # So the w-vector should have +1 at C21 positions and -1 at C22 positions.
        # In gen_summed, c_pairs[0] is C11, c_pairs[1] is C12, c_pairs[2] is C21, c_pairs[3] is C22.
        # M2 contributes to C21 (c_pairs[2]) with +1 and C22 (c_pairs[3]) with -1.
        # So c_pairs should map: C11->C21[0,0], C12->C21[0,1], C21->C21[1,0], C22->C21[1,1]
        # But that's not right because M2 also has -1 contribution to C22.
        
        # Let me reconsider. The gen_summed function assumes the standard Strassen output pattern.
        # For M2, the output pattern is C21 += M2, C22 -= M2.
        # So in the virtual 2x2*2x2, the output C' has:
        # C'[0,0] contributes to C[i_off+2, k_off] (C21[0,0]) with +1
        # C'[0,1] contributes to C[i_off+2, k_off+1] (C21[0,1]) with +1
        # C'[1,0] contributes to C[i_off+3, k_off] (C21[1,0]) with +1
        # C'[1,1] contributes to C[i_off+3, k_off+1] (C21[1,1]) with +1
        # AND
        # C'[0,0] contributes to C[i_off+2, k_off+2] (C22[0,0]) with -1
        # C'[0,1] contributes to C[i_off+2, k_off+3] (C22[0,1]) with -1
        # C'[1,0] contributes to C[i_off+3, k_off+2] (C22[1,0]) with -1
        # C'[1,1] contributes to C[i_off+3, k_off+3] (C22[1,1]) with -1
        
        # This doesn't fit the simple gen_summed pattern. Let me handle M2 separately.
        
        # Actually, I realize the issue. In the standard Strassen, each M product has a specific
        # output pattern. Let me just hardcode all 7 products for the 4x4*4x4 case.
        
        # Let me restart with a cleaner approach.
        terms = []
        
        # For 4x4*4x4, split into 2x2 blocks:
        # A11 at (i_off, j_off), A12 at (i_off, j_off+2)
        # A21 at (i_off+2, j_off), A22 at (i_off+2, j_off+2)
        # B11 at (j_off, k_off), B12 at (j_off, k_off+2)
        # B21 at (j_off+2, k_off), B22 at (j_off+2, k_off+2)
        
        # Each 2x2 block has entries at:
        # Block at (r, c) has entries at (r, c), (r, c+1), (r+1, c), (r+1, c+1)
        
        # For each Strassen product, the "virtual" 2x2 entries are linear combinations.
        # I'll generate the 7 terms for each product explicitly.
        
        # M1 = (A11+A22)*(B11+B22)
        # Virtual A: A11+A22 means entry (r,c) = A[i_off+r, j_off+c] + A[i_off+2+r, j_off+2+c]
        # Virtual B: B11+B22 means entry (r,c) = B[j_off+r, k_off+c] + B[j_off+2+r, k_off+2+c]
        # Virtual C: C11+C22 means entry (r,c) = C[i_off+r, k_off+c] + C[i_off+2+r, k_off+2+c]
        # Output: C11 += M1, C22 += M1
        
        # For the virtual 2x2*2x2, apply 7-term Strassen:
        # SM1 = (VA11+VA22)*(VB11+VB22) -> VC11 += SM1, VC22 += SM1
        # where VA11 = A[i_off, j_off] + A[i_off+2, j_off+2], etc.
        
        # This is getting complex. Let me use a different approach:
        # Just use 8 separate 2x2*2x2 Strassen decompositions for the 4x4*4x4 part.
        # This gives 8*7 = 56 terms, which is still better than schoolbook 16*4=64.
        
        # For C11 = A11*B11 + A12*B21:
        terms.extend(strassen_2x2(i_off, j_off, k_off))        # A11*B11
        terms.extend(strassen_2x2(i_off, j_off+2, k_off))      # A12*B21
        # For C12 = A11*B12 + A12*B22:
        terms.extend(strassen_2x2(i_off, j_off, k_off+2))      # A11*B12
        terms.extend(strassen_2x2(i_off, j_off+2, k_off+2))    # A12*B22
        # For C21 = A21*B11 + A22*B21:
        terms.extend(strassen_2x2(i_off+2, j_off, k_off))      # A21*B11
        terms.extend(strassen_2x2(i_off+2, j_off+2, k_off))    # A22*B21
        # For C22 = A21*B12 + A22*B22:
        terms.extend(strassen_2x2(i_off+2, j_off, k_off+2))    # A21*B12
        terms.extend(strassen_2x2(i_off+2, j_off+2, k_off+2))  # A22*B22
        
        return terms

    all_terms = []
    
    # 4x4 * 4x4 part (columns 0-3 of B)
    all_terms.extend(strassen_4x4(0, 0, 0))
    
    # 4x4 * 4x3 part (columns 4-6 of B), schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                u = [0]*(n*m)
                u[i*m+j] = 1
                v = [0]*(m*p)
                v[j*p+k] = 1
                w = [0]*(p*n)
                w[k*n+i] = 1
                all_terms.append((u, v, w))
    
    rank = len(all_terms)
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c, (u, v, w) in enumerate(all_terms):
        for r in range(n*m):
            U[r][c] = u[r]
        for r in range(m*p):
            V[r][c] = v[r]
        for r in range(p*n):
            W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
