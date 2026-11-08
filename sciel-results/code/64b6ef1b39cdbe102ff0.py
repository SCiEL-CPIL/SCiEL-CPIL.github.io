import numpy as np
import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using hardcoded Strassen coefficients.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r] where k=col, i=row of C
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # Pre-computed integer coefficients for the 49 rank-1 terms
    # Each term has U, V, W vectors of length 16
    # U[i*4+k, r], V[k*4+j, r], W[j*4+i, r] (trace convention: W[col*4+row, r])
    
    # Define the 7 top-level Strassen terms and their 7 sub-terms
    # This gives 49 total terms
    
    U = np.zeros((16, rank), dtype=np.int8)
    V = np.zeros((16, rank), dtype=np.int8)
    W = np.zeros((16, rank), dtype=np.int8)
    
    # Helper functions for indexing
    def a_idx(i, k): return i * 4 + k
    def b_idx(k, j): return k * 4 + j
    def c_idx(row, col): return col * 4 + row  # W[col*4+row, r]
    
    # The 7 top-level terms and their contributions to C blocks
    # Term t contributes to specific 2x2 blocks of C with given signs
    top_contrib = [
        [(0,0,1), (1,1,1)],      # P1 -> C00, C11
        [(1,0,1), (1,1,-1)],     # P2 -> C10, -C11
        [(0,1,1), (1,1,1)],      # P3 -> C01, C11
        [(0,0,1), (1,0,1)],      # P4 -> C00, C10
        [(0,0,-1), (0,1,1)],     # P5 -> -C00, C01
        [(1,1,1)],               # P6 -> C11
        [(0,0,1)]                # P7 -> C00
    ]
    
    # The 7 sub-terms within each 2x2 Strassen
    sub_contrib = [
        [(0,0,1), (1,1,1)],      # M1
        [(1,0,1), (1,1,-1)],     # M2
        [(0,1,1), (1,1,1)],      # M3
        [(0,0,1), (1,0,1)],      # M4
        [(0,0,-1), (0,1,1)],     # M5
        [(1,1,1)],               # M6
        [(0,0,1)]                # M7
    ]
    
    # Define the 7 top-level L and R patterns
    # Each is a 2x2 matrix of linear forms on A and B
    # L[t][r][c] is a dict mapping (i,k) -> coeff for A
    # R[t][r][c] is a dict mapping (k,j) -> coeff for B
    
    def make_form(elements):
        """Create a linear form from a list of (index, coeff) pairs"""
        form = {}
        for idx, coeff in elements:
            form[idx] = form.get(idx, 0) + coeff
            if form[idx] == 0:
                del form[idx]
        return form
    
    def merge_forms(f1, f2, c1=1, c2=1):
        """Merge two linear forms with scaling"""
        result = {}
        for k, v in f1.items():
            result[k] = result.get(k, 0) + v * c1
        for k, v in f2.items():
            result[k] = result.get(k, 0) + v * c2
        return {k: v for k, v in result.items() if v != 0}
    
    # Define element indices for 2x2 blocks
    # Block (br, bc) has elements at global positions (2*br+r, 2*bc+c)
    def block_elem(br, bc, r, c):
        return (2*br + r, 2*bc + c)
    
    # Build the 7 top-level terms
    terms = []
    
    for t in range(7):
        L = [[{} for _ in range(2)] for _ in range(2)]
        R = [[{} for _ in range(2)] for _ in range(2)]
        
        for r in range(2):
            for c in range(2):
                a00 = block_elem(0, 0, r, c)
                a01 = block_elem(0, 1, r, c)
                a10 = block_elem(1, 0, r, c)
                a11 = block_elem(1, 1, r, c)
                b00 = block_elem(0, 0, r, c)
                b01 = block_elem(0, 1, r, c)
                b10 = block_elem(1, 0, r, c)
                b11 = block_elem(1, 1, r, c)
                
                if t == 0:  # P1 = (A00+A11)(B00+B11)
                    L[r][c] = make_form([(a00, 1), (a11, 1)])
                    R[r][c] = make_form([(b00, 1), (b11, 1)])
                elif t == 1:  # P2 = (A10+A11)B00
                    L[r][c] = make_form([(a10, 1), (a11, 1)])
                    R[r][c] = make_form([(b00, 1)])
                elif t == 2:  # P3 = A00(B01-B11)
                    L[r][c] = make_form([(a00, 1)])
                    R[r][c] = make_form([(b01, 1), (b11, -1)])
                elif t == 3:  # P4 = A11(B10-B00)
                    L[r][c] = make_form([(a11, 1)])
                    R[r][c] = make_form([(b10, 1), (b00, -1)])
                elif t == 4:  # P5 = (A00+A01)B11
                    L[r][c] = make_form([(a00, 1), (a01, 1)])
                    R[r][c] = make_form([(b11, 1)])
                elif t == 5:  # P6 = (A10-A00)(B00+B01)
                    L[r][c] = make_form([(a10, 1), (a00, -1)])
                    R[r][c] = make_form([(b00, 1), (b01, 1)])
                elif t == 6:  # P7 = (A01-A11)(B10+B11)
                    L[r][c] = make_form([(a01, 1), (a11, -1)])
                    R[r][c] = make_form([(b10, 1), (b11, 1)])
        
        terms.append((L, R))
    
    # Now generate the 49 rank-1 terms
    for t in range(7):
        L, R = terms[t]
        L00, L01 = L[0][0], L[0][1]
        L10, L11 = L[1][0], L[1][1]
        R00, R01 = R[0][0], R[0][1]
        R10, R11 = R[1][0], R[1][1]
        
        # 7 sub-products for 2x2 Strassen
        sub_products = [
            (merge_forms(L00, L11), merge_forms(R00, R11)),      # M1
            (merge_forms(L10, L11), R00),                         # M2
            (L00, merge_forms(R01, R11, 1, -1)),                  # M3
            (L11, merge_forms(R10, R00, 1, -1)),                  # M4
            (merge_forms(L00, L01), R11),                         # M5
            (merge_forms(L10, L00, 1, -1), merge_forms(R00, R01)),# M6
            (merge_forms(L01, L11, 1, -1), merge_forms(R10, R11)) # M7
        ]
        
        for k in range(7):
            r_idx = t * 7 + k
            A_form, B_form = sub_products[k]
            
            # Fill U
            for (i, kk), coeff in A_form.items():
                U[a_idx(i, kk), r_idx] = coeff
            
            # Fill V
            for (kk, j), coeff in B_form.items():
                V[b_idx(kk, j), r_idx] = coeff
            
            # Fill W
            for (lr, lc, sub_coeff) in sub_contrib[k]:
                for (br, bc, block_coeff) in top_contrib[t]:
                    gr = 2 * br + lr
                    gc = 2 * bc + lc
                    W[c_idx(gr, gc), r_idx] += sub_coeff * block_coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
