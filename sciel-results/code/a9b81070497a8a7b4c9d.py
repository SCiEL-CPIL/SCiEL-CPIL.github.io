"""Matrix multiplication 3,4,7 - rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Return {'U', 'V', 'W'} with shapes (12, rank), (28, rank), (21, rank)
    that exactly decompose the 3x4 * 4x7 matrix multiplication tensor.
    Strategy: combine Strassen for 2x2x2 blocks (rank 7) with schoolbook for the rest.
    """
    n, m, p = 3, 4, 7
    
    # We'll build the decomposition term by term.
    # Each term is a triple (u_vec, v_vec, w_vec) where:
    #   u_vec has length 12 (index i*m+j for i in 0..n-1, j in 0..m-1)
    #   v_vec has length 28 (index j*p+k for j in 0..m-1, k in 0..p-1)
    #   w_vec has length 21 (index k*n+i for k in 0..p-1, i in 0..n-1)
    
    terms = []  # list of (u, v, w) sparse vectors
    
    # We partition the matrices into blocks:
    # A is 3x4, B is 4x7, C is 3x7
    # 
    # Strategy: Use Strassen-like approach on overlapping 2x2 blocks.
    # 
    # Let's think about this more carefully.
    # 
    # The schoolbook approach gives rank 3*4*7 = 84.
    # 
    # For Strassen on 2x2x2 blocks: if we can group the computation into 
    # 2x2 subproblems, each of rank 7 instead of 8.
    #
    # A is 3x4: we can split rows into {0,1} and {2}, cols into {0,1} and {2,3}
    # B is 4x7: split rows into {0,1} and {2,3}, cols into various 2-col groups
    # C is 3x7: output
    #
    # Let me think about a different partitioning.
    #
    # Actually, let's use a more systematic approach:
    # 
    # Partition A (3x4) as:
    #   A = [[A00, A01], [A10, A11]] where A00 is 2x2, A01 is 2x2, A10 is 1x2, A11 is 1x2
    # 
    # Partition B (4x7) as:
    #   B = [[B00, B01], [B10, B11]] where B00 is 2x2, B01 is 2x2 (cols 0-1, 2-3), etc.
    #   Actually B is 4x7, so row split: {0,1} and {2,3}, col split into groups of 2 with last being 1
    #
    # This is getting complex. Let me use a cleaner approach.
    #
    # Approach: Use the fact that we can apply Strassen to any 2x2x2 sub-block.
    # 
    # Let's partition:
    # - A rows: {0,1} (top 2) and {2} (bottom 1)
    # - A cols / B rows: {0,1} (top 2) and {2,3} (bottom 2)  
    # - B cols: {0,1}, {2,3}, {4,5}, {6}
    #
    # Block C[i,j] = sum_k A[i,k]*B[k,j]
    #
    # For the 2x2x2 blocks (i in {0,1}, k in {0,1}, j in {0,1}):
    #   C[{0,1},{0,1}] contribution from A[{0,1},{0,1}] * B[{0,1},{0,1}] can use Strassen (rank 7)
    #   C[{0,1},{2,3}] contribution from A[{0,1},{0,1}] * B[{0,1},{2,3}] can use Strassen (rank 7)
    #   C[{0,1},{4,5}] contribution from A[{0,1},{0,1}] * B[{0,1},{4,5}] can use Strassen (rank 7)
    #   C[{0,1},{6}]    contribution from A[{0,1},{0,1}] * B[{0,1},{6}]    needs special handling
    #
    # Hmm, this is getting messy. Let me use a simpler and more effective strategy.
    
    # STRATEGY: Use a block decomposition where we group columns of B in pairs.
    # 
    # A is 3x4, B is 4x7.
    # 
    # Split B's columns into: {0,1}, {2,3}, {4,5}, {6}
    # Split A's rows into: {0,1}, {2}
    # Split A's cols / B's rows into: {0,1}, {2,3}
    #
    # For each pair of B-columns (j1,j2) and each pair of A-rows (i1,i2):
    #   The subproblem is A[i1,i2; 0:4] * B[0:4; j1,j2] -> C[i1,i2; j1,j2]
    #   This is a 2x4 * 4x2 = 2x2 problem.
    #
    # For 2x4 * 4x2: we can further decompose.
    #   A_sub is 2x4, B_sub is 4x2.
    #   C_sub[i,j] = sum_k A_sub[i,k] * B_sub[k,j]
    #
    # We can use the standard 2x2x2 Strassen if we can express 2x4*4x2 as combinations of 2x2x2.
    #
    # Actually, 2x4*4x2 has rank at most 2*4*2 = 16 (schoolbook).
    # With Strassen-like: we can split A_sub into A_sub[:, :2] and A_sub[:, 2:], 
    # and B_sub into B_sub[:2, :] and B_sub[2:, :].
    # Then C_sub = A_sub[:,:2]*B_sub[:2,:] + A_sub[:,2:]*B_sub[2:,:]
    # Each term is 2x2 * 2x2 = 2x2, which has Strassen rank 7.
    # So total rank for 2x4*4x2 is at most 7+7 = 14.
    #
    # But we also have the single-row (i=2) and single-col (j=6) cases.
    #
    # Let me count:
    # Row pairs: (0,1), (2,) -> 2 groups
    # Col pairs: (0,1), (2,3), (4,5), (6) -> 4 groups
    #
    # For row pair (0,1) and col pair (j1,j2) [j1,j2 both exist]:
    #   2x4 * 4x2, rank <= 14 (two Strassen 2x2x2)
    # For row pair (0,1) and col (6):
    #   2x4 * 4x1, rank <= 2*4*1 = 8 (schoolbook) or we can do better
    # For row (2) and col pair (j1,j2):
    #   1x4 * 4x2, rank <= 1*4*2 = 8 (schoolbook)
    # For row (2) and col (6):
    #   1x4 * 4x1, rank <= 4 (schoolbook)
    #
    # Total: 3 * 14 + 8 + 3 * 8 + 4 = 42 + 8 + 24 + 4 = 78
    #
    # That's better than 84! Let me see if we can do even better.
    #
    # For 2x4 * 4x1: C[i] = sum_k A[i,k]*B[k,0]
    # We can write this as 2x2*2x1 + 2x2*2x1 using Strassen-like:
    # Actually 2x2*2x1 has rank at most 2*2*1=4, but we can use 3 multiplications:
    #   (A[0,0]+A[0,1])*(B[0,0]+B[1,0]) - A[0,0]*B[0,0] - A[0,1]*B[1,0]
    # That's still 4. Hmm, for 2x2*2x1, the tensor rank is actually 4 (I think).
    # Wait: 2x2*2x1: C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0], C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # This is like 2 independent dot products of length 2, so rank 4? No...
    # Actually it's a 2x2x1 tensor. The rank of a 2x2x1 tensor... 
    # C[0] = a*b + c*d, C[1] = e*b + f*d
    # This can be done with 4 multiplications (schoolbook). Can we do 3?
    # Let's see: we need to compute two bilinear forms. The tensor is (a,b,c,d,e,f) -> (ab+cd, eb+fd)
    # I believe the rank is 4 for this.
    #
    # So 2x4*4x1: split into (2x2*2x1) + (2x2*2x1) = 4+4 = 8. Same as schoolbook.
    #
    # For 1x4*4x2: C[j] = sum_k A[k]*B[k,j] for j=0,1
    # = (A[0]*B[0,j] + A[1]*B[1,j]) + (A[2]*B[2,j] + A[3]*B[3,j])
    # = 1x2*2x2 + 1x2*2x2
    # 1x2*2x2: C[j] = A[0]*B[0,j] + A[1]*B[1,j]
    # This is rank 4 (schoolbook). Can we do better? 
    # C[0] = a*b + c*d, C[1] = a*e + c*f
    # This is like computing a*col0 + c*col1 of B. Rank 4.
    #
    # Hmm, so we're at 78. Let me see if there's a better partitioning.
    
    # Alternative: What if we use Strassen more aggressively?
    # 
    # Let me try a different approach: direct rank reduction via known algorithms.
    #
    # For m x n x p matrix multiplication, the best known upper bounds for small dimensions:
    # 3x4x7: I don't have a specific known bound, but let's try to get close to 78 or lower.
    
    # Let me implement the 78-rank construction:
    
    def idx_a(i, j):
        """Index into U: i is row of A (0..n-1), j is col of A / row of B (0..m-1)"""
        return i * m + j
    
    def idx_b(j, k):
        """Index into V: j is row of B (0..m-1), k is col of B (0..p-1)"""
        return j * p + k
    
    def idx_w(k, i):
        """Index into W: k is col of B / row of C (0..p-1), i is row of C (0..n-1)"""
        return k * n + i
    
    # For Strassen 2x2x2 on A block A[ia:ia+2, ja:ja+2] * B block B[ja:ja+2, kb:kb+2] -> C block C[ia:ia+2, kb:kb+2]
    # The 7 Strassen terms:
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    #
    # C[0,0] = M1 - M5 + M6 - M4 + M2  -- wait let me get this right
    # C[0,0] = M1 - M3 - M4 + M7  -- no
    #
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B22-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M5 + M6 - M4
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    #
    # Let me use 0-indexed: A[0,0], A[0,1], A[1,0], A[1,1] etc.
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[1,0]-B[0,0])
    # M4 = A[1,1] * (B[1,1]-B[0,1])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    #
    # C[0,0] = M1 - M5 + M6 - M4
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M7

    def strassen_222(ia, ja, kb):
        """
        Add Strassen terms for A[ia:ia+2, ja:ja+2] * B[ja:ja+2, kb:kb+2] -> C[ia:ia+2, kb:kb+2]
        ia: starting row in A (0-indexed), ja: starting col in A / row in B, kb: starting col in B
        Returns list of (u, v, w) sparse terms (each a dict from index to coefficient)
        """
        terms = []
        
        # A indices
        a00 = idx_a(ia, ja)
        a01 = idx_a(ia, ja+1)
        a10 = idx_a(ia+1, ja)
        a11 = idx_a(ia+1, ja+1)
        
        # B indices
        b00 = idx_b(ja, kb)
        b01 = idx_b(ja, kb+1)
        b10 = idx_b(ja+1, kb)
        b11 = idx_b(ja+1, kb+1)
        
        # C indices
        c00 = idx_w(kb, ia)
        c01 = idx_w(kb+1, ia)
        c10 = idx_w(kb, ia+1)
        c11 = idx_w(kb+1, ia+1)
        
        def make_term(u_coeffs, v_coeffs, w_coeffs):
            """u_coeffs, v_coeffs, w_coeffs are dicts: index -> coefficient"""
            return (u_coeffs, v_coeffs, w_coeffs)
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        terms.append(make_term(
            {a00: 1, a11: 1},
            {b00: 1, b11: 1},
            {c00: 1, c01: 0, c10: 0, c11: -1}  # C[0,0] += 1, C[1,1] -= 1... 
            # Wait, I need to recheck the Strassen formulas
        ))
        
        # Let me redo this more carefully.
        # Standard Strassen (using 1-indexed A, B, C):
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B11)
        # M4 = A22(B22-B12)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 - M5 + M6 - M4
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        #
        # In 0-indexed:
        # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
        # M2 = (A[1,0]+A[1,1])B[0,0]
        # M3 = A[0,0](B[1,0]-B[0,0])
        # M4 = A[1,1](B[1,1]-B[0,1])
        # M5 = (A[0,0]+A[0,1])B[1,1]
        # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
        # M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
        #
        # C[0,0] = M1 - M5 + M6 - M4
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M7
        
        # Each M contributes to multiple C entries with coefficients.
        # For a term M with A-coeffs and B-coeffs, its contribution to C is:
        #   For each (ci, cj) where C[ci,cj] gets coefficient alpha from M:
        #     The term contributes alpha * (A_coeffs * B_coeffs) to C[ci,cj]
        #     In the decomposition, this means: u = A_coeffs, v = B_coeffs, w[ci,cj_index] = alpha
        
        # M1: A={a00:1, a11:1}, B={b00:1, b11:1}
        #   C[0,0] += 1, C[1,1] += 1
        terms.append(make_term(
            {a00: 1, a11: 1},
            {b00: 1, b11: 1},
            {c00: 1, c11: 1}
        ))
        
        # M2: A={a10:1, a11:1}, B={b00:1}
        #   C[1,0] += 1, C[1,1] -= 1
        terms.append(make_term(
            {a10: 1, a11: 1},
            {b00: 1},
            {c10: 1, c11: -1}
        ))
        
        # M3: A={a00:1}, B={b10:1, b00:-1}
        #   C[0,1] += 1, C[1,1] += 1
        terms.append(make_term(
            {a00: 1},
            {b10: 1, b00: -1},
            {c01: 1, c11: 1}
        ))
        
        # M4: A={a11:1}, B={b11:1, b01:-1}
        #   C[0,0] -= 1, C[1,0] += 1
        terms.append(make_term(
            {a11: 1},
            {b11: 1, b01: -1},
            {c00: -1, c10: 1}
        ))
        
        # M5: A={a00:1, a01:1}, B={b11:1}
        #   C[0,0] -= 1, C[0,1] += 1
        terms.append(make_term(
            {a00: 1, a01: 1},
            {b11: 1},
            {c00: -1, c01: 1}
        ))
        
        # M6: A={a10:1, a00:-1}, B={b00:1, b01:1}
        #   C[0,0] += 1
        terms.append(make_term(
            {a10: 1, a00: -1},
            {b00: 1, b01: 1},
            {c00: 1}
        ))
        
        # M7: A={a01:1, a11:-1}, B={b10:1, b11:1}
        #   C[1,1] += 1
        terms.append(make_term(
            {a01: 1, a11: -1},
            {b10: 1, b11: 1},
            {c11: 1}
        ))
        
        return terms
    
    # Schoolbook for a sub-block: A[ia:ia+ra, ja:ja+ca] * B[ja:ja+ca, kb:kb+cb] -> C[ia:ia+ra, kb:kb+cb]
    def schoolbook(ia, ja, ra, ca, kb, cb):
        """
        Schoolbook decomposition for A[ia:ia+ra, ja:ja+ca] * B[ja:ja+ca, kb:kb+cb] -> C[ia:ia+ra, kb:kb+cb]
        ra: number of rows in A block, ca: number of cols in A block (= rows in B block)
        cb: number of cols in B block
        """
        terms = []
        for di in range(ra):
            for dj in range(ca):
                for dk in range(cb):
                    a_idx = idx_a(ia+di, ja+dj)
                    b_idx = idx_b(ja+dj, kb+dk)
                    w_idx = idx_w(kb+dk, ia+di)
                    terms.append(({a_idx: 1}, {b_idx: 1}, {w_idx: 1}))
        return terms
    
    # Now build the full decomposition:
    # 
    # Partition:
    # A rows: {0,1} (2 rows) and {2} (1 row)
    # A cols / B rows: {0,1} (2 cols) and {2,3} (2 cols)
    # B cols: {0,1}, {2,3}, {4,5}, {6}
    #
    # Block (row_group, col_group):
    # (0,1) x (0,1): 2x4 * 4x2 -> use two Strassen 2x2x2 (split A cols into {0,1} and {2,3})
    #   = Strassen(ia=0, ja=0, kb=0) + Strassen(ia=0, ja=2, kb=0)
    # (0,1) x (2,3): 2x4 * 4x2 -> Strassen(ia=0, ja=0, kb=2) + Strassen(ia=0, ja=2, kb=2)
    # (0,1) x (4,5): 2x4 * 4x2 -> Strassen(ia=0, ja=0, kb=4) + Strassen(ia=0, ja=2, kb=4)
    # (0,1) x (6):    2x4 * 4x1 -> schoolbook (rank 8)
    # (2,) x (0,1):   1x4 * 4x2 -> schoolbook (rank 8)
    # (2,) x (2,3):   1x4 * 4x2 -> schoolbook (rank 8)
    # (2,) x (4,5):   1x4 * 4x2 -> schoolbook (rank 8)
    # (2,) x (6):     1x4 * 4x1 -> schoolbook (rank 4)
    
    all_terms = []
    
    # (0,1) x (0,1): two Strassen blocks
    all_terms.extend(strassen_222(ia=0, ja=0, kb=0))
    all_terms.extend(strassen_222(ia=0, ja=2, kb=0))
    
    # (0,1) x (2,3)
    all_terms.extend(strassen_222(ia=0, ja=0, kb=2))
    all_terms.extend(strassen_222(ia=0, ja=2, kb=2))
    
    # (0,1) x (4,5)
    all_terms.extend(strassen_222(ia=0, ja=0, kb=4))
    all_terms.extend(strassen_222(ia=0, ja=2, kb=4))
    
    # (0,1) x (6): 2x4 * 4x1, schoolbook
    all_terms.extend(schoolbook(ia=0, ja=0, ra=2, ca=4, kb=6, cb=1))
    
    # (2,) x (0,1): 1x4 * 4x2, schoolbook
    all_terms.extend(schoolbook(ia=2, ja=0, ra=1, ca=4, kb=0, cb=2))
    
    # (2,) x (2,3)
    all_terms.extend(schoolbook(ia=2, ja=0, ra=1, ca=4, kb=2, cb=2))
    
    # (2,) x (4,5)
    all_terms.extend(schoolbook(ia=2, ja=0, ra=1, ca=4, kb=4, cb=2))
    
    # (2,) x (6)
    all_terms.extend(schoolbook(ia=2, ja=0, ra=1, ca=4, kb=6, cb=1))
    
    # Now convert all_terms (list of (u_dict, v_dict, w_dict)) into matrices U, V, W
    rank = len(all_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for idx, val in u_dict.items():
            U[idx][c] = val
        for idx, val in v_dict.items():
            V[idx][c] = val
        for idx, val in w_dict.items():
            W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
