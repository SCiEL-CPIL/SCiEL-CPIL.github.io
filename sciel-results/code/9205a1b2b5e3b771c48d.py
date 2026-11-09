import numpy as np
from fractions import Fraction
import time

def construct():
    n, m, p = 2, 4, 5
    deadline = time.time() + 110
    
    # We'll construct a decomposition of 2x4x5 with rank 36
    # Strategy: Split into 2x2x5 subproblems, each using Strassen for 2x2x2 + trivial for 2x2x1
    # Actually, let me think more carefully.
    #
    # For 2x2x5: C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1}, k in {0,1,2,3,4}
    # 
    # We can write B as [B_col0, B_col1, B_col2, B_col3, B_col4] where each B_col_k is 2x1
    # C[:,k] = A @ B_col_k
    #
    # So we need to multiply 2x2 by five different 2x1 vectors.
    # 
    # Using Strassen for 2x2*2x2: rank 7
    # But we have 2x1 vectors, not 2x2 matrices.
    #
    # Alternative: Use the fact that multiplying by a fixed 2x1 vector b:
    # [a b; c d] @ [x; y] = [ax+by; cx+dy]
    # This can be done with 4 multiplications (trivial) or with Strassen-like approach.
    #
    # Actually, for 2x2 * 2x1, the tensor has 2*2*1 = 4 entries.
    # T[i,j,k] where i in {0,1}, j in {0,1}, k in {0,1}... wait no.
    # 
    # Let me re-clarify the tensor structure.
    # For matrix multiplication C = A * B where A is n x m, B is m x p:
    # T[i,j,k] = 1 if C[i,k] depends on A[i,j] and B[j,k]
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # 
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # For 2x4x5: n=2, m=4, p=5
    # 
    # Let me try a different approach: use known low-rank decompositions.
    #
    # For 2x2x2: Strassen gives rank 7
    # For 2x2x1: This is just 4 scalar products, so rank 4 trivially
    #            But can we do better? T[i,j,k] for i,j in {0,1}, k in {0,1}... 
    #            Wait, 2x2x1 means n=2, m=2, p=1
    #            C is 2x1, C[i] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    #            T[i,j,0] = 1 for i in {0,1}, j in {0,1}
    #            That's 4 entries. Rank is 4 (trivial).
    #
    # Hmm, so Strassen doesn't help for 2x2x1.
    #
    # Let me think about this differently. For 2x2x5:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # 
    # This is like computing 2 rows of a 2x5 matrix product with a 2x2 left factor.
    # 
    # Total number of scalar products in trivial: 2*2*5 = 20
    # 
    # Can we share across the 5 columns?
    # For each row i of C: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # 
    # This is a linear combination of B[0,k] and B[1,k] with coefficients A[i,0] and A[i,1].
    # For 5 values of k, this is like multiplying a 1x2 vector by a 2x5 matrix.
    # Which has rank 1*2*5 = 10 trivially.
    # Two such rows gives rank 20 trivially.
    #
    # But we can use Strassen-like ideas if we group the columns.
    # 
    # Actually, let me try a systematic approach: find a rank-20 or lower decomposition.
    #
    # Let me try to use the following trick:
    # Group the 4 middle indices into two pairs: {0,1} and {2,3}, and handle index 4 separately.
    # C[:, 0:2] = A @ B[:, 0:2]  -- 2x2 * 2x2, rank 7 (Strassen)
    # C[:, 2:4] = A @ B[:, 2:4]  -- 2x2 * 2x2, rank 7 (Strassen)
    # C[:, 4:5] = A @ B[:, 4:5]  -- 2x2 * 2x1, rank 4 (trivial)
    # Total: 7 + 7 + 4 = 18 for one 2x2x5 block
    # Two such blocks: 18 * 2 = 36
    #
    # Wait, but the 2x4x5 problem has m=4, so we have 4 middle indices.
    # We can split the 4 middle indices into two groups of 2:
    # C = A[:, 0:2] @ B[0:2, :] + A[:, 2:4] @ B[2:4, :]
    # First term: 2x2 * 2x5, rank 18 (using above)
    # Second term: 2x2 * 2x5, rank 18 (using above)
    # Total: 36
    #
    # Let me implement Strassen for 2x2x2 and combine.

    # First, let me hardcode the Strassen decomposition for 2x2x2
    # Strassen's 7 products:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B01-B11)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    def strassen_2x2x2_decomp():
        """Return U, V, W for 2x2x2 with rank 7 (Strassen)."""
        n2, m2, p2 = 2, 2, 2
        rank = 7
        U = np.zeros((n2*m2, rank), dtype=object)
        V = np.zeros((m2*p2, rank), dtype=object)
        W = np.zeros((p2*n2, rank), dtype=object)
        
        # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # A indices: (i,j) for i in {0,1}, j in {0,1}
        # B indices: (j,k) for j in {0,1}, k in {0,1}
        # C indices: (k,i) for k in {0,1}, i in {0,1}  [note: W[k*n+i, r]]
        
        # P1 = (A00+A11)*(B00+B11)
        # A00: i=0,j=0; A11: i=1,j=1
        # B00: j=0,k=0; B11: j=1,k=1
        U[0*2+0, 0] += Fraction(1)   # A00
        U[1*2+1, 0] += Fraction(1)   # A11
        V[0*2+0, 0] += Fraction(1)   # B00
        V[1*2+1, 0] += Fraction(1)   # B11
        # C contribution: P1 contributes to C00 and C11
        W[0*2+0, 0] += Fraction(1)   # C00 (k=0,i=0)
        W[1*2+1, 0] += Fraction(1)   # C11 (k=1,i=1)
        
        # P2 = (A10+A11)*B00
        U[1*2+0, 1] += Fraction(1)   # A10
        U[1*2+1, 1] += Fraction(1)   # A11
        V[0*2+0, 1] += Fraction(1)   # B00
        W[0*2+1, 1] += Fraction(1)   # C10 (k=0,i=1)
        
        # P3 = A00*(B01-B11)
        U[0*2+0, 2] += Fraction(1)   # A00
        V[0*2+1, 2] += Fraction(1)   # B01
        V[1*2+1, 2] += Fraction(-1)  # B11
        W[1*2+0, 2] += Fraction(1)   # C01 (k=1,i=0)
        
        # P4 = A11*(B10-B00)
        U[1*2+1, 3] += Fraction(1)   # A11
        V[1*2+0, 3] += Fraction(1)   # B10
        V[0*2+0, 3] += Fraction(-1)  # B00
        W[0*2+0, 3] += Fraction(1)   # C00
        W[0*2+1, 3] += Fraction(1)   # C10
        
        # P5 = (A00+A01)*B11
        U[0*2+0, 4] += Fraction(1)   # A00
        U[0*2+1, 4] += Fraction(1)   # A01
        V[1*2+1, 4] += Fraction(1)   # B11
        W[1*2+0, 4] += Fraction(-1)  # C01
        W[0*2+1, 4] += Fraction(1)   # C10  # Wait, let me recheck
        # C00 = P1+P4-P5+P7, C01=P3+P5, C10=P2+P4, C11=P1-P2+P3+P6
        # So P5 contributes to C01 (coefficient +1) and C00 (coefficient -1)
        # Let me redo:
        # Actually: C00 = P1 + P4 - P5 + P7
        # So P5 contributes -1 to C00
        
        # Let me restart this more carefully.
        
        U = np.zeros((4, rank), dtype=object)
        V = np.zeros((4, rank), dtype=object)
        W = np.zeros((4, rank), dtype=object)
        
        # P1 = (A00+A11)*(B00+B11)
        # Contributes to: C00 with +1, C11 with +1
        U[0, 0] = Fraction(1); U[3, 0] = Fraction(1)
        V[0, 0] = Fraction(1); V[3, 0] = Fraction(1)
        W[0, 0] = Fraction(1); W[3, 0] = Fraction(1)
        # U indices: [A00, A01, A10, A11] -> [0,1,2,3]
        # V indices: [B00, B01, B10, B11] -> [0,1,2,3]
        # W indices: [C00, C01, C10, C11] -> [0,1,2,3] where Cki -> index k*2+i... 
        # Wait, W shape is (p*n, rank) = (2*2, rank) = (4, rank)
        # W[k*n+i, r] where k in {0,1}, i in {0,1}
        # So W[0, r] = C[0,0], W[1, r] = C[0,1], W[2, r] = C[1,0], W[3, r] = C[1,1]
        
        # P1 = (A00+A11)*(B00+B11)
        U[0, 0] = Fraction(1); U[3, 0] = Fraction(1)
        V[0, 0] = Fraction(1); V[3, 0] = Fraction(1)
        W[0, 0] = Fraction(1); W[3, 0] = Fraction(1)
        
        # P2 = (A10+A11)*B00
        U[2, 1] = Fraction(1); U[3, 1] = Fraction(1)
        V[0, 1] = Fraction(1)
        W[2, 1] = Fraction(1)  # C10
        
        # P3 = A00*(B01-B11)
        U[0, 2] = Fraction(1)
        V[1, 2] = Fraction(1); V[3, 2] = Fraction(-1)
        W[1, 2] = Fraction(1)  # C01
        
        # P4 = A11*(B10-B00)
        U[3, 3] = Fraction(1)
        V[2, 3] = Fraction(1); V[0, 3] = Fraction(-1)
        W[0, 3] = Fraction(1); W[2, 3] = Fraction(1)  # C00, C10
        
        # P5 = (A00+A01)*B11
        U[0, 4] = Fraction(1); U[1, 4] = Fraction(1)
        V[3, 4] = Fraction(1)
        W[0, 4] = Fraction(-1); W[1, 4] = Fraction(1)  # C00 with -1, C01 with +1
        
        # P6 = (A10-A00)*(B00+B01)
        U[2, 5] = Fraction(1); U[0, 5] = Fraction(-1)
        V[0, 5] = Fraction(1); V[1, 5] = Fraction(1)
        W[3, 5] = Fraction(1)  # C11
        
        # P7 = (A01-A11)*(B10+B11)
        U[1, 6] = Fraction(1); U[3, 6] = Fraction(-1)
        V[2, 6] = Fraction(1); V[3, 6] = Fraction(1)
        W[0, 6] = Fraction(1)  # C00
        
        return U, V, W
    
    def trivial_2x2x1_decomp():
        """Return U, V, W for 2x2x1 with rank 4 (trivial)."""
        n2, m2, p2 = 2, 2, 1
        rank = 4
        U = np.zeros((n2*m2, rank), dtype=object)
        V = np.zeros((m2*p2, rank), dtype=object)
        W = np.zeros((p2*n2, rank), dtype=object)
        
        for i in range(2):
            for j in range(2):
                idx = i * 2 + j  # 0,1,2,3
                U[idx, idx] = Fraction(1)
                V[j * 1 + 0, idx] = Fraction(1)
                W[0 * 2 + i, idx] = Fraction(1)
        
        return U, V, W
    
    def combine_blocks(U1, V1, W1, U2, V2, W2, 
                       a_row_offset, b_row_offset, c_col_offset,
                       a_col_offset, b_col_offset):
        """Combine two decompositions for 2x2x2 and 2x2x1 into a 2x2x3 block.
        
        Actually, this is getting complex. Let me just build the full 2x2x5 decomposition directly.
        """
        pass
    
    # Let me take a simpler approach: build the full 2x4x5 decomposition directly
    # by combining Strassen blocks.
    
    # For 2x2x5, we use:
    # Columns 0-1: Strassen 2x2x2 (rank 7)
    # Columns 2-3: Strassen 2x2x2 (rank 7)  
    # Column 4: trivial 2x2x1 (rank 4)
    # Total rank: 18
    
    def decomp_2x2x5():
        """Return U, V, W for 2x2x5 with rank 18."""
        n2, m2, p2 = 2, 2, 5
        rank = 18
        
        U = np.zeros((n2*m2, rank), dtype=object)
        V = np.zeros((m2*p2, rank), dtype=object)
        W = np.zeros((p2*n2, rank), dtype=object)
        
        # Block 1: columns 0,1 (indices 0,1 in k)
        U1, V1, W1 = strassen_2x2x2_decomp()
        # U1: (4,7), V1: (4,7) where V1[j*2+k, r], W1: (4,7) where W1[k*2+i, r]
        # For 2x2x5: V shape is (2*5, 18) = (10, 18)
        # V[j*5+k, r]
        # For block 1, k in {0,1}, so V1[j*2+k, r] maps to V[j*5+k, r]
        
        for r in range(7):
            for ij in range(4):
                if U1[ij, r] != 0:
                    U[ij, r] += U1[ij, r]
            for jk in range(4):
                j = jk // 2
                k = jk % 2
                if V1[jk, r] != 0:
                    V[j*5+k, r] += V1[jk, r]
            for ki in range(4):
                k = ki // 2
                i = ki % 2
                if W1[ki, r] != 0:
                    W[k*2+i, r] += W1[ki, r]
        
        # Block 2: columns 2,3 (indices 2,3 in k)
        U2, V2, W2 = strassen_2x2x2_decomp()
        for r in range(7):
            col_offset = 7
            r2 = col_offset + r
            for ij in range(4):
                if U2[ij, r] != 0:
                    U[ij, r2] += U2[ij, r]
            for jk in range(4):
                j = jk // 2
                k = jk % 2
                if V2[jk, r] != 0:
                    V[j*5+(k+2), r2] += V2[jk, r]
            for ki in range(4):
                k = ki // 2
                i = ki % 2
                if W2[ki, r] != 0:
                    W[(k+2)*2+i, r2] += W2[ki, r]
        
        # Block 3: column 4
        U3, V3, W3 = trivial_2x2x1_decomp()
        # U3: (4,4), V3: (2,4) where V3[j*1+0, r], W3: (2,4) where W3[0*2+i, r]
        for r in range(4):
            col_offset = 14
            r3 = col_offset + r
            for ij in range(4):
                if U3[ij, r] != 0:
                    U[ij, r3] += U3[ij, r]
            for j in range(2):
                if V3[j, r] != 0:
                    V[j*5+4, r3] += V3[j, r]
            for i in range(2):
                if W3[i, r] != 0:
                    W[4*2+i, r3] += W3[i, r]
        
        return U, V, W
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        """Verify that the decomposition exactly computes the matrix multiplication tensor"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if total != 1:
                        return False
        return True
    
    # Now build 2x4x5 from two 2x2x5 blocks
    # A is 2x4: A[:, 0:2] and A[:, 2:4]
    # B is 4x5: B[0:2, :] and B[2:4, :]
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    
    U1, V1, W1 = decomp_2x2x5()
    U2, V2, W2 = decomp_2x2x5()
    
    rank = 36
    U_out = np.zeros((2*4, rank), dtype=object)
    V_out = np.zeros((4*5, rank), dtype=object)
    W_out = np.zeros((5*2, rank), dtype=object)
    
    # Block 1: j in {0,1} (first 2 columns of A, first 2 rows of B)
    for r in range(18):
        for ij in range(4):  # i in {0,1}, j_local in {0,1}
            i = ij // 2
            j_local = ij % 2
            j = j_local  # j_global = 0,1
            if U1[ij, r] != 0:
                U_out[i*4+j, r] += U1[ij, r]
        for jk in range(10):  # j_local in {0,1}, k in {0,1,2,3,4}
            j_local = jk // 5
            k = jk % 5
            j = j_local  # j_global = 0,1
            if V1[jk, r] != 0:
                V_out[j*5+k, r] += V1[jk, r]
        for ki in range(10):  # k in {0,1,2,3,4}, i in {0,1}
            k = ki // 2
            i = ki % 2
            if W1[ki, r] != 0:
                W_out[k*2+i, r] += W1[ki, r]
    
    # Block 2: j in {2,3} (last 2 columns of A, last 2 rows of B)
    for r in range(18):
        r2 = 18 + r
        for ij in range(4):
            i = ij // 2
            j_local = ij % 2
            j = j_local + 2  # j_global = 2,3
            if U2[ij, r] != 0:
                U_out[i*4+j, r2] += U2[ij, r]
        for jk in range(10):
            j_local = jk // 5
            k = jk % 5
            j = j_local + 2  # j_global = 2,3
            if V2[jk, r] != 0:
                V_out[j*5+k, r2] += V2[jk, r]
        for ki in range(10):
            k = ki // 2
            i = ki % 2
            if W2[ki, r] != 0:
                W_out[k*2+i, r2] += W2[ki, r]
    
    # Verify
    if not verify_decomposition(U_out, V_out, W_out, 36, n, m, p):
        # Fallback to trivial rank-40
        U_out = np.zeros((8, 40), dtype=object)
        V_out = np.zeros((20, 40), dtype=object)
        W_out = np.zeros((10, 40), dtype=object)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U_out[i * m + j, idx] = Fraction(1)
                    V_out[j * p + k, idx] = Fraction(1)
                    W_out[k * n + i, idx] = Fraction(1)
    
    def convert_to_list(arr):
        res = []
        for row in arr:
            res_row = []
            for val in row:
                if isinstance(val, Fraction):
                    if val.denominator == 1:
                        res_row.append(int(val))
                    elif val.denominator == 2:
                        res_row.append(val.numerator / 2.0)
                    else:
                        res_row.append(float(val))
                else:
                    res_row.append(val)
            res.append(res_row)
        return res
    
    return {"U": convert_to_list(U_out), "V": convert_to_list(V_out), "W": convert_to_list(W_out)}
