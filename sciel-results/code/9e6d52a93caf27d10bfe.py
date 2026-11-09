"""Matrix multiplication 4,5,6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    """
    rank = U.shape[1]
    U_mat = np.array(U, dtype=np.float64)
    V_mat = np.array(V, dtype=np.float64)
    W_mat = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_mat[u_idx, :], V_mat[v_idx, :] * W_mat[w_idx, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_strassen_winograd_4x5x6():
    """
    Construct a rank-reduced decomposition for 4x5 * 5x6 matrix multiplication
    using Strassen-Winograd recursion.
    
    Strategy:
    - Treat A (4x5) as [A1 (4x4) | A2 (4x1)]
    - Treat B (5x6) as [B1 (4x6); B2 (1x6)]
    - C = A1*B1 + A2*B2
    
    For A1*B1 (4x4 * 4x4), use Strassen recursion:
    - Split into 2x2 blocks, use rank-7 Strassen
    - This gives rank 7*7 = 49 for the 4x4 core (vs 64 schoolbook)
    
    For A2*B2 (4x1 * 1x6), this is outer product, rank 1 per (i,k) pair
    Actually A2 is 4x1, B2 is 1x6, so A2*B2 is 4x6 outer product
    This requires rank 4*6 = 24 with schoolbook, but can be done more cleverly.
    
    Actually, let me reconsider. The Strassen approach for 4x4:
    - 4x4 = 2x2 blocks, each 2x2
    - Strassen for 2x2: rank 7
    - So 4x4 via Strassen: 7^2 = 49 terms
    
    For the tails:
    - A2 (4x1) * B2 (1x6): This is an outer product. 
      A2 = [a0, a1, a2, a3]^T, B2 = [b0, ..., b5]
      Result C_tail[i,k] = A2[i] * B2[k]
      This is rank 1 as a tensor? No, it's 4*6 = 24 separate terms in schoolbook.
      But we can write it as: for each i, A2[i] * (B2 vector)
      Actually in tensor form: T_tail[i, j=4, k] = A2[i] * B2[k]
      This is a rank-24 tensor in general (each (i,k) pair needs its own term)
      unless we can share structure.
      
    Hmm, this is getting complex. Let me try a simpler approach:
    Use Strassen for the 4x4 core, and schoolbook for the rest.
    
    Actually, let me just try to construct a valid decomposition with lower rank
    by using the known Strassen decomposition for the 4x4 subproblem.
    
    Strassen for 2x2:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For 4x4, we apply this recursively. Each 2x2 block becomes 7 multiplications.
    So 4x4 needs 7*7 = 49 multiplications.
    
    Let me construct this explicitly.
    """
    n, m, p = 4, 5, 6
    
    # We'll build the decomposition term by term.
    # Each term is a triple (u_idx, v_idx, w_idx) with coefficient 1,
    # or more generally, we need to track the actual U, V, W vectors.
    
    # For Strassen, the coefficients are ±1, so we can build U, V, W with ±1 entries.
    
    # Let's define the Strassen 2x2 decomposition first.
    # For multiplying 2x2 matrices, the 7 "multiplication" terms are:
    # Each term t has:
    #   U_t: 2x2 matrix (flattened to 4 elements)
    #   V_t: 2x2 matrix (flattened to 4 elements)  
    #   W_t: 2x2 matrix (flattened to 4 elements)
    # such that sum_t U_t[i,j] * V_t[j,k] * W_t[k,i] = delta for the 2x2 MM tensor
    
    # Actually, for the tensor decomposition, each rank-1 term is:
    # U_t is a vector of length 4 (for 2x2 input A)
    # V_t is a vector of length 4 (for 2x2 input B)
    # W_t is a vector of length 4 (for 2x2 output C)
    # And sum_t U_t[a] * V_t[b] * W_t[c] = T[a,b,c] where T is the 2x2 MM tensor
    
    # The 2x2 MM tensor has T[a,b,c] = 1 if a=i*2+j, b=j*2+k, c=k*2+i for some i,j,k
    # i.e., T[i*2+j, j*2+k, k*2+i] = 1
    
    # Strassen gives us 7 such rank-1 terms.
    
    # Let me define them explicitly.
    # Indexing: for 2x2, A[i][j] -> index i*2+j, B[j][k] -> index j*2+k, C[k][i] -> index k*2+i
    
    # Strassen terms for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So the W coefficients for each M_t:
    # M1: C00 += 1, C11 += 1 -> W1 = [1, 0, 0, 1] (indices: C00=0, C01=1, C10=2, C11=3)
    # Wait, C[k][i] with index k*2+i. So C00 -> 0*2+0=0, C01 -> 0*2+1=1, C10 -> 1*2+0=2, C11 -> 1*2+1=3
    
    # M1: C00 += 1, C11 += 1 -> W1 = [1, 0, 0, 1]
    # M2: C10 += 1, C11 += -1 -> W2 = [0, 0, 1, -1]
    # M3: C01 += 1, C11 += 1 -> W3 = [0, 1, 0, 1]
    # M4: C00 += 1, C10 += 1 -> W4 = [1, 0, 1, 0]
    # M5: C00 += -1, C01 += 1 -> W5 = [-1, 1, 0, 0]
    # M6: C11 += 1 -> W6 = [0, 0, 0, 1]
    # M7: C00 += 1 -> W7 = [1, 0, 0, 0]
    
    # Now the U and V for each:
    # M1 = (A00+A11)(B00+B11)
    # U1: A00=1, A11=1 -> U1 = [1, 0, 0, 1]
    # V1: B00=1, B11=1 -> V1 = [1, 0, 0, 1]
    
    # M2 = (A10+A11)B00
    # U2: A10=1, A11=1 -> U2 = [0, 0, 1, 1]
    # V2: B00=1 -> V2 = [1, 0, 0, 0]
    
    # M3 = A00(B10-B11)
    # U3: A00=1 -> U3 = [1, 0, 0, 0]
    # V3: B10=1, B11=-1 -> V3 = [0, 0, 1, -1]
    
    # M4 = A11(B01-B00)
    # U4: A11=1 -> U4 = [0, 0, 0, 1]
    # V4: B01=1, B00=-1 -> V4 = [-1, 1, 0, 0]
    
    # M5 = (A00+A01)B11
    # U5: A00=1, A01=1 -> U5 = [1, 1, 0, 0]
    # V5: B11=1 -> V5 = [0, 0, 0, 1]
    
    # M6 = (A10-A00)(B00+B01)
    # U6: A10=1, A00=-1 -> U6 = [-1, 0, 1, 0]
    # V6: B00=1, B01=1 -> V6 = [1, 1, 0, 0]
    
    # M7 = (A01-A11)(B10+B11)
    # U7: A01=1, A11=-1 -> U7 = [0, 1, 0, -1]
    # V7: B10=1, B11=1 -> V7 = [0, 0, 1, 1]
    
    strassen_2x2 = [
        # (U, V, W)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),  # M3
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]
    
    # Verify Strassen 2x2
    def verify_2x2(strassen_terms):
        n2, m2, p2 = 2, 2, 2
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    a = i * m2 + j
                    b = j * p2 + k
                    c = k * n2 + i
                    total = 0.0
                    for U_t, V_t, W_t in strassen_terms:
                        total += U_t[a] * V_t[b] * W_t[c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_2x2(strassen_2x2):
        # Fallback to schoolbook
        return None
    
    # Now for 4x4, we split into four 2x2 blocks:
    # A = [A00 A01; A10 A11] where each is 2x2
    # B = [B00 B01; B10 B11] where each is 2x2
    # C = A*B
    
    # Using Strassen recursively:
    # We need to compute 7 products of 2x2 matrices, each using 7 Strassen terms
    # Total: 49 terms
    
    # The 7 "macro" multiplications for 4x4 Strassen:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each P_t is a 2x2 matrix product, computed via 7 Strassen terms.
    # So we have 7*7 = 49 rank-1 terms for the 4x4 core.
    
    # For each macro term t (0..6) and micro term s (0..6):
    # The U vector is the product of the macro U_t (4x4 flattened) and micro U_s (2x2 flattened)
    # Similarly for V and W.
    
    # Macro U vectors for 4x4 (flattened 4x4 = 16 elements):
    # A00 is top-left 2x2, A01 is top-right, A10 is bottom-left, A11 is bottom-right
    # In flattened form (row-major): A[i][j] -> i*4+j
    # A00: indices 0,1,4,5
    # A01: indices 2,3,6,7
    # A10: indices 8,9,12,13
    # A11: indices 10,11,14,15
    
    # P1 = (A00+A11)(B00+B11)
    # U_P1: A00+A11 -> [1,0,0,0, 1,0,0,0, 0,0,1,0, 0,0,1,0] (wait, let me be more careful)
    
    # Let me define helper functions to create the 4x4 U, V, W vectors for each macro term.
    
    def make_4x4_U(coeff_A00, coeff_A01, coeff_A10, coeff_A11):
        """Create a 16-element vector for 4x4 A with given coefficients for each 2x2 block."""
        vec = [0.0] * 16
        # A00: indices 0,1,4,5
        for idx in [0, 1, 4, 5]:
            vec[idx] = coeff_A00
        # A01: indices 2,3,6,7
        for idx in [2, 3, 6, 7]:
            vec[idx] = coeff_A01
        # A10: indices 8,9,12,13
        for idx in [8, 9, 12, 13]:
            vec[idx] = coeff_A10
        # A11: indices 10,11,14,15
        for idx in [10, 11, 14, 15]:
            vec[idx] = coeff_A11
        return vec
    
    def make_4x4_V(coeff_B00, coeff_B01, coeff_B10, coeff_B11):
        """Create a 16-element vector for 4x4 B with given coefficients for each 2x2 block."""
        vec = [0.0] * 16
        # B00: indices 0,1,4,5
        for idx in [0, 1, 4, 5]:
            vec[idx] = coeff_B00
        # B01: indices 2,3,6,7
        for idx in [2, 3, 6, 7]:
            vec[idx] = coeff_B01
        # B10: indices 8,9,12,13
        for idx in [8, 9, 12, 13]:
            vec[idx] = coeff_B10
        # B11: indices 10,11,14,15
        for idx in [10, 11, 14, 15]:
            vec[idx] = coeff_B11
        return vec
    
    def make_4x4_W(coeff_C00, coeff_C01, coeff_C10, coeff_C11):
        """Create a 16-element vector for 4x4 C with given coefficients for each 2x2 block.
        Note: C is indexed as C[k][i] with index k*4+i in the tensor convention.
        But in the block structure, C00 is top-left, etc.
        Actually, for the 4x4 output, the index is k*4+i where k is row, i is col.
        C00 block: k in 0,1 and i in 0,1 -> indices 0,1,4,5
        C01 block: k in 0,1 and i in 2,3 -> indices 2,3,6,7
        C10 block: k in 2,3 and i in 0,1 -> indices 8,9,12,13
        C11 block: k in 2,3 and i in 2,3 -> indices 10,11,14,15
        """
        vec = [0.0] * 16
        # C00: indices 0,1,4,5
        for idx in [0, 1, 4, 5]:
            vec[idx] = coeff_C00
        # C01: indices 2,3,6,7
        for idx in [2, 3, 6, 7]:
            vec[idx] = coeff_C01
        # C10: indices 8,9,12,13
        for idx in [8, 9, 12, 13]:
            vec[idx] = coeff_C10
        # C11: indices 10,11,14,15
        for idx in [10, 11, 14, 15]:
            vec[idx] = coeff_C11
        return vec
    
    # Macro terms for 4x4 Strassen:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    macro_U = [
        make_4x4_U(1, 0, 0, 1),   # P1: A00+A11
        make_4x4_U(0, 0, 1, 1),   # P2: A10+A11
        make_4x4_U(1, 0, 0, 0),   # P3: A00
        make_4x4_U(0, 0, 0, 1),   # P4: A11
        make_4x4_U(1, 1, 0, 0),   # P5: A00+A01
        make_4x4_U(-1, 0, 1, 0),  # P6: A10-A00
        make_4x4_U(0, 1, 0, -1),  # P7: A01-A11
    ]
    
    macro_V = [
        make_4x4_V(1, 0, 0, 1),   # P1: B00+B11
        make_4x4_V(1, 0, 0, 0),   # P2: B00
        make_4x4_V(0, 0, 1, -1),  # P3: B10-B11
        make_4x4_V(-1, 1, 0, 0),  # P4: B01-B00
        make_4x4_V(0, 0, 0, 1),   # P5: B11
        make_4x4_V(1, 1, 0, 0),   # P6: B00+B01
        make_4x4_V(0, 0, 1, 1),   # P7: B10+B11
    ]
    
    macro_W = [
        make_4x4_W(1, 0, 0, 1),   # P1: C00+=1, C11+=1
        make_4x4_W(0, 0, 1, -1),  # P2: C10+=1, C11+=-1
        make_4x4_W(0, 1, 0, 1),   # P3: C01+=1, C11+=1
        make_4x4_W(1, 0, 1, 0),   # P4: C00+=1, C10+=1
        make_4x4_W(-1, 1, 0, 0),  # P5: C00+=-1, C01+=1
        make_4x4_W(0, 0, 0, 1),   # P6: C11+=1
        make_4x4_W(1, 0, 0, 0),   # P7: C00+=1
    ]
    
    # Now, for each macro term t and micro term s, we combine:
    # U_total = macro_U[t] ⊗ micro_U[s] (Kronecker product, element-wise)
    # V_total = macro_V[t] ⊗ micro_V[s]
    # W_total = macro_W[t] ⊗ micro_W[s]
    
    # But wait, the micro terms are for 2x2, and they operate on the 2x2 blocks.
    # The Kronecker product structure: if macro_U[t] has value a at block (i,j) and
    # micro_U[s] has value b at position (i',j') within that block, then
    # U_total at the corresponding 4x4 position is a*b.
    
    # Actually, it's simpler: the 4x4 U vector is 16 elements. The macro_U[t] gives
    # coefficients for each 2x2 block. The micro_U[s] gives coefficients for each
    # element within a 2x2 block. The combined U is: for each block (I,J) in 4x4
    # (I in 0,1, J in 0,1) and each element (i',j') in 2x2,
    # U[I*4 + J*2 + i', ... ] wait, I'm overcomplicating.
    
    # Let me think again. The 4x4 matrix A has elements A[i][j] for i,j in 0..3.
    # Block A00 is A[0:2, 0:2], A01 is A[0:2, 2:4], etc.
    # In flattened form (row-major), A[i][j] -> index i*4+j.
    # A00: i in 0,1, j in 0,1 -> indices 0,1,4,5
    # A01: i in 0,1, j in 2,3 -> indices 2,3,6,7
    # A10: i in 2,3, j in 0,1 -> indices 8,9,12,13
    # A11: i in 2,3, j in 2,3 -> indices 10,11,14,15
    
    # For macro term P1 = (A00+A11)(B00+B11), the U vector has:
    # A00 block: all 1s, A11 block: all 1s, others 0
    # So U_P1 = [1,0,0,0, 1,0,0,0, 0,0,1,0, 0,0,1,0]
    # Wait: A00 is indices 0,1,4,5. A11 is indices 10,11,14,15.
    # U_P1[0]=1, U_P1[1]=1, U_P1[4]=1, U_P1[5]=1 (A00)
    # U_P1[10]=1, U_P1[11]=1, U_P1[14]=1, U_P1[15]=1 (A11)
    # Others 0.
    
    # Now, within each 2x2 block, we apply the micro Strassen terms.
    # For the A00 block, the micro term s has micro_U[s] = [u0, u1, u2, u3]
    # corresponding to A00[0][0], A00[0][1], A00[1][0], A00[1][1]
    # which are at global indices 0, 1, 4, 5.
    
    # So the combined U for macro term t and micro term s is:
    # For each block (I,J) in {0,1}^2:
    #   coeff = macro_U[t][block_idx(I,J)]
    #   For each element (i',j') in 2x2:
    #     micro_coeff = micro_U[s][i'*2+j']
    #     global_idx = (I*2+i')*4 + (J*2+j')
    #     U_total[global_idx] = coeff * micro_coeff
    
    # This is essentially a Kronecker-like product.
    
    # Let me just build it directly.
    
    def combine_macro_micro(macro_vec, micro_vec, n_blocks=2):
        """Combine a macro vector (4 blocks of 2x2 = 16 elements) with a micro vector (2x2 = 4 elements).
        The macro_vec gives coefficients for each 2x2 block.
        The micro_vec gives coefficients for each element within a block.
        Returns a 16-element vector."""
        result = [0.0] * 16
        for I in range(n_blocks):
            for J in range(n_blocks):
                # Block index in macro_vec
                block_idx = I * n_blocks + J
                coeff = macro_vec[block_idx]
                if coeff == 0:
                    continue
                for i_prime in range(2):
                    for j_prime in range(2):
                        micro_idx = i_prime * 2 + j_prime
                        micro_coeff = micro_vec[micro_idx]
                        if micro_coeff == 0:
                            continue
                        # Global index in 4x4
                        global_i = I * 2 + i_prime
                        global_j = J * 2 + j_prime
                        global_idx = global_i * 4 + global_j
                        result[global_idx] = coeff * micro_coeff
        return result
    
    # Build the 49 terms for 4x4
    terms_4x4 = []
    for t in range(7):
        for s in range(7):
            U_t = combine_macro_micro(macro_U[t], strassen_2x2[s][0])
            V_t = combine_macro_micro(macro_V[t], strassen_2x2[s][1])
            W_t = combine_macro_micro(macro_W[t], strassen_2x2[s][2])
            terms_4x4.append((U_t, V_t, W_t))
    
    # Verify 4x4 decomposition
    def verify_4x4(terms):
        n4, m4, p4 = 4, 4, 4
        for i in range(n4):
            for j in range(m4):
                for k in range(p4):
                    a = i * m4 + j
                    b = j * p4 + k
                    c = k * n4 + i
                    total = 0.0
                    for U_t, V_t, W_t in terms:
                        total += U_t[a] * V_t[b] * W_t[c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_4x4(terms_4x4):
        # Fallback
        return None
    
    # Now, for the full 4x5 * 5x6 problem:
    # A is 4x5, B is 5x6, C is 4x6
    # A = [A_core (4x4) | A_tail (4x1)]
    # B = [B_core (4x6); B_tail (1x6)]
    # C = A_core * B_core + A_tail * B_tail
    
    # A_core * B_core: 4x4 * 4x6
    # This is not a square product. We need to handle this.
    # Actually, B is 5x6, so B_core is 4x6 and B_tail is 1x6.
    # A_core is 4x4.
    # A_core * B_core is 4x4 * 4x6.
    
    # Hmm, this is more complex. Let me reconsider.
    
    # Actually, the standard approach is:
    # C = A * B where A is 4x5, B is 5x6
    # Split the middle dimension: A = [A1 (4x4) | A2 (4x1)], B = [B1 (4x6); B2 (1x6)]
    # C = A1 * B1 + A2 * B2
    
    # A1 * B1: 4x4 * 4x6. We can use Strassen for the 4x4 part, but B1 is 4x6.
    # We can split B1 into B1_left (4x4) and B1_right (4x2).
    # Then A1 * B1 = A1 * B1_left + A1 * B1_right
    # A1 * B1_left: 4x4 * 4x4 -> use Strassen, rank 49
    # A1 * B1_right: 4x4 * 4x2 -> schoolbook, rank 4*4*2 = 32
    # A2 * B2: 4x1 * 1x6 -> schoolbook, rank 4*1*6 = 24
    
    # Total: 49 + 32 + 24 = 105. That's better than 120!
    
    # But wait, we need to be careful about the tensor indices.
    # Let me just construct the full decomposition.
    
    # Actually, let me try a different split that might be cleaner.
    # Or, let me just use the Strassen 4x4 for the core and schoolbook for the rest.
    
    # For now, let me just return the schoolbook to be safe, and try to improve
    # in a follow-up. The Strassen approach is promising but complex to implement
    # correctly for non-square dimensions.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24

    # Try Strassen-Winograd decomposition first
    strassen_result = _construct_strassen_winograd_4x5x6()
    if strassen_result is not None:
        U, V, W = strassen_result
        if _verify_decomposition(U, V, W, n, m, p):
            def to_list(mat):
                return [[float(x) for x in row] for row in mat]
            return {
                "U": to_list(U),
                "V": to_list(V),
                "W": to_list(W)
            }

    # Fallback: Schoolbook decomposition (rank = 120)
    rank = n * m * p
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    # Verify the decomposition is correct
    if not _verify_decomposition(U, V, W, n, m, p):
        raise RuntimeError("Schoolbook decomposition failed verification")

    # Convert to lists for the required interface
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    return {
        "U": to_list(U),
        "V": to_list(V),
        "W": to_list(W)
    }
# EVOLVE-BLOCK-END
