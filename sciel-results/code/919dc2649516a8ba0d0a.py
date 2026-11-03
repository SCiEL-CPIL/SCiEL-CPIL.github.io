"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Block decomposition: rows 2+1, middle 2+2+1, cols 2+2+2
    row_blocks = [(0, 2), (2, 3)]       # sizes 2, 1
    mid_blocks = [(0, 2), (2, 4), (4, 5)]  # sizes 2, 2, 1
    col_blocks = [(0, 2), (2, 4), (4, 6)]  # sizes 2, 2, 2
    
    U = []
    V = []
    W = []
    
    def add_term(u_val, v_val, w_val, i, j, k):
        """Add a rank-1 term with given values at positions (i,j,k)."""
        U.append(u_val)
        V.append(v_val)
        W.append(w_val)
    
    def add_direct_block(i_start, i_end, j_start, j_end, k_start, k_end):
        """Add direct (schoolbook) terms for a block."""
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    U.append(1)
                    V.append(1)
                    W.append(1)
    
    def add_strassen_222(i0, j0, k0):
        """
        Add Strassen decomposition for 2x2x2 block starting at (i0, j0, k0).
        A is 2x2: rows i0,i0+1; cols j0,j0+1
        B is 2x2: rows j0,j0+1; cols k0,k0+1
        C is 2x2: rows i0,i0+1; cols k0,k0+1
        
        W index convention: W[k*n + i, r]
        """
        # Strassen's 7 multiplications:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For each multiplication Mk, we need to express it as sum of rank-1 terms
        # Mk involves products of linear combinations of A entries and B entries
        
        # A entries: A[i0,j0], A[i0,j0+1], A[i0+1,j0], A[i0+1,j0+1]
        # B entries: B[j0,k0], B[j0,k0+1], B[j0+1,k0], B[j0+1,k0+1]
        
        # Each Mk = (sum of A terms) * (sum of B terms)
        # This expands to multiple rank-1 terms
        
        # M1 = (A00+A11)(B00+B11)
        # = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # Each Aij*Bjk contributes to C with appropriate signs
        
        # Let's define the Strassen terms more carefully.
        # A00 = A[i0, j0], A01 = A[i0, j0+1], A10 = A[i0+1, j0], A11 = A[i0+1, j0+1]
        # B00 = B[j0, k0], B01 = B[j0, k0+1], B10 = B[j1, k0], B11 = B[j1, k0+1]
        
        # The 7 products expand into rank-1 terms:
        # A product (a_linear)(b_linear) where a_linear = sum_i c_i * A[i,j_a] and b_linear = sum_k d_k * B[j_b, k]
        # contributes terms: for each (i, j_a) and (j_b, k) with c_i * d_k:
        #   U[i*m + j_a] += c_i
        #   V[j_a*p + k] += d_k  (wait, V is indexed by [j*p+k])
        #   W[k*n + i] += 1
        
        # Actually, each rank-1 term in the tensor is:
        # T[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # So for a product A[i,j]*B[j,k], the rank-1 term has:
        # U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
        
        # For a linear combination, we add multiple such terms.
        
        # Let me just enumerate all 7 Strassen products and their expansions:
        
        # Helper to add a term: A[i,j] * B[j,k] with coefficient (ci, ck)
        def add_a_b_term(i, j, k, coeff_a, coeff_b):
            """Add rank-1 term: coeff_a * A[i,j] * coeff_b * B[j,k] -> contributes to C[i,k]"""
            U.append(coeff_a)
            V.append(coeff_b)
            W.append(1)
        
        # M1 = (A00+A11)(B00+B11)
        # = A00*B00 + A00*B11 + A11*B00 + A11*B11
        add_a_b_term(i0, j0, k0, 1, 1)       # A00*B00
        add_a_b_term(i0, j0, k0+1, 1, 1)     # A00*B01  -- wait, B11 = B[j1,k1]
        
        # Let me redo this carefully.
        # i0, i0+1 are the two row indices
        # j0, j0+1 are the two middle indices  
        # k0, k0+1 are the two col indices
        
        # A[i0,j0], A[i0,j0+1], A[i0+1,j0], A[i0+1,j0+1]
        # B[j0,k0], B[j0,k0+1], B[j0+1,k0], B[j0+1,k0+1]
        
        # M1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        # Expands to:
        # A[i0,j0]*B[j0,k0] + A[i0,j0]*B[j0+1,k0+1] + A[i0+1,j0+1]*B[j0,k0] + A[i0+1,j0+1]*B[j0+1,k0+1]
        
        # M2 = (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
        # = A[i0+1,j0]*B[j0,k0] + A[i0+1,j0+1]*B[j0,k0]
        
        # M3 = A[i0,j0] * (B[j0+1,k0] - B[j0+1,k0+1])
        # = A[i0,j0]*B[j0+1,k0] - A[i0,j0]*B[j0+1,k0+1]
        
        # M4 = A[i0+1,j0+1] * (B[j0,k0+1] - B[j0,k0])
        # = A[i0+1,j0+1]*B[j0,k0+1] - A[i0+1,j0+1]*B[j0,k0]
        
        # M5 = (A[i0,j0] + A[i0,j0+1]) * B[j0+1,k0+1]
        # = A[i0,j0]*B[j0+1,k0+1] + A[i0,j0+1]*B[j0+1,k0+1]
        
        # M6 = (A[i0+1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k0+1])
        # = A[i0+1,j0]*B[j0,k0] + A[i0+1,j0]*B[j0,k0+1] - A[i0,j0]*B[j0,k0] - A[i0,j0]*B[j0,k0+1]
        
        # M7 = (A[i0,j0+1] - A[i0+1,j0+1]) * (B[j0+1,k0+1] + B[j0+1,k0])
        # Wait, M7 = (A01-A11)(B10+B11) where B10=B[j0+1,k0], B11=B[j0+1,k0+1]
        # = (A[i0,j0+1] - A[i0+1,j0+1]) * (B[j0+1,k0] + B[j0+1,k0+1])
        # = A[i0,j0+1]*B[j0+1,k0] + A[i0,j0+1]*B[j0+1,k0+1] - A[i0+1,j0+1]*B[j0+1,k0] - A[i0+1,j0+1]*B[j0+1,k0+1]
        
        # Now, each product A[i,j]*B[j,k] is a rank-1 term in the tensor with:
        # U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
        # But when we have linear combinations, we combine them.
        
        # The key insight: we don't need to expand into individual A[i,j]*B[j,k] terms.
        # Instead, each "multiplication" in Strassen is one rank-1 term in the 
        # *Strassen* sense, but in the tensor decomposition, it expands.
        
        # Wait, I'm confusing two levels. Let me restart.
        
        # The tensor T for 2x2x2 matmul has rank 7 (Strassen).
        # A rank-1 term in the 2x2x2 tensor is: sum over (i,j,k) of u[i,j]*v[j,k]*w[k,i] * A[i,j]*B[j,k]
        # No, that's not right either.
        
        # The multiplication tensor T has T[i,j,k] = e_{ij} ⊗ e_{jk} ⊗ e_{ki}
        # A rank-1 decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
        # where U_r is a 2x2 matrix, V_r is a 2x2 matrix, W_r is a 2x2 matrix.
        # T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
        
        # For Strassen, the 7 terms have specific U, V, W matrices.
        
        # Strassen's 7 terms for 2x2x2:
        # Term 1: U1 = [[1,0],[0,1]], V1 = [[1,0],[0,1]], W1 = [[1,1],[0,0]]  -- no, let me look this up properly.
        
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11): U has A00,A11 entries; V has B00,B11; W combines for C
        # This is getting complicated. Let me use a different approach.
        
        # Direct approach: just use schoolbook for all blocks. Rank = 3*5*6 = 90.
        # The finding says rank 84 is achievable with correct Strassen W-index.
        
        # Let me implement Strassen properly for the 2x2x2 blocks.
        # The 7 rank-1 terms of Strassen 2x2x2:
        
        # Using the standard formulation where:
        # C = A*B, A,B,C are 2x2
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
        
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e  
        # M3 = a(g-h)
        # M4 = d(f-g)
        # M5 = (a+b)h
        # M6 = (c-a)(e+g)
        # M7 = (b-d)(f+h)
        
        # i = M1 + M4 - M5 + M7
        # j = M3 + M5
        # k = M2 + M4
        # l = M1 - M2 + M3 + M6
        
        # Each Mk is a product of a linear form in A and a linear form in B.
        # In the tensor decomposition, each Mk contributes one rank-1 term.
        # The U, V, W for each term:
        
        # For M1 = (a+d)(e+h):
        # U1: a->1, d->1 (positions (0,0) and (1,1) in A)
        # V1: e->1, h->1 (positions (0,0) and (1,1) in B)
        # W1: i->1, l->1 (positions (0,0) and (1,1) in C), but with the combination...
        
        # Actually, W1 for M1 contributes to C with: i+=1, l+=1 (coefficient +1 for both)
        # But i = M1+M4-M5+M7 and l = M1-M2+M3+M6
        # So W1[i_pos]=1, W1[l_pos]=1, others 0
        
        # Let me define the 7 terms with their U, V, W matrices (2x2 each):
        
        # Positions: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
        #           B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
        #           C[0,0]=i, C[0,1]=j, C[1,0]=k, C[1,1]=l
        
        terms_222 = [
            # (U_matrix, V_matrix, W_matrix) as flat lists [00,01,10,11]
            # M1 = (a+d)(e+h)
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),  # W: i+=1, l+=1
            # M2 = (c+d)e
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),   # W: k+=1
            # M3 = a(g-h)
            ([1,0,0,0], [0,0,1,-1], [0,1,0,0]),   # W: j+=1
            # M4 = d(f-g)
            ([0,0,0,1], [0,1,-1,0], [1,0,0,0]),   # W: i+=1... wait
            # i = M1+M4-M5+M7, so M4 contributes +1 to i
            # k = M2+M4, so M4 contributes +1 to k
            # Let me redo: M4 should have W=[1,0,1,0] (i and k)
            
            # I'm making errors. Let me be very careful.
            # i = M1 + M4 - M5 + M7
            # j = M3 + M5
            # k = M2 + M4
            # l = M1 - M2 + M3 + M6
        ]
        
        # This is getting error-prone. Let me just use the direct/schoolbook approach
        # which is guaranteed correct, and accept rank 90.
        
        # Actually, let me implement it correctly using a systematic approach.
        pass
    
    # Given the complexity and risk of errors, let me use a cleaner approach:
    # Use schoolbook for the full 3x5x6, which gives rank 90.
    # This is the safe baseline.
    
    # But the finding shows rank 84 is achievable. Let me try to implement it correctly.
    # 
    # Approach: Split into blocks and use Strassen for 2x2x2 sub-blocks.
    # Rows: 2+1, Middle: 2+2+1, Cols: 2+2+2
    #
    # For each combination of (row_block, mid_block, col_block):
    # If all three sizes are 2: use Strassen (rank 7)
    # Otherwise: use schoolbook for that sub-block
    
    # But Strassen applies to the full 2x2x2 tensor, not just one (i,j,k) triple.
    # The 7 rank-1 terms of Strassen 2x2x2 each involve multiple (i,j,k) positions.
    
    # Let me implement Strassen 2x2x2 correctly:
    
    def strassen_222_terms(i0, j0, k0):
        """
        Return list of (U_entries, V_entries, W_entries) for Strassen 2x2x2.
        i0: starting row index, j0: starting mid index, k0: starting col index
        U_entries: dict mapping (i, j) -> value
        V_entries: dict mapping (j, k) -> value  
        W_entries: dict mapping (k, i) -> value  (note: W index is k*n+i)
        """
        # A[i0,j0], A[i0,j0+1], A[i0+1,j0], A[i0+1,j0+1]
        # B[j0,k0], B[j0,k0+1], B[j0+1,k0], B[j0+1,k0+1]
        # C[i0,k0], C[i0,k0+1], C[i0+1,k0], C[i0+1,k0+1]
        
        # Strassen's 7 terms:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # A00=A[i0,j0], A01=A[i0,j0+1], A10=A[i0+1,j0], A11=A[i0+1,j0+1]
        # B00=B[j0,k0], B01=B[j0,k0+1], B10=B[j0+1,k0], B11=B[j0+1,k0+1]
        
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00->1, A11->1
        # V: B00->1, B11->1
        # W: C00->+1, C11->+1
        terms.append({
            'U': {(i0,j0): 1, (i0+1,j0+1): 1},
            'V': {(j0,k0): 1, (j0+1,k0+1): 1},
            'W': {(k0,i0): 1, (k0+1,i0+1): 1}
        })
        
        # M2 = (A10+A11)B00
        # U: A10->1, A11->1
        # V: B00->1
        # W: C10->+1
        terms.append({
            'U': {(i0+1,j0): 1, (i0+1,j0+1): 1},
            'V': {(j0,k0): 1},
            'W': {(k0,i0+1): 1}
        })
        
        # M3 = A00(B10-B11)
        # U: A00->1
        # V: B10->1, B11->-1
        # W: C01->+1
        terms.append({
            'U': {(i0,j0): 1},
            'V': {(j0+1,k0): 1, (j0+1,k0+1): -1},
            'W': {(k0+1,i0): 1}
        })
        
        # M4 = A11(B01-B10)
        # U: A11->1
        # V: B01->1, B10->-1
        # W: C00->+1, C10->+1
        terms.append({
            'U': {(i0+1,j0+1): 1},
            'V': {(j0,k0+1): 1, (j0+1,k0): -1},
            'W': {(k0,i0): 1, (k0,i0+1): 1}
        })
        
        # M5 = (A00+A01)B11
        # U: A00->1, A01->1
        # V: B11->1
        # W: C00->-1, C01->+1
        terms.append({
            'U': {(i0,j0): 1, (i0,j0+1): 1},
            'V': {(j0+1,k0+1): 1},
            'W': {(k0,i0): -1, (k0+1,i0): 1}
        })
        
        # M6 = (A10-A00)(B00+B10)
        # U: A10->1, A00->-1
        # V: B00->1, B10->1
        # W: C11->+1
        terms.append({
            'U': {(i0+1,j0): 1, (i0,j0): -1},
            'V': {(j0,k0): 1, (j0+1,k0): 1},
            'W': {(k0+1,i0+1): 1}
        })
        
        # M7 = (A01-A11)(B01+B11)
        # U: A01->1, A11->-1
        # V: B01->1, B11->1
        # W: C00->+1
        terms.append({
            'U': {(i0,j0+1): 1, (i0+1,j0+1): -1},
            'V': {(j0,k0+1): 1, (j0+1,k0+1): 1},
            'W': {(k0,i0): 1}
        })
        
        return terms
    
    # Now build the full decomposition:
    # For each (row_block, mid_block, col_block):
    #   If all sizes are 2: use Strassen
    #   Else: use schoolbook for that sub-block
    
    U_list = []
    V_list = []
    W_list = []
    
    row_blocks = [(0, 2), (2, 3)]
    mid_blocks = [(0, 2), (2, 4), (4, 5)]
    col_blocks = [(0, 2), (2, 4), (4, 6)]
    
    for (i_start, i_end) in row_blocks:
        for (j_start, j_end) in mid_blocks:
            for (k_start, k_end) in col_blocks:
                i_size = i_end - i_start
                j_size = j_end - j_start
                k_size = k_end - k_start
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Use Strassen
                    for term in strassen_222_terms(i_start, j_start, k_start):
                        # Convert to flat entries
                        u_dict = {}
                        for (i, j), val in term['U'].items():
                            u_dict[i * m + j] = val
                        v_dict = {}
                        for (j, k), val in term['V'].items():
                            v_dict[j * p + k] = val
                        w_dict = {}
                        for (k, i), val in term['W'].items():
                            w_dict[k * n + i] = val
                        
                        # Create full-length vectors
                        u_vec = [0] * (n * m)
                        v_vec = [0] * (m * p)
                        w_vec = [0] * (p * n)
                        for idx, val in u_dict.items():
                            u_vec[idx] = val
                        for idx, val in v_dict.items():
                            v_vec[idx] = val
                        for idx, val in w_dict.items():
                            w_vec[idx] = val
                        
                        U_list.append(u_vec)
                        V_list.append(v_vec)
                        W_list.append(w_vec)
                else:
                    # Schoolbook for this sub-block
                    for i in range(i_start, i_end):
                        for j in range(j_start, j_end):
                            for k in range(k_start, k_end):
                                u_vec = [0] * (n * m)
                                v_vec = [0] * (m * p)
                                w_vec = [0] * (p * n)
                                u_vec[i * m + j] = 1
                                v_vec[j * p + k] = 1
                                w_vec[k * n + i] = 1
                                U_list.append(u_vec)
                                V_list.append(v_vec)
                                W_list.append(w_vec)
    
    # Count the rank
    rank = len(U_list)
    
    # Verify we're within time
    if time.time() > deadline - 5.0:
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Quick verification with random matrices
    try:
        import numpy as np
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u = np.array(U_list[r])
            v = np.array(V_list[r])
            w = np.array(W_list[r])
            # u[i*m+j], v[j*p+k], w[k*n+i]
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += w[k * n + i] * (
                        sum(u[i * m + j] * v[j * p + k] for j in range(m))
                    )
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Verification failed, fall back to schoolbook
            U_list = []
            V_list = []
            W_list = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_vec = [0] * (n * m)
                        v_vec = [0] * (m * p)
                        w_vec = [0] * (p * n)
                        u_vec[i * m + j] = 1
                        v_vec[j * p + k] = 1
                        w_vec[k * n + i] = 1
                        U_list.append(u_vec)
                        V_list.append(v_vec)
                        W_list.append(w_vec)
            rank = n * m * p
    except:
        pass
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

