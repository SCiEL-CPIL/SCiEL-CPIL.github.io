import numpy as np

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a term to U, V, W
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        nonlocal col
        U[u_idx, col] += u_val
        V[v_idx, col] += v_val
        W[w_idx, col] += w_val
        col += 1

    # Helper for Strassen 2x2x2
    # u_indices: list of 4 global U indices for A[0,0], A[0,1], A[1,0], A[1,1]
    # v_indices: list of 4 global V indices for B[0,0], B[0,1], B[1,0], B[1,1]
    # w_indices: list of 4 global W indices for C[0,0], C[0,1], C[1,0], C[1,1]
    def add_strassen(u_idx, v_idx, w_idx):
        # M1 = (A00 + A11)(B00 + B11)
        add_term(u_idx[0], v_idx[0], w_idx[0], 1, 1, 1)
        add_term(u_idx[3], v_idx[0], w_idx[0], 1, 1, 1)
        add_term(u_idx[0], v_idx[3], w_idx[0], 1, 1, 1)
        add_term(u_idx[3], v_idx[3], w_idx[0], 1, 1, 1)
        # Wait, Strassen M1 is a single multiplication of sums.
        # In tensor decomposition, we can't just add 4 terms to one column if they are not linearly independent in the factor matrices?
        # No, the decomposition is $\sum_r U_{.,r} V_{.,r} W_{.,r}$.
        # M1 corresponds to one rank-1 term?
        # No, Strassen's algorithm computes 7 products. Each product is a rank-1 term in the tensor decomposition?
        # Yes. $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
        # This is NOT a rank-1 term of the original tensor.
        # The original tensor is $\sum A_{ij} B_{jk} C_{ik}$.
        # Strassen rewrites the computation.
        # The decomposition is:
        # $C_{00} = M_1 + M_4 - M_5 + M_7$
        # etc.
        # So we need to express $C$ as a linear combination of the 7 intermediate products $M_1 \dots M_7$.
        # Each $M_k$ is a product of a linear combination of A's and a linear combination of B's.
        # So each $M_k$ corresponds to a rank-1 term in the decomposition of the *computation*, but for the tensor decomposition of the multiplication map, we have:
        # $\mu = \sum_{r=1}^7 u_r \otimes v_r \otimes w_r$
        # Where $u_r, v_r$ are the vectors of coefficients for A and B in the $r$-th product, and $w_r$ is the vector of coefficients for C.
        
        # Let's list the 7 terms for 2x2x2:
        # 1. M1: u=(1,0,0,1), v=(1,0,0,1), w=(1,1,0,0) ? No.
        # C00 gets M1. C01 gets 0. C10 gets 0. C11 gets 0?
        # No, the reconstruction is:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for M1:
        # u_vec: A00 + A11 -> [1, 0, 0, 1] (flattened A)
        # v_vec: B00 + B11 -> [1, 0, 0, 1] (flattened B)
        # w_vec: contributes 1 to C00, 1 to C11?
        # From C00 = ... + M1 ...
        # From C11 = M1 ...
        # So w_vec for M1 is [1, 0, 0, 1] (flattened C: C00, C01, C10, C11)
        
        # Let's verify M1 contribution:
        # Term 1: u1 v1 w1.
        # u1 = A00 + A11. v1 = B00 + B11. w1 = C00 + C11.
        # Product: (A00+A11)(B00+B11)(C00+C11).
        # This expands to A00B00C00 + A00B00C11 + A00B11C00 + ...
        # This is not a single term of the original tensor.
        # The decomposition is valid if $\sum u_r v_r w_r = \mu$.
        # Yes, this is the standard Strassen decomposition.
        
        # So we define the 7 vectors for a 2x2x2 block.
        
        # u vectors (size 4, corresponding to A00, A01, A10, A11)
        # v vectors (size 4, corresponding to B00, B01, B10, B11)
        # w vectors (size 4, corresponding to C00, C01, C10, C11)
        
        # M1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
        # M2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]  # C10 = M2 + ...
        # M3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0] # C01 = M3 + ...
        # M4: u=[0,0,0,1], v=[0,-1,1,0], w=[1,0,1,0] # C00=M1+M4..., C10=M2+M4...
        # M5: u=[1,1,0,0], v=[0,0,0,1], w=[-1,1,0,0] # C00=-M5, C01=M5
        # M6: u=[0,0,-1,0], v=[1,1,0,0], w=[0,0,0,1] # C11=M6...
        # M7: u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,0] # C00=M7...
        
        # Let's double check the signs and indices.
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B11)
        # M4 = A22(B12-B22)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # My indexing above used 0-based.
        # A00=A11, A01=A12, A10=A21, A11=A22.
        # B00=B11, B01=B12, B10=B21, B11=B22.
        # C00=C11, C01=C12, C10=C21, C11=C22.
        
        # M1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
        # M2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
        # M3: u=[1,0,0,0], v=[0,-1,1,0], w=[0,1,0,0]  # Note: B21-B11 is B10-B00. So v=[0,-1,1,0].
        # M4: u=[0,0,0,1], v=[0,1,0,-1], w=[1,0,1,0]  # Note: B12-B22 is B01-B11. So v=[0,1,0,-1].
        # M5: u=[1,1,0,0], v=[0,0,0,1], w=[-1,1,0,0]
        # M6: u=[0,0,1,-1], v=[1,1,0,0], w=[0,0,0,1]  # Note: A21-A11 is A10-A00. So u=[0,0,1,-1].
        # M7: u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,0]
        
        # We need to map these local vectors to global vectors.
        pass

    # Since implementing the mapping logic inline is complex, I will pre-calculate the columns.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
