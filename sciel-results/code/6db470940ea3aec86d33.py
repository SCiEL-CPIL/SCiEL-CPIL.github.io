import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using Strassen-like decomposition for 2x4x4
    # Split into 4 blocks of 2x2x2
    # Each 2x2x2 uses rank 7
    # Total rank 28
    
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Define Strassen factors for 2x2 multiplication
    # Local indices i, j in {0,1}
    # U_local[i, j], V_local[j, k], W_local[i, k]
    
    # M1: (A00+A11)(B00+B11)
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: C00=1, C01=-1, C10=-1, C11=1 ? 
    # Wait, the W coefficients come from the reconstruction formulas.
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to accumulate W contributions.
    
    # Let's define the 7 factors
    # Factor 1
    # U: A00, A11
    # V: B00, B11
    # W: C00, C11 (from M1 terms in C00, C11)
    # C00 gets +1 from M1
    # C11 gets +1 from M1
    
    # It is easier to define the U, V, W vectors for each of the 7 multiplications
    # and then sum their contributions to the output C.
    # But the output C is determined by the sum of the products.
    # So for each rank-1 term r, we have:
    # u_r, v_r, w_r such that C = sum w_r * (u_r . A) * (v_r . B) ?
    # No, the tensor decomposition is T = sum U_r V_r W_r.
    # This means C_{ik} = sum_r U_{i,j?} ...
    # The standard form is:
    # C_{ik} = sum_r W_{ik, r} * (sum_j U_{ij, r} A_{ij}) * (sum_j V_{jk, r} B_{jk}) ?
    # No, the indices are:
    # U has shape (n*m, rank). U[i*m+j, r]
    # V has shape (m*p, rank). V[j*p+k, r]
    # W has shape (p*n, rank). W[k*n+i, r]
    
    # C_{ik} = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A_{ij}) * (sum_j V[j*p+k, r] B_{jk})
    # This is NOT correct. The product is not distributed like that.
    # The tensor is T_{i,j,k} = delta_{ij} delta_{jk}? No.
    # T_{i,j,k} = 1 if the indices match the multiplication structure.
    # Actually, the decomposition is:
    # C_{ik} = sum_r W_{ik, r} * U_{i, r} * V_{k, r} ? No.
    
    # Let's stick to the definition:
    # C = sum_r W_r * (U_r * A) * (B * V_r) ?
    # No, the rank-1 term is a tensor of rank 1.
    # A rank-1 tensor is $u \otimes v \otimes w$.
    # $T = \sum_r u_r \otimes v_r \otimes w_r$.
    # $T_{i,j,k} = \sum_r u_{i,j,r} v_{j,k,r} w_{k,i,r}$?
    # Indices:
    # U corresponds to A (n x m).
    # V corresponds to B (m x p).
    # W corresponds to C (n x p).
    
    # The multiplication is $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # The tensor is $T_{i,j,k} = \delta_{i, i'} \delta_{j, j'} \delta_{k, k'}$?
    # No, the tensor of matrix multiplication $M_{n,m,p}$ has entries:
    # $M_{i,j,k} = 1$ if $i \in \{0..n-1\}, j \in \{0..m-1\}, k \in \{0..p-1\}$?
    # No, it's a 3-tensor.
    # The decomposition is:
    # $\sum_j A_{ij} B_{jk} = \sum_r W_{ik,r} (\sum_j U_{ij,r} A_{ij}) (\sum_j V_{jk,r} B_{jk})$?
    # This formula is incorrect.
    
    # Correct formula:
    # $C_{ik} = \sum_r W_{ik,r} \cdot L_A(r) \cdot L_B(r)$
    # where $L_A(r) = \sum_{i',j'} U_{i'j', r} A_{i'j'}$
    # $L_B(r) = \sum_{j',k'} V_{j'k', r} B_{j'k'}$
    
    # Let's verify this with the trivial decomposition.
    # Rank $nmp$.
    # For each $i,j,k$, we have a term.
    # $U_{ij, r} = 1$ if $r$ corresponds to $(i,j,k)$.
    # $V_{jk, r} = 1$ if $r$ corresponds to $(i,j,k)$.
    # $W_{ik, r} = 1$ if $r$ corresponds to $(i,j,k)$.
    # $L_A(r) = A_{ij}$.
    # $L_B(r) = B_{jk}$.
    # $W_{ik,r} L_A(r) L_B(r) = 1 \cdot A_{ij} \cdot B_{jk}$.
    # Summing over $r$ (which iterates over $i,j,k$):
    # $\sum_{i,j,k} \delta_{ik, ik'} A_{ij} B_{jk} = \sum_j A_{ij'} B_{jk'} = C_{i'k'}$.
    # Yes, this works.
    
    # So for Strassen:
    # We define 7 terms for a 2x2 block.
    # Each term r has:
    # $U_r$: linear combo of A elements.
    # $V_r$: linear combo of B elements.
    # $W_r$: linear combo of C elements (coefficients for reconstruction).
    
    # For Block 1 (A_L, B_LL -> C_LL):
    # Local A indices: (0,0), (0,1), (1,0), (1,1) map to Global (0,0), (0,1), (1,0), (1,1).
    # Local B indices: (0,0), (0,1), (1,0), (1,1) map to Global (0,0), (0,1), (1,0), (1,1).
    # Local C indices: (0,0), (0,1), (1,0), (1,1) map to Global (0,0), (0,1), (1,0), (1,1).
    
    # We will add the contributions of the 7 Strassen factors to U, V, W.
    
    def add_strassen_block(offset_u, offset_v, offset_w, block_id):
        # block_id is 0..3
        # We assume the local 2x2 structure
        # We return the rank offset used
        
        # Factors:
        # 1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1 (from C00=M1..., C11=M1...)
        
        # 2: (A10+A11)B00
        # U: A10=1, A11=1
        # V: B00=1
        # W: C10=1, C11=-1 (from C10=M2..., C11=-M2...)
        
        # 3: A00(B01-B11)
        # U: A00=1
        # V: B01=1, B11=-1
        # W: C01=1, C11=1 (from C01=M3..., C11=M3...)
        
        # 4: A11(B10-B00)
        # U: A11=1
        # V: B10=1, B00=-1
        # W: C00=-1, C10=1 (from C00=-M4..., C10=M4...)
        
        # 5: (A00+A01)B11
        # U: A00=1, A01=1
        # V: B11=1
        # W: C00=1, C11=0? 
        # C00 = M1 - M4 + M5 - M7. So C00 gets +1 from M5.
        # C11 = M1 - M2 + M3 + M6. M5 is not in C11.
        
        # 6: (A10-A00)(B00+B01)
        # U: A10=1, A00=-1
        # V: B00=1, B01=1
        # W: C11=1 (from C11=...+M6)
        
        # 7: (A01-A11)(B10+B11)
        # U: A01=1, A11=-1
        # V: B10=1, B11=1
        # W: C01=1, C00=-1 (from C01=M7..., C00=-M7...)
        
        # We need to map local indices to global.
        # Local A (i,j) -> Global (i, j + offset_u) ?
        # offset_u is the column offset for A.
        # Local B (j,k) -> Global (j + offset_v_row, k + offset_v_col) ?
        # V index is j*p + k.
        # offset_v is a tuple (row_offset, col_offset) for B.
        # Local C (i,k) -> Global (i, k + offset_w_col) ?
        # W index is k*n + i.
        # offset_w is a tuple (row_offset, col_offset) for C?
        # C is n x p.
        # W index k*n + i.
        # So k is row, i is col.
        
        # Let's define the mappings explicitly for the 4 blocks.
        
        pass

    # Block 0: A_L (cols 0,1), B_LL (rows 0,1, cols 0,1), C_LL (cols 0,1)
    # A offset: col 0
    # B offset: row 0, col 0
    # C offset: col 0
    
    # Block 1: A_R (cols 2,3), B_RL (rows 2,3, cols 0,1), C_LL (cols 0,1)
    # A offset: col 2
    # B offset: row 2, col 0
    # C offset: col 0
    
    # Block 2: A_L (cols 0,1), B_LR (rows 0,1, cols 2,3), C_LR (cols 2,3)
    # A offset: col 0
    # B offset: row 0, col 2
    # C offset: col 2
    
    # Block 3: A_R (cols 2,3), B_RR (rows 2,3, cols 2,3), C_LR (cols 2,3)
    # A offset: col 2
    # B offset: row 2, col 2
    # C offset: col 2

    blocks = [
        (0, (0, 0), 0), # A_col_offset, B_row_col_offset, C_col_offset
        (2, (2, 0), 0),
        (0, (0, 2), 2),
        (2, (2, 2), 2)
    ]
    
    rank_offset = 0
    for a_off, (b_row_off, b_col_off), c_off in blocks:
        # Define the 7 factors for this block
        # We will add them to U, V, W
        
        # Factor 1
        U[rank_offset, a_off + 0] = 1 # A00
        U[rank_offset, a_off + 1] = 1 # A11 (local j=1, global j=a_off+1)
        # Wait, local A is 2x2.
        # Local (0,0) -> Global (0, a_off)
        # Local (0,1) -> Global (0, a_off+1)
        # Local (1,0) -> Global (1, a_off)
        # Local (1,1) -> Global (1, a_off+1)
        
        # U index is i*m + j.
        # m=4.
        # Local i=0, j=0 -> Global i=0, j=a_off. U_idx = 0*4 + a_off.
        # Local i=1, j=1 -> Global i=1, j=a_off+1. U_idx = 1*4 + a_off+1.
        
        U[0*4 + a_off, rank_offset] = 1
        U[1*4 + a_off + 1, rank_offset] = 1
        
        # V index is j*p + k.
        # Local (0,0) -> Global (b_row_off, b_col_off). V_idx = b_row_off*4 + b_col_off.
        # Local (1,1) -> Global (b_row_off+1, b_col_off+1). V_idx = (b_row_off+1)*4 + (b_col_off+1).
        
        V[b_row_off*4 + b_col_off, rank_offset] = 1
        V[(b_row_off+1)*4 + (b_col_off+1), rank_offset] = 1
        
        # W index is k*n + i.
        # Local (0,0) -> Global (0, c_off). W_idx = 0*2 + 0 = 0.
        # Local (1,1) -> Global (1, c_off+1). W_idx = 1*2 + 1 = 3.
        
        W[0*2 + 0, rank_offset] = 1 # C00
        W[1*2 + 1, rank_offset] = 1 # C11
        
        rank_offset += 1
        
        # Factor 2
        # (A10+A11)B00
        U[1*4 + a_off, rank_offset] = 1 # A10
        U[1*4 + a_off + 1, rank_offset] = 1 # A11
        
        V[b_row_off*4 + b_col_off, rank_offset] = 1 # B00
        
        # C10 = M2... -> W[0*2 + 1] = 1
        # C11 = -M2... -> W[1*2 + 1] = -1
        
        W[0*2 + 1, rank_offset] = 1
        W[1*2 + 1, rank_offset] = -1
        
        rank_offset += 1
        
        # Factor 3
        # A00(B01-B11)
        U[0*4 + a_off, rank_offset] = 1 # A00
        
        V[b_row_off*4 + b_col_off + 1, rank_offset] = 1 # B01
        V[(b_row_off+1)*4 + (b_col_off+1), rank_offset] = -1 # B11
        
        # C01 = M3... -> W[0*2 + 1] = 1
        # C11 = M3... -> W[1*2 + 1] = 1
        
        W[0*2 + 1, rank_offset] = 1
        W[1*2 + 1, rank_offset] = 1
        
        rank_offset += 1
        
        # Factor 4
        # A11(B10-B00)
        U[1*4 + a_off + 1, rank_offset] = 1 # A11
        
        V[(b_row_off+1)*4 + b_col_off, rank_offset] = 1 # B10
        V[b_row_off*4 + b_col_off, rank_offset] = -1 # B00
        
        # C00 = -M4... -> W[0*2 + 0] = -1
        # C10 = M4... -> W[0*2 + 1] = 1
        
        W[0*2 + 0, rank_offset] = -1
        W[0*2 + 1, rank_offset] = 1
        
        rank_offset += 1
        
        # Factor 5
        # (A00+A01)B11
        U[0*4 + a_off, rank_offset] = 1 # A00
        U[0*4 + a_off + 1, rank_offset] = 1 # A01
        
        V[(b_row_off+1)*4 + (b_col_off+1), rank_offset] = 1 # B11
        
        # C00 = M5... -> W[0*2 + 0] = 1
        
        W[0*2 + 0, rank_offset] = 1
        
        rank_offset += 1
        
        # Factor 6
        # (A10-A00)(B00+B01)
        U[1*4 + a_off, rank_offset] = 1 # A10
        U[0*4 + a_off, rank_offset] = -1 # A00
        
        V[b_row_off*4 + b_col_off, rank_offset] = 1 # B00
        V[b_row_off*4 + b_col_off + 1, rank_offset] = 1 # B01
        
        # C11 = M6... -> W[1*2 + 1] = 1
        
        W[1*2 + 1, rank_offset] = 1
        
        rank_offset += 1
        
        # Factor 7
        # (A01-A11)(B10+B11)
        U[0*4 + a_off + 1, rank_offset] = 1 # A01
        U[1*4 + a_off + 1, rank_offset] = -1 # A11
        
        V[(b_row_off+1)*4 + b_col_off, rank_offset] = 1 # B10
        V[(b_row_off+1)*4 + (b_col_off+1), rank_offset] = 1 # B11
        
        # C01 = M7... -> W[0*2 + 1] = 1
        # C00 = -M7... -> W[0*2 + 0] = -1
        
        W[0*2 + 1, rank_offset] = 1
        W[0*2 + 0, rank_offset] = -1
        
        rank_offset += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
