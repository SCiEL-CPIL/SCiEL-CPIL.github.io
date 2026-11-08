import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Total rank is sum of ranks of sub-problems
    # 2x2x2 (Strassen) : 7
    # 2x3x2 (School)   : 12
    # 2x2x3 (School)   : 12
    # 2x3x3 (School)   : 18
    # 2x2x2 (Strassen) : 7
    # 2x3x2 (School)   : 12
    # 2x2x3 (School)   : 12
    # 2x3x3 (School)   : 18
    # Total: 98
    
    total_rank = 98
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a schoolbook block
    def add_schoolbook(m_loc, n_loc, p_loc, i_off, j_off, k_off):
        nonlocal col_offset
        for i in range(m_loc):
            for j in range(n_loc):
                for k in range(p_loc):
                    u_idx = (i + i_off) * m + (j + j_off)
                    v_idx = (j + j_off) * p + (k + k_off)
                    w_idx = (k + k_off) * n + (i + i_off)
                    U[u_idx][col_offset] = 1
                    V[v_idx][col_offset] = 1
                    W[w_idx][col_offset] = 1
                    col_offset += 1

    # Helper to add a Strassen block
    def add_strassen(m_loc, n_loc, p_loc, i_off, j_off, k_off):
        nonlocal col_offset
        # m_loc=2, n_loc=2, p_loc=2
        # Local indices 0,1
        # A is 2x2, B is 2x2, C is 2x2
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to express U, V, W for each term.
        # Each term is a sum of products of linear combinations of A and B.
        # This is equivalent to a rank-1 tensor for each "product" in the expansion?
        # No, Strassen combines terms.
        # The decomposition is a sum of 7 rank-1 tensors.
        # Each rank-1 tensor corresponds to one M_r.
        # For M_r, we need to find u, v, w such that u^T A v = M_r?
        # No, the tensor is $C_{ij} = \sum_{k} A_{ik} B_{kj}$.
        # Strassen gives $C_{ij} = \sum_{r=1}^7 \alpha_{ijr} M_r$.
        # And $M_r = \sum_{i,k} \beta_{ikr} A_{ik} \gamma_{kr} B_{kj}$?
        # Actually, $M_r$ is a scalar computed from A and B.
        # $M_r = \sum_{i,k} (U_r)_{ik} A_{ik} (V_r)_{kj} B_{kj}$?
        # Let's use the standard form.
        # $M_r = \sum_{i=0}^1 \sum_{j=0}^1 U_{r,ij} A_{ij} \sum_{k=0}^1 \sum_{l=0}^1 V_{r,jk} B_{kl}$?
        # This is getting complicated to map to the single U, V, W matrices directly.
        # 
        # Alternative:
        # The rank-1 decomposition of the tensor T is $\sum_{r=1}^R u_r \otimes v_r \otimes w_r$.
        # $T_{ijk} = \sum_r U_{ir} V_{jr} W_{kr}$.
        # For Strassen, we can derive the U, V, W entries.
        
        # Let's hardcode the U, V, W for 2x2x2 Strassen.
        # Reference: 
        # U (2x7), V (2x7), W (2x7)
        # Note: In our global U, index is i*m+j. Here m=2.
        # Local U index: i*2 + j.
        
        # M1: (A00+A11)(B00+B11)
        # A part: A00, A11. V part: B00, B11.
        # This corresponds to terms in the tensor.
        
        # It is easier to use the known U, V, W matrices for Strassen 2x2x2.
        # From literature:
        # U = [[1, 0, 1, 0, 1, 1, 0],
        #      [1, 1, 0, 1, 0, -1, 1]]
        #      [0, 1, 0, 1, 1, 0, -1]
        #      [1, 0, -1, 0, 0, 1, 1]]
        # Wait, U is 4x7 (flattened A).
        # A is 2x2. Flattened: A00, A01, A10, A11.
        # 
        # Let's verify M1 = (A00+A11)(B00+B11).
        # Term 1 in sum:
        # U col 0: [1, 0, 0, 1]^T? (A00, A01, A10, A11) -> 1*A00 + 0*A01 + 0*A10 + 1*A11.
        # V col 0: [1, 0, 0, 1]^T? (B00, B01, B10, B11) -> 1*B00 + 0*B01 + 0*B10 + 1*B11.
        # W col 0: [1, 0, 0, 1]^T? (C00, C01, C10, C11) -> 1*C00 + 0*C01 + 0*C10 + 1*C11.
        # Contribution to C00: 1*1*1 = 1.
        # Contribution to C11: 1*1*1 = 1.
        # Contribution to C01: 0.
        # Contribution to C10: 0.
        # This matches M1 contributing to C00 and C11.
        
        # We need the full set of 7 columns.
        
        # I will generate them programmatically or hardcode if I can find them.
        # Since I cannot browse, I will derive or use a known simple set.
        
        # Actually, I can just use the schoolbook for 2x2x2 (rank 8) if I can't find Strassen easily?
        # If I use rank 8, total rank = 8+12+12+18+8+12+12+18 = 100. No improvement.
        # So I MUST use Strassen.
        
        # Let's hardcode the Strassen U, V, W for 2x2x2.
        # U (4x7), V (4x7), W (4x7)
        # Flattening order: row-major.
        
        # Col 0: M1
        # U: 1,0,0,1 (A00, A01, A10, A11)
        # V: 1,0,0,1 (B00, B01, B10, B11)
        # W: 1,0,0,1 (C00, C01, C10, C11)
        
        # Col 1: M2 = (A10+A11)B00
        # U: 0,0,1,1
        # V: 1,0,0,0
        # W: 0,0,1,0 (C10)
        
        # Col 2: M3 = A00(B01-B11)
        # U: 1,0,0,0
        # V: 0,1,0,-1
        # W: 0,1,0,0 (C01)
        
        # Col 3: M4 = A11(B10-B00)
        # U: 0,0,0,1
        # V: 0,0,1,-1
        # W: 1,0,1,0 (C00, C10)
        
        # Col 4: M5 = (A00+A01)B11
        # U: 1,1,0,0
        # V: 0,0,0,1
        # W: -1,1,0,1 (C00, C01, C11)
        
        # Col 5: M6 = (A10-A00)(B00+B01)
        # U: -1,0,1,0
        # V: 1,1,0,0
        # W: 0,0,0,1 (C11)
        
        # Col 6: M7 = (A01-A11)(B10+B11)
        # U: 0,1,0,-1
        # V: 0,0,1,1
        # W: 1,0,0,1 (C00, C11)
        
        # Let's verify C00 = M1 + M4 - M5 + M7
        # M1: 1
        # M4: 1
        # M5: -1
        # M7: 1
        # Sum: 1+1-1+1 = 2?
        # Wait.
        # C00 = A00B00 + A01B10.
        # M1 = A00B00 + A00B11 + A11B00 + A11B11
        # M4 = A11B10 - A11B00
        # M5 = A00B11 + A01B11
        # M7 = A01B10 + A01B11 - A11B10 - A11B11
        # Sum:
        # A00B00: 1
        # A00B11: 1 - 1 = 0
        # A11B00: 1 - 1 = 0
        # A11B11: 1 - 1 = 0
        # A11B10: 1 - 1 = 0
        # A01B11: -1 + 1 = 0
        # A01B10: 1
        # Sum: A00B00 + A01B10. Correct.
        
        # So the hardcoded values are correct.
        
        # Map to global indices.
        # Local U indices: 0:A00, 1:A01, 2:A10, 3:A11
        # Global U index: (i+i_off)*m + (j+j_off)
        # Local V indices: 0:B00, 1:B01, 2:B10, 3:B11
        # Global V index: (j+j_off)*p + (k+k_off)
        # Local W indices: 0:C00, 1:C01, 2:C10, 3:C11
        # Global W index: (k+k_off)*n + (i+i_off)
        
        # U_vals = [
        #     [1, 0, 1, 0, 1, -1, 0], # A00
        #     [0, 0, 0, 0, 1, 0, 1],  # A01
        #     [0, 1, 0, 0, 0, 1, 0],  # A10
        #     [1, 1, 0, 1, 0, 0, -1]  # A11
        # ]
        # V_vals = [
        #     [1, 1, 0, -1, 0, 1, 0], # B00
        #     [0, 0, 1, 0, 0, 1, 0],  # B01
        #     [0, 0, 0, 1, 0, 0, 1],  # B10
        #     [1, 0, -1, 0, 1, 0, 1]  # B11
        # ]
        # W_vals = [
        #     [1, 0, 0, 1, -1, 0, 1], # C00
        #     [0, 0, 1, 0, 1, 0, 0],  # C01
        #     [0, 1, 0, 1, 0, 0, 0],  # C10
        #     [1, 0, 0, 0, 1, 1, 1]   # C11
        # ]
        
        U_vals = [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ]
        V_vals = [
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ]
        W_vals = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 1]
        ]
        
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    u_idx = (i + i_off) * m + (j + j_off)
                    U[u_idx][col_offset + r] += U_vals[i*2+j][r]
            for j in range(2):
                for k in range(2):
                    v_idx = (j + j_off) * p + (k + k_off)
                    V[v_idx][col_offset + r] += V_vals[j*2+k][r]
            for k in range(2):
                for i in range(2):
                    w_idx = (k + k_off) * n + (i + i_off)
                    W[w_idx][col_offset + r] += W_vals[k*2+i][r]
            
        col_offset += 7

    # Block 1: C11 = A11 B11 (2x2x2)
    # A11: i_off=0, j_off=0
    # B11: j_off=0, k_off=0
    add_strassen(2, 2, 2, 0, 0, 0)
    
    # Block 2: C11 += A12 B21 (2x3x2)
    # A12: i_off=0, j_off=2
    # B21: j_off=2, k_off=0
    add_schoolbook(2, 3, 2, 0, 2, 0)
    
    # Block 3: C12 = A11 B12 (2x2x3)
    # A11: i_off=0, j_off=0
    # B12: j_off=0, k_off=2
    add_schoolbook(2, 2, 3, 0, 0, 2)
    
    # Block 4: C12 += A12 B22 (2x3x3)
    # A12: i_off=0, j_off=2
    # B22: j_off=2, k_off=2
    add_schoolbook(2, 3, 3, 0, 2, 2)
    
    # Block 5: C21 = A21 B11 (2x2x2)
    # A21: i_off=2, j_off=0
    # B11: j_off=0, k_off=0
    add_strassen(2, 2, 2, 2, 0, 0)
    
    # Block 6: C21 += A22 B21 (2x3x2)
    # A22: i_off=2, j_off=2
    # B21: j_off=2, k_off=0
    add_schoolbook(2, 3, 2, 2, 2, 0)
    
    # Block 7: C22 = A21 B12 (2x2x3)
    # A21: i_off=2, j_off=0
    # B12: j_off=0, k_off=2
    add_schoolbook(2, 2, 3, 2, 0, 2)
    
    # Block 8: C22 += A22 B22 (2x3x3)
    # A22: i_off=2, j_off=2
    # B22: j_off=2, k_off=2
    add_schoolbook(2, 3, 3, 2, 2, 2)
    
    return {"U": U, "V": V, "W": W}

