"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # We will build U, V, W as lists of columns (each column is a vector)
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_entries, v_entries, w_entries):
        """
        Add a rank-1 term.
        u_entries: dict {index: value} for U vector (length n*m)
        v_entries: dict {index: value} for V vector (length m*p)
        w_entries: dict {index: value} for W vector (length p*n)
        """
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        for idx, val in u_entries.items():
            u_vec[idx] = val
        for idx, val in v_entries.items():
            v_vec[idx] = val
        for idx, val in w_entries.items():
            w_vec[idx] = val
            
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Helper to get global index for U (i, j) where i in [0,n), j in [0,m)
    def u_idx(i, j):
        return i * m + j
        
    # Helper to get global index for V (j, k) where j in [0,m), k in [0,p)
    def v_idx(j, k):
        return j * p + k
        
    # Helper to get global index for W (k, i) where k in [0,p), i in [0,n)
    def w_idx(k, i):
        return k * n + i

    # Strassen decomposition for 2x2 x 2x2
    # Local indices: A(i,j), B(j,k), C(i,k) with i,j,k in {0,1}
    # Global offsets: i0, j0 for A; j0, k0 for B; i0, k0 for C
    # Note: In our problem, A is 4x4, B is 4x7.
    # We split A into 2x2 blocks. A_block(r, c) covers rows 2r..2r+1, cols 2c..2c+1.
    # We split B into 2x? blocks. B_block(r, c) covers rows 2r..2r+1, cols 2c..2c+1 (if c<2) or 2c..2c+2 (if c==2).
    
    # We will iterate over the 4x4 A matrix blocks and 4x7 B matrix blocks.
    # A has 2x2 blocks (4 total).
    # B has 2x2 blocks (2 total) and 2x3 blocks (2 total).
    
    # Let's define the Strassen terms for 2x2 x 2x2.
    # The 7 terms are sums of products of linear combinations of A and B entries.
    # Each term contributes to specific C entries.
    # We need to map the local 2x2 indices to global indices.
    
    # Strassen 2x2 x 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each term Mx, we have a specific combination of A entries and B entries.
    # We need to add this to U, V, W.
    # U gets the A coefficients, V gets the B coefficients, W gets the C coefficients.
    
    # Let's define the terms as (u_dict, v_dict, w_dict) for the local 2x2x2 case.
    # We will generate these dynamically for each block position.
    
    # Block positions:
    # A blocks: (r, c) with r,c in {0,1}. Rows 2r, 2r+1; Cols 2c, 2c+1.
    # B blocks: (r, c) with r in {0,1}, c in {0,1,2}.
    #   If c < 2: Rows 2r, 2r+1; Cols 2c, 2c+1. (2x2)
    #   If c == 2: Rows 2r, 2r+1; Cols 4, 5, 6. (2x3)
    # C blocks: (r, c) with r in {0,1}, c in {0,1,2}.
    #   If c < 2: Rows 2r, 2r+1; Cols 2c, 2c+1. (2x2)
    #   If c == 2: Rows 2r, 2r+1; Cols 4, 5, 6. (2x3)
    
    # For each pair of A block (ra, ca) and B block (rb, cb):
    # If ra != rb, no contribution to C.
    # If ra == rb, the result is in C block (ra, cb).
    
    # So we iterate r in {0,1}.
    # For each r:
    #   A blocks involved: (r, 0) and (r, 1).
    #   B blocks involved: (r, 0), (r, 1), (r, 2).
    
    # Case 1: B block is 2x2 (cb in {0,1}).
    #   We have A11=(r,0), A12=(r,1), A21=(r+1,0), A22=(r+1,1) -- Wait, indices are local to the 2x2 block of A?
    #   No, Strassen is applied to the 2x2 block of A and 2x2 block of B.
    #   The "2x2" refers to the block size.
    #   So for A block (r, c), the local indices 0,1 map to global rows 2r, 2r+1 and cols 2c, 2c+1.
    
    # Let's implement a helper to add a Strassen 2x2x2 decomposition for specific block offsets.
    
    def add_strassen_2x2x2x2(ar, ac, br, bc):
        """
        Adds Strassen decomposition for A block (ar, ac) and B block (br, bc).
        Both are 2x2.
        A block (ar, ac): Rows 2*ar, 2*ar+1; Cols 2*ac, 2*ac+1.
        B block (br, bc): Rows 2*br, 2*br+1; Cols 2*bc, 2*bc+1.
        Result C block: Rows 2*ar, 2*ar+1; Cols 2*bc, 2*bc+1.
        Requires ar == br.
        """
        if ar != br:
            return
            
        # Local indices mapping
        # A00 -> (2*ar, 2*ac), A01 -> (2*ar, 2*ac+1)
        # A10 -> (2*ar+1, 2*ac), A11 -> (2*ar+1, 2*ac+1)
        # B00 -> (2*br, 2*bc), B01 -> (2*br, 2*bc+1)
        # B10 -> (2*br+1, 2*bc), B11 -> (2*br+1, 2*bc+1)
        # C00 -> (2*bc, 2*ar), C01 -> (2*bc+1, 2*ar)
        # C10 -> (2*bc, 2*ar+1), C11 -> (2*bc+1, 2*ar+1)
        
        # Define the 7 terms.
        # Each term is a sum of (u, v, w) contributions.
        # We will define the linear combinations for U, V, W.
        
        # M1 = (A00 + A11) * (B00 + B11)
        # U: A00:1, A11:1
        # V: B00:1, B11:1
        # W: C00:1, C11:1
        add_term(
            {u_idx(2*ar, 2*ac): 1, u_idx(2*ar+1, 2*ac+1): 1},
            {v_idx(2*br, 2*bc): 1, v_idx(2*br+1, 2*bc+1): 1},
            {w_idx(2*bc, 2*ar): 1, w_idx(2*bc+1, 2*ar+1): 1}
        )
        
        # M2 = (A10 + A11) * B00
        # U: A10:1, A11:1
        # V: B00:1
        # W: C10:1
        add_term(
            {u_idx(2*ar+1, 2*ac): 1, u_idx(2*ar+1, 2*ac+1): 1},
            {v_idx(2*br, 2*bc): 1},
            {w_idx(2*bc, 2*ar+1): 1}
        )
        
        # M3 = A00 * (B01 - B11)
        # U: A00:1
        # V: B01:1, B11:-1
        # W: C01:1
        add_term(
            {u_idx(2*ar, 2*ac): 1},
            {v_idx(2*br, 2*bc+1): 1, v_idx(2*br+1, 2*bc+1): -1},
            {w_idx(2*bc+1, 2*ar): 1}
        )
        
        # M4 = A11 * (B10 - B00)
        # U: A11:1
        # V: B10:1, B00:-1
        # W: C00:1, C10:1
        add_term(
            {u_idx(2*ar+1, 2*ac+1): 1},
            {v_idx(2*br+1, 2*bc): 1, v_idx(2*br, 2*bc): -1},
            {w_idx(2*bc, 2*ar): 1, w_idx(2*bc, 2*ar+1): 1}
        )
        
        # M5 = (A00 + A01) * B11
        # U: A00:1, A01:1
        # V: B11:1
        # W: C00:-1, C01:1
        add_term(
            {u_idx(2*ar, 2*ac): 1, u_idx(2*ar, 2*ac+1): 1},
            {v_idx(2*br+1, 2*bc+1): 1},
            {w_idx(2*bc, 2*ar): -1, w_idx(2*bc+1, 2*ar): 1}
        )
        
        # M6 = (A10 - A00) * (B00 + B01)
        # U: A10:1, A00:-1
        # V: B00:1, B01:1
        # W: C11:1
        add_term(
            {u_idx(2*ar+1, 2*ac): 1, u_idx(2*ar, 2*ac): -1},
            {v_idx(2*br, 2*bc): 1, v_idx(2*br, 2*bc+1): 1},
            {w_idx(2*bc+1, 2*ar+1): 1}
        )
        
        # M7 = (A01 - A11) * (B10 + B11)
        # U: A01:1, A11:-1
        # V: B10:1, B11:1
        # W: C00:1, C11:-1
        add_term(
            {u_idx(2*ar, 2*ac+1): 1, u_idx(2*ar+1, 2*ac+1): -1},
            {v_idx(2*br+1, 2*bc): 1, v_idx(2*br+1, 2*bc+1): 1},
            {w_idx(2*bc, 2*ar): 1, w_idx(2*bc+1, 2*ar+1): -1}
        )

    # Case 2: B block is 2x3 (bc == 2).
    # A block (ar, ac) is 2x2.
    # B block (br, 2) is 2x3 (cols 4,5,6).
    # Result C block (ar, 2) is 2x3 (cols 4,5,6).
    # We use schoolbook for this part: 2*2*3 = 6 terms.
    
    def add_schoolbook_2x2x2x3(ar, ac, br, bc):
        """
        Adds schoolbook decomposition for A block (ar, ac) [2x2] and B block (br, bc) [2x3].
        A block (ar, ac): Rows 2*ar, 2*ar+1; Cols 2*ac, 2*ac+1.
        B block (br, 2): Rows 2*br, 2*br+1; Cols 4, 5, 6.
        Result C block: Rows 2*bc, 2*bc+1; Cols 2*ar, 2*ar+1. (Wait, C is p x n, so rows are k, cols are i)
        C block (br, 2) in terms of C indices:
        C rows (k) correspond to B cols. B cols 4,5,6 -> C rows 4,5,6.
        C cols (i) correspond to A rows. A rows 2*ar, 2*ar+1 -> C cols 2*ar, 2*ar+1.
        """
        if ar != br:
            return
            
        # A entries:
        # A00: (2*ar, 2*ac), A01: (2*ar, 2*ac+1)
        # A10: (2*ar+1, 2*ac), A11: (2*ar+1, 2*ac+1)
        
        # B entries:
        # B00: (2*br, 4), B01: (2*br, 5), B02: (2*br, 6)
        # B10: (2*br+1, 4), B11: (2*br+1, 5), B12: (2*br+1, 6)
        
        # C entries:
        # C00: (4, 2*ar), C01: (5, 2*ar), C02: (6, 2*ar)
        # C10: (4, 2*ar+1), C11: (5, 2*ar+1), C12: (6, 2*ar+1)
        
        # C00 = A00*B00 + A10*B10
        # C01 = A00*B01 + A10*B11
        # C02 = A00*B02 + A10*B12
        # C10 = A01*B00 + A11*B10
        # C11 = A01*B01 + A11*B11
        # C12 = A01*B02 + A11*B12
        
        # We add 6 terms.
        # Term 1: A00 * B00 -> C00
        add_term(
            {u_idx(2*ar, 2*ac): 1},
            {v_idx(2*br, 4): 1},
            {w_idx(4, 2*ar): 1}
        )
        # Term 2: A10 * B10 -> C00
        add_term(
            {u_idx(2*ar+1, 2*ac): 1},
            {v_idx(2*br+1, 4): 1},
            {w_idx(4, 2*ar): 1}
        )
        # Term 3: A00 * B01 -> C01
        add_term(
            {u_idx(2*ar, 2*ac): 1},
            {v_idx(2*br, 5): 1},
            {w_idx(5, 2*ar): 1}
        )
        # Term 4: A10 * B11 -> C01
        add_term(
            {u_idx(2*ar+1, 2*ac): 1},
            {v_idx(2*br+1, 5): 1},
            {w_idx(5, 2*ar): 1}
        )
        # Term 5: A00 * B02 -> C02
        add_term(
            {u_idx(2*ar, 2*ac): 1},
            {v_idx(2*br, 6): 1},
            {w_idx(6, 2*ar): 1}
        )
        # Term 6: A10 * B12 -> C02
        add_term(
            {u_idx(2*ar+1, 2*ac): 1},
            {v_idx(2*br+1, 6): 1},
            {w_idx(6, 2*ar): 1}
        )
        # Term 7: A01 * B00 -> C10
        add_term(
            {u_idx(2*ar, 2*ac+1): 1},
            {v_idx(2*br, 4): 1},
            {w_idx(4, 2*ar+1): 1}
        )
        # Term 8: A11 * B10 -> C10
        add_term(
            {u_idx(2*ar+1, 2*ac+1): 1},
            {v_idx(2*br+1, 4): 1},
            {w_idx(4, 2*ar+1): 1}
        )
        # Term 9: A01 * B01 -> C11
        add_term(
            {u_idx(2*ar, 2*ac+1): 1},
            {v_idx(2*br, 5): 1},
            {w_idx(5, 2*ar+1): 1}
        )
        # Term 10: A11 * B11 -> C11
        add_term(
            {u_idx(2*ar+1, 2*ac+1): 1},
            {v_idx(2*br+1, 5): 1},
            {w_idx(5, 2*ar+1): 1}
        )
        # Term 11: A01 * B02 -> C12
        add_term(
            {u_idx(2*ar, 2*ac+1): 1},
            {v_idx(2*br, 6): 1},
            {w_idx(6, 2*ar+1): 1}
        )
        # Term 12: A11 * B12 -> C12
        add_term(
            {u_idx(2*ar+1, 2*ac+1): 1},
            {v_idx(2*br+1, 6): 1},
            {w_idx(6, 2*ar+1): 1}
        )

    # Iterate over all blocks
    for r in range(2):
        # A col 0 (2x2)
        # B col 0 (2x2)
        add_strassen_2x2x2x2(r, 0, r, 0)
        # B col 1 (2x2)
        add_strassen_2x2x2x2(r, 1, r, 1)
        # B col 2 (2x3)
        add_schoolbook_2x2x2x3(r, 0, r, 2)
        add_schoolbook_2x2x2x3(r, 1, r, 2)

    # Convert lists of columns to matrices
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for r_idx in range(n * m):
            U[r_idx][c] = U_cols[c][r_idx]
        for r_idx in range(m * p):
            V[r_idx][c] = V_cols[c][r_idx]
        for r_idx in range(p * n):
            W[r_idx][c] = W_cols[c][r_idx]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
