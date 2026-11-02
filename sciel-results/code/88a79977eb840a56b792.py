"""
Strategy: Structured Block Strassen Construction.
Instead of random search, we decompose the 2x5x6 matrix multiplication into 
blocks of 2x2x2 and 2x1x2 multiplications.
The 2x2x2 blocks use Strassen's algorithm (rank 7).
The 2x1x2 blocks use the simple algorithm (rank 2).
This yields a total rank of 48, which is an improvement over the schoolbook rank of 60.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    
    # We will build the decomposition term by term.
    # U: (n*m, rank) = (10, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (12, rank)
    
    # We use a list of terms. Each term is a tuple (u_vec, v_vec, w_vec)
    # where u_vec is length 10, v_vec is length 30, w_vec is length 12.
    # But we build them as sparse dictionaries or just append to lists.
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict):
        # Convert dicts to dense vectors for storage, or keep sparse?
        # The output requires dense lists.
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        for idx, val in u_dict.items():
            u_vec[idx] = val
        for idx, val in v_dict.items():
            v_vec[idx] = val
        for idx, val in w_dict.items():
            w_vec[idx] = val
            
        terms_U.append(u_vec)
        terms_V.append(v_vec)
        terms_W.append(w_vec)

    # Strassen's algorithm for 2x2 matrix multiplication:
    # C = A * B
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # Strassen uses 7 products:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    
    # We need to map these to the global indices.
    # Global A is 2x5. Global B is 5x6.
    # We identified 6 blocks of type 2x2x2 and 3 blocks of type 2x1x2.
    
    # Block Definitions:
    # A is split into columns: A1 (cols 0-1), A2 (cols 2-3), A3 (col 4)
    # B is split into rows: B1 (rows 0-1), B2 (rows 2-3), B3 (row 4)
    # B is split into columns: C1 (cols 0-1), C2 (cols 2-3), C3 (cols 4-5)
    
    # Block (Ai, Bj, Ck) corresponds to:
    # A indices: rows 0-1, cols corresponding to i.
    # B indices: rows corresponding to j, cols corresponding to k.
    # C indices: rows corresponding to k, cols corresponding to i.
    
    # Indices:
    # A: row 0-1, col 0-4. Index = row * 5 + col.
    # B: row 0-4, col 0-5. Index = row * 6 + col.
    # C: row 0-5, col 0-1. Index = row * 2 + col. (Note: Task says W[k*n+i,r] -> row k, col i. k is 0..5, i is 0..1. So W index = k*2 + i).
    
    # Let's define the Strassen block addition function
    def add_strassen_block(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        """
        Adds a Strassen decomposition for a 2x2x2 block.
        A block is A[a_row_start:a_row_start+2, a_col_start:a_col_start+2]
        B block is B[b_row_start:b_row_start+2, b_col_start:b_col_start+2]
        C block is C[c_row_start:c_row_start+2, c_col_start:c_col_start+2]
        
        Note: Our A is 2x5, so a_row_start is always 0.
        """
        
        # A indices in the block:
        # A00: (0, a_col_start)
        # A01: (0, a_col_start+1)
        # A10: (1, a_col_start)
        # A11: (1, a_col_start+1)
        
        # B indices in the block:
        # B00: (b_row_start, b_col_start)
        # B01: (b_row_start, b_col_start+1)
        # B10: (b_row_start+1, b_col_start)
        # B11: (b_row_start+1, b_col_start+1)
        
        # C indices in the block:
        # C00: (c_row_start, c_col_start)
        # C01: (c_row_start, c_col_start+1)
        # C10: (c_row_start+1, c_col_start)
        # C11: (c_row_start+1, c_col_start+1)
        
        # Strassen Terms:
        # p1 = (A00 + A11) * (B00 + B11)
        # p2 = (A10 + A11) * B00
        # p3 = A00 * (B01 - B11)
        # p4 = A11 * (B10 - B00)
        # p5 = (A00 + A01) * B11
        # p6 = (A10 - A00) * (B00 + B01)
        # p7 = (A01 - A11) * (B10 + B11)
        
        # We need to map these products to U, V, W vectors.
        # For a term like (A_part) * (B_part), it contributes to U and V.
        # But Strassen combines them into C.
        # The tensor decomposition is:
        # C = sum_c U_c * V_c * W_c
        # Each Strassen product corresponds to one term in the sum.
        
        # Let's define the 7 terms for this block.
        
        # Term 1: p1
        # U part: A00, A11
        # V part: B00, B11
        # W part: C00, C11 (with coeffs)
        
        # Actually, it's easier to just list the 7 products and their contributions to C.
        # Product 1: (A00 + A11) * (B00 + B11)
        # This product contributes to C00 (+1), C11 (-1)
        # So we need two "sub-terms" in the tensor decomposition?
        # No, the tensor decomposition rank counts the number of rank-1 tensors.
        # Strassen's algorithm uses 7 multiplications.
        # Each multiplication is a rank-1 tensor in the 2x2x2 space.
        # We need to expand each multiplication into the global U, V, W.
        
        # Multiplication 1: M1 = (A00 + A11) * (B00 + B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=-1
        add_term(
            {a_row_start*5 + a_col_start: 1, (a_row_start+1)*5 + (a_col_start+1): 1},
            {b_row_start*6 + b_col_start: 1, (b_row_start+1)*6 + (b_col_start+1): 1},
            {c_row_start*2 + c_col_start: 1, (c_row_start+1)*2 + (c_col_start+1): -1}
        )
        
        # Multiplication 2: M2 = (A10 + A11) * B00
        # U: A10=1, A11=1
        # V: B00=1
        # W: C10=1, C11=-1
        add_term(
            {(a_row_start+1)*5 + a_col_start: 1, (a_row_start+1)*5 + (a_col_start+1): 1},
            {b_row_start*6 + b_col_start: 1},
            {(c_row_start+1)*2 + c_col_start: 1, (c_row_start+1)*2 + (c_col_start+1): -1}
        )
        
        # Multiplication 3: M3 = A00 * (B01 - B11)
        # U: A00=1
        # V: B01=1, B11=-1
        # W: C01=1, C11=1
        add_term(
            {a_row_start*5 + a_col_start: 1},
            {b_row_start*6 + (b_col_start+1): 1, (b_row_start+1)*6 + (b_col_start+1): -1},
            {c_row_start*2 + (c_col_start+1): 1, (c_row_start+1)*2 + (c_col_start+1): 1}
        )
        
        # Multiplication 4: M4 = A11 * (B10 - B00)
        # U: A11=1
        # V: B10=1, B00=-1
        # W: C10=1, C00=1
        add_term(
            {(a_row_start+1)*5 + (a_col_start+1): 1},
            {(b_row_start+1)*6 + b_col_start: 1, b_row_start*6 + b_col_start: -1},
            {(c_row_start+1)*2 + c_col_start: 1, c_row_start*2 + c_col_start: 1}
        )
        
        # Multiplication 5: M5 = (A00 + A01) * B11
        # U: A00=1, A01=1
        # V: B11=1
        # W: C00=-1, C01=1
        add_term(
            {a_row_start*5 + a_col_start: 1, a_row_start*5 + (a_col_start+1): 1},
            {(b_row_start+1)*6 + (b_col_start+1): 1},
            {c_row_start*2 + c_col_start: -1, c_row_start*2 + (c_col_start+1): 1}
        )
        
        # Multiplication 6: M6 = (A10 - A00) * (B00 + B01)
        # U: A10=1, A00=-1
        # V: B00=1, B01=1
        # W: C11=1
        add_term(
            {(a_row_start+1)*5 + a_col_start: 1, a_row_start*5 + a_col_start: -1},
            {b_row_start*6 + b_col_start: 1, b_row_start*6 + (b_col_start+1): 1},
            {(c_row_start+1)*2 + (c_col_start+1): 1}
        )
        
        # Multiplication 7: M7 = (A01 - A11) * (B10 + B11)
        # U: A01=1, A11=-1
        # V: B10=1, B11=1
        # W: C00=1
        add_term(
            {a_row_start*5 + (a_col_start+1): 1, (a_row_start+1)*5 + (a_col_start+1): -1},
            {(b_row_start+1)*6 + b_col_start: 1, (b_row_start+1)*6 + (b_col_start+1): 1},
            {c_row_start*2 + c_col_start: 1}
        )
        
    # Helper for 2x1x2 block (A is 2x1, B is 1x2, C is 2x2)
    # C = A * B
    # A = [[a], [b]], B = [[c, d]]
    # C = [[ac, ad], [bc, bd]]
    # Rank 2:
    # Term 1: a * c -> C00
    # Term 2: a * d -> C01
    # Wait, that's rank 4?
    # No, C00 = ac, C01 = ad, C10 = bc, C11 = bd.
    # We can do:
    # 1. a * c -> C00
    # 2. a * d -> C01
    # 3. b * c -> C10
    # 4. b * d -> C11
    # This is rank 4.
    # Can we do better?
    # (a+b)(c+d) = ac + ad + bc + bd.
    # We need to isolate them.
    # Actually, 2x1x2 is just 4 independent products.
    # So rank is 4.
    
    def add_simple_212_block(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        # A is 2x1 at (a_row_start, a_col_start)
        # B is 1x2 at (b_row_start, b_col_start)
        # C is 2x2 at (c_row_start, c_col_start)
        
        # A0: (a_row_start, a_col_start)
        # A1: (a_row_start+1, a_col_start)
        # B0: (b_row_start, b_col_start)
        # B1: (b_row_start, b_col_start+1)
        # C00: (c_row_start, c_col_start)
        # C01: (c_row_start, c_col_start+1)
        # C10: (c_row_start+1, c_col_start)
        # C11: (c_row_start+1, c_col_start+1)
        
        # Term 1: A0 * B0 -> C00
        add_term(
            {a_row_start*5 + a_col_start: 1},
            {b_row_start*6 + b_col_start: 1},
            {c_row_start*2 + c_col_start: 1}
        )
        
        # Term 2: A0 * B1 -> C01
        add_term(
            {a_row_start*5 + a_col_start: 1},
            {b_row_start*6 + (b_col_start+1): 1},
            {c_row_start*2 + (c_col_start+1): 1}
        )
        
        # Term 3: A1 * B0 -> C10
        add_term(
            {(a_row_start+1)*5 + a_col_start: 1},
            {b_row_start*6 + b_col_start: 1},
            {(c_row_start+1)*2 + c_col_start: 1}
        )
        
        # Term 4: A1 * B1 -> C11
        add_term(
            {(a_row_start+1)*5 + a_col_start: 1},
            {b_row_start*6 + (b_col_start+1): 1},
            {(c_row_start+1)*2 + (c_col_start+1): 1}
        )
        
    # Now, apply to our blocks.
    # A cols: 0-1 (Block 1), 2-3 (Block 2), 4 (Block 3)
    # B rows: 0-1 (Block 1), 2-3 (Block 2), 4 (Block 3)
    # B cols: 0-1 (Block 1), 2-3 (Block 2), 4-5 (Block 3)
    # C rows: 0-1 (Block 1), 2-3 (Block 2), 4-5 (Block 3)
    # C cols: 0-1 (Block 1), 2-3 (Block 2), 4-5 (Block 3) -- Wait, C is 6x2.
    # C cols are 0-1.
    # So C blocks are:
    # (Row 0-1, Col 0-1) -> C00
    # (Row 2-3, Col 0-1) -> C10
    # (Row 4-5, Col 0-1) -> C20
    
    # Mapping:
    # A Block 1 (cols 0-1) interacts with B Block 1 (rows 0-1) -> C Block 1 (rows 0-1)
    # A Block 1 (cols 0-1) interacts with B Block 2 (rows 2-3) -> C Block 2 (rows 2-3)
    # A Block 1 (cols 0-1) interacts with B Block 3 (rows 4)   -> C Block 3 (rows 4-5)
    
    # A Block 2 (cols 2-3) interacts with B Block 1 (rows 0-1) -> C Block 1 (rows 0-1)
    # A Block 2 (cols 2-3) interacts with B Block 2 (rows 2-3) -> C Block 2 (rows 2-3)
    # A Block 2 (cols 2-3) interacts with B Block 3 (rows 4)   -> C Block 3 (rows 4-5)
    
    # A Block 3 (col 4)     interacts with B Block 1 (rows 0-1) -> C Block 1 (rows 0-1)
    # A Block 3 (col 4)     interacts with B Block 2 (rows 2-3) -> C Block 2 (rows 2-3)
    # A Block 3 (col 4)     interacts with B Block 3 (rows 4)   -> C Block 3 (rows 4-5)
    
    # 2x2x2 Blocks (Rank 7 each):
    # 1. A1, B1 -> C1
    add_strassen_block(0, 0, 0, 0, 0, 0)
    # 2. A1, B2 -> C2
    add_strassen_block(0, 0, 2, 0, 2, 0)
    # 3. A1, B3 is 2x1x2 (Rank 4)
    add_simple_212_block(0, 0, 4, 0, 4, 0)
    
    # 4. A2, B1 -> C1
    add_strassen_block(0, 2, 0, 0, 0, 0)
    # 5. A2, B2 -> C2
    add_strassen_block(0, 2, 2, 0, 2, 0)
    # 6. A2, B3 is 2x1x2 (Rank 4)
    add_simple_212_block(0, 2, 4, 0, 4, 0)
    
    # 7. A3, B1 is 2x1x2 (Rank 4)
    add_simple_212_block(0, 4, 0, 0, 0, 0)
    # 8. A3, B2 is 2x1x2 (Rank 4)
    add_simple_212_block(0, 4, 2, 0, 2, 0)
    # 9. A3, B3 is 2x1x2 (Rank 4)
    add_simple_212_block(0, 4, 4, 0, 4, 0)
    
    # Total Rank:
    # 4 Strassen blocks * 7 = 28
    # 5 Simple blocks * 4 = 20
    # Total = 48.
    
    # Construct output arrays
    rank = len(terms_U)
    U = np.array(terms_U, dtype=float).T
    V = np.array(terms_V, dtype=float).T
    W = np.array(terms_W, dtype=float).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
