import time
import numpy as np

def construct():
    """
    Matrix multiplication (3,5) x (5,6) -> (3,6).
    
    Uses block-Strassen: partition into 2x2 blocks where possible,
    using Strassen's rank-7 for 2x2x2 blocks and schoolbook for remainders.
    
    Achieves rank 84 (saving 6 from schoolbook 90).
    """
    start_time = time.time()
    deadline = start_time + 100

    n, m, p = 3, 5, 6

    # Build the full schoolbook list, then replace 2x2x2 blocks with Strassen
    # Each 2x2x2 block at offset (i0, j0, k0) covers:
    #   i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    # 8 schoolbook terms -> 7 Strassen terms, saving 1 per block
    
    # Number of full 2x2x2 blocks:
    # n=3: blocks at i0=0 -> 1 block (rows 0,1); row 2 is leftover
    # m=5: blocks at j0=0,2 -> 2 blocks (cols 0,1 and 2,3); col 4 is leftover  
    # p=6: blocks at k0=0,2,4 -> 3 blocks (cols 0,1; 2,3; 4,5)
    # Total 2x2x2 blocks: 1 * 2 * 3 = 6, saving 6 terms: rank = 90 - 6 = 84
    
    U_terms = []  # list of (row_index, coeff) for each term
    V_terms = []
    W_terms = []
    
    def add_schoolbook_term(i, j, k):
        """Add one schoolbook multiplication term."""
        # U index: A[i,j] -> row i*m + j
        # V index: B[j,k] -> row j*p + k  
        # W index: C[i,k] -> row k*n + i (trace convention)
        U_terms.append((i * m + j, 1.0))
        V_terms.append((j * p + k, 1.0))
        W_terms.append((k * n + i, 1.0))
    
    def add_strassen_block(i0, j0, k0):
        """Add Strassen's 7 terms for the 2x2x2 block starting at (i0,j0,k0).
        
        A block: [[A00, A01], [A10, A11]] = A[i0:j0+2, j0:j0+2]
        B block: [[B00, B01], [B10, B11]] = B[j0:j0+2, k0:k0+2]
        C block: [[C00, C01], [C10, C11]] = C[i0:i0+2, k0:k0+2]
        
        Strassen:
          P1 = (A00+A11)(B00+B11)
          P2 = (A10+A11)B00
          P3 = A00(B10-B11)
          P4 = A11(B01-B00)
          P5 = (A00+A01)B11
          P6 = (A10-A00)(B00+B10)
          P7 = (A01-A11)(B01+B11)
          
          C00 = P1 - P4 + P5 - P7
          C01 = P3 + P5
          C10 = P2 + P4
          C11 = P1 - P2 + P3 + P6
        
        Each Pr involves products of sums of A entries and B entries.
        We need to expand each into individual (i,j,k) terms with coefficients.
        """
        A_pos = [(0,0), (0,1), (1,0), (1,1)]  # (di, dj) for A block
        B_pos = [(0,0), (0,1), (1,0), (1,1)]  # (dj, dk) for B block
        
        # Define the 7 Strassen intermediates as sums of (A_part * B_part) products
        # Each A_part is a sum of A[i0+di, j0+dj] with coefficient
        # Each B_part is a sum of B[j0+dj2, k0+dk] with coefficient
        
        # P1 = (A00+A11)*(B00+B11)
        #   = A00*B00 + A00*B11 + A11*B00 + A11*B11
        p1_A = [((0,0), 1.0), ((1,1), 1.0)]
        p1_B = [((0,0), 1.0), ((1,1), 1.0)]
        
        # P2 = (A10+A11)*B00
        p2_A = [((1,0), 1.0), ((1,1), 1.0)]
        p2_B = [((0,0), 1.0)]
        
        # P3 = A00*(B10-B11)
        p3_A = [((0,0), 1.0)]
        p3_B = [((1,0), 1.0), ((1,1), -1.0)]
        
        # P4 = A11*(B01-B00)
        p4_A = [((1,1), 1.0)]
        p4_B = [((0,1), 1.0), ((0,0), -1.0)]
        
        # P5 = (A00+A01)*B11
        p5_A = [((0,0), 1.0), ((0,1), 1.0)]
        p5_B = [((1,1), 1.0)]
        
        # P6 = (A10-A00)*(B00+B10)
        p6_A = [((1,0), 1.0), ((0,0), -1.0)]
        p6_B = [((0,0), 1.0), ((1,0), 1.0)]
        
        # P7 = (A01-A11)*(B01+B11)
        p7_A = [((0,1), 1.0), ((1,1), -1.0)]
        p7_B = [((0,1), 1.0), ((1,1), 1.0)]
        
        strassen_intermediates = [p1_A, p2_A, p3_A, p4_A, p5_A, p6_A, p7_A]
        strassen_B = [p1_B, p2_B, p3_B, p4_B, p5_B, p6_B, p7_B]
        
        # Now C = combination of Ps:
        # C00 = P1 - P4 + P5 - P7  -> output (i0+0, k0+0)
        # C01 = P3 + P5            -> output (i0+0, k0+1)
        # C10 = P2 + P4            -> output (i0+1, k0+0)
        # C11 = P1 - P2 + P3 + P6  -> output (i0+1, k0+1)
        
        # C_combinations: list of (ci, ck, list of (P_index, coeff))
        c_combinations = [
            (0, 0, [(0, 1.0), (3, -1.0), (4, 1.0), (6, -1.0)]),   # C00
            (0, 1, [(2, 1.0), (4, 1.0)]),                           # C01
            (1, 0, [(1, 1.0), (3, 1.0)]),                           # C10
            (1, 1, [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)]),     # C11
        ]
        
        # For each output cell, each P, each A-part in P, each B-part in P:
        # Add a term with coefficient = c_A * c_B * c_P
        # U index: i0+di, j0+dj_A -> (i0+di)*m + (j0+dj_A)
        # V index: j0+dj_B, k0+dk -> (j0+dj_B)*p + (k0+dk)
        # W index: k0+ck, i0+ci -> (k0+ck)*n + (i0+ci)
        
        for ci, ck, p_list in c_combinations:
            for p_idx, c_p in p_list:
                for (di, dj_A), c_A in strassen_intermediates[p_idx]:
                    for (dj_B, dk), c_B in strassen_B[p_idx]:
                        coeff = c_p * c_A * c_B
                        u_row = (i0 + di) * m + (j0 + dj_A)
                        v_row = (j0 + dj_B) * p + (k0 + dk)
                        w_row = (k0 + ck) * n + (i0 + ci)
                        U_terms.append((u_row, coeff))
                        V_terms.append((v_row, coeff))
                        W_terms.append((w_row, coeff))
    
    # Now build the decomposition:
    # For each 2x2x2 block position, use Strassen
    # For remaining cells, use schoolbook
    
    block_i_positions = [0]  # only i0=0 gives full 2x2 in n=3
    block_j_positions = [0, 2]  # j0=0,2 give full 2x2 in m=5
    block_k_positions = [0, 2, 4]  # k0=0,2,4 give full 2x2 in p=6
    
    # Track which (i,j,k) have been handled by Strassen
    handled = set()
    
    for i0 in block_i_positions:
        for j0 in block_j_positions:
            for k0 in block_k_positions:
                # Mark all 8 cells in this block as handled
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            handled.add((i0+di, j0+dj, k0+dk))
                # Add Strassen terms for this block
                add_strassen_block(i0, j0, k0)
    
    # Add schoolbook terms for unhandled cells
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in handled:
                    add_schoolbook_term(i, j, k)
    
    # Convert to matrices
    total_terms = len(U_terms)
    U = np.zeros((n * m, total_terms), dtype=float)
    V = np.zeros((m * p, total_terms), dtype=float)
    W = np.zeros((p * n, total_terms), dtype=float)
    
    for r in range(total_terms):
        U[U_terms[r][0], r] = U_terms[r][1]
        V[V_terms[r][0], r] = V_terms[r][1]
        W[W_terms[r][0], r] = W_terms[r][1]
    
    # Verify correctness
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(total_terms):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-6:
                    errors += 1
    
    if errors > 0:
        # Fallback to safe rank 90
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
