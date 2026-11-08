"""Tensor decomposition for matrix multiplication 4x4 * 4x9."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# Precompute the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
# where the three indices correspond to U-row, V-row, W-row.
# U has n*m = 16 rows, V has m*p = 36 rows, W has p*n = 36 rows.

def _build_tensor():
    n, m, p = DIMENSIONS
    # T is a dict mapping (u_idx, v_idx, w_idx) -> 1
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[(u_idx, v_idx, w_idx)] = 1
    return T, n, m, p

def _schoolbook():
    """Return the schoolbook decomposition (rank 144)."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, T, n, m, p):
    """Check that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == T[i,j,k] for all."""
    rank = len(U[0])
    # For each (u,v) pair, check that the W-column vectors span correctly
    # Actually check: for each (i,j,k), sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(rank):
                    s += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if s != 1:
                    return False
    # Also check that entries are 0 or 1 in T (they should be)
    return True

def _try_strassen_style(n, m, p, T):
    """
    Try to find a lower-rank decomposition using a structured approach.
    For 4x4 * 4x9, we can think of this as 4x4 * (4x9).
    
    One approach: use Strassen on the 4x4 block structure.
    Partition A (4x4) into 2x2 blocks of 2x2, and B (4x9) into 2x2 blocks of 2x2 (with the last block being 2x1 or padded).
    
    Actually, let's try a different approach: use the known optimal rank for 2x2x2 (7) 
    and try to compose.
    
    For 4x4 * 4x9:
    We can write A as [[A11, A12],[A21, A22]] where each is 2x2.
    B as [[B11, B12],[B21, B22]] where B11, B21 are 2x9 and B12, B22 are 2x9.
    
    This doesn't directly help with Strassen since B isn't square.
    
    Let's try a more direct approach: use a greedy/random search for a lower-rank solution.
    """
    pass

def _random_search(T, n, m, p, max_rank, time_limit=10.0):
    """
    Try to find a decomposition of rank <= max_rank using random initialization 
    and local search.
    """
    start = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    # For each rank from max_rank down, try random initialization
    for rank in range(max_rank, 143, -1):  # 144 is schoolbook
        if time.time() - start > time_limit:
            break
        # Try a few random initializations
        for trial in range(50):
            if time.time() - start > time_limit:
                break
            # Random sparse initialization with +/-1 entries
            import random
            random.seed(trial * 12345 + rank)
            U = [[0] * rank for _ in range(nm)]
            V = [[0] * rank for _ in range(mp)]
            W = [[0] * rank for _ in range(pn)]
            
            # Initialize with random +/-1
            for c in range(rank):
                # Pick random non-zero pattern
                for _ in range(nm // 3):
                    u_idx = random.randint(0, nm - 1)
                    U[u_idx][c] = random.choice([-1, 1])
                for _ in range(mp // 3):
                    v_idx = random.randint(0, mp - 1)
                    V[v_idx][c] = random.choice([-1, 1])
                for _ in range(pn // 3):
                    w_idx = random.randint(0, pn - 1)
                    W[w_idx][c] = random.choice([-1, 1])
            
            # Check if this works (very unlikely for random, but let's try)
            if _verify(U, V, W, T, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    T, n, m, p = _build_tensor()
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p  # 144
    
    # Try structured Strassen-like decomposition
    # For 4x4 * 4x9, we can partition into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] (each 2x2)
    # B = [[B11, B12], [B21, B22]] (B11, B21 are 2x9, B12, B22 are 2x9)
    # 
    # Using Strassen for the 2x2 block multiplication:
    # M1 = (A11+A22)(B11+B22)  -> but B11+B22 is 2x9
    # M2 = (A21+A22)B11        -> (A21+A22) is 2x2, B11 is 2x9
    # M3 = A11(B12-B22)        -> A11 is 2x2, (B12-B22) is 2x9
    # M4 = A22(B21-B11)        -> A22 is 2x2, (B21-B11) is 2x9
    # M5 = (A11+A12)B22        -> (A11+A12) is 2x2, B22 is 2x9
    # M6 = (A21-A11)(B11+B12)  -> (A21-A11) is 2x2, (B11+B12) is 2x9
    # M7 = (A12-A22)(B21+B22)  -> (A12-A22) is 2x2, (B21+B22) is 2x9
    #
    # Each Mi is a 2x2 * 2x9 multiplication.
    # For 2x2 * 2x9: schoolbook rank = 2*2*9 = 36
    # So total rank = 7 * 36 = 252, which is worse than 144.
    #
    # Better: use Strassen for 2x2 * 2x2 part and handle the 9 columns separately.
    # Actually, 2x2 * 2x9 can be done with rank 2*2*9=36 schoolbook, or potentially less.
    # The optimal rank for 2x2 * 2x2 is 7. For 2x2 * 2x9, it's at most 7*9/4... no.
    # 
    # Let's think differently. The tensor for 2x2 * 2x9 has dimensions 4 x 18 x 18.
    # Schoolbook rank = 4*18*18... no, it's 2*2*9 = 36.
    # 
    # Actually, let's just try to find a better decomposition directly.
    # 
    # Alternative: use the fact that 4x4 * 4x9 = 4x4 * (4x4 + 4x5)
    # = (4x4 * 4x4) + (4x4 * 4x5)
    # Rank of 4x4*4x4 is at most 49 (Strassen), rank of 4x4*4x5 is at most 80 schoolbook.
    # Total: 49 + 80 = 129 < 144.
    
    if time.time() < deadline - 5.0:
        # Try 4x4 * 4x4 (Strassen rank 49) + 4x4 * 4x5 (schoolbook rank 80) = 129
        # But we need to be careful about the tensor structure.
        # 
        # Actually, the decomposition needs to be for the specific tensor, not just a sum.
        # Let's try a different partition.
        #
        # Partition B (4x9) into B1 (4x4) and B2 (4x5).
        # Then A*B = A*B1 + A*B2.
        # If we have a rank-49 decomposition for 4x4*4x4 and rank-80 for 4x4*4x5,
        # we can combine them for total rank 129.
        #
        # But we need to construct these explicitly.
        #
        # For 4x4 * 4x4, use Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # Each Mi is 2x2*2x2, which has rank 7 (Strassen).
        # So total rank = 7*7 = 49.
        #
        # For 4x4 * 4x5, schoolbook rank = 4*4*5 = 80.
        #
        # Total: 49 + 80 = 129.
        
        # Let's construct the Strassen 4x4*4x4 decomposition
        def strassen_4x4_4x4():
            """Return U, V, W for 4x4 * 4x4 with rank 49."""
            # Each 2x2*2x2 Strassen has rank 7
            # We need to map the 7 intermediate products to the final 4x4 result.
            # 
            # For 2x2 * 2x2 Strassen:
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            #
            # For 4x4 * 4x4, the 2x2 blocks of C are:
            # C11 = M1 + M4 - M5 + M7  (where Mj are 2x2 matrices from the 7 Strassen products)
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            #
            # Each Mj is a 2x2 matrix computed as a 2x2*2x2 product.
            # Using Strassen for each Mj, we get 7 sub-products per Mj, each of rank 7.
            # Wait, no. Each Mj = Xj * Yj where Xj, Yj are 2x2.
            # The rank of a 2x2*2x2 multiplication is 7.
            # So each Mj contributes 7 to the rank.
            # Total: 7 * 7 = 49.
            
            # Let's build it step by step.
            # U has 16 rows (4x4), V has 16 rows (4x4), W has 16 rows (4x4).
            rank = 49
            U = [[0] * rank for _ in range(16)]
            V = [[0] * rank for _ in range(16)]
            W = [[0] * rank for _ in range(16)]
            
            # The 7 Strassen intermediate matrices for 4x4*4x4:
            # M1 = (A11+A22) * (B11+B22)
            # M2 = (A21+A22) * B11
            # M3 = A11 * (B12-B22)
            # M4 = A22 * (B21-B11)
            # M5 = (A11+A12) * B22
            # M6 = (A21-A11) * (B11+B12)
            # M7 = (A12-A22) * (B21+B22)
            
            # Each Mj is a 2x2*2x2 product, which we decompose with Strassen (rank 7).
            # For a 2x2*2x2 Strassen:
            # P1 = (X11+X22)(Y11+Y22)
            # P2 = (X21+X22)Y11
            # P3 = X11(Y12-Y22)
            # P4 = X22(Y21-Y11)
            # P5 = (X11+X12)Y22
            # P6 = (X21-X11)(Y11+Y12)
            # P7 = (X12-X22)(Y21+Y22)
            # C11 = P1+P4-P5+P7, C12 = P3+P5, C21 = P2+P4, C22 = P1-P2+P3+P6
            
            # For each of the 7 Mj, we have 7 sub-products Pj1..Pj7.
            # Total: 49 sub-products.
            
            # Map: for Mj (j=0..6), sub-product k (k=0..6), global index = j*7+k
            # U rows: 0-3 = A11, 4-7 = A12, 8-11 = A21, 12-15 = A22
            # V rows: 0-3 = B11, 4-7 = B12, 8-11 = B21, 12-15 = B22
            # W rows: 0-3 = C11, 4-7 = C12, 8-11 = C21, 12-15 = C22
            
            # Define the 7 Strassen combinations for 4x4*4x4:
            # Mj = Xj * Yj where Xj is a 2x2 linear combo of A-blocks, Yj is 2x2 linear combo of B-blocks
            # Xj blocks: (row_block, col_block) -> coefficient
            # A-blocks: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
            # B-blocks: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
            
            M_defs = [
                # M1 = (A11+A22)(B11+B22)
                {('X', (0,0)): 1, ('X', (1,1)): 1, ('Y', (0,0)): 1, ('Y', (1,1)): 1},
                # M2 = (A21+A22)B11
                {('X', (1,0)): 1, ('X', (1,1)): 1, ('Y', (0,0)): 1},
                # M3 = A11(B12-B22)
                {('X', (0,0)): 1, ('Y', (0,1)): 1, ('Y', (1,1)): -1},
                # M4 = A22(B21-B11)
                {('X', (1,1)): 1, ('Y', (1,0)): 1, ('Y', (0,0)): -1},
                # M5 = (A11+A12)B22
                {('X', (0,0)): 1, ('X', (0,1)): 1, ('Y', (1,1)): 1},
                # M6 = (A21-A11)(B11+B12)
                {('X', (1,0)): 1, ('X', (0,0)): -1, ('Y', (0,0)): 1, ('Y', (0,1)): 1},
                # M7 = (A12-A22)(B21+B22)
                {('X', (0,1)): 1, ('X', (1,1)): -1, ('Y', (1,0)): 1, ('Y', (1,1)): 1},
            ]
            
            # For each Mj, decompose Xj*Yj using Strassen (7 sub-products)
            # Xj is 2x2 with entries from A-blocks, Yj is 2x2 with entries from B-blocks
            # Xj = [[x00, x01], [x10, x11]] where xrc is a linear combo of A-blocks
            # Yj = [[y00, y01], [y10, y11]] where yrc is a linear combo of B-blocks
            
            for j, Mdef in enumerate(M_defs):
                # Extract Xj and Yj coefficients
                Xcoeff = {}  # (r, c) -> {A_block: coeff}
                Ycoeff = {}  # (r, c) -> {B_block: coeff}
                for (side, block), coeff in Mdef.items():
                    if side == 'X':
                        r, c = block
                        for ar in range(2):
                            for ac in range(2):
                                # A_block (r,c) entry (ar,ac) contributes to Xj[r,c]
                                a_block = (r, c)
                                if a_block not in Xcoeff.get((r,c), {}):
                                    Xcoeff.setdefault((r,c), {})
                                Xcoeff[(r,c)][(ar, ac)] = Xcoeff[(r,c)].get((ar, ac), 0) + coeff
                    else:
                        r, c = block
                        for br in range(2):
                            for bc in range(2):
                                b_block = (r, c)
                                if b_block not in Ycoeff.get((r,c), {}):
                                    Ycoeff.setdefault((r,c), {})
                                Ycoeff[(r,c)][(br, bc)] = Ycoeff[(r,c)].get((br, bc), 0) + coeff
                
                # Now Xj is 2x2, Yj is 2x2
                # Apply Strassen to Xj*Yj
                # P1 = (X00+X11)(Y00+Y11)
                # P2 = (X10+X11)Y00
                # P3 = X00(Y01-Y11)
                # P4 = X11(Y10-Y00)
                # P5 = (X00+X01)Y11
                # P6 = (X10-X00)(Y00+Y01)
                # P7 = (X01-X11)(Y10+Y11)
                
                strassen_defs = [
                    # P1 = (X00+X11)(Y00+Y11)
                    {('X', (0,0)): 1, ('X', (1,1)): 1, ('Y', (0,0)): 1, ('Y', (1,1)): 1},
                    # P2 = (X10+X11)Y00
                    {('X', (1,0)): 1, ('X', (1,1)): 1, ('Y', (0,0)): 1},
                    # P3 = X00(Y01-Y11)
                    {('X', (0,0)): 1, ('Y', (0,1)): 1, ('Y', (1,1)): -1},
                    # P4 = X11(Y10-Y00)
                    {('X', (1,1)): 1, ('Y', (1,0)): 1, ('Y', (0,0)): -1},
                    # P5 = (X00+X01)Y11
                    {('X', (0,0)): 1, ('X', (0,1)): 1, ('Y', (1,1)): 1},
                    # P6 = (X10-X00)(Y00+Y01)
                    {('X', (1,0)): 1, ('X', (0,0)): -1, ('Y', (0,0)): 1, ('Y', (0,1)): 1},
                    # P7 = (X01-X11)(Y10+Y11)
                    {('X', (0,1)): 1, ('X', (1,1)): -1, ('Y', (1,0)): 1, ('Y', (1,1)): 1},
                ]
                
                for k, Sdef in enumerate(strassen_defs):
                    gidx = j * 7 + k
                    # Build U, V, W contributions
                    # U: for each (r,c) in X, for each A-block (ar,ac) with coeff, 
                    #   U[A_block_row*4 + A_block_col, gidx] += coeff
                    # V: for each (r,c) in Y, for each B-block (br,bc) with coeff,
                    #   V[B_block_row*4 + B_block_col, gidx] += coeff
                    # W: C = X*Y, C_rc = sum... 
                    #   For Strassen: C11 = P1+P4-P5+P7, C12 = P3+P5, C21 = P2+P4, C22 = P1-P2+P3+P6
                    #   So W contribution depends on k (which P) and the C-block position.
                    
                    # U coefficients
                    for (side, block), coeff in Sdef.items():
                        if side == 'X':
                            r, c = block
                            # X[r,c] has contributions from A-blocks
                            # X[r,c] = sum over (ar,ac) in Xcoeff[(r,c)] of Xcoeff[(r,c)][(ar,ac)] * A[ar,ac]
                            # But Xcoeff[(r,c)] maps (ar,ac) -> total_coeff
                            # The A-block (ar,ac) corresponds to A rows ar*2..ar*2+1, cols ac*2..ac*2+1
                            # U row index for A[ar,ac] is ar*2 + ac (in the 4x4 A matrix, row ar*2, col ac*2 gives index ar*2*4+ac*2... wait)
                            # A is 4x4, U has 16 rows indexed by i*4+j for A[i,j]
                            # A-block (br, bc) is A[br*2:br*2+2, bc*2:bc*2+2]
                            # So A[br*2+ar, bc*2+ac] has U row index (br*2+ar)*4 + (bc*2+ac)
                            if (r, c) in Xcoeff:
                                for (ar, ac), a_coeff in Xcoeff[(r, c)].items():
                                    total_coeff = coeff * a_coeff
                                    u_row = (r * 2 + ar) * 4 + (c * 2 + ac)
                                    U[u_row][gidx] += total_coeff
                    
                    # V coefficients
                    for (side, block), coeff in Sdef.items():
                        if side == 'Y':
                            r, c = block
                            if (r, c) in Ycoeff:
                                for (br, bc), b_coeff in Ycoeff[(r, c)].items():
                                    total_coeff = coeff * b_coeff
                                    v_row = (r * 2 + br) * 4 + (c * 2 + bc)
                                    V[v_row][gidx] += total_coeff
                    
                    # W coefficients: C = X*Y, C is 2x2
                    # C_rc = sum_s X_rs * Y_sc
                    # For Strassen Pk, the contribution to C is:
                    # P1 -> C11, C22 (coeff 1)
                    # P2 -> C21 (coeff 1)
                    # P3 -> C12, C22 (coeff 1)
                    # P4 -> C11, C21 (coeff 1)
                    # P5 -> C11 (coeff -1), C12 (coeff 1)
                    # P6 -> C22 (coeff 1)
                    # P7 -> C11 (coeff 1)
                    
                    w_contribs = {
                        0: [(0,0, 1), (1,1, 1)],  # P1 -> C11, C22
                        1: [(1,0, 1)],              # P2 -> C21
                        2: [(0,1, 1), (1,1, 1)],   # P3 -> C12, C22
                        3: [(0,0, 1), (1,0, 1)],   # P4 -> C11, C21
                        4: [(0,0, -1), (0,1, 1)],  # P5 -> -C11, C12
                        5: [(1,1, 1)],              # P6 -> C22
                        6: [(0,0, 1)],              # P7 -> C11
                    }
                    
                    for (cr, cc), w_coeff in w_contribs[k]:
                        # C-block (cr,cc) is C[cr*2:cr*2+2, cc*2:cc*2+2]
                        # C[cr*2+dr, cc*2+dc] has W row index (cr*2+dr)*4 + (cc*2+dc)
                        # But wait, W is indexed by k*n+i for the 4x4*4x9 case.
                        # For 4x4*4x4, W row = k*4+i where C[k,i]... 
                        # Actually W[k*n+i, c] where n=4, so W[k*4+i, c] for C[k,i]
                        # C is 4x4, so C[k,i] where k=0..3, i=0..3
                        # C-block (cr,cc) contains C[cr*2+dr, cc*2+dc]
                        for dr in range(2):
                            for dc in range(2):
                                w_row = (cr * 2 + dr) * 4 + (cc * 2 + dc)
                                W[w_row][gidx] += w_coeff
            
            return U, V, W, rank
        
        # Now for 4x4 * 4x5, schoolbook rank 80
        def schoolbook_4x4_4x5():
            n, m, p = 4, 4, 5
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            return U, V, W, rank
        
        # Combine: 4x4*4x4 (rank 49) + 4x4*4x5 (rank 80) = rank 129
        # But we need to map this to the 4x4*4x9 tensor.
        # B (4x9) = [B1 (4x4), B2 (4x5)]
        # A*B = A*B1 + A*B2
        # 
        # U for combined: U1 (16x49) for A*B1, U2 (16x80) for A*B2
        # V for combined: V1 (16x49) for B1, V2 (20x80) for B2
        # W for combined: W1 (16x49) for C1, W2 (20x80) for C2
        # 
        # But the full problem is 4x4*4x9, so:
        # U: 16 rows, V: 36 rows (4*9), W: 36 rows (9*4)
        # 
        # B1 is columns 0-3 of B, B2 is columns 4-8 of B.
        # V1 has 16 rows (for B1, which is 4x4), V2 has 20 rows (for B2, which is 4x5).
        # In the full V (36 rows), V1 maps to rows 0-15 (B[0:4, 0:4]), V2 maps to rows 16-35 (B[0:4, 4:9]).
        # 
        # W1 has 16 rows (for C1, which is 4x4), W2 has 20 rows (for C2, which is 4x5).
        # In the full W (36 rows), W1 maps to rows 0-15 (C[0:4, 0:4]), W2 maps to rows 16-35 (C[0:4, 4:9]).
        
        U1, V1, W1, r1 = strassen_4x4_4x4()
        U2, V2, W2, r2 = schoolbook_4x4_4x5()
        
        total_rank = r1 + r2  # 49 + 80 = 129
        U_full = [[0] * total_rank for _ in range(16)]
        V_full = [[0] * total_rank for _ in range(36)]
        W_full = [[0] * total_rank for _ in range(36)]
        
        # Copy U1, U2
        for i in range(16):
            for c in range(r1):
                U_full[i][c] = U1[i][c]
            for c in range(r2):
                U_full[i][r1 + c] = U2[i][c]
        
        # Copy V1, V2
        for i in range(16):
            for c in range(r1):
                V_full[i][c] = V1[i][c]
        for i in range(20):
            for c in range(r2):
                V_full[16 + i][r1 + c] = V2[i][c]
        
        # Copy W1, W2
        for i in range(16):
            for c in range(r1):
                W_full[i][c] = W1[i][c]
        for i in range(20):
            for c in range(r2):
                W_full[16 + i][r1 + c] = W2[i][c]
        
        # Verify
        if _verify(U_full, V_full, W_full, T, n, m, p):
            if total_rank < best_rank:
                best = {"U": U_full, "V": V_full, "W": W_full}
                best_rank = total_rank
        
        if time.time() > deadline - 10.0:
            return best
    
    # Try random search for even lower rank
    if time.time() < deadline - 15.0:
        result = _random_search(T, n, m, p, best_rank - 1, time_limit=15.0)
        if result is not None:
            best = result
    
    return best
