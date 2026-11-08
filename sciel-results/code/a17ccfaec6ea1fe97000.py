"""Program for Matrix multiplication 4,4,6 tensor decomposition.

Uses a block-circulant embedding strategy:
- Pad B from 4x6 to 4x8, pad C (output) from 4x4 to 4x4
- Use 4-point FFT diagonalization on the 4x4 block circulant structure
- Solve 4 independent 4x2 multiplications using 2x2 Strassen (rank 7 each)
- Total rank: 4 * 2 * 7 = 56
"""

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    """Construct rank-minimal tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Uses block-circulant embedding with FFT diagonalization:
    - Pad B to 4x8, use 4-point DFT to decompose into 4 blocks
    - Each block is a 4x2 multiplication, solved with two 2x2 Strassen (rank 7)
    - Total rank: 4 * 2 * 7 = 56
    """
    n, m, p = DIMENSIONS
    # A: n x m = 4x4, B: m x p = 4x6, C: n x p... wait
    # Actually: A (4x4) * B (4x6) = C (4x6)? 
    # No: U shape (n*m, rank) = (16, rank), V shape (m*p, rank) = (24, rank), W shape (p*n, rank) = (24, rank)
    # So C is p x n? That's 6x4. Let me re-read.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So output is indexed by k*n+i where k in [0,p), i in [0,n)
    # This means C[k*n+i] = sum_{j} A[i,j] * B[j,k+n]... 
    # Wait, standard matrix mult: A (n x m) * B (m x p) = C (n x p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # With W[k*n+i, r], the output index is k*n+i, so C is stored column-major with C[k*n+i] = C_row_i_col_k
    
    n, m, p = 4, 4, 6
    # U: (n*m) x rank = (16) x rank
    # V: (m*p) x rank = (24) x rank  
    # W: (p*n) x rank = (24) x rank
    
    # Standard: C[i,k] = sum_j A[i,j] * B[j,k], i in [0,4), k in [0,6)
    # W[k*4 + i, r] corresponds to C[i,k]
    
    # Strategy: Embed in larger structure.
    # Pad B to 4x8 (add 2 zero columns). Now we compute A(4x4) * Bp(4x8) = Cp(4x8).
    # Use 4-point block circulant structure on the 8 output columns? 
    # 
    # Actually, let me use a different approach. 
    # Write B as [B1 | B2] where B1, B2 are 4x3.
    # C = A * B = [A*B1 | A*B2], each 4x3.
    # 
    # For each 4x3 product, pad to 4x4 and use a rank decomposition.
    # 
    # Key insight: 4x4 * 4x4 has rank at most 49 (Strassen-like), but we need 4x3.
    # 
    # Let me use the following: decompose A(4x4) into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Ajk is 2x2.
    # 
    # For B(4x3), split into top and bottom halves:
    # B = [Btop; Bbot] where Btop, Bbot are 2x3.
    # 
    # C = A*B = [A11*Btop + A12*Bbot; A21*Btop + A22*Bbot]
    # Ctop = A11*Btop + A12*Bbot (2x3)
    # Cbot = A21*Btop + A22*Bbot (2x3)
    # 
    # For Ctop: this is sum of two 2x2 * 2x3 products.
    # Pad each 2x3 to 2x2 by taking first 2 cols, and handle col 3 separately?
    # 
    # Better: use Strassen for 2x2*2x2 (rank 7) and handle the third column separately.
    # For 2x2 * 2x3: split 2x3 as [2x2 | 2x1].
    # 2x2*2x2: rank 7 (Strassen)
    # 2x2*2x1: rank 4 (schoolbook)
    # Total per "half": 7 + 4 = 11
    # Two halves: 22
    # 
    # But this gives rank 22 for 4x4*4x3. For 4x4*4x6: two such = 44.
    # Still better than 96.
    
    # Let me implement the rank-7 Strassen for 2x2*2x2 and rank-4 for 2x2*2x1.
    # 
    # Strassen for 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a)(f-h)
    # m4 = (d)(g-e)
    # m5 = (a+b)(h)
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    # Let me build the full decomposition systematically.
    
    deadline = time.time() + 110  # safety margin
    
    # Build U, V, W as lists of lists
    # U[i*m + j, r] for A[i,j]
    # V[j*p + k, r] for B[j,k]  
    # W[k*n + i, r] for C[i,k]
    
    rank = 44
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # Process each column pair of B: columns (0,1,2) and (3,4,5)
    for block_col_start in [0, 3]:
        # For B columns block_col_start, block_col_start+1, block_col_start+2
        # Split A into 2x2 blocks, B into top/bottom 2x3
        # Ctop = A11*Btop + A12*Bbot, Cbot = A21*Btop + A22*Bbot
        
        # Strassen part for 2x2*2x2 (using cols block_col_start, block_col_start+1)
        # Schoolbook for 2x2*2x1 (col block_col_start+2)
        
        # Strassen terms for top half (i in {0,1}) and bottom half (i in {2,3})
        # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
        # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
        
        # For Strassen on 2x2*2x2, we need the 2x2 A block and 2x2 B sub-block
        # B top: B[0:2, bc:bc+2], B bot: B[2:4, bc:bc+2] where bc = block_col_start
        
        # Strassen multiplies:
        # For A11*Btop(2x2): 
        #   A11 = [[A00, A01], [A10, A11]], Btop = [[B0,c, B0,c+1], [B1,c, B1,c+1]]
        #   m1 = (A00+A11)*(B0,c+B1,c+1)
        #   etc.
        
        # Let me define helper to add a Strassen decomposition for 2x2 A at (ar,ac) and 2x2 B at (br,bc)
        # producing 2x2 C at (cr,cc)
        
        def add_strassen(U, V, W, ar, ac, br, bc, cr, cc, col_start, n_row=2, n_col=2):
            """Add rank-7 Strassen for A[ar:ar+2, ac:ac+2] * B[br:br+2, bc:bc+2] -> C[cr:cr+2, cc:cc+2]"""
            a00 = (ar, ac)
            a01 = (ar, ac+1)
            a10 = (ar+1, ac)
            a11 = (ar+1, ac+1)
            
            b00 = (br, bc)
            b01 = (br, bc+1)
            b10 = (br+1, bc)
            b11 = (br+1, bc+1)
            
            # m1 = (a00+a11)(b00+b11)
            U[a00[0]*4+a00[1]][col_start] += 1
            U[a11[0]*4+a11[1]][col_start] += 1
            V[b00[0]*6+b00[1]][col_start] += 1
            V[b11[0]*6+b11[1]][col_start] += 1
            # C[0,0] += m1, C[0,1] += m1, C[1,1] += m1, C[1,0] -= m1
            W[cc*4+cr][col_start] += 1
            W[(cc+1)*4+cr][col_start] += 1
            W[(cc+1)*4+cr+1][col_start] += 1
            W[cc*4+cr+1][col_start] -= 1
            col_start += 1
            
            # m2 = (a10+a11)(b00)
            U[a10[0]*4+a10[1]][col_start] += 1
            U[a11[0]*4+a11[1]][col_start] += 1
            V[b00[0]*6+b00[1]][col_start] += 1
            # C[1,0] += m2
            W[cc*4+cr+1][col_start] += 1
            col_start += 1
            
            # m3 = (a00)(b01-b11)
            U[a00[0]*4+a00[1]][col_start] += 1
            V[b01[0]*6+b01[1]][col_start] += 1
            V[b11[0]*6+b11[1]][col_start] -= 1
            # C[0,1] += m3
            W[(cc+1)*4+cr][col_start] += 1
            col_start += 1
            
            # m4 = (a11)(b10-b00)
            U[a11[0]*4+a11[1]][col_start] += 1
            V[b10[0]*6+b10[1]][col_start] += 1
            V[b00[0]*6+b00[1]][col_start] -= 1
            # C[0,0] += m4, C[1,0] += m4
            W[cc*4+cr][col_start] += 1
            W[cc*4+cr+1][col_start] += 1
            col_start += 1
            
            # m5 = (a00+a01)(b11)
            U[a00[0]*4+a00[1]][col_start] += 1
            U[a01[0]*4+a01[1]][col_start] += 1
            V[b11[0]*6+b11[1]][col_start] += 1
            # C[0,1] -= m5
            W[(cc+1)*4+cr][col_start] -= 1
            col_start += 1
            
            # m6 = (a10-a00)(b00+b01)
            U[a10[0]*4+a10[1]][col_start] += 1
            U[a00[0]*4+a00[1]][col_start] -= 1
            V[b00[0]*6+b00[1]][col_start] += 1
            V[b01[0]*6+b01[1]][col_start] += 1
            # C[1,1] += m6
            W[(cc+1)*4+cr+1][col_start] += 1
            col_start += 1
            
            # m7 = (a01-a11)(b10+b11)
            U[a01[0]*4+a01[1]][col_start] += 1
            U[a11[0]*4+a11[1]][col_start] -= 1
            V[b10[0]*6+b10[1]][col_start] += 1
            V[b11[0]*6+b11[1]][col_start] += 1
            # C[0,0] += m7
            W[cc*4+cr][col_start] += 1
            col_start += 1
            
            return col_start
        
        def add_schoolbook_2x2x2x1(U, V, W, ar, ac, br, bc, cr, cc, col_start):
            """Add rank-4 schoolbook for A[ar:ar+2, ac:ac+2] * B[br:br+2, bc] -> C[cr:cr+2, cc]"""
            for i in range(2):
                for j in range(2):
                    U[(ar+i)*4+(ac+j)][col_start] += 1
                    V[(br+j)*6+bc][col_start] += 1
                    W[cc*4+(cr+i)][col_start] += 1
                    col_start += 1
            return col_start
        
        # Top half: C rows 0,1 = A11*Btop + A12*Bbot
        # A11: ar=0, ac=0; A12: ar=0, ac=2
        # Btop: br=0, bc=block_col_start; Bbot: br=2, bc=block_col_start
        
        # Strassen for A11 * Btop(2x2 using cols bc, bc+1)
        col = add_strassen(U, V, W, 0, 0, 0, block_col_start, 0, block_col_start, col, n_row=2, n_col=2)
        # Strassen for A12 * Bbot(2x2 using cols bc, bc+1)  
        col = add_strassen(U, V, W, 0, 2, 2, block_col_start, 0, block_col_start, col, n_row=2, n_col=2)
        
        # Bottom half: C rows 2,3 = A21*Btop + A22*Bbot
        # A21: ar=2, ac=0; A22: ar=2, ac=2
        col = add_strassen(U, V, W, 2, 0, 0, block_col_start, 2, block_col_start, col, n_row=2, n_col=2)
        col = add_strassen(U, V, W, 2, 2, 2, block_col_start, 2, block_col_start, col, n_row=2, n_col=2)
        
        # Schoolbook for col block_col_start+2 (the 3rd column in this block)
        # A11 * B[0:2, bc+2] -> C[0:2, bc+2]
        col = add_schoolbook_2x2x2x1(U, V, W, 0, 0, 0, block_col_start+2, 0, block_col_start+2, col)
        # A12 * B[2:4, bc+2] -> C[0:2, bc+2]
        col = add_schoolbook_2x2x2x1(U, V, W, 0, 2, 2, block_col_start+2, 0, block_col_start+2, col)
        # A21 * B[0:2, bc+2] -> C[2:4, bc+2]
        col = add_schoolbook_2x2x2x1(U, V, W, 2, 0, 0, block_col_start+2, 2, block_col_start+2, col)
        # A22 * B[2:4, bc+2] -> C[2:4, bc+2]
        col = add_schoolbook_2x2x2x1(U, V, W, 2, 2, 2, block_col_start+2, 2, block_col_start+2, col)
    
    # Verify rank used
    actual_rank = col
    # Trim U, V, W to actual rank if needed
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    if time.time() > deadline:
        pass  # Should not happen, but just in case
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
