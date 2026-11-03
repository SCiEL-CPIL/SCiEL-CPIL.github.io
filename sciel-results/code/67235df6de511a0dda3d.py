"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses block-Strassen approach: partition into 2x2 blocks and apply Strassen where possible.
    """
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Strategy: Use the fact that 2x5*5x6 can be decomposed by splitting dimensions.
    # 2 = 1+1, 5 = 2+3, 6 = 3+3
    # We can use Strassen on the 2x2x2 sub-blocks.
    
    # Let's think about this more carefully.
    # A (2x5) times B (5x6) = C (2x6)
    # 
    # Split A into [A1 | A2] where A1 is 2x2, A2 is 2x3
    # Split B into [B1; B2] where B1 is 2x6, B2 is 3x6
    # C = A1*B1 + A2*B2
    #
    # For A1*B1: 2x2 * 2x6. Split B1 into [B1a | B1b] where each is 2x3.
    #   A1*B1a: 2x2 * 2x3 -> can use Strassen-like
    #   A1*B1b: 2x2 * 2x3 -> can use Strassen-like
    #
    # Actually, let me use a simpler approach based on known results.
    # 
    # The schoolbook rank is 2*5*6 = 60.
    # Using Strassen on 2x2x2 blocks: we have floor(2/2)*floor(5/2)*floor(6/2) = 1*2*3 = 6 blocks of 2x2x2.
    # Each 2x2x2 block uses 7 instead of 8, saving 6*1 = 6 ranks.
    # Remaining: 2x(5-4)*6 + 2*4*(6-6) + ... let me be more careful.
    #
    # Better approach: partition 2=2, 5=4+1, 6=6
    # Actually, let's use the standard block decomposition.
    #
    # Let me use a different strategy. The known minimum rank for 2x5*5x6...
    # Let me just implement a good block-Strassen decomposition.
    #
    # Partition: n=2, m=5, p=6
    # Split m into 4+1, p into 6
    # A = [A_2x4 | A_2x1], B = [B_4x6; B_1x6]
    # C = A_2x4 * B_4x6 + A_2x1 * B_1x6
    #
    # For A_2x4 * B_4x6: split into two 2x2*2x6 blocks
    #   A_2x4 = [Aa_2x2 | Ab_2x2], B_4x6 = [Ba_2x6; Bb_2x6]
    #   Aa_2x2 * Ba_2x6: use Strassen on the 2x2 part, treating the 6 as 3+3
    #     Ba_2x6 = [Baa_2x3 | Bab_2x3]
    #     Aa*Baa: 2x2*2x3, Aa*Bab: 2x2*2x3
    #     For 2x2*2x3: split the 3 into... hmm, not clean.
    #
    # Let me try a cleaner approach. Use the fact that we can do:
    # 2x5 * 5x6 by embedding in larger structures or using known decompositions.
    #
    # Actually, the simplest improvement: use Strassen on all possible 2x2x2 blocks.
    # n=2, m=5=2+2+1, p=6=2+2+2
    # Blocks: 1 (in n) x 3 (in m: two 2s and one 1) x 3 (in p: three 2s)
    # For 2x2x2 blocks: use Strassen (rank 7)
    # For 2x2x1 blocks: schoolbook (rank 2)
    # For 2x1x2 blocks: schoolbook (rank 2)
    # For 2x1x1 blocks: schoolbook (rank 1)
    
    # Let me just do a careful manual decomposition.
    
    # I'll use the following block structure:
    # n=2, m=5, p=6
    # Split m = 2 + 2 + 1, p = 2 + 2 + 2
    # 
    # For each (i, j_block, k_block):
    #   If j_block size=2 and k_block size=2: Strassen (7 terms)
    #   Otherwise: schoolbook (size_j * size_k terms)
    
    # Number of 2x2x2 blocks: 1 * 2 * 3 = 6, each rank 7 -> 42
    # Number of 2x2x1 blocks: 1 * 1 * 3 = 3, each rank 2 -> 6
    # Total: 42 + 6 = 48? Wait, m=2+2+1, so there are 2 blocks of size 2 and 1 block of size 1.
    # p=2+2+2, so there are 3 blocks of size 2.
    # 2x2x2 blocks: 2*3 = 6, rank 7 each -> 42
    # 2x1x2 blocks: 1*3 = 3, rank 2 each -> 6
    # Total: 48
    
    # But wait, can we do better? Let me check if there's a known result.
    # The rank of 2x5*5x6 matrix multiplication...
    # 
    # Actually, let me just implement the 48-rank decomposition and verify it works.
    
    rank = 48
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    # m splits: [2, 2, 1]
    # p splits: [2, 2, 2]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    m_offsets = [0, 2, 4]
    p_offsets = [0, 2, 4]
    
    for jb in range(3):
        for kb in range(3):
            mj = m_splits[jb]
            pk = p_splits[kb]
            mo = m_offsets[jb]
            po = p_offsets[kb]
            
            if mj == 2 and pk == 2:
                # Strassen 2x2x2
                # A block: rows 0,1 (n=2), cols mo, mo+1 (m)
                # B block: rows mo, mo+1 (m), cols po, po+1 (p)
                # C block: rows po, po+1 (p), cols 0,1 (n)
                #
                # Strassen:
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
                
                # M1
                U[0 * m + mo + 0, col] = 1  # A00
                U[1 * m + mo + 1, col] = 1  # A11
                V[mo + 0, po + 0] = 1  # B00
                V[mo + 1, po + 1] = 1  # B11
                W[po + 0, 0] = 1  # C00
                W[po + 1, 1] = 1  # C11
                col += 1
                
                # M2
                U[1 * m + mo + 0, col] = 1  # A10
                U[1 * m + mo + 1, col] = 1  # A11
                V[mo + 0, po + 0] = 1  # B00
                W[po + 1, 0] = 1  # C10
                col += 1
                
                # M3
                U[0 * m + mo + 0, col] = 1  # A00
                V[mo + 1, po + 0] = 1  # B10
                V[mo + 1, po + 1] = -1  # B11
                W[po + 0, 1] = 1  # C01
                W[po + 1, 1] = 1  # C11
                col += 1
                
                # M4
                U[1 * m + mo + 1, col] = 1  # A11
                V[mo + 0, po + 1] = 1  # B01
                V[mo + 0, po + 0] = -1  # B10
                W[po + 0, 0] = 1  # C00
                W[po + 1, 0] = 1  # C10
                col += 1
                
                # M5
                U[0 * m + mo + 0, col] = 1  # A00
                U[0 * m + mo + 1, col] = 1  # A01
                V[mo + 1, po + 1] = 1  # B11
                W[po + 0, 0] = -1  # C00
                W[po + 0, 1] = 1  # C01
                col += 1
                
                # M6
                U[1 * m + mo + 0, col] = 1  # A10
                U[0 * m + mo + 0, col] = -1  # A00
                V[mo + 0, po + 0] = 1  # B00
                V[mo + 0, po + 1] = 1  # B10
                W[po + 1, 1] = 1  # C11
                col += 1
                
                # M7
                U[0 * m + mo + 1, col] = 1  # A01
                U[1 * m + mo + 1, col] = -1  # A11
                V[mo + 0, po + 1] = 1  # B01
                V[mo + 1, po + 1] = 1  # B11
                W[po + 0, 0] = 1  # C00
                col += 1
                
            elif mj == 2 and pk == 1:
                # Schoolbook 2x2x1: 2 terms
                for a in range(2):
                    for b in range(2):
                        U[a * m + mo + b, col] = 1
                        V[mo + b, po + 0] = 1
                        W[po + 0, a] = 1
                        col += 1
                        
            elif mj == 1 and pk == 2:
                # Schoolbook 2x1x2: 2 terms
                for a in range(2):
                    for b in range(2):
                        U[a * m + mo + 0, col] = 1
                        V[mo + 0, po + b] = 1
                        W[po + b, a] = 1
                        col += 1
                        
            else:
                # 2x1x1: 1 term
                U[0 * m + mo + 0, col] = 1
                U[1 * m + mo + 0, col] = 1
                V[mo + 0, po + 0] = 1
                W[po + 0, 0] = 1
                W[po + 0, 1] = 1
                col += 1
    
    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Convert to list of lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
