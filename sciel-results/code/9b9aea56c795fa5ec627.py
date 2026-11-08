"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_6x6_block_strassen():
    """
    Construct a rank-189 decomposition for 6x6 matrix multiplication
    by using a 3x3 block structure with each 2x2 block multiplication
    done via Strassen's 7-multiplication algorithm.
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0.0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0.0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0.0] * rank for _ in range(p * n)]  # 36 x 189
    
    col = 0
    for i in range(3):  # row block of A / row block of C
        for k in range(3):  # col block of A / row block of B
            for j in range(3):  # col block of B / col block of C
                # Strassen products for A_ik * B_kj
                # A_ik has elements at flat indices:
                #   (a,b) -> (2i+a)*6 + (2k+b) = 12i + 6a + 2k + b
                # B_kj has elements at flat indices:
                #   (a,b) -> (2k+a)*6 + (2j+b) = 12k + 6a + 2j + b
                # C_ij has elements at flat indices in W (column-major):
                #   C element (a,b) of block ij -> row a, col b in C block -> 
                #   flat C row = 2i+a, flat C col = 2j+b -> W index = (2j+b)*6 + (2i+a)
                
                aA = lambda a, b: 12*i + 6*a + 2*k + b  # flat index in A
                bB = lambda a, b: 12*k + 6*a + 2*j + b  # flat index in B
                cW = lambda a, b: (2*j + b)*6 + (2*i + a)  # flat index in W (col-major)
                
                # P0 = (A00+A11)(B00+B11)
                U[aA(0,0)][col] += 1
                U[aA(1,1)][col] += 1
                V[bB(0,0)][col] += 1
                V[bB(1,1)][col] += 1
                # C00 += P0
                W[cW(0,0)][col] += 1
                col += 1
                
                # P1 = (A10+A11)*B00
                U[aA(1,0)][col] += 1
                U[aA(1,1)][col] += 1
                V[bB(0,0)][col] += 1
                # C10 += P1
                W[cW(1,0)][col] += 1
                col += 1
                
                # P2 = A00*(B10-B11)
                U[aA(0,0)][col] += 1
                V[bB(1,0)][col] += 1
                V[bB(1,1)][col] += -1
                # C01 += P2
                W[cW(0,1)][col] += 1
                col += 1
                
                # P3 = A11*(B01-B00)
                U[aA(1,1)][col] += 1
                V[bB(0,1)][col] += 1
                V[bB(0,0)][col] += -1
                # C00 += P3
                W[cW(0,0)][col] += 1
                # C10 += P3
                W[cW(1,0)][col] += 1
                col += 1
                
                # P4 = (A00+A01)*B11
                U[aA(0,0)][col] += 1
                U[aA(0,1)][col] += 1
                V[bB(1,1)][col] += 1
                # C00 += -P4
                W[cW(0,0)][col] += -1
                # C01 += P4
                W[cW(0,1)][col] += 1
                col += 1
                
                # P5 = (A10-A00)*(B00+B10)
                U[aA(1,0)][col] += 1
                U[aA(0,0)][col] += -1
                V[bB(0,0)][col] += 1
                V[bB(1,0)][col] += 1
                # C11 += P5
                W[cW(1,1)][col] += 1
                col += 1
                
                # P6 = (A01-A11)*(B01+B11)
                U[aA(0,1)][col] += 1
                U[aA(1,1)][col] += -1
                V[bB(0,1)][col] += 1
                V[bB(1,1)][col] += 1
                # C00 += P6
                W[cW(0,0)][col] += 1
                col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}

def construct():
    t0 = time.time()
    
    # Build the block-Strassen decomposition
    result = _build_6x6_block_strassen()
    
    # Safety margin - if we're running out of time, return what we have
    if time.time() - t0 > 110:
        return result
    
    return result

# EVOLVE-BLOCK-END
