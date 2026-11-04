"""Rewritten program for 4x5x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    # Target tensor: for each (i,j,k) in [n]x[m]x[p], we need
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1
    
    def build_decomposition():
        """Build a rank decomposition using block structure.
        
        Partition: n=4, m=5, p=6
        Strategy: Use Strassen-like decompositions for sub-blocks.
        
        Partition A (4x5) into:
          - Top-left: 2x2 (rows 0-1, cols 0-1)
          - Top-right: 2x3 (rows 0-1, cols 2-4)
          - Bottom-left: 2x2 (rows 2-3, cols 0-1)
          - Bottom-right: 2x3 (rows 2-3, cols 2-4)
        
        Partition B (5x6) into:
          - Top-left: 2x2 (rows 0-1, cols 0-1)
          - Top-right: 2x4 (rows 0-1, cols 2-5)
          - Bottom-left: 3x2 (rows 2-4, cols 0-1)
          - Bottom-right: 3x4 (rows 2-4, cols 2-5)
        
        This gets complex. Let me use a simpler approach:
        Direct construction with rank = n*m*p = 120 as baseline,
        then try to reduce using known fast multiplication results.
        
        For 4x5x6, we can use:
        - 4x5 = 4x4 + 4x1, 5x6 = 4x6 + 1x6
        - Or use the fact that 4x4x4 has rank ~49 (Lader), 
          4x4x5, 4x5x4, 5x4x4, 5x4x5, 5x5x4, 4x5x5, 5x4x5, 5x5x4, 5x5x5
        """
        # Start with schoolbook
        rank = n * m * p  # 120
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        
        return U, V, W, rank
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    # Try to build a better decomposition using block structure
    # Partition: n=2+2, m=3+2, p=3+3
    # Blocks:
    #   A_ij * B_jk -> C_ki
    #   (0,0,0): 2x3x3, (0,0,1): 2x3x3
    #   (0,1,0): 2x2x3, (0,1,1): 2x2x3
    #   (1,0,0): 2x3x3, (1,0,1): 2x3x3
    #   (1,1,0): 2x2x3, (1,1,1): 2x2x3
    
    def strassen_2x2x2():
        """Standard Strassen 2x2x2 decomposition, rank 7."""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # M1=(A11+A22)(B11+B22): C11+=M1, C22+=M1
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        # M2=(A21+A22)B21: C21+=M2
        U[2][1]=1; U[3][1]=1; V[2][1]=1; W[2][1]=1
        # M3=A11(B12-B22): C12+=M3
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1
        # M4=A22(B21-B11): C11+=M4, C21+=M4
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        # M5=(A11+A12)B22: C11-=M5, C12+=M5
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        # M6=(A21-A11)(B11+B12): C22+=M6
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        # M7=(A12-A22)(B21+B22): C11+=M7, C22+=M7
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[3][6]=1
        return U, V, W
    
    def build_block_decomp():
        """Build decomposition using 2x3x3 and 2x2x3 blocks with Strassen.
        
        2x3x3 block: A(2x3) * B(3x3) -> C(3x2)
        Split B = [B1(3x2) | B2(3x1)]
        C = A*B1 + A*B2
        A*B1: 2x3 * 3x2 = 2x2, use Strassen-like (rank 7 for 2x2x2 part + extras)
        Actually, let's compute directly:
        (A*B1)[i,k] = sum_j A[i,j]*B1[j,k] for i in [0,2), k in [0,2)
        This is 2x2x2 tensor, rank 7 via Strassen
        (A*B2)[i,0] = sum_j A[i,j]*B2[j,0] for i in [0,2)
        This is 2x1x1, rank 2
        
        Total for 2x3x3: 7 + 2 = 9? No, that's not right.
        Let me think more carefully.
        
        For 2x3x3: A is 2x3, B is 3x3, C is 3x2
        C[k,i] = sum_j A[i,j] * B[j,k]
        
        Split j into j=0,1,2:
        C[k,i] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
        
        For fixed i, this is a 1x3 * 3x2 = 1x2 multiplication, rank 2
        For 2 values of i, total rank = 2*2 = 4 if independent
        But we can do better by grouping.
        
        Actually, the tensor T_{i,j,k} = 1 if we're computing C[k,i] from A[i,j]*B[j,k]
        The rank of the 2x3x3 tensor... let me just use schoolbook: 2*3*3=18
        Or use: for each j (3 values), we have a 2x1x3 * 1x3x2 structure
        T = sum_j U_j ⊗ V_j ⊗ W_j where U_j[i] = delta(i,j) for i in [0,2), j in [0,3)
        Wait, this is getting complicated. Let me just use direct construction.
        """
        total_rank = 0
        U_blocks = []
        V_blocks = []
        W_blocks = []
        
        # Define block boundaries
        # n: [0,2) and [2,4)
        # m: [0,3) and [3,5)
        # p: [0,3) and [3,6)
        
        n_splits = [0, 2, 4]
        m_splits = [0, 3, 5]
        p_splits = [0, 3, 6]
        
        for ni in range(2):
            for mj in range(2):
                for pk in range(2):
                    ni_start, ni_end = n_splits[ni], n_splits[ni+1]
                    mj_start, mj_end = m_splits[mj], m_splits[mj+1]
                    pk_start, pk_end = p_splits[pk], p_splits[pk+1]
                    
                    ni_size = ni_end - ni_start  # 2
                    mj_size = mj_end - mj_start  # 3 or 2
                    pk_size = pk_end - pk_start  # 3
                    
                    # A block: ni_size x mj_size
                    # B block: mj_size x pk_size
                    # C block: pk_size x ni_size
                    
                    # Direct schoolbook for this block
                    for i in range(ni_size):
                        for j in range(mj_size):
                            for k in range(pk_size):
                                a = (ni_start + i) * m + (mj_start + j)
                                b = (mj_start + j) * p + (pk_start + k)
                                c = (pk_start + k) * n + (ni_start + i)
                                U_blocks.append((a, 1))
                                V_blocks.append((b, 1))
                                W_blocks.append((c, 1))
                                total_rank += 1
        
        # This gives schoolbook rank = 120, not helpful.
        # Let me try a different approach: use known fast algorithms.
        
        # For 4x4x4, the best known rank is 49 (Lader et al.)
        # For 4x4x5, 4x5x4, 5x4x4, 5x4x5, 5x5x4, 4x5x5, 5x4x5, 5x5x5
        # I don't have explicit formulas memorized, so let me construct one.
        
        # Alternative: Use the fact that 4x5x6 can be decomposed as:
        # 4x5x6 = 4x4x6 + 4x1x6
        # 4x4x6 = 4x4x4 + 4x4x2
        # 4x4x4: rank 49
        # 4x4x2: rank ?
        # 4x1x6: rank 6
        # Total: 49 + ? + 6
        
        # This is getting too complex without explicit formulas.
        # Let me just return the schoolbook decomposition for now,
        # and try to optimize with a simple local search.
        
        return None
    
    # Build schoolbook
    U, V, W, rank = build_decomposition()
    
    # Verify
    if not verify(U, V, W, rank):
        raise ValueError("Schoolbook decomposition failed verification")
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to improve with a simple approach: 
    # Check if we can merge any pairs of terms
    # For a tensor term (a,b,c) with weight w, we need U[a,r]*V[b,r]*W[c,r] to contribute
    # If two terms have the same (a,b,c) pattern, they can be merged
    
    # For now, return the schoolbook result
    return best_result
# EVOLVE-BLOCK-END
