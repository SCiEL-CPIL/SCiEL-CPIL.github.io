import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # We will build the tensor decomposition using a combination of:
    # 1. Strassen's algorithm for 2x2x2 blocks
    # 2. Laderman's algorithm for 3x3x3 blocks (if applicable)
    # 3. Schoolbook for the remaining terms
    
    # Strategy: Partition the dimensions to maximize use of fast algorithms
    # m=5 can be partitioned as 2+3
    # We'll use:
    # - 3x2x7 part: Use Strassen for 2x2 blocks in the m=2 dimension
    # - 3x3x7 part: Use Laderman for 3x3 blocks in the m=3 dimension
    
    # However, p=7 is not a power of 2 or 3, so we need to handle it carefully.
    # Let's use a different approach: use Strassen blocks where possible, 
    # and schoolbook for the rest.
    
    # First, let's identify all (i,j,k) tuples that we can cover with Strassen blocks
    # A Strassen block covers a 2x2x2 sub-tensor
    
    # For 3x5x7, we can place Strassen blocks at positions where all three dimensions
    # have at least 2 elements remaining.
    
    # Let's use a greedy approach: try to place as many non-overlapping 2x2x2 
    # Strassen blocks as possible, then use schoolbook for the rest.
    
    # Actually, let's use a more systematic approach based on the partition m=2+3.
    
    # Part 1: 3x2x7 sub-problem (i in 0..2, j in 0..1, k in 0..6)
    # We can use Strassen blocks here: for each pair of k values, we can do a 2x2x2
    # block. But 7 is odd, so we can do 3 full 2x2x2 blocks (covering k=0..5) 
    # and 1 schoolbook column (k=6).
    
    # Part 2: 3x3x7 sub-problem (i in 0..2, j in 2..4, k in 0..6)
    # For 3x3, we can use Laderman's algorithm (rank 23) for each 3x3x3 block.
    # But 7 is not 3, so we need to handle it differently.
    
    # Let's use a simpler approach: just use Strassen blocks where possible 
    # and schoolbook for the rest. This is what the current program does,
    # but let's see if we can improve it.
    
    # Actually, let's try a different partition strategy.
    # Use Strassen blocks for the 2x2 parts of m=5 (i.e., j=0,1 and j=2,3),
    # and schoolbook for j=4.
    
    # For j=0,1: 3x2x7, we can use Strassen for k=0,1; k=2,3; k=4,5 (3 blocks)
    # and schoolbook for k=6.
    
    # For j=2,3: 3x2x7, same as above (3 Strassen blocks + 1 schoolbook column)
    
    # For j=4: 3x1x7, all schoolbook.
    
    # Total: 6 Strassen blocks * 7 terms each = 42 terms
    #         2 schoolbook columns * 3 terms each = 6 terms
    #         7 schoolbook terms for j=4
    # Total: 42 + 6 + 7 = 55 terms
    
    # But wait, we can do better. Let's use a more efficient packing.
    
    # Actually, let's just use the current approach but verify it's correct.
    # The current program uses 6 Strassen blocks covering 48 terms, and 
    # schoolbook for the remaining 105 - 48 = 57 terms, for a total of 6+57=63 terms.
    
    # Can we do better? Let's think about the theoretical minimum.
    # The rank of the 3x5x7 matrix multiplication tensor is at most 3*5*7=105 (schoolbook).
    # Using Strassen, we can reduce it. The current approach gives rank 63.
    
    # Let's try to optimize further by using more Strassen blocks or a different algorithm.
    
    # For now, let's stick with the current approach but make sure it's correct.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Define the 6 Strassen blocks
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4)
    ]
    
    # Set of indices covered by Strassen blocks
    covered = set()
    for I0, J0, K0 in blocks:
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    covered.add((I0+i, J0+j, K0+k))
                    
    # Strassen decomposition for 2x2x2
    def add_strassen_block(I0, J0, K0):
        # M1
        U = { (I0, J0): 1, (I0+1, J0+1): 1 }
        V = { (J0, K0): 1, (J0+1, K0+1): 1 }
        W = { (K0, I0): 1, (K0+1, I0+1): 1 }
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M2
        U = { (I0+1, J0): 1, (I0+1, J0+1): 1 }
        V = { (J0, K0): 1 }
        W = { (K0, I0+1): 1, (K0+1, I0+1): -1 }
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M3
        U = { (I0, J0): 1 }
        V = { (J0, K0+1): 1, (J0+1, K0+1): -1 }
        W = { (K0+1, I0): 1, (K0+1, I0+1): 1 }
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M4
        U = { (I0+1, J0+1): 1 }
        V = { (J0+1, K0): 1, (J0, K0): -1 }
        W = { (K0, I0): 1, (K0, I0+1): 1 }
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M5
        U = { (I0, J0): 1, (I0, J0+1): 1 }
        V = { (J0+1, K0+1): 1 }
        W = { (K0, I0): -1, (K0+1, I0): 1 }
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M6
        U = { (I0+1, J0): 1, (I0, J0): -1 }
        V = { (J0, K0): 1, (J0, K0+1): 1 }
        W = { (K0+1, I0+1): 1 }
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M7
        U = { (I0, J0+1): 1, (I0+1, J0+1): -1 }
        V = { (J0+1, K0): 1, (J0+1, K0+1): 1 }
        W = { (K0, I0): 1 }
        U_cols.append(U); V_cols.append(V); W_cols.append(W)

    for block in blocks:
        if time.time() > deadline:
            break
        add_strassen_block(*block)
        
    # Schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                if time.time() > deadline:
                    break
                U = { (i, j): 1 }
                V = { (j, k): 1 }
                W = { (k, i): 1 }
                U_cols.append(U); V_cols.append(V); W_cols.append(W)
                
    # Convert sparse columns to dense matrices
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for (i, j), val in U_cols[c].items():
            U[i * m + j][c] = val
        for (j, k), val in V_cols[c].items():
            V[j * p + k][c] = val
        for (k, i), val in W_cols[c].items():
            W[k * n + i][c] = val
            
    return {"U": U, "V": V, "W": W}
