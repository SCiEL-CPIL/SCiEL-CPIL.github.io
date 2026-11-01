"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 9
    
    # Rank 34 construction for 2x2xp
    # This is the known minimal rank for p >= 2
    rank = 4 * p - 2
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: j*p+k
    # W indices: k*n+i
    
    # We use the following strategy:
    # For k=0:
    #   4 terms: a*B00->C00, b*B10->C00, c*B00->C10, d*B10->C10
    # For k=1..p-1:
    #   4 terms? No, we need to save 2 terms total.
    #   
    # Actually, the construction is:
    # 1. a*B[0,k] -> C[0,k] for all k (p terms)
    # 2. c*B[0,k] -> C[1,k] for all k (p terms)
    # 3. b*B[1,k] -> C[0,k] for k=0..p-2 (p-1 terms)
    # 4. d*B[1,k] -> C[1,k] for k=0..p-2 (p-1 terms)
    # 5. (b+d)*B[1,p-1] -> C[0,p-1] + C[1,p-1] (1 term)
    # 6. (d-b)*B[1,p-1] -> C[1,p-1] - C[0,p-1] (1 term)
    #    Wait, this requires W to have two entries.
    #    Term 5: U_b=1, U_d=1? No, U must be rank 1.
    #    
    # Let's use the standard 4p-2 construction:
    # Terms 1..p: a * B[0,k] -> C[0,k]
    # Terms p+1..2p: c * B[0,k] -> C[1,k]
    # Terms 2p+1..3p-1: b * B[1,k] -> C[0,k] for k=0..p-2
    # Terms 3p..4p-2: d * B[1,k] -> C[1,k] for k=0..p-2
    # Term 4p-1: (b+d) * B[1,p-1] -> C[0,p-1] + C[1,p-1] ? No, U must be single.
    
    # Correct construction:
    # 1. a * B[0,k] -> C[0,k] (p terms)
    # 2. c * B[0,k] -> C[1,k] (p terms)
    # 3. b * B[1,k] -> C[0,k] (p-1 terms, k=0..p-2)
    # 4. d * B[1,k] -> C[1,k] (p-1 terms, k=0..p-2)
    # 5. (b+d) * B[1,p-1] -> C[0,p-1] (1 term, but U has b and d? No)
    
    # This is hard to construct correctly without a reference.
    # I will use the schoolbook rank 36 to be safe, as 34 is risky.
    # But 36 is the current score. I need to improve.
    # I'll try to implement 34 with the "shared last term" trick.
    
    # Let's try:
    # For k=0..p-1:
    #   Term 2k: a*B[0,k] -> C[0,k]
    #   Term 2k+1: c*B[0,k] -> C[1,k]
    # For k=0..p-2:
    #   Term 2p+k: b*B[1,k] -> C[0,k]
    #   Term 2p+k+1: d*B[1,k] -> C[1,k]
    # Last two terms for k=p-1:
    #   Term 4p-3: (b+d)*B[1,p-1] -> C[0,p-1] + C[1,p-1] (Invalid U)
    
    # I will fallback to 36.
    rank = 36
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((18, rank), dtype=float)
    W = np.zeros((18, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
