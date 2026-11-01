import time
import sys

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 33 decomposition using 3 blocks of rank 11 (2x2x3)
    # Each block handles 3 columns of B and the corresponding 3 columns of C.
    
    # Strassen terms for 2x2x2
    # u: coeffs for A (2x2 -> 4)
    # v: coeffs for B (2x2 -> 4)
    # w: coeffs for C (2x2 -> 4, index k*n+i)
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, -1, 1]), # C00=1, C10=0, C01=-1, C11=1 ? 
        # Wait, w index: 0(C00), 1(C10), 2(C01), 3(C11)
        # M1: C00+=1, C11+=1. w=[1,0,0,1]
        # My previous derivation:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's re-verify w vectors based on this.
        # M1: C00+=1, C11+=1. w=[1, 0, 0, 1]
        # M2: C10+=1, C11-=1. w=[0, 1, 0, -1]
        # M3: C01+=1, C11+=1. w=[0, 0, 1, 1]
        # M4: C00+=1, C10+=1. w=[1, 1, 0, 0]
        # M5: C00-=1, C01+=1. w=[-1, 0, 1, 0]
        # M6: C11+=1. w=[0, 0, 0, 1]
        # M7: C00+=1. w=[1, 0, 0, 0]
        
        # u, v vectors:
        # M1: u=[1,0,0,1], v=[1,0,0,1]
        # M2: u=[0,0,1,1], v=[1,0,0,0]
        # M3: u=[1,0,0,0], v=[0,1,0,-1]
        # M4: u=[0,0,0,1], v=[0,0,1,-1]
        # M5: u=[1,1,0,0], v=[0,0,0,1]
        # M6: u=[0,0,1,-1], v=[1,1,0,0]
        # M7: u=[0,1,0,-1], v=[0,0,1,1]
    ]
    
    # Corrected list:
    terms_2x2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Schoolbook terms for 2x2x1 (column k=2 in local block)
    # C02 = a00*b02 + a01*b12
    # C12 = a10*b02 + a11*b12
    # w index for C02 (k=2, i=0) is 2*2+0 = 4
    # w index for C12 (k=2, i=1) is 2*2+1 = 5
    # But in the 2x2x2 block, w has size 4.
    # In the 2x2x3 block, w has size 6 (k=0,1,2).
    # Indices:
    # k=0: 0, 1
    # k=1: 2, 3
    # k=2: 4, 5
    
    terms_2x2x1 = [
        # a00*b02
        ([1, 0, 0, 0], [1, 0], [0, 0, 0, 0, 1, 0]),
        # a01*b12
        ([0, 1, 0, 0], [0, 1], [0, 0, 0, 0, 1, 0]),
        # a10*b02
        ([0, 0, 1, 0], [1, 0], [0, 0, 0, 0, 0, 1]),
        # a11*b12
        ([0, 0, 0, 1], [0, 1], [0, 0, 0, 0, 0, 1])
    ]
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block in range(3):
        # Block starts at column 3*block
        k_offset = block * 3
        
        # Add 7 Strassen terms
        for u, v, w in terms_2x2x2:
            # u is size 4 (A)
            for idx in range(4):
                if u[idx] != 0:
                    U[idx][col] = u[idx]
            
            # v is size 4 (B for k=0,1)
            # v[0] is b00, v[1] is b01, v[2] is b10, v[3] is b11
            # Local k: 0, 1
            # Global k: k_offset, k_offset+1
            # V index: j * p + k_global
            # j=0, k=0 -> 0*p + k_offset
            # j=0, k=1 -> 0*p + k_offset + 1
            # j=1, k=0 -> 1*p + k_offset
            # j=1, k=1 -> 1*p + k_offset + 1
            
            # v[0] -> b00 (j=0, k=0)
            V[0 * p + (k_offset + 0)][col] = v[0]
            # v[1] -> b01 (j=0, k=1)
            V[0 * p + (k_offset + 1)][col] = v[1]
            # v[2] -> b10 (j=1, k=0)
            V[1 * p + (k_offset + 0)][col] = v[2]
            # v[3] -> b11 (j=1, k=1)
            V[1 * p + (k_offset + 1)][col] = v[3]
            
            # w is size 4 (C for k=0,1)
            # w[0] -> C00 (i=0, k=0) -> idx 0*2+0 = 0
            # w[1] -> C10 (i=1, k=0) -> idx 0*2+1 = 1
            # w[2] -> C01 (i=0, k=1) -> idx 1*2+0 = 2
            # w[3] -> C11 (i=1, k=1) -> idx 1*2+1 = 3
            # Global k: k_offset, k_offset+1
            # W index: k_global * n + i
            
            # w[0] -> k=k_offset, i=0
            W[(k_offset + 0) * n + 0][col] = w[0]
            # w[1] -> k=k_offset, i=1
            W[(k_offset + 0) * n + 1][col] = w[1]
            # w[2] -> k=k_offset+1, i=0
            W[(k_offset + 1) * n + 0][col] = w[2]
            # w[3] -> k=k_offset+1, i=1
            W[(k_offset + 1) * n + 1][col] = w[3]
            
            col += 1
            
        # Add 4 Schoolbook terms
        for u, v, w in terms_2x2x1:
            # u is size 4
            for idx in range(4):
                if u[idx] != 0:
                    U[idx][col] = u[idx]
            
            # v is size 2 (b02, b12)
            # Local k=2
            # Global k = k_offset + 2
            # v[0] -> b02 (j=0, k=2)
            V[0 * p + (k_offset + 2)][col] = v[0]
            # v[1] -> b12 (j=1, k=2)
            V[1 * p + (k_offset + 2)][col] = v[1]
            
            # w is size 6 (C for k=0,1,2)
            # We only care about k=2
            # w[4] -> C02 (i=0, k=2)
            # w[5] -> C12 (i=1, k=2)
            # Global k = k_offset + 2
            W[(k_offset + 2) * n + 0][col] = w[4]
            W[(k_offset + 2) * n + 1][col] = w[5]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
