"""Rewritten program for Matrix multiplication 2,2,13"""

import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=13
    # Output C = A @ B where A is 2x2, B is 2x13, C is 2x13
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (26, rank)  
    # W shape: (p*n, rank) = (26, rank)
    # W index: k*n + i (column k, row i)
    # V index: j*p + k (row j, column k)
    # U index: i*m + j (row i, col j)
    
    rank = 46
    U = np.zeros((4, rank), dtype=np.float64)
    V = np.zeros((26, rank), dtype=np.float64)
    W = np.zeros((26, rank), dtype=np.float64)
    
    col = 0
    
    def add_term(u_dict, v_dict, w_dict):
        nonlocal col
        for idx, val in u_dict.items():
            U[idx, col] += val
        for idx, val in v_dict.items():
            V[idx, col] += val
        for idx, val in w_dict.items():
            W[idx, col] += val
        col += 1
    
    # Process 6 blocks of 2 columns (k=0,1; 2,3; ...; 10,11)
    for block_idx in range(6):
        k0 = block_idx * 2
        k1 = block_idx * 2 + 1
        
        # U indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # V indices: j*p + k; j=0 -> k, j=1 -> 13+k
        # W indices: k*n + i; i=0 -> 2k, i=1 -> 2k+1
        
        # Strassen 7 products for 2x2x2 block
        # M1 = (a00+a11)(b00+b11)
        add_term(
            {0: 1, 3: 1}, 
            {k0: 1, 13+k1: 1}, 
            {2*k0: 1, 2*k1+1: 1}
        )
        
        # M2 = (a10+a11)b00
        add_term(
            {2: 1, 3: 1}, 
            {k0: 1}, 
            {2*k0+1: 1, 2*k1+1: -1}
        )
        
        # M3 = a00(b01-b11)
        add_term(
            {0: 1}, 
            {k1: 1, 13+k1: -1}, 
            {2*k1: 1, 2*k1+1: 1}
        )
        
        # M4 = a11(b10-b00)
        add_term(
            {3: 1}, 
            {13+k0: 1, k0: -1}, 
            {2*k0: 1, 2*k0+1: 1}
        )
        
        # M5 = (a00+a01)b11
        add_term(
            {0: 1, 1: 1}, 
            {13+k1: 1}, 
            {2*k0: -1, 2*k1: 1}
        )
        
        # M6 = (a10-a00)(b00+b01)
        add_term(
            {2: 1, 0: -1}, 
            {k0: 1, k1: 1}, 
            {2*k1+1: 1}
        )
        
        # M7 = (a01-a11)(b10+b11)
        add_term(
            {1: 1, 3: -1}, 
            {13+k0: 1, 13+k1: 1}, 
            {2*k0: 1}
        )
    
    # Process last column k=12 (schoolbook 2x2x1, rank 4)
    k = 12
    # C0k = A00*B0k + A01*B1k
    add_term({0: 1}, {k: 1}, {2*k: 1})
    add_term({1: 1}, {13+k: 1}, {2*k: 1})
    # C1k = A10*B0k + A11*B1k
    add_term({2: 1}, {k: 1}, {2*k+1: 1})
    add_term({3: 1}, {13+k: 1}, {2*k+1: 1})
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
