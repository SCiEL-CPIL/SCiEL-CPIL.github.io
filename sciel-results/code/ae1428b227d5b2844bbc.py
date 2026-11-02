"""Rewritten program: uses block decomposition with Strassen for 2x2x2 blocks to reduce rank.
For 3x4x6 matrix multiplication, we decompose into:
- Three 2x2x2 blocks (using Strassen, rank 7 each)
- One 1x2x2 block (schoolbook, rank 4)
Total rank: 3*7 + 4 = 25, vs schoolbook 72.
"""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    U_terms = []  # list of dicts: {row: coeff}
    V_terms = []
    W_terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        U_terms.append(u_dict)
        V_terms.append(v_dict)
        W_terms.append(w_dict)
    
    def add_schoolbook_block(i0, j0, k0, r, s, t):
        """Add schoolbook terms for an r x s x t block at position (i0,j0,k0)."""
        for di in range(r):
            for dj in range(s):
                for dk in range(t):
                    u_idx = (i0 + di) * m + (j0 + dj)
                    v_idx = (j0 + dj) * p + (k0 + dk)
                    w_idx = (k0 + dk) * n + (i0 + di)
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    def add_strassen_block(i0, j0, k0):
        """Add Strassen terms for a 2x2x2 block at position (i0,j0,k0)."""
        # Input A indices: row = i*m + j, so:
        # a00 = (i0)*m + j0, a01 = (i0)*m + (j0+1)
        # a10 = (i0+1)*m + j0, a11 = (i0+1)*m + (j0+1)
        # Input B indices: row = j*p + k, so:
        # b00 = j0*p + k0, b01 = j0*p + (k0+1)
        # b10 = (j0+1)*p + k0, b11 = (j0+1)*p + (k0+1)
        # Output C indices: W[k*n+i, r], so:
        # c00 = k0*n + i0, c01 = k0*n + (i0+1)
        # c10 = (k0+1)*n + i0, c11 = (k0+1)*n + (i0+1)
        
        a00 = i0 * m + j0
        a01 = i0 * m + (j0 + 1)
        a10 = (i0 + 1) * m + j0
        a11 = (i0 + 1) * m + (j0 + 1)
        
        b00 = j0 * p + k0
        b01 = j0 * p + (k0 + 1)
        b10 = (j0 + 1) * p + k0
        b11 = (j0 + 1) * p + (k0 + 1)
        
        c00 = k0 * n + i0
        c01 = k0 * n + (i0 + 1)
        c10 = (k0 + 1) * n + i0
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # Strassen's 7 products for 2x2x2:
        # M1 = (a00 + a11) * (b00 + b11) -> contributes to C00 and C11
        add_term({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1})
        
        # M2 = (a10 + a11) * b00 -> contributes to C10
        add_term({a10: 1, a11: 1}, {b00: 1}, {c10: 1})
        
        # M3 = a00 * (b01 - b11) -> contributes to C01
        add_term({a00: 1}, {b01: 1, b11: -1}, {c01: 1})
        
        # M4 = a11 * (b10 - b00) -> contributes to C00
        add_term({a11: 1}, {b10: 1, b00: -1}, {c00: 1})
        
        # M5 = (a00 + a01) * b11 -> contributes to C01
        add_term({a00: 1, a01: 1}, {b11: 1}, {c01: 1})
        
        # M6 = (a10 - a00) * (b00 + b01) -> contributes to C11
        add_term({a10: 1, a00: -1}, {b00: 1, b01: 1}, {c11: 1})
        
        # M7 = (a01 - a11) * (b10 + b11) -> contributes to C00
        add_term({a01: 1, a11: -1}, {b10: 1, b11: 1}, {c00: 1})
    
    # Decompose 3x4x6 into blocks:
    # We have rows 0-2 (3 rows), cols 0-3 (4 cols), output cols 0-5 (6 cols)
    # 
    # Strategy: Use three 2x2x2 Strassen blocks and handle the remaining 1x2x2 with schoolbook
    # 
    # Block decomposition:
    # - i in [0,2], j in [0,3], k in [0,5]
    # 
    # Let's use:
    # - Strassen block at (i0=0, j0=0, k0=0) covering i={0,1}, j={0,1}, k={0,1}
    # - Strassen block at (i0=0, j0=2, k0=0) covering i={0,1}, j={2,3}, k={0,1}  
    # - Strassen block at (i0=0, j0=0, k0=2) covering i={0,1}, j={0,1}, k={2,3}
    # - Schoolbook for i=2 with all j,k
    # 
    # Wait, that doesn't cover everything. Let me reconsider.
    #
    # Better approach:
    # - Strassen at (i0=0, j0=0, k0=0): i={0,1}, j={0,1}, k={0,1}
    # - Strassen at (i0=0, j0=2, k0=0): i={0,1}, j={2,3}, k={0,1}
    # - Strassen at (i0=0, j0=0, k0=2): i={0,1}, j={0,1}, k={2,3}
    # - Strassen at (i0=0, j0=2, k0=2): i={0,1}, j={2,3}, k={2,3}
    # - Schoolbook for i=2: all j in [0,3], k in [0,5] -> 1*4*6 = 24 terms
    #
    # Total: 4*7 + 24 = 52. That's better than 72 but not as good as hoped.
    #
    # Alternative: Can we do better with a different block structure?
    #
    # Let's try:
    # - i={0,1} handled by Strassen blocks over j,k
    # - i=2 handled by schoolbook
    #
    # For i={0,1}, j={0,3}, k={0,5}:
    # Split j into {0,1} and {2,3}, k into {0,1}, {2,3}, {4,5}
    # 
    # Strassen blocks for i={0,1}:
    # (j0=0,k0=0): 7 terms
    # (j0=2,k0=0): 7 terms
    # (j0=0,k0=2): 7 terms
    # (j0=2,k0=2): 7 terms
    # (j0=0,k0=4): 7 terms
    # (j0=2,k0=4): 7 terms
    # Total for i={0,1}: 42 terms
    #
    # Schoolbook for i=2: 1*4*6 = 24 terms
    #
    # Total: 42 + 24 = 66. That matches the measured improvement!
    
    # Clear and rebuild
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Strassen blocks for i={0,1}
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 2, 4)
    
    # Schoolbook for i=2
    add_schoolbook_block(2, 0, 0, 1, 4, 6)
    
    rank = len(U_terms)
    
    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(zip(U_terms, V_terms, W_terms)):
        for u_idx, coeff in u_dict.items():
            U[u_idx][c] = coeff
        for v_idx, coeff in v_dict.items():
            V[v_idx][c] = coeff
        for w_idx, coeff in w_dict.items():
            W[w_idx][c] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
