import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank: 12 Strassen blocks (7 each) + 2 Rank6 blocks (6 each)
    # 12 * 7 + 2 * 6 = 84 + 12 = 96
    rank = 96
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def add_strassen(i0, j0, k0, r_offset):
        # Local indices a,b,c in 0..1
        # Global i = i0+a, j = j0+b, k = k0+c
        
        # M1: (A00+A11)(B00+B11) -> C00, C01, C10, C11
        # U: a=0,b=0: 1; a=1,b=1: 1
        U[i0*m + j0, r_offset] = 1
        U[(i0+1)*m + j0+1, r_offset] = 1
        # V: b=0,c=0: 1; b=1,c=1: 1
        V[j0*p + k0, r_offset] = 1
        V[(j0+1)*p + k0+1, r_offset] = 1
        # W: k=0,i=0: 1; k=0,i=1: 1; k=1,i=0: 1; k=1,i=1: 1
        W[k0*n + i0, r_offset] = 1
        W[k0*n + i0+1, r_offset] = 1
        W[(k0+1)*n + i0, r_offset] = 1
        W[(k0+1)*n + i0+1, r_offset] = 1
        
        # M2: (A10+A11)B00 -> C10, C11
        r_offset += 1
        U[(i0+1)*m + j0, r_offset] = 1
        U[(i0+1)*m + j0+1, r_offset] = 1
        V[j0*p + k0, r_offset] = 1
        W[k0*n + i0+1, r_offset] = 1
        W[(k0+1)*n + i0+1, r_offset] = 1
        
        # M3: A00(B01-B11) -> C01, C11
        r_offset += 1
        U[i0*m + j0, r_offset] = 1
        V[j0*p + k0+1, r_offset] = 1
        V[(j0+1)*p + k0+1, r_offset] = -1
        W[(k0+1)*n + i0, r_offset] = 1
        W[(k0+1)*n + i0+1, r_offset] = 1
        
        # M4: A11(B10-B00) -> C00, C10
        r_offset += 1
        U[(i0+1)*m + j0+1, r_offset] = 1
        V[(j0+1)*p + k0, r_offset] = 1
        V[j0*p + k0, r_offset] = -1
        W[k0*n + i0, r_offset] = 1
        W[k0*n + i0+1, r_offset] = 1
        
        # M5: (A00+A01)B11 -> C00(-), C01
        r_offset += 1
        U[i0*m + j0, r_offset] = 1
        U[i0*m + j0+1, r_offset] = 1
        V[(j0+1)*p + k0+1, r_offset] = 1
        W[k0*n + i0, r_offset] = -1
        W[(k0+1)*n + i0, r_offset] = 1
        
        # M6: (A10-A00)(B00+B01) -> C11
        r_offset += 1
        U[(i0+1)*m + j0, r_offset] = 1
        U[i0*m + j0, r_offset] = -1
        V[j0*p + k0, r_offset] = 1
        V[j0*p + k0+1, r_offset] = 1
        W[(k0+1)*n + i0+1, r_offset] = 1
        
        # M7: (A01-A11)(B10+B11) -> C00, C10
        r_offset += 1
        U[i0*m + j0+1, r_offset] = 1
        U[(i0+1)*m + j0+1, r_offset] = -1
        V[(j0+1)*p + k0, r_offset] = 1
        V[(j0+1)*p + k0+1, r_offset] = 1
        W[k0*n + i0, r_offset] = 1
        W[k0*n + i0+1, r_offset] = 1
        
        return r_offset + 1

    def add_rank6(i0, j0, k0, r_offset):
        # Local i in 0,1; j=0; k in 0..5
        # r in 0..5
        for r in range(6):
            # U[i, r] = 1
            U[i0*m + j0, r_offset + r] = 1
            U[(i0+1)*m + j0, r_offset + r] = 1
            # V[0, k, r] = delta(k, r)
            # Global V index: (j0+0)*p + (k0+r)
            V[j0*p + k0 + r, r_offset + r] = 1
            # W[k, i, r] = 1
            # Global W index: (k0+r)*n + i
            W[(k0+r)*n + i0, r_offset + r] = 1
            W[(k0+r)*n + i0+1, r_offset + r] = 1
            
        return r_offset + 6

    current_rank = 0
    
    # Top half (i0=0)
    # A_top B_top: j in 0..1
    # k in 0..1, 2..3, 4..5
    current_rank = add_strassen(0, 0, 0, current_rank)
    current_rank = add_strassen(0, 0, 2, current_rank)
    current_rank = add_strassen(0, 0, 4, current_rank)
    
    # A_top B_bot: j in 2..4
    # j=2 (width 1) -> Rank6
    current_rank = add_rank6(0, 2, 0, current_rank)
    
    # j=3..4 (width 2) -> Strassen
    current_rank = add_strassen(0, 3, 0, current_rank)
    current_rank = add_strassen(0, 3, 2, current_rank)
    current_rank = add_strassen(0, 3, 4, current_rank)
    
    # Bottom half (i0=2)
    # A_bot B_top: j in 0..1
    current_rank = add_strassen(2, 0, 0, current_rank)
    current_rank = add_strassen(2, 0, 2, current_rank)
    current_rank = add_strassen(2, 0, 4, current_rank)
    
    # A_bot B_bot: j in 2..4
    current_rank = add_rank6(2, 2, 0, current_rank)
    
    current_rank = add_strassen(2, 3, 0, current_rank)
    current_rank = add_strassen(2, 3, 2, current_rank)
    current_rank = add_strassen(2, 3, 4, current_rank)
    
    # Convert to lists as required by interface?
    # The interface says "integer/half-integer entries".
    # The return type is {'U', 'V', 'W'}.
    # The seed returned lists of lists.
    # I will return numpy arrays or lists. The evaluator likely handles both or expects lists.
    # Seed used: U = [[0] * rank ...]
    # I will return numpy arrays to be safe, or convert to lists.
    # "factor matrices of shapes ... with integer/half-integer entries"
    # Let's return numpy arrays.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
