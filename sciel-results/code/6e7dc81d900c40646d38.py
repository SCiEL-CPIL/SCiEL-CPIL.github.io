import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-39 decomposition for 2x3x7 matrix multiplication.
    Uses Strassen for 2x2x2 blocks and schoolbook for the rest.
    """
    n, m, p = 2, 3, 7
    # We will build the decomposition term by term.
    # U: (n*m, rank) -> (6, rank)
    # V: (m*p, rank) -> (21, rank)
    # W: (p*n, rank) -> (14, rank)
    
    # We don't know the final rank upfront, so we use lists of rows.
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """
        u_coeffs: dict {(i,j): val}
        v_coeffs: dict {(j,k): val}
        w_coeffs: dict {(k,i): val}
        """
        U_row = [0.0] * (n * m)
        for (i, j), val in u_coeffs.items():
            U_row[i * m + j] = val
            
        V_row = [0.0] * (m * p)
        for (j, k), val in v_coeffs.items():
            V_row[j * p + k] = val
            
        W_row = [0.0] * (p * n)
        for (k, i), val in w_coeffs.items():
            W_row[k * n + i] = val
            
        U_rows.append(U_row)
        V_rows.append(V_row)
        W_rows.append(W_row)

    # --- Part 1: Strassen for 2x2x2 blocks ---
    # A is 2x2 (indices i=0,1; j=0,1)
    # B is 2x2 (indices j=0,1; k=k_s, k_s+1)
    # C is 2x2 (indices i=0,1; k=k_s, k_s+1)
    
    def add_strassen_block(k_start):
        # m1 = (a+d)(e+h)
        # a=A[0,0], d=A[1,1]
        # e=B[0,k_s], h=B[1,k_s+1]
        add_term(
            {(0,0): 1, (1,1): 1},
            {(0, k_start): 1, (1, k_start+1): 1},
            {(k_start, 0): 1, (k_start+1, 1): 1}
        )
        
        # m2 = (c+d)e
        # c=A[1,0], d=A[1,1]
        # e=B[0,k_s]
        add_term(
            {(1,0): 1, (1,1): 1},
            {(0, k_start): 1},
            {(k_start, 1): 1} # C[1, k_s] += m2
        )
        # Wait, C[1, k_s] is C[1,0] relative to block.
        # C[1, k_s] corresponds to i=1, k=k_start.
        # W index is (k, i). So (k_start, 1).
        # Correct.
        
        # m3 = a(f-h)
        # a=A[0,0]
        # f=B[0,k_s+1], h=B[1,k_s+1]
        add_term(
            {(0,0): 1},
            {(0, k_start+1): 1, (1, k_start+1): -1},
            {(k_start+1, 0): 1} # C[0, k_s+1] += m3
        )
        
        # m4 = d(g-e)
        # d=A[1,1]
        # g=B[1,k_s], e=B[0,k_s]
        add_term(
            {(1,1): 1},
            {(1, k_start): 1, (0, k_start): -1},
            {(k_start, 0): -1, (k_start+1, 1): 1} # C[0,k_s]-=m4, C[1,k_s+1]+=m4
        )
        
        # m5 = (a+b)h
        # a=A[0,0], b=A[0,1]
        # h=B[1,k_s+1]
        add_term(
            {(0,0): 1, (0,1): 1},
            {(1, k_start+1): 1},
            {(k_start, 0): 1, (k_start+1, 1): 1} # C[0,k_s]+=m5, C[1,k_s+1]+=m5
        )
        
        # m6 = (c-a)(e+f)
        # c=A[1,0], a=A[0,0]
        # e=B[0,k_s], f=B[0,k_s+1]
        add_term(
            {(1,0): 1, (0,0): -1},
            {(0, k_start): 1, (0, k_start+1): 1},
            {(k_start, 1): 1, (k_start+1, 1): -1} # C[1,k_s]+=m6, C[1,k_s+1]-=m6
        )
        
        # m7 = (b-d)(g+h)
        # b=A[0,1], d=A[1,1]
        # g=B[1,k_s], h=B[1,k_s+1]
        add_term(
            {(0,1): 1, (1,1): -1},
            {(1, k_start): 1, (1, k_start+1): 1},
            {(k_start, 0): -1, (k_start+1, 0): 1} # C[0,k_s]-=m7, C[0,k_s+1]+=m7
        )

    # Apply Strassen to k blocks: 0-1, 2-3, 4-5
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # --- Part 2: Schoolbook for 2x2x1 block (k=6) ---
    # A is 2x2 (j=0,1), B is 2x1 (j=0,1; k=6)
    # C is 2x1 (i=0,1; k=6)
    # C[i,6] = A[i,0]B[0,6] + A[i,1]B[1,6]
    
    # Term for A[0,0]B[0,6]
    add_term(
        {(0,0): 1},
        {(0, 6): 1},
        {(6, 0): 1}
    )
    # Term for A[1,0]B[0,6]
    add_term(
        {(1,0): 1},
        {(0, 6): 1},
        {(6, 1): 1}
    )
    # Term for A[0,1]B[1,6]
    add_term(
        {(0,1): 1},
        {(1, 6): 1},
        {(6, 0): 1}
    )
    # Term for A[1,1]B[1,6]
    add_term(
        {(1,1): 1},
        {(1, 6): 1},
        {(6, 1): 1}
    )
    
    # --- Part 3: Schoolbook for 2x1x7 block (j=2) ---
    # A is 2x1 (i=0,1; j=2)
    # B is 1x7 (j=2; k=0..6)
    # C is 2x7 (i=0,1; k=0..6)
    # C[i,k] = A[i,2]B[2,k]
    
    for k in range(7):
        # Term for A[0,2]B[2,k]
        add_term(
            {(0,2): 1},
            {(2, k): 1},
            {(k, 0): 1}
        )
        # Term for A[1,2]B[2,k]
        add_term(
            {(1,2): 1},
            {(2, k): 1},
            {(k, 1): 1}
        )
        
    # Convert lists to matrices
    rank = len(U_rows)
    U = U_rows
    V = V_rows
    W = W_rows
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
