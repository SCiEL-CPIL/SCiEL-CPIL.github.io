import numpy as np

def construct():
    n, m, p = 3, 5, 7
    U, V, W = get_factors(n, m, p)
    return {"U": U, "V": V, "W": W}

def get_factors(n, m, p):
    # Base case: 1x1x1
    if n == 1 and m == 1 and p == 1:
        return [[1]], [[1]], [[1]]
    
    # Base case: 2x2x2 Strassen
    if n == 2 and m == 2 and p == 2:
        return strassen_factors()
    
    # Recursive case: Split m
    # We choose to split m to allow Strassen on 2x2x2 blocks if possible
    # If m is even, split m/2, m/2
    # If m is odd, split m//2, m - m//2
    m1 = m // 2
    m2 = m - m1
    
    # Get factors for part 1: n x m1 x p
    U1, V1, W1 = get_factors(n, m1, p)
    # Get factors for part 2: n x m2 x p
    U2, V2, W2 = get_factors(n, m2, p)
    
    # Concatenate U
    # U1 is n*m1 x rank1, U2 is n*m2 x rank2
    # We need to map U2's rows to the correct positions in the full n*m space
    # U1 rows correspond to A[0:n, 0:m1]
    # U2 rows correspond to A[0:n, m1:m]
    
    rank1 = len(U1[0]) if U1 else 0
    rank2 = len(U2[0]) if U2 else 0
    total_rank = rank1 + rank2
    
    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]
    
    # Fill U
    # U1 rows are for A indices (i, j) where 0<=j<m1
    # In full A (n x m), index is i*m + j
    for i in range(n):
        for j in range(m1):
            row_idx = i * m + j
            U_full[row_idx] = U1[i * m1 + j] + [0] * rank2
            
    # U2 rows are for A indices (i, j) where m1<=j<m
    # In U2, the local index is i*m2 + (j - m1)
    for i in range(n):
        for j in range(m1, m):
            row_idx = i * m + j
            local_j = j - m1
            U_full[row_idx] = [0] * rank1 + U2[i * m2 + local_j]
            
    # Fill V
    # V1 rows are for B indices (j, k) where 0<=j<m1
    # In full B (m x p), index is j*p + k
    for j in range(m1):
        for k in range(p):
            row_idx = j * p + k
            V_full[row_idx] = V1[j * p + k] + [0] * rank2
            
    # V2 rows are for B indices (j, k) where m1<=j<m
    # In V2, local index is (j - m1)*p + k
    for j in range(m1, m):
        for k in range(p):
            row_idx = j * p + k
            local_j = j - m1
            V_full[row_idx] = [0] * rank1 + V2[local_j * p + k]
            
    # Fill W
    # W is just concatenation of columns
    for c in range(total_rank):
        if c < rank1:
            for r in range(p * n):
                W_full[r][c] = W1[r][c]
        else:
            for r in range(p * n):
                W_full[r][c] = W2[r][c - rank1]
                
    return U_full, V_full, W_full

def strassen_factors():
    # 2x2x2 Strassen
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U = [
        [1, 0, 0, 0, 0, 0, 0], # M1: A00
        [0, 0, 0, 0, 1, 0, 0], # M1: A11 (Wait, M1 uses A00+A11)
        # Let's list the 7 terms properly
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B10-B11)
        # M4: A11(B01-B00)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B01)
        # M7: (A01-A11)(B10+B11)
        
        # U column c corresponds to factor c
        # U[row, c] is coeff of A[row] in factor c
        # V[row, c] is coeff of B[row] in factor c
        # W[row, c] is coeff of C[row] in result from factor c
        
        # Factor 0 (M1):
        # U: A00(1), A11(1) -> rows 0, 3
        # V: B00(1), B11(1) -> rows 0, 3
        # W: C00(1), C11(1) -> rows 0, 3
        # Factor 1 (M2):
        # U: A10(1), A11(1) -> rows 2, 3
        # V: B00(1) -> row 0
        # W: C10(1), C11(-1) -> rows 2, 3
        # Factor 2 (M3):
        # U: A00(1) -> row 0
        # V: B10(1), B11(-1) -> rows 2, 3
        # W: C01(1), C11(1) -> rows 1, 3
        # Factor 3 (M4):
        # U: A11(1) -> row 3
        # V: B01(1), B00(-1) -> rows 1, 0
        # W: C00(1), C10(1) -> rows 0, 2
        # Factor 4 (M5):
        # U: A00(1), A01(1) -> rows 0, 1
        # V: B11(1) -> row 3
        # W: C00(-1), C01(1) -> rows 0, 1
        # Factor 5 (M6):
        # U: A10(1), A00(-1) -> rows 2, 0
        # V: B00(1), B01(1) -> rows 0, 1
        # W: C11(1) -> row 3
        # Factor 6 (M7):
        # U: A01(1), A11(-1) -> rows 1, 3
        # V: B10(1), B11(1) -> rows 2, 3
        # W: C00(1) -> row 0
        
        [[1, 0, 1, 0, 1, -1, 0], # A00
         [0, 0, 0, 0, 1, 0, 1],  # A01
         [0, 1, 0, 0, 0, 1, 0],  # A10
         [1, 1, 0, 1, 0, 0, -1]  # A11
        ],
        [[1, 1, 0, -1, 0, 1, 0], # B00
         [0, 0, 0, 1, 0, 1, 0],  # B01
         [0, 0, 1, 0, 0, 0, 1],  # B10
         [1, 0, -1, 0, 1, 0, 1]  # B11
        ],
        [[1, 0, 0, 1, -1, 0, 1], # C00
         [0, 0, 1, 0, 1, 0, 0],  # C01
         [0, 1, 0, 1, 0, 0, 0],  # C10
         [1, -1, 1, 0, 0, 1, 0]  # C11
        ]
    ]
    return U, V, W
