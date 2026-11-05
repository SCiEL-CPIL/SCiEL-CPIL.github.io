import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U, V, W = build(n, m, p)
    return {"U": U, "V": V, "W": W}

def build(n, m, p):
    # Returns U (n*m, rank), V (m*p, rank), W (p*n, rank)
    
    # Base case: Schoolbook
    if n == 1 and m == 1 and p == 1:
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W

    # If m == 1, rank is n*p
    if m == 1:
        rank = n * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(p):
                U[i][col] = 1
                V[k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        return U, V, W

    # If n == 1, rank is m*p
    if n == 1:
        rank = m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for j in range(m):
            for k in range(p):
                U[j][col] = 1
                V[j * p + k][col] = 1
                W[k][col] = 1
                col += 1
        return U, V, W

    # If p == 1, rank is n*m
    if p == 1:
        rank = n * m
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return U, V, W

    # Specific optimized cases
    
    if n == 2 and m == 2 and p == 2:
        return build_strassen_2x2x2()

    if n == 2 and m == 2 and p == 1:
        return build(2, 2, 1) # Will hit p==1 base case? No, p==1 is handled above.
        # Actually p==1 case is rank 4.

    if n == 2 and m == 1 and p == 3:
        return build(2, 1, 3) # m==1 case handled above. Rank 6.

    # Recursive splits
    
    # Split p if p >= 2 and n,m are not small?
    # My logic:
    # 4,5,6 -> split p
    # 4,5,3 -> split n
    # 2,5,3 -> split m
    # 2,2,3 -> split p
    # 2,3,3 -> split m
    
    if n == 4 and m == 5 and p == 6:
        # Split p into 3, 3
        U1, V1, W1 = build(4, 5, 3)
        U2, V2, W2 = build(4, 5, 3)
        # Embed
        # U1 is for B[:, :3], U2 is for B[:, 3:]
        # V1 is for B[:, :3], V2 is for B[:, 3:]
        # W1 is for C[:, :3], W2 is for C[:, 3:]
        
        # U shape: 20 x rank
        # V shape: 30 x rank
        # W shape: 24 x rank
        
        r1 = len(U1[0])
        r2 = len(U2[0])
        rank = r1 + r2
        
        U = [[0] * rank for _ in range(20)]
        V = [[0] * rank for _ in range(30)]
        W = [[0] * rank for _ in range(24)]
        
        for i in range(20):
            U[i] = U1[i] + [0]*r2
        for i in range(30):
            # V1 rows 0..14 correspond to B rows 0..4, cols 0..2
            # V2 rows 0..14 correspond to B rows 0..4, cols 3..5
            # We need to map V2 indices to global V indices
            # V index = j * p + k.
            # In sub-problem, p_sub=3.
            # V1 index = j * 3 + k.
            # Global V index for V1: j * 6 + k.
            # Global V index for V2: j * 6 + (k + 3).
            
            # Wait, V1 is for the first sub-problem.
            # The sub-problem is 4x5x3.
            # V1 shape is 5x3 = 15 rows.
            # Global V has 5x6 = 30 rows.
            
            # We need to copy V1 into V[0:15] -> V[0, 3, 6, 9, 12, 15, 18, 21, 24, 27]? No.
            # V index j*p+k.
            # For k in 0..2:
            # V1[j*3+k] -> V[j*6+k]
            
            # For V2:
            # V2[j*3+k] -> V[j*6+k+3]
            
            # It is easier to construct a mapping.
            pass # Handled in generic combine below

    # Generic Combine Logic
    # This is getting complex to hardcode every case.
    # Let's write a helper `combine` that takes sub-decomps and their index mappings.
    
    # Actually, I will just implement the specific cases explicitly to avoid bugs.
    
    if n == 4 and m == 5 and p == 6:
        U1, V1, W1 = build(4, 5, 3)
        U2, V2, W2 = build(4, 5, 3)
        return combine_p_split(U1, V1, W1, U2, V2, W2, n, m, 3, 3)

    if n == 4 and m == 5 and p == 3:
        U1, V1, W1 = build(2, 5, 3)
        U2, V2, W2 = build(2, 5, 3)
        return combine_n_split(U1, V1, W1, U2, V2, W2, 2, 5, 3, 2, 2)

    if n == 2 and m == 5 and p == 3:
        U1, V1, W1 = build(2, 2, 3)
        U2, V2, W2 = build(2, 3, 3)
        return combine_m_split(U1, V1, W1, U2, V2, W2, 2, 2, 3, 2, 3)

    if n == 2 and m == 2 and p == 3:
        U1, V1, W1 = build(2, 2, 2)
        U2, V2, W2 = build(2, 2, 1)
        return combine_p_split(U1, V1, W1, U2, V2, W2, 2, 2, 2, 1)

    if n == 2 and m == 3 and p == 3:
        U1, V1, W1 = build(2, 2, 3)
        U2, V2, W2 = build(2, 1, 3)
        return combine_m_split(U1, V1, W1, U2, V2, W2, 2, 2, 3, 2, 1)

    # Fallback: Schoolbook
    return build(n, m, p) # This will recurse infinitely if not handled.
    # Add base cases for remaining small sizes if any.
    
    # Actually, the base cases at the top handle m=1, n=1, p=1.
    # 2,2,1 is handled by p=1 case.
    # 2,1,3 is handled by m=1 case.
    # 2,2,2 is handled by Strassen.
    # So the recursion terminates.

def combine_p_split(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
    # Splits p into p1, p2.
    # A is n x m.
    # B1 is m x p1, B2 is m x p2.
    # C1 is n x p1, C2 is n x p2.
    # C = [C1 C2].
    
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 + r2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * (p1 + p2))]
    W = [[0] * rank for _ in range((p1 + p2) * n)]
    
    # U is same for both (A is same)
    for i in range(n * m):
        U[i] = U1[i] + U2[i]
        
    # V
    # V1 indices: j * p1 + k
    # V2 indices: j * p2 + k
    # Global V indices: j * (p1+p2) + k (for first part)
    #                  j * (p1+p2) + p1 + k (for second part)
    for j in range(m):
        for k in range(p1):
            V[j * (p1 + p2) + k] = V1[j * p1 + k] + [0] * r2
        for k in range(p2):
            V[j * (p1 + p2) + p1 + k] = [0] * r1 + V2[j * p2 + k]
            
    # W
    # W1 indices: k * n + i
    # W2 indices: k * n + i
    # Global W indices: k * n + i (for first part)
    #                  (p1+k) * n + i (for second part)
    for k in range(p1):
        for i in range(n):
            W[k * n + i] = W1[k * n + i] + [0] * r2
    for k in range(p2):
        for i in range(n):
            W[(p1 + k) * n + i] = [0] * r1 + W2[k * n + i]
            
    return U, V, W

def combine_n_split(U1, V1, W1, U2, V2, W2, n1, m, p, n2, n3):
    # Splits n into n1, n2.
    # A1 is n1 x m, A2 is n2 x m.
    # B is m x p.
    # C1 is n1 x p, C2 is n2 x p.
    # C = [C1; C2].
    
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 + r2
    
    U = [[0] * rank for _ in range((n1 + n2) * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * (n1 + n2))]
    
    # U
    # U1 indices: i * m + j
    # U2 indices: i * m + j
    # Global U indices: i * m + j (for first part)
    #                  (n1+i) * m + j (for second part)
    for i in range(n1):
        for j in range(m):
            U[i * m + j] = U1[i * m + j] + [0] * r2
    for i in range(n2):
        for j in range(m):
            U[(n1 + i) * m + j] = [0] * r1 + U2[i * m + j]
            
    # V is same for both
    for i in range(m * p):
        V[i] = V1[i] + V2[i]
        
    # W
    # W1 indices: k * n1 + i
    # W2 indices: k * n2 + i
    # Global W indices: k * (n1+n2) + i (for first part)
    #                  k * (n1+n2) + n1 + i (for second part)
    for k in range(p):
        for i in range(n1):
            W[k * (n1 + n2) + i] = W1[k * n1 + i] + [0] * r2
    for k in range(p):
        for i in range(n2):
            W[k * (n1 + n2) + n1 + i] = [0] * r1 + W2[k * n2 + i]
            
    return U, V, W

def combine_m_split(U1, V1, W1, U2, V2, W2, n, m1, p, m2, m3):
    # Splits m into m1, m2.
    # A1 is n x m1, A2 is n x m2.
    # B1 is m1 x p, B2 is m2 x p.
    # C1 = A1 B1, C2 = A2 B2.
    # C = C1 + C2.
    
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 + r2
    
    U = [[0] * rank for _ in range(n * (m1 + m2))]
    V = [[0] * rank for _ in range((m1 + m2) * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U
    # U1 indices: i * m1 + j
    # U2 indices: i * m2 + j
    # Global U indices: i * (m1+m2) + j (for first part)
    #                  i * (m1+m2) + m1 + j (for second part)
    for i in range(n):
        for j in range(m1):
            U[i * (m1 + m2) + j] = U1[i * m1 + j] + [0] * r2
    for i in range(n):
        for j in range(m2):
            U[i * (m1 + m2) + m1 + j] = [0] * r1 + U2[i * m2 + j]
            
    # V
    # V1 indices: j * p + k
    # V2 indices: j * p + k
    # Global V indices: j * p + k (for first part)
    #                  (m1+j) * p + k (for second part)
    for j in range(m1):
        for k in range(p):
            V[j * p + k] = V1[j * p + k] + [0] * r2
    for j in range(m2):
        for k in range(p):
            V[(m1 + j) * p + k] = [0] * r1 + V2[j * p + k]
            
    # W is same for both (summed)
    for i in range(p * n):
        W[i] = W1[i] + W2[i]
        
    return U, V, W

def build_strassen_2x2x2():
    # A: 2x2, B: 2x2, C: 2x2
    # U: 4x7, V: 4x7, W: 4x7
    U = [
        [1, 0, 1, 0, 1, 0, 0], # a
        [0, 0, 0, 0, 1, 1, 1], # b
        [0, 1, 0, 0, 0, -1, 0], # c
        [1, 1, 0, 1, 0, 0, -1]  # d
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0], # e
        [1, 0, 1, 0, 0, 1, 1], # f
        [0, 1, 0, 1, 0, 0, 1], # g
        [1, 0, -1, 0, 1, 0, 1]  # h
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0], # C01
        [0, 1, 0, 1, 0, 0, 0], # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    return U, V, W
# EVOLVE-BLOCK-END
