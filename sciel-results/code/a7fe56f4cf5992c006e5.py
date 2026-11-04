"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.
    
    We use a structured approach:
    1. Use the known rank-5 algorithm for 2x3x3 multiplication (Strassen-like for 2x3).
    2. Compose with 3x3x7 multiplication to get 2x3x7.
    
    The 2x3x3 multiplication can be done with rank 5 (it's known that R(2,3,3) = 5).
    The 3x3x7 multiplication: we can use schoolbook (rank 21) or try to find better.
    
    Actually, let me think more carefully. We want to multiply A (2x3) by B (3x7) to get C (2x7).
    
    One approach: decompose the multiplication into blocks.
    Let A = [A1 A2 A3] where Ai is 2x1, and B = [B1 B2 B3] where Bi is 1x7.
    Then C = A1*B1 + A2*B2 + A3*B3, where each Ai*Bi is 2x7.
    
    Alternatively, we can use the fact that we can write:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For rank minimization, we can use tensor decomposition techniques.
    
    Let me try a different approach: use the known rank for 2x3x3 and extend.
    
    Actually, let's use a more direct approach. We'll try to find a decomposition
    by exploiting the rectangular structure.
    
    Key insight: The multiplication 2x3 * 3x7 can be viewed as:
    - First, compute 6 scalar products (schoolbook: rank 42)
    - But we can do better by grouping.
    
    Let's use the following strategy:
    Split the 3x7 matrix B into three 3x? blocks or use the structure differently.
    
    Actually, here's a cleaner approach using tensor rank properties:
    
    For M_{n,m,p}, we know:
    - R(2,3,3) = 5 (there exists a rank-5 algorithm)
    - We can use this to help with 2x3x7
    
    Method: Write B as a 3x7 matrix. We can think of the computation as:
    For each output column k (k=0..6), compute C[:,k] = A * B[:,k]
    where B[:,k] is a 3x1 vector.
    
    So we need to compute 7 instances of 2x3 * 3x1 = 2x1 multiplication.
    Each 2x3 * 3x1 multiplication has rank at most 6 (schoolbook), but can it be done with less?
    
    Actually, 2x3 * 3x1 is just a 2x3 matrix times a 3-vector, giving a 2-vector.
    The tensor for this is M_{2,3,1}. R(2,3,1) = 6 (since it's essentially 6 independent products, 
    but actually we can check: C[0] = A[0,0]*B[0] + A[0,1]*B[1] + A[0,2]*B[2]
    C[1] = A[1,0]*B[0] + A[1,1]*B[1] + A[1,2]*B[2]
    This is a 2x3x1 tensor. Its rank is 6 in general (no known lower bound below 6, and 
    it's unlikely to be less than 6 since we have 6 "independent" terms... but actually,
    for 2x3x1, the rank is exactly 6? Let me think... 
    
    Actually, for M_{2,3,1}, we can write it as a sum of 6 rank-1 tensors, and I believe
    the rank is 6. But we have 7 such columns, so naively that's 42.
    
    However, we can share computation across columns! This is the key.
    
    Let me use a different decomposition. Consider:
    C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    We can use the following identity (inspired by Strassen):
    Define intermediate quantities that combine A and B entries.
    
    For 2x3 * 3x7, here's an approach using the rank-5 2x3x3 result:
    
    Step 1: Split B (3x7) into B_left (3x3), B_mid (3x3), B_right (3x1)
    Step 2: Compute A * B_left, A * B_mid, A * B_right using rank-5 2x3x3 for the first two
            and rank-6 2x3x1 for the last.
    
    Total rank: 5 + 5 + 6 = 16? But this doesn't account for sharing.
    
    Actually, this composition approach doesn't directly give a tensor decomposition of M_{2,3,7}.
    
    Let me try a more systematic approach. I'll use the following known result:
    R(2,3,7) <= R(2,3,3) * 2 + R(2,3,1) = 5*2 + 6 = 16
    
    But we need to construct the actual decomposition.
    
    Here's the construction:
    Split B into B1 (3x3), B2 (3x3), B3 (3x1).
    C = A*B1 + A*B2 + A*B3
    
    For A*B1 (2x3 * 3x3 = 2x3): use rank-5 decomposition
    For A*B2 (2x3 * 3x3 = 2x3): use rank-5 decomposition  
    For A*B3 (2x3 * 3x1 = 2x1): use rank-6 decomposition
    
    Total rank = 5 + 5 + 6 = 16
    
    Now I need the actual rank-5 decomposition for 2x3x3.
    
    The rank-5 algorithm for 2x3x3 multiplication:
    Let A be 2x3, B be 3x3, C = A*B be 2x3.
    
    One known rank-5 decomposition:
    M1 = A[0,0] * (B[0,:] + B[1,:] + B[2,:])  -- wait, this isn't quite right.
    
    Let me look up or derive the rank-5 2x3x3 algorithm.
    
    Actually, I recall that for 2x3x3, we can use:
    Let a1=A[0,0], a2=A[0,1], a3=A[0,2], a4=A[1,0], a5=A[1,1], a6=A[1,2]
    Let b11,b12,b13,b21,b22,b23,b31,b32,b33 be entries of B.
    
    C[0,0] = a1*b11 + a2*b21 + a3*b31
    C[0,1] = a1*b12 + a2*b22 + a3*b32
    C[0,2] = a1*b13 + a2*b23 + a3*b33
    C[1,0] = a4*b11 + a5*b21 + a6*b31
    C[1,1] = a4*b12 + a5*b22 + a6*b32
    C[1,2] = a4*b13 + a5*b23 + a6*b33
    
    A rank-5 decomposition would express these 6 outputs using 5 products.
    
    Here's one approach:
    P1 = a1 * (b11 + b12 + b13)
    P2 = a2 * (b21 + b22 + b23)
    P3 = a3 * (b31 + b32 + b33)
    P4 = (a1 + a4) * (b11 + b12 + b13)
    P5 = (a4) * (b11 + b12 + b13)
    
    Hmm, this doesn't seem to work cleanly. Let me think differently.
    
    Actually, I think the cleanest approach for this task is to just implement the schoolbook
    decomposition (rank 42) since finding an exact lower-rank decomposition is complex and
    error-prone. The schoolbook is guaranteed to be correct.
    
    But wait - the task says "Find a rank-minimal exact tensor decomposition". If I just
    return schoolbook, I get rank 42. Can I do better?
    
    Let me try a different approach: use the fact that we can factor the multiplication.
    
    Actually, for the purposes of this task, let me implement a correct schoolbook decomposition
    and see if I can find any simple improvements.
    
    For now, let me just return the schoolbook decomposition to ensure correctness.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
