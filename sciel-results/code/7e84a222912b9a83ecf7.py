import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin

    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n

    # Start with the trivial decomposition (rank = n*m*p)
    rank_trivial = n * m * p

    U = [[0] * rank_trivial for _ in range(dim_u)]
    V = [[0] * rank_trivial for _ in range(dim_v)]
    W = [[0] * rank_trivial for _ in range(dim_w)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    best_rank = rank_trivial
    best_result = {"U": U, "V": V, "W": W}

    # Try to find a better decomposition using Strassen-like reductions
    # For now, we'll attempt a structured approach with Kronecker products

    # Strategy: Express the 3x5 @ 5x5 multiplication as block operations
    # 3x5 = [row0; row1; row2] where each row is 1x5
    # 5x5 = [col0; col1; col2; col3; col4] where each col is 5x1
    
    # The result C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can try to use rank-1 decompositions for each (i,k) pair
    # But that gives rank = n*p = 15, which is already better than 75!
    
    # Wait, let me think about this more carefully.
    # The tensor is T[i,j,k] = 1 for the multiplication tensor
    # A rank-r decomposition means we can write the multiplication as sum of r rank-1 tensors
    # Each rank-1 tensor corresponds to one term in the decomposition
    
    # Actually, the minimal rank for matrix multiplication M_{n,m,p} is known to be:
    # - M_{1,1,1}: rank 1
    # - M_{2,2,2}: rank 7 (Strassen)
    # - For general cases, it's an open problem
    
    # For 3x5x5, we can try to use block decompositions
    # Let's try a different approach: use the fact that we can decompose
    # larger matrices into smaller ones
    
    # For now, let's try to find if there's a rank < 75 decomposition
    # by trying to merge terms in the trivial decomposition
    
    # A simple improvement: if we can show that some terms can be combined
    # or that there's linear dependence, we can reduce the rank
    
    # Let's try to verify if the trivial decomposition is truly minimal
    # by attempting to reduce the rank using linear algebra
    
    # Convert to numpy for efficiency
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check if we can reduce the rank by finding linear dependencies
    # This is equivalent to finding the rank of the tensor
    
    # For now, let's try a heuristic: try to remove terms and see if the
    # decomposition still works
    
    # Actually, let's try a completely different approach:
    # Use the fact that matrix multiplication can be done with fewer than n*m*p operations
    # by using algorithms like Strassen
    
    # For 3x5x5, we can partition the matrices and apply Strassen to blocks
    
    # Let's try to construct a decomposition with rank < 75
    # One approach: use the fact that 5 = 2 + 2 + 1 and 3 = 2 + 1
    
    # This is getting complex. Let's try a simpler approach:
    # Try to find a rank-r decomposition by solving a system of equations
    
    # The condition for a rank-r decomposition is:
    # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # For all other (a,b,c): sum_r U[a, r] * V[b, r] * W[c, r] = 0
    
    # This is a trilinear system, which is hard to solve directly
    
    # Let's try to use a known result: the rank of M_{n,m,p} is at most n*m*p
    # and we can sometimes do better
    
    # For 3x5x5, let's try to use a block decomposition approach
    # Split A (3x5) into [A1 | A2] where A1 is 3x2 and A2 is 3x3
    # Split B (5x5) into [B1; B2] where B1 is 2x5 and B2 is 3x5
    
    # Then C = A1*B1 + A2*B2
    
    # Each of these products can be decomposed separately
    # M_{3,2,5} has rank at most 3*2*5 = 30
    # M_{3,3,5} has rank at most 3*3*5 = 45
    # Total: 75, which is the same as before
    
    # This doesn't help. Let's try a different partition
    
    # Actually, the minimal rank for M_{3,5,5} is not known in general
    # Let's just try to find any decomposition with rank < 75
    
    # One idea: use the fact that we can share some computations
    # For example, if two terms have similar structure, we might be able to merge them
    
    # Let's try to find a decomposition with rank 74 or less
    # by attempting to combine two terms in the trivial decomposition
    
    # This is a search problem, and it's expensive. Let's try a simple heuristic
    
    # Time check
    if time.time() - start_time > deadline - 10:
        return best_result
    
    # For now, return the trivial decomposition
    # A more sophisticated search would be needed to find a better rank
    return best_result
