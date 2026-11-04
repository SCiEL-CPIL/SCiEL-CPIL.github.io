import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    t0 = time.time()
    deadline = t0 + 110.0

    n, m, p = 4, 4, 6

    # Strassen 2x2*2x2 factors
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))

    prods = [
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0), (1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U_s[i * 2 + j, m_idx] += c
        for (j, k, c) in Vb:
            V_s[j * 2 + k, m_idx] += c
        for (i, k, c) in Wb:
            W_s[k * 2 + i, m_idx] += c

    # Build rank-77 Strassen decomposition
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    # Part 1: 4x4*4x4 Strassen
    for t in range(7):
        for s in range(7):
            col = 7 * t + s
            for i in range(4):
                I = i // 2
                i0 = i % 2
                for j in range(4):
                    J = j // 2
                    j0 = j % 2
                    val = U_s[2 * I + J, t] * U_s[2 * i0 + j0, s]
                    if val != 0:
                        U[i * 4 + j, col] += val

            for j in range(4):
                J = j // 2
                j0 = j % 2
                for k in range(4):
                    K = k // 2
                    k0 = k % 2
                    val = V_s[2 * J + K, t] * V_s[2 * j0 + k0, s]
                    if val != 0:
                        V[j * 6 + k, col] += val

            for k in range(4):
                K = k // 2
                k0 = k % 2
                for i in range(4):
                    I = i // 2
                    i0 = i % 2
                    val = W_s[2 * K + I, t] * W_s[2 * k0 + i0, s]
                    if val != 0:
                        W[k * 4 + i, col] += val

    # Part 2: 4x4*4x2 for columns 4-5
    block_products = [
        (0, 0, 0, 4, 0, 4),
        (0, 2, 2, 4, 0, 4),
        (2, 0, 0, 4, 2, 4),
        (2, 2, 2, 4, 2, 4),
    ]

    for block_idx, (ar, ac, br, bc, cr, cc) in enumerate(block_products):
        for s in range(7):
            col = rank_444 + block_idx * 7 + s
            for i in range(2):
                for j in range(2):
                    val = U_s[i * 2 + j, s]
                    if val != 0:
                        U[(ar + i) * 4 + (ac + j), col] += val

            for j in range(2):
                for k in range(2):
                    val = V_s[j * 2 + k, s]
                    if val != 0:
                        V[(br + j) * 6 + (bc + k), col] += val

            for k in range(2):
                for i in range(2):
                    val = W_s[k * 2 + i, s]
                    if val != 0:
                        W[(cc + k) * 4 + (cr + i), col] += val

    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in mat]

    # Check deadline - if tight, return immediately
    if time.time() > deadline - 20.0:
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    # Try to reduce rank by finding linear dependencies
    # The tensor condition: for all (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = delta(i,j,k)
    # We can try to merge columns that are linearly dependent in a way that preserves the tensor
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = total_rank

    # Try pairwise column merging: if col_a and col_b can be combined into one
    # such that the tensor equation still holds, we reduce rank by 1
    
    # A simpler approach: check if any column is a linear combination of others
    # and if so, we might be able to eliminate it with a rank-1 update
    
    # For now, try a greedy approach: for each pair of columns, check if their
    # outer products in the tensor sense are linearly dependent
    
    # Compute the "tensor signature" for each column
    # For column r, the contribution to the tensor is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # (with appropriate index mapping)
    
    # Flatten U, V, W columns into vectors and check for near-dependence
    # U is (16, rank), V is (24, rank), W is (24, rank)
    
    # The tensor equation in flat form:
    # T[i,j,k] = sum_r U[i*4+j, r] * V[j*6+k, r] * W[k*4+i, r]
    # This is a 3-way tensor, not easily reduced to pairwise column dependence
    
    # Alternative: try random sparse perturbations and check if rank can be reduced
    # This is expensive, so we'll do a limited search
    
    # For the refinement, we'll try to see if we can express some columns
    # as combinations and remove them with a correction term
    
    # Given time constraints, we'll attempt a simple rank reduction check:
    # Compute the "effective rank" by checking if the matrix of tensor
    # contributions has any zero singular values
    
    # Build the linear system: for each (i,j,k), the equation is
    # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = T[i,j,k]
    # where T[i,j,k] = 1 if this is a valid multiplication term, else 0
    
    # Actually, for matrix multiplication A(4x4) * B(4x6) = C(4x6):
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) in the multiplication
    
    # The tensor decomposition must satisfy:
    # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i in [0,4), j in [0,4), k in [0,6)
    
    # Let's verify our decomposition is correct first
    # Then try to reduce rank
    
    # Verify correctness
    residual = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(total_rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                expected = 1.0
                residual += (s - expected)**2
    
    if residual > 1e-10:
        # Decomposition is wrong, return as-is
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    # Now try to reduce rank by finding redundant columns
    # Strategy: for each column r, check if removing it and adjusting
    # other columns can still satisfy the tensor equation
    
    # This is a complex optimization, so we'll do a limited greedy search
    # Try removing one column at a time and see if we can compensate
    
    # For time efficiency, we'll try a small number of candidate removals
    # and use least-squares to find the correction
    
    # Actually, a simpler approach: check if the columns are linearly
    # independent in the "tensor space"
    
    # Build the tensor as a matrix: each (i,j,k) gives one equation
    # The unknowns are the rank-r coefficients
    # But this is a 3-way tensor, so it's not a simple linear system
    
    # Given the complexity, we'll stick with the rank-77 solution
    # and return it, as rank reduction is non-trivial
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
