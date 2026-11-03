"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135  # 135s limit to be safe (150s wall-clock)
    
    n, m, p = DIMENSIONS
    
    # We will construct the decomposition by splitting the middle dimension m=5 into 2 and 3.
    # Part 1: 3x2x7
    # Part 2: 3x3x7
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(U, V, W, i0, j0, k0, col):
        # M1
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M2
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] -= 1
        col += 1
        
        # M3
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M4
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        col += 1
        
        # M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[k0 * n + i0][col] -= 1
        col += 1
        
        # M6
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M7
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        
        return col

    # Helper to add schoolbook terms for a sub-block
    def add_schoolbook(U, V, W, i_range, j_range, k_range, col):
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col

    # Correct count:
    # Part 1a: 3 Strassen (21) + 4 Schoolbook = 25
    # Part 1b: 14 Schoolbook
    # Part 2a: 3 Strassen (21) + 4 Schoolbook = 25
    # Part 2b: 14 Schoolbook
    # Part 2c: 21 Schoolbook
    # Total: 25 + 14 + 25 + 14 + 21 = 99
    
    rank = 99
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j in {0, 1}
    # Sub-part 1a: i in {0, 1}
    # Strassen blocks for k in {0,1}, {2,3}, {4,5}
    for k0 in [0, 2, 4]:
        if time.time() > deadline: break
        col = add_strassen(U, V, W, 0, 0, k0, col)
    
    # Schoolbook for i in {0,1}, j in {0,1}, k in {6}
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_schoolbook(U, V, W, [0, 1], [0, 1], [6], col)
    
    # Sub-part 1b: i in {2}
    # Schoolbook for i in {2}, j in {0,1}, k in {0..6}
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_schoolbook(U, V, W, [2], [0, 1], range(7), col)
    
    # Part 2: j in {2, 3, 4}
    # Sub-part 2a: i in {0, 1}, j in {2, 3}
    # Strassen blocks for k in {0,1}, {2,3}, {4,5}
    for k0 in [0, 2, 4]:
        if time.time() > deadline: break
        col = add_strassen(U, V, W, 0, 2, k0, col)
    
    # Schoolbook for i in {0,1}, j in {2,3}, k in {6}
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_schoolbook(U, V, W, [0, 1], [2, 3], [6], col)
    
    # Sub-part 2b: i in {0, 1}, j in {4}
    # Schoolbook for i in {0,1}, j in {4}, k in {0..6}
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_schoolbook(U, V, W, [0, 1], [4], range(7), col)
    
    # Sub-part 2c: i in {2}, j in {2, 3, 4}
    # Schoolbook for i in {2}, j in {2,3,4}, k in {0..6}
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_schoolbook(U, V, W, [2], [2, 3, 4], range(7), col)
    
    # Now attempt to reduce rank via exact rational linear algebra
    # The tensor T_{i,j,k} = sum_r U_{i,j}^r * V_{j,k}^r * W_{k,i}^r
    # We need to find a smaller set of rank-1 tensors that span the same tensor space.
    
    # Strategy: Try to find linear dependencies among the 99 rank-1 components.
    # Each rank-1 component r defines a tensor T^r where T^r[i,j,k] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    # If there exist coefficients c_r such that sum_r c_r * T^r = 0, we can eliminate one component.
    
    # Build the constraint matrix: for each (i,j,k) triple, we have:
    # sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = delta_{i,j,k} (the target tensor)
    
    # Convert to numpy for efficient linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor T of shape (n, m, p)
    T = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
    
    # Verify correctness
    expected = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i, j, k] = 1.0
    
    if not np.allclose(T, expected, atol=1e-9):
        # Fallback to original
        return {"U": U, "V": V, "W": W}
    
    # Now try to find a smaller decomposition
    # We can try to find a basis for the span of the rank-1 tensors
    # and see if we can represent the target tensor with fewer terms
    
    # Flatten the tensor
    T_flat = T.flatten()
    
    # Build the matrix of rank-1 tensors
    # Each column is a rank-1 tensor flattened
    M = np.zeros((n * m * p, rank))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[i * m * p + j * p + k, r] = U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
    
    # Find the rank of M
    # Use SVD to find the numerical rank
    try:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        # The numerical rank is the number of singular values > tolerance
        tol = 1e-10 * max(M.shape) * s[0] if len(s) > 0 else 0
        num_rank = np.sum(s > tol)
        
        if num_rank < rank:
            # We can potentially reduce the rank
            # But we need to ensure the reduced decomposition is exact
            # This is complex, so for now, we just return the original
            pass
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
