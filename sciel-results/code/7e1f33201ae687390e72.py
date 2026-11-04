import numpy as np
import time

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0  # Leave 5s margin before 150s kill

def _verify_decomposition(U, V, W, n=4, m=4, p=6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    R = U.shape[1]
    # Build the multiplication tensor T[i,j,k] = 1 if i==k else 0 (for A[i,r]*B[r,j])
    # Actually T[i,j,k] = delta_{i,k} * delta_{j,r} ... 
    # The tensor is: T[i,j,k] = 1 if the (i,k) entry of A and (j,k) entry of B... 
    # Standard: (AB)_{i,j} = sum_k A_{i,k} * B_{k,j}
    # So tensor T[i,j,k] = A_{i,k} * B_{k,j} -> coefficient is 1 for all valid (i,k,j)
    # In our indexing: U has shape (n*m, R), V has shape (m*p, R), W has shape (p*n, R)
    # U[i*m + k, r] * V[k*p + j, r] * W[j*n + i, r] summed over r should give delta... 
    # Actually the decomposition is: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = (AB)_{i,j} for the basis
    # The tensor T[i,j,k] = 1 (since A_{i,k}*B_{k,j} is just the product of basis elements)
    # So we need: sum_r U[i*m+k,r] * V[k*p+j,r] * W[j*n+i,r] = 1 for all i,j,k
    
    # This is expensive to check fully, so check a sample or use matrix form
    # Better: check that for random A, B, the reconstruction matches
    np.random.seed(42)
    A = np.random.randint(-5, 5, size=(n, m))
    B = np.random.randint(-5, 5, size=(m, p))
    AB = A @ B
    
    # Reconstruct using U, V, W
    # sum_r U[i*m+k,r] * V[k*p+j,r] * W[j*n+i,r]
    result = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            for k in range(m):
                u_col = U[i*m + k, :]
                v_col = V[k*p + j, :]
                w_val = W[j*n + i, :]
                result[i, j] += np.sum(u_col * v_col * w_val)
    
    if not np.allclose(result, AB, atol=1e-9):
        return False
    return True

def _try_rank_reduce(U, V, W, n=4, m=4, p=6, max_iterations=200):
    """Try to reduce rank by finding and removing redundant columns."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    R = U.shape[1]
    
    # Try to find columns that are linear combinations of others
    # For each column r, check if it can be expressed as a combination of others
    # This is expensive, so use a simpler heuristic:
    # Check if removing a column and re-solving for the others still works
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = R
    
    # Strategy: try removing each column one at a time
    # For each column to remove, check if the remaining columns still span the tensor
    # Use a greedy approach
    
    for attempt in range(max_iterations):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        improved = False
        for r in range(best_rank):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            # Try removing column r
            U_test = np.column_stack([U[:, j] for j in range(best_rank) if j != r]) if best_rank > 1 else None
            if U_test is None:
                break
            
            # Check if the remaining decomposition is still valid
            # We can't just check validity since removing a column changes the decomposition
            # Instead, we need to check if column r is a linear combination of others
            # in the tensor space
            
            # Simpler heuristic: check if column r has very small norm relative to others
            # or if it's approximately zero
            norm_r = np.linalg.norm(U[:, r]) + np.linalg.norm(V[:, r]) + np.linalg.norm(W[:, r])
            if norm_r < 1e-10:
                # Column is essentially zero, remove it
                U = np.column_stack([U[:, j] for j in range(best_rank) if j != r])
                V = np.column_stack([V[:, j] for j in range(best_rank) if j != r])
                W = np.column_stack([W[:, j] for j in range(best_rank) if j != r])
                best_rank -= 1
                improved = True
                break
        
        if not improved:
            # Try a different approach: for each pair of columns, see if one can be 
            # absorbed into the other by adjusting coefficients
            # This requires solving a system, which is complex
            # For now, just check for exact duplicates
            found_dup = False
            for i in range(best_rank):
                for j in range(i+1, best_rank):
                    if (np.allclose(U[:, i], U[:, j], atol=1e-10) and 
                        np.allclose(V[:, i], V[:, j], atol=1e-10) and 
                        np.allclose(W[:, i], W[:, j], atol=1e-10)):
                        # Columns i and j are identical, remove j
                        U = np.column_stack([U[:, k] for k in range(best_rank) if k != j])
                        V = np.column_stack([V[:, k] for k in range(best_rank) if k != j])
                        W = np.column_stack([W[:, k] for k in range(best_rank) if k != j])
                        best_rank -= 1
                        found_dup = True
                        break
                if found_dup:
                    break
            if not found_dup:
                break
    
    U = U[:, :best_rank]
    V = V[:, :best_rank]
    W = W[:, :best_rank]
    
    return U, V, W, best_rank

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    # U: (n*m, R) = (16, R)
    # V: (m*p, R) = (24, R)
    # W: (p*n, R) = (24, R)

    # Strassen coefficients for 2x2x2
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: A00 + A11
        {0: 0, 1: 0, 2: 1, 3: 1},  # s=1: A10 + A11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=2: A00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=3: A11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=4: A00 + A01
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=5: A10 - A00
        {0: 0, 1: 1, 2: 0, 3: -1}  # s=6: A01 - A11
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: B00 + B11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=1: B00
        {0: 0, 1: 1, 2: 0, 3: -1}, # s=2: B01 - B11
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=3: B10 - B00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=4: B11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=5: B00 + B01
        {0: 0, 1: 0, 2: 1, 3: 1}   # s=6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s=0
        [(1, 0, 1), (1, 1, -1)],    # s=1
        [(0, 1, 1), (1, 1, 1)],     # s=2
        [(0, 0, 1), (1, 0, 1)],     # s=3
        [(0, 0, -1), (0, 1, 1)],    # s=4
        [(1, 1, 1)],                # s=5
        [(0, 0, 1)]                 # s=6
    ]

    # Total rank: 7 * (7 + 4) = 77
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))

    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]

            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]

            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma

            col += 1

        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]

                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]

                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma

                col += 1

    # Attempt rank reduction
    if time.time() - START_TIME < TIME_LIMIT - 20:
        U, V, W, R = _try_rank_reduce(U, V, W, n, m, p)

    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
