"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
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

    # Build the 99-term solution
    rank = 99
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # i in {0,1}, j in {0,1}, k in {0,1}, {2,3}, {4,5}: Strassen
    for k0 in [0, 2, 4]:
        col = add_strassen(U, V, W, 0, 0, k0, col)
    # i in {0,1}, j in {0,1}, k in {6}: schoolbook
    col = add_schoolbook(U, V, W, [0, 1], [0, 1], [6], col)
    # i in {0,1}, j in {2,3}, k in {0,1}, {2,3}, {4,5}: Strassen
    for k0 in [0, 2, 4]:
        col = add_strassen(U, V, W, 0, 2, k0, col)
    # i in {0,1}, j in {2,3}, k in {6}: schoolbook
    col = add_schoolbook(U, V, W, [0, 1], [2, 3], [6], col)
    # i in {0,1}, j in {4}, k in {0..6}: schoolbook
    col = add_schoolbook(U, V, W, [0, 1], [4], range(7), col)
    # i in {2}, j in {0..4}, k in {0..6}: schoolbook
    col = add_schoolbook(U, V, W, [2], range(5), range(7), col)
    
    # Verify
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    # Convert to numpy for efficient verification and rank reduction
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verify correctness with numpy
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                assert abs(total - 1) < 1e-10, f"Failed at ({i},{j},{k}): got {total}"
    
    # Try to reduce rank by finding linearly dependent columns
    # Stack U, V, W into a single matrix and find the rank
    # Actually, we need to find if any column r can be expressed as a linear combination of others
    # such that the product U[:,r] * V[:,r] * W[:,r] can be reproduced by other terms.
    
    # A simpler approach: check if the tensor can be decomposed with fewer terms
    # by looking for columns that are linear combinations in a way that preserves the product.
    
    # For now, let's try a simple greedy elimination:
    # For each column r, check if it can be removed by adjusting other columns.
    # This is complex, so let's just return the 99-term solution for now.
    
    # Convert back to lists
    U_out = U_np.tolist()
    V_out = V_np.tolist()
    W_out = W_np.tolist()
    
    # Round to nearest integer/half-integer
    def round_to_half(x):
        return round(x * 2) / 2
    
    U_out = [[round_to_half(x) for x in row] for row in U_out]
    V_out = [[round_to_half(x) for x in row] for row in V_out]
    W_out = [[round_to_half(x) for x in row] for row in W_out]
    
    # Final verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = sum(U_out[i*m+j][r] * V_out[j*p+k][r] * W_out[k*n+i][r] for r in range(rank))
                assert abs(total - 1) < 1e-10, f"Failed at ({i},{j},{k}): got {total}"
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
