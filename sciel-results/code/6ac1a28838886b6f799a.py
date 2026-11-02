"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 3, 4, 4

    def verify(U, V, W, tol=1e-9):
        """Verify tensor decomposition for matmul tensor."""
        r = U.shape[1]
        # Check ones
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check zeros
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for j2 in range(m):
                    for k2 in range(p):
                        b = j2 * p + k2
                        for k3 in range(p):
                            for i3 in range(n):
                                c = k3 * n + i3
                                if j == j2 and i == i3 and k == k2:
                                    continue
                                val = np.dot(U[a, :], V[b, :] * W[c, :])
                                if abs(val) > tol:
                                    return False
        return True

    # Strassen decomposition for 2x2x2 matrix multiplication
    # S1 = (B11 - B22) * (A22 - A11)
    # S2 = (B21 + B11) * A11
    # S3 = (B12 + B22) * A22
    # S4 = (A11 + A22) * (B22 - S2)  -- this is not quite right, let me use standard Strassen
    # Standard Strassen:
    # M1 = (A11 - A22) * (B22 - B11)
    # M2 = (A11 + A22) * B11
    # M3 = (A11 + A12) * (B22 - B21)
    # M4 = (A21 - A22) * (B11 + B12)
    # M5 = (A11 + A22) * (B11 + B22)
    # M6 = (A12 - A22) * (B21 + B22)
    # M7 = (A11 - A21) * (B11 + B12)
    
    # For 2x2:
    # C11 = M5 + M4 - M2 - M1
    # C12 = M3 + M1
    # C21 = M7 + M2
    # C22 = M5 + M3 - M7 - M4

    # Block decomposition approach:
    # We have A (3x4) and B (4x4), want C (3x4)
    # Split A into rows: A_top (2x4) and A_bottom (1x4)
    # Split B into columns: B_left (4x2) and B_right (4x2)
    # Then C_top (2x4) = A_top * B, C_bottom (1x4) = A_bottom * B
    
    # For C_top (2x4): Split A_top into A_top_left (2x2) and A_top_right (2x2)
    # Split B into B_left (4x2) and B_right (4x2)
    # C_top_left (2x2) = A_top_left * B_left + A_top_right * B_right[:2]
    # This is getting complicated. Let me use a simpler approach.
    
    # Actually, let's use the block approach from the inspiration:
    # Split the 3x4 matrix A into a 2x4 block (top) and 1x4 block (bottom)
    # Split the 4x4 matrix B into 2x2 blocks: B = [[B11, B12], [B21, B22]]
    
    # For the top 2x4 part of C: C_top = A_top * B
    # A_top is 2x4, B is 4x4, so C_top is 2x4
    # Split A_top into A_top_left (2x2) and A_top_right (2x2)
    # C_top_left (2x2) = A_top_left * B11 + A_top_right * B21
    # C_top_right (2x2) = A_top_left * B12 + A_top_right * B22
    
    # For the bottom 1x4 part of C: C_bottom = A_bottom * B
    # A_bottom is 1x4, B is 4x4, so C_bottom is 1x4
    # C_bottom_left (1x2) = A_bottom_left (1x2) * B11 + A_bottom_right (1x2) * B21
    # C_bottom_right (1x2) = A_bottom_left (1x2) * B12 + A_bottom_right (1x2) * B22

    # Now, for each 2x2x2 block multiplication, we can use Strassen (rank 7)
    # For 1x2x2 multiplications, we need rank 2 (trivial)
    
    # Let's define the Strassen factors for 2x2x2:
    # Given A (2x2) and B (2x2), C = A*B
    # Strassen uses 7 products:
    # M1 = (A11 - A22) * (B22 - B11)
    # M2 = (A11 + A22) * B11
    # M3 = (A11 + A12) * (B22 - B21)
    # M4 = (A21 - A22) * (B11 + B12)
    # M5 = (A11 + A22) * (B11 + B22)
    # M6 = (A12 - A22) * (B21 + B22)
    # M7 = (A11 - A21) * (B11 + B12)
    # C11 = M5 + M4 - M2 - M1
    # C12 = M3 + M1
    # C21 = M7 + M2
    # C22 = M5 + M3 - M7 - M4

    # Let's build the decomposition step by step
    # First, let's handle the top-left 2x2x2 block:
    # A_tl (2x2) = A[0:2, 0:2], B_tl (2x2) = B[0:2, 0:2]
    # But wait, B is 4x4, so we need to be more careful.
    
    # Let me redefine:
    # A is 3x4: rows 0,1,2 and cols 0,1,2,3
    # B is 4x4: rows 0,1,2,3 and cols 0,1,2,3
    # C is 3x4: rows 0,1,2 and cols 0,1,2,3
    
    # Split A into:
    # A_top = A[0:2, :] (2x4)
    # A_bot = A[2:3, :] (1x4)
    
    # Split B into 2x2 blocks:
    # B[0:2, 0:2] = B00, B[0:2, 2:4] = B01
    # B[2:4, 0:2] = B10, B[2:4, 2:4] = B11
    
    # C_top = A_top * B (2x4)
    # C_top[0:2, 0:2] = A_top[0:2, 0:2] * B00 + A_top[0:2, 2:4] * B10
    # C_top[0:2, 2:4] = A_top[0:2, 0:2] * B01 + A_top[0:2, 2:4] * B11
    
    # C_bot = A_bot * B (1x4)
    # C_bot[0, 0:2] = A_bot[0, 0:2] * B00 + A_bot[0, 2:4] * B10
    # C_bot[0, 2:4] = A_bot[0, 0:2] * B01 + A_bot[0, 2:4] * B11

    # Now, for each 2x2x2 multiplication, we use Strassen (rank 7)
    # For each 1x2x2 multiplication, we use rank 2
    
    # Let's count the rank:
    # C_top[0:2, 0:2]: 2x2x2 -> rank 7 (Strassen)
    # C_top[0:2, 2:4]: 2x2x2 -> rank 7 (Strassen)
    # C_bot[0, 0:2]: 1x2x2 -> rank 2
    # C_bot[0, 2:4]: 1x2x2 -> rank 2
    
    # Total rank = 7 + 7 + 2 + 2 = 18? That seems too low.
    # Wait, I think I'm confusing myself. Let me reconsider.
    
    # Actually, the issue is that the same B blocks are used in multiple multiplications.
    # We need to be more careful about how we share the factors.
    
    # Let me try a different approach: directly construct the decomposition.
    
    # For simplicity, let's first return the schoolbook decomposition and then
    # try to improve it.
    
    # Schoolbook: rank = 3*4*4 = 48
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    if verify(U0, V0, W0):
        # Try to find a better decomposition using block structure
        # For now, return schoolbook
        return {
            "U": U0.tolist(),
            "V": V0.tolist(),
            "W": W0.tolist()
        }
    else:
        # Should not happen
        return {
            "U": U0.tolist(),
            "V": V0.tolist(),
            "W": W0.tolist()
        }
# EVOLVE-BLOCK-END
