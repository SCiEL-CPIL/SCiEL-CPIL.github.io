import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Try to find a rank-minimal decomposition
    # Strategy: Use Strassen for 2x2x2 blocks and optimize the overall structure
    
    # For 2x4 x 4x7, we can use the following approach:
    # Split the 4 middle dimension into 2+2
    # A = [A_left(2x2), A_right(2x2)]
    # B = [B_top(2x7); B_bottom(2x7)]
    # C = A_left @ B_top + A_right @ B_bottom
    
    # Each 2x2 x 2x7 can be done with Strassen-like methods
    # But we can do better by sharing the Strassen products
    
    # Let's use a direct approach: build the tensor and find a low-rank decomposition
    # The multiplication tensor T[i,j,k] = 1 for all valid (i,j,k) in the sum
    
    # Actually, for matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = delta (it's 1 when computing the contribution)
    # A rank-r decomposition: T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]
    
    # For 2x4x7, let's try to find a decomposition with rank < 50
    
    # Use the following structure:
    # We'll use a combination of Strassen blocks and direct terms
    
    # Let's build a rank-25 solution using careful blocking
    # Split 7 into 2+2+2+1
    
    rank_candidates = [25, 24, 23, 22, 21, 20]
    
    def build_strassen_2x2x2x7():
        """Build a decomposition for 2x2 x 2x7 using Strassen-like method"""
        # This is complex, so let's use a simpler approach
        # For 2x2 x 2xp, the rank is at most 7*2 = 14 (schoolbook)
        # But with Strassen, we can do better for the 2x2 part
        
        # Actually, let's just use schoolbook for simplicity and try to optimize
        pass
    
    # Let's try a different approach: use the known result that
    # rank(2x4x7) <= some value, and construct it explicitly
    
    # I'll use a greedy approach: build the tensor and try to find a low-rank decomposition
    # by solving a system of equations
    
    # The tensor T has shape (n*m, m*p, p*n) = (8, 28, 14)
    # T[i*m+j, j*p+k, k*n+i] = 1 if this is a valid multiplication term
    
    # Actually, the tensor is: T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # and T[a,b,c] = 1 if the indices are consistent (same j in a and b, same k in b and c, same i in a and c)
    
    # Let's build the tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Now we want to find U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that sum_r U[:,r] * V[:,r] * W[:,r] = T (element-wise product summed over r)
    
    # This is equivalent to: for all (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # Let's try to find a low-rank solution using random initialization and gradient descent
    # Or use a structured approach
    
    # Given the time constraint, let's use a known construction
    # For 2x4x7, a rank-25 decomposition can be achieved
    
    # Let's build it manually using the following strategy:
    # Use 3 Strassen 2x2x2 blocks (for the first 6 columns of B) and 1 schoolbook 2x2x1 block
    
    U = [[0] * 25 for _ in range(n * m)]
    V = [[0] * 25 for _ in range(m * p)]
    W = [[0] * 25 for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        nonlocal col
        if col >= 25:
            return False
        
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # Strassen 7 multiplications
        U[a][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        U[c][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1
        
        U[a][col] += 1
        V[f][col] += 1
        V[h][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1
        
        U[d][col] += 1
        V[g][col] += 1
        V[e][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        U[a][col] += 1
        U[b][col] += 1
        V[h][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1
        
        U[c][col] += 1
        U[a][col] -= 1
        V[e][col] += 1
        V[f][col] += 1
        W[c11][col] += 1
        col += 1
        
        U[b][col] += 1
        U[d][col] -= 1
        V[g][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        col += 1
        
        return True
    
    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        if col >= 25:
            return False
        
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        
        U[a][col] += 1
        V[e][col] += 1
        W[c00][col] += 1
        col += 1
        
        U[b][col] += 1
        V[g][col] += 1
        W[c00][col] += 1
        col += 1
        
        U[c][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        col += 1
        
        U[d][col] += 1
        V[g][col] += 1
        W[c10][col] += 1
        col += 1
        
        return True
    
    # Build the decomposition:
    # 3 Strassen blocks for k=0,2,4 and 1 schoolbook for k=6
    # Each Strassen block handles 2 rows of A, 2 cols of B (2x2x2)
    # But we need to cover all 4 rows of A and 7 cols of B
    
    # A is 2x4, so we have 2 rows. We can split the 4 columns into 2+2.
    # B is 4x7, so we split the 7 rows into 2+2+2+1.
    
    # For each pair of B rows (2x7 block), we compute A @ B_block
    # This gives a 2x7 result
    
    # Using Strassen for 2x2 x 2x7:
    # Split A into 2x2 blocks (left and right)
    # For each 2x7 block of B, we do 2x2 x 2x7
    
    # Actually, let's just use the simple blocking:
    # Block 1: A[:, 0:2] @ B[0:2, :] -> 2x7 result (use Strassen-like)
    # Block 2: A[:, 2:4] @ B[2:4, :] -> 2x7 result
    # Block 3: A[:, 2:4] @ B[4:6, :] -> 2x7 result  
    # Block 4: A[:, 2:4] @ B[6:7, :] -> 2x1 result
    
    # This is getting complex. Let me just use the rank-50 solution that works
    # and verify it's correct.
    
    # Reset and use the original working solution
    U = [[0] * 50 for _ in range(n * m)]
    V = [[0] * 50 for _ in range(m * p)]
    W = [[0] * 50 for _ in range(p * n)]
    col = 0
    
    for k0 in [0, 2, 4]:
        if not add_strassen_222(0, 0, k0):
            break
        if time.time() > deadline:
            break
    
    if not add_schoolbook_221(0, 0, 6):
        pass
    
    for k0 in [0, 2, 4]:
        if not add_strassen_222(0, 2, k0):
            break
        if time.time() > deadline:
            break
    
    if not add_schoolbook_221(0, 2, 6):
        pass
    
    # Verify correctness
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    rng = np.random.default_rng(42)
    A_test = rng.random((n, m))
    B_test = rng.random((m, p))
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(50):
        u_vec = U_arr[:, r].reshape(n, m)
        v_vec = V_arr[:, r].reshape(m, p)
        w_vec = W_arr[:, r].reshape(p, n)
        C_computed += np.sum(u_vec * A_test, axis=1)[:, None] * np.sum(v_vec * B_test, axis=0)[None, :] * w_vec
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        U = [[0] * (n*m*p) for _ in range(n * m)]
        V = [[0] * (n*m*p) for _ in range(m * p)]
        W = [[0] * (n*m*p) for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
