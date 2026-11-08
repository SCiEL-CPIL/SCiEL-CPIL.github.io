"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns (U, V, W) lists of shape (4, 7), (4, 7), (4, 7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen's 7 products for 2x2 multiplication
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # U rows: (0,0), (0,1), (1,0), (1,1)
    # V rows: (0,0), (0,1), (1,0), (1,1)
    # W rows: (0,0), (0,1), (1,0), (1,1)
    
    # M1 = (a11+a22)(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a21+a22)b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = a11(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = a22(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (a11+a12)b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a12-a22)(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1
    
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(len(U[0])):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    deadline = time.time() + 110  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to build a Strassen-based decomposition
    # For 4x5x6, we can use a tiling approach
    # Split into blocks and apply Strassen where possible
    
    # Approach: Use 2x2 Strassen on the 4x4 part, handle remaining parts separately
    # 4x5x6 = (4x4x4) + (4x1x4) + (4x4x2) + (4x1x2)
    # But we need to be more careful about the dimensions
    
    # Let's try a direct construction using Strassen for the 4x4x4 core
    # and schoolbook for the remaining parts
    
    # 4x4x4: Strassen gives rank 7 (using 2x2 Strassen recursively, but 4x4 needs more)
    # Actually, for 4x4x4, the best known is 49 (using 2x2 Strassen tiling)
    # Let's use the 2x2 Strassen tiling for 4x4x4
    
    def build_4x4x4_strassen():
        """
        Build Strassen decomposition for 4x4 * 4x4 -> 4x4.
        Uses 2x2 Strassen tiling: split into four 2x2 blocks.
        Rank = 7 * 4 = 28? No, that's not right.
        
        Actually, for 4x4x4 using 2x2 Strassen:
        We have 4 blocks of A (each 2x2), 4 blocks of B (each 2x2)
        The 2x2 Strassen has 7 multiplications
        For 4x4, we need to combine the 2x2 blocks
        
        Let me think more carefully.
        A = [[A11, A12], [A21, A22]] where each is 2x2
        B = [[B11, B12], [B21, B22]] where each is 2x2
        C = A*B where each C block is 2x2
        
        C11 = A11*B11 + A12*B21
        C12 = A11*B12 + A12*B22
        C21 = A21*B11 + A22*B21
        C22 = A21*B12 + A22*B22
        
        Each of these is a 2x2*2x2->2x2 multiplication
        Using Strassen for each: 7 multiplications each
        Total: 4 * 7 = 28 multiplications
        
        But wait, we can share some computations.
        Actually, the standard 4x4 Strassen tiling gives rank 49
        (7^2 = 49, using recursive Strassen)
        
        Let me just use schoolbook for 4x4x4 = 64, which is worse.
        Or use the 2x2 Strassen directly: 28 multiplications.
        """
        # For 4x4x4, use 2x2 Strassen tiling
        # This gives rank 28
        # Each Strassen multiplication handles a 2x2*2x2->2x2
        # We have 4 such multiplications (one for each output block)
        
        U = [[0]*28 for _ in range(16)]
        V = [[0]*28 for _ in range(16)]
        W = [[0]*28 for _ in range(16)]
        
        # For each output block (ci, cj), we need to compute
        # C[cj*2+cj2, cj*2+cj2] for cj2 in {0,1}
        # This involves A[ci*2+ai, :]*B[:, cj*2+bj]
        
        # Let's use Strassen for each 2x2 block multiplication
        # Block (ci, cj) of C = sum over blocks
        
        # Actually, let me just do schoolbook for 4x4x4 = 64
        # And then see if I can improve
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        
        return U, V, W, 64
    
    # For now, use schoolbook for the whole thing
    # The key insight is that we need a better decomposition
    
    # Let's try a different approach: use the fact that 4x5x6 can be
    # decomposed using Strassen on sub-blocks
    
    # For 4x5x6:
    # Split A into 4x4 and 4x1
    # Split B into 4x6 and 1x6
    # C = A4x4 * B4x6 + A4x1 * B1x6
    
    # A4x4 * B4x6: this is 4x4 times 4x6
    # We can split B4x6 into two 4x3 blocks
    # Or use Strassen on the 4x4 part
    
    # This is getting complex. Let me try a numerical search for a low-rank
    # decomposition with integer/half-integer entries.
    
    # Actually, let me try a simpler approach: use the known result that
    # 4x4x4 can be done in rank 49 (Strassen), and build from there.
    
    # For 4x4x4, the Strassen rank is 49 (using recursive 2x2 Strassen)
    # Let me implement that properly.
    
    def strassen_4x4x4():
        """
        Recursive Strassen for 4x4x4.
        Split into 2x2 blocks, use 2x2 Strassen for each.
        Rank = 7 * 7 = 49
        """
        # The 2x2 Strassen has 7 multiplications
        # For 4x4, we have a 2x2 grid of 2x2 blocks
        # Each block multiplication uses 7 Strassen multiplications
        # Total: 4 * 7 = 28? No...
        
        # Actually, the recursive Strassen for 4x4:
        # A = [[A11, A12], [A21, A22]] (each 2x2)
        # B = [[B11, B12], [B21, B22]] (each 2x2)
        # C = A*B (each C block is 2x2)
        
        # C11 = A11*B11 + A12*B21
        # C12 = A11*B12 + A12*B22
        # C21 = A21*B11 + A22*B21
        # C22 = A21*B12 + A22*B22
        
        # Each Ci,j is a 2x2*2x2->2x2, which uses 7 Strassen mults
        # Total: 4 * 7 = 28 multiplications
        
        # But we can share: the Strassen products involve sums/differences
        # of the 2x2 blocks, which are themselves 2x2 matrices
        
        # Let me just use 28 multiplications (4 blocks * 7 Strassen each)
        
        U = [[0]*28 for _ in range(16)]
        V = [[0]*28 for _ in range(16)]
        W = [[0]*28 for _ in range(16)]
        
        # For each output block (ci, cj) where ci,cj in {0,1}
        # Ci,j = Ai,1*B1,j + Ai,2*B2,j
        # Each term is a 2x2*2x2->2x2 multiplication
        
        # Let me index the 2x2 Strassen products
        # For multiplication of two 2x2 matrices X*Y->Z:
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y01-Y11)
        # M4 = X11(Y10-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # For 4x4, the input blocks are 2x2 matrices
        # A block (bi, bj) has entries A[bi*2+i2, bj*2+j2] for i2,j2 in {0,1}
        
        # This is getting very complex. Let me just use schoolbook and
        # try to find improvements through search.
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        
        return U, V, W, 64
    
    # Given the complexity, let me try a numerical optimization approach
    # to find a low-rank decomposition
    
    # Try various target ranks
    for target_rank in [100, 96, 90, 84, 80, 78, 72, 70, 66, 64, 60]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        np.random.seed(42 + target_rank)
        
        # Initialize with random values
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Iterate to satisfy constraints
        max_iters = 20
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Check current error
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        errors.append(s - 1.0)
            errors = np.array(errors)
            max_err = np.max(np.abs(errors))
            
            if max_err < 1e-6:
                # Found a solution, try to round to integers/half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify the rounded solution
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U_int[i*m+j, :], V_int[j*p+k, :] * W_int[k*n+i, :])
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    U_list = [[U_int[i, c] for c in range(target_rank)] for i in range(n * m)]
                    V_list = [[V_int[i, c] for c in range(target_rank)] for i in range(m * p)]
                    W_list = [[W_int[i, c] for c in range(target_rank)] for i in range(p * n)]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    break
            
            # Adjust one random variable to reduce error
            idx = np.random.randint(len(errors))
            idx_flat = idx
            k = idx_flat % p
            idx_flat //= p
            j = idx_flat % m
            idx_flat //= m
            i = idx_flat
            
            err = errors[idx]
            if abs(err) < 1e-10:
                continue
            
            grad = V[j*p+k, :] * W[k*n+i, :]
            if np.linalg.norm(grad) < 1e-10:
                continue
            step = err / (np.dot(grad, grad) + 1e-10)
            U[i*m+j, :] -= step * grad
        
        if time.time() > deadline:
            break
    
    return best_result
# EVOLVE-BLOCK-END
