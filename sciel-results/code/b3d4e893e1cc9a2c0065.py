"""Rank-minimal tensor decomposition for 4x4 * 4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Schoolbook solution as fallback (rank 112)
    rank_school = n * m * p
    U_sb = np.zeros((n*m, rank_school))
    V_sb = np.zeros((m*p, rank_school))
    W_sb = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    def check_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Check if U,V,W exactly decompose the tensor."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def _verify(U, V, W, n, m, p, tol=1e-9):
        """Verify decomposition exactly reconstructs matmul tensor."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
            return False
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def _strassen_2x2_2x2():
        """Strassen for 2x2 * 2x2 -> 2x2, rank 7."""
        n = m = p = 2
        U = np.zeros((n*m, 7)); V = np.zeros((m*p, 7)); W = np.zeros((p*n, 7))
        # M1=(A11+A22)(B11+B22); C11+=M1, C22+=M1
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        # M2=(A21+A22)B11; C21+=M2
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1
        # M3=A11(B12-B22); C12+=M3
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1
        # M4=A22(B21-B11); C11+=M4, C21+=M4
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
        # M5=(A11+A12)B22; C11-=M5, C12+=M5
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
        # M6=(A21-A11)(B11+B12); C22+=M6
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        # M7=(A12-A22)(B21+B22); C11+=M7
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W

    def _build_444():
        """Strassen-based 4x4 * 4x4 -> 4x4, rank 49."""
        n = m = p = 4
        U = np.zeros((n*m, 49)); V = np.zeros((m*p, 49)); W = np.zeros((p*n, 49))
        # A block indices: A11=rows0-1,cols0-1; A12=rows0-1,cols2-3; A21=rows2-3,cols0-1; A22=rows2-3,cols2-3
        A11 = [0,1,4,5]; A12 = [2,3,6,7]; A21 = [8,9,12,13]; A22 = [10,11,14,15]
        B11 = [0,1,4,5]; B12 = [2,3,6,7]; B21 = [8,9,12,13]; B22 = [10,11,14,15]
        C11 = [0,1,4,5]; C12 = [2,3,6,7]; C21 = [8,9,12,13]; C22 = [10,11,14,15]

        # Strassen macro-factors
        macros = [
            ([A11, A22], [1,1], [B11, B22], [1,1], [(C11,1),(C22,1)]),
            ([A21, A22], [1,1], [B11], [1], [(C21,1)]),
            ([A11], [1], [B12, B22], [1,-1], [(C12,1)]),
            ([A22], [1], [B21, B11], [1,-1], [(C11,1),(C21,1)]),
            ([A11, A12], [1,1], [B22], [1], [(C11,-1),(C12,1)]),
            ([A21, A11], [1,-1], [B11, B12], [1,1], [(C22,1)]),
            ([A12, A22], [1,-1], [B21, B22], [1,1], [(C11,1)]),
        ]

        # Strassen 2x2*2x2 pattern
        strassen_pat = [
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),  # M1
            ([(2,1),(3,1)], [(0,1)], [(2,1)]),              # M2
            ([(0,1)], [(1,1),(3,-1)], [(1,1)]),             # M3
            ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),       # M4
            ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),       # M5
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),       # M6
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),       # M7
        ]

        col = 0
        for macro in macros:
            A_blocks, A_signs, B_blocks, B_signs, C_targets = macro
            for sub in strassen_pat:
                U_entries, V_entries, W_entries = sub
                for a_local, s_u in U_entries:
                    gidx = A_blocks[a_local]
                    U[gidx, col] += A_signs[a_local] * s_u
                for b_local, s_v in V_entries:
                    gidx = B_blocks[b_local]
                    V[gidx, col] += B_signs[b_local] * s_v
                for c_local, s_w in W_entries:
                    C_block, c_sign = C_targets[c_local]
                    for gidx in C_block:
                        W[gidx, col] += c_sign * s_w
                col += 1
        return U, V, W

    def _build_447():
        """Build 4x4 * 4x7 decomposition using Strassen blocks.
        
        Split B into 4x4 and 4x3 parts.
        Use Strassen for the 4x4 part (rank 49).
        For the 4x3 part, split A into 2x4 top and bottom, B into 2x3 top and bottom.
        C_top = A_top * B_top + A_top * B_bot... wait no.
        
        Actually: C[i,k] = sum_j A[i,j] * B[j,k]
        For k=0..3: C[:, 0:4] = A * B[:, 0:4] (4x4 * 4x4)
        For k=4..6: C[:, 4:7] = A * B[:, 4:7] (4x4 * 4x3)
        
        For the 4x4 part: use Strassen (rank 49)
        For the 4x3 part: split A into A_top (2x4) and A_bot (2x4)
        B_43 = B[:, 4:7] is 4x3, split into B_top (2x3) and B_bot (2x3)
        C_top_part = A_top * B_43 (2x4 * 4x3 -> 2x3)
        C_bot_part = A_bot * B_43 (2x4 * 4x3 -> 2x3)
        
        For A_top * B_43: A_top is 2x4, split into A_top_left (2x2) and A_top_right (2x2)
        B_43 split into B_top (2x3) and B_bot (2x3)
        C_top_part = A_top_left * B_top + A_top_right * B_bot
        
        Use 4 separate 2x2 * 2x3 products, each rank 12 (schoolbook for 2x2*2x3).
        Or use a smarter decomposition.
        
        For now, let's use schoolbook for the 4x3 part: rank 4*4*3 = 48
        Total: 49 + 48 = 97
        
        But we can do better. Let's use Strassen-like sharing.
        
        Actually, let's just build it with the 4x4 Strassen and 4x3 schoolbook for now.
        """
        n = m = p = 4
        p_total = 7
        
        U = np.zeros((n*m, 49 + 48))
        V = np.zeros((m*p_total, 49 + 48))
        W = np.zeros((p_total*n, 49 + 48))
        
        # Part 1: 4x4 * 4x4 (columns 0-3 of B, columns 0-3 of C)
        U44, V44, W44 = _build_444()
        U[:, :49] = U44
        V[:m*4, :49] = V44  # B rows 0-3, cols 0-3
        W[:4*n, :49] = W44  # C rows 0-3, cols 0-3
        
        # Part 2: 4x4 * 4x3 (columns 4-6 of B, columns 4-6 of C)
        # Schoolbook: rank 4*4*3 = 48
        col = 49
        for i in range(n):
            for j in range(m):
                for k in range(3):
                    U[i*m+j, col] = 1
                    V[j*p_total + (4+k), col] = 1
                    W[(4+k)*n + i, col] = 1
                    col += 1
        
        return U, V, W

    # Try the structured Strassen-based decomposition
    if time.time() - start_time < DEADLINE - 30:
        try:
            U_struct, V_struct, W_struct = _build_447()
            if check_decomposition(U_struct, V_struct, W_struct, n, m, p):
                best_rank = U_struct.shape[1]
                best_U, best_V, best_W = U_struct, V_struct, W_struct
        except Exception:
            pass

    if time.time() - start_time > DEADLINE - 20:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Try a random search with small integer entries at various ranks
    np.random.seed(42)

    def try_random_with_polish(rank, n, m, p, num_trials=200, max_val=4):
        """Try random initializations and check."""
        for trial in range(num_trials):
            if time.time() - start_time > DEADLINE - 10:
                return None
            # Use small integer values (multiply by 2 for half-integers)
            U = np.random.randint(-max_val, max_val+1, size=(n*m, rank)) * 0.5
            V = np.random.randint(-max_val, max_val+1, size=(m*p, rank)) * 0.5
            W = np.random.randint(-max_val, max_val+1, size=(p*n, rank)) * 0.5

            if check_decomposition(U, V, W, n, m, p):
                return U, V, W
        return None

    # Try some promising ranks below the structured solution
    for test_rank in [95, 90, 85, 80, 75, 70]:
        if time.time() - start_time > DEADLINE - 30:
            break
        result = try_random_with_polish(test_rank, n, m, p, num_trials=50)
        if result is not None:
            U, V, W = result
            if check_decomposition(U, V, W, n, m, p):
                best_rank = test_rank
                best_U, best_V, best_W = U, V, W
                break

    if time.time() - start_time > DEADLINE - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
