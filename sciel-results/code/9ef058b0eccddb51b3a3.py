"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    U_dim = n * m  # 12
    V_dim = m * p  # 24
    W_dim = p * n  # 18
    
    # Build target tensor for verification
    target = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m + j, j*p + k, k*n + i] = 1.0
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the target tensor."""
        M = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
        for r in range(U.shape[1]):
            M += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(M, target, atol=1e-9)
    
    def to_integer_list(M):
        """Convert numpy array to list of lists with integer values."""
        return [[int(round(x)) for x in row] for row in M]
    
    def schoolbook_factors(n, m, p):
        """Return U, V, W for schoolbook decomposition of n x m * m x p."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Start with schoolbook solution as baseline
    U_school, V_school, W_school = schoolbook_factors(n, m, p)
    best_rank = n * m * p  # 72
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Strategy: Use block decomposition with Strassen-like methods
    # For 3x4 * 4x6:
    # Split m=4 into 2+2
    # A = [A1 | A2] where A1 is 3x2, A2 is 3x2
    # B = [B1; B2] where B1 is 2x6, B2 is 2x6
    # C = A1*B1 + A2*B2
    
    # For each 3x2 * 2x6 subproblem:
    # Split n=3 into 1+2, p=6 into 3+3
    # A = [A_top (1x2); A_bot (2x2)]
    # B = [B_left (2x3), B_right (2x3)]
    # Result:
    #   C_top_left = A_top * B_left (1x2 * 2x3, rank 6)
    #   C_top_right = A_top * B_right (1x2 * 2x3, rank 6)
    #   C_bot_left = A_bot * B_left (2x2 * 2x3, rank 11 with Strassen)
    #   C_bot_right = A_bot * B_right (2x2 * 2x3, rank 11 with Strassen)
    # Total per 3x2*2x6: 6+6+11+11 = 34
    # Total for 3x4*4x6: 34+34 = 68
    
    # Now construct the 68-rank solution explicitly
    
    def strassen_2x2_factors():
        """
        Return U, V, W for Strassen decomposition of 2x2 * 2x2.
        U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
        where indices are:
        U: (0,0), (0,1), (1,0), (1,1)
        V: (0,0), (0,1), (1,0), (1,1)
        W: (0,0), (0,1), (1,0), (1,1)
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)*(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A21+A22)*B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = 1
        
        # M3 = A11*(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4 = A22*(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[0, 3] = 1
        
        # M5 = (A11+A12)*B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[3, 4] = 1; W[2, 4] = 1
        
        # M6 = (A21-A11)*(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1
        
        # M7 = (A12-A22)*(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[1, 6] = 1
        
        return U, V, W
    
    def build_2x2_2x3_factors():
        """
        Build factors for 2x2 * 2x3 using Strassen on the 2x2 part + schoolbook for 2x1.
        A is 2x2, B is 2x3 = [B_left (2x1), B_right (2x2)]
        Result C is 2x3 = [C_left (2x1), C_right (2x2)]
        
        C_left = A * B_left (2x2 * 2x1, rank 4)
        C_right = A * B_right (2x2 * 2x2, rank 7 with Strassen)
        Total rank: 11
        """
        # U_dim for 2x2 is 4, V_dim for 2x3 is 6, W_dim for 2x3 is 6
        # But we need to map to the local coordinate system
        
        # For 2x2 * 2x1:
        # A[i,j] at i*2+j, B[j,k] at j*1+k = j+k, C[k,i] at k*2+i
        U_2x1 = np.zeros((4, 4))
        V_2x1 = np.zeros((2, 4))
        W_2x1 = np.zeros((2, 4))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(1):
                    U_2x1[i*2+j, col] = 1
                    V_2x1[j*1+k, col] = 1
                    W_2x1[k*2+i, col] = 1
                    col += 1
        
        # For 2x2 * 2x2 with Strassen:
        U_s, V_s, W_s = strassen_2x2_factors()
        
        # Combine: total rank = 4 + 7 = 11
        U_total = np.zeros((4, 11))
        V_total = np.zeros((6, 11))
        W_total = np.zeros((6, 11))
        
        # First 4 columns: 2x2 * 2x1 (left part)
        # B is 2x3, so B_left is columns 0-1 (j*3+k for j in 0-1, k=0)
        # Actually B[j,k] at position j*3+k
        # B_left: j in 0-1, k=0 -> positions 0, 3
        # B_right: j in 0-1, k in 1-2 -> positions 1,2,4,5
        
        # For C_left (2x1): C[k,i] at position k*2+i
        # For C_right (2x2): C[k,i] at position k*3+i
        
        # U is same for both: A[i,j] at i*2+j
        U_total[:, 0:4] = U_2x1
        
        # V for left part: B[j,0] at j*3+0 = j*3
        V_total[0, 0] = 1; V_total[3, 1] = 1  # j=0,k=0 -> pos 0; j=1,k=0 -> pos 3
        V_total[0, 2] = 1; V_total[3, 3] = 1
        # Wait, let me redo this. V_2x1 has shape (2, 4) where row j corresponds to B[j,0]
        # In the 2x3 matrix, B[j,k] is at position j*3+k
        # So B[0,0] is at 0, B[1,0] is at 3
        V_total[0, 0:4] = V_2x1[0, :]
        V_total[3, 0:4] = V_2x1[1, :]
        
        # W for left part: C[k,i] at k*2+i
        # In the 2x3 result, C[k,i] for the left part (k=0) is at position 0*3+i = i
        # So C[0,0] at 0, C[0,1] at 1
        W_total[0:2, 0:4] = W_2x1[:, :]
        
        # Next 7 columns: 2x2 * 2x2 (right part)
        # B_right: B[j,k] for j in 0-1, k in 1-2
        # Positions: j*3+k -> (0,1)=1, (0,2)=2, (1,1)=4, (1,2)=5
        # In Strassen, B[j',k'] where j' in 0-1, k' in 0-1 corresponds to B[j, k'+1]
        # So B_strassen[0,0] = B[0,1] at pos 1
        # B_strassen[0,1] = B[0,2] at pos 2
        # B_strassen[1,0] = B[1,1] at pos 4
        # B_strassen[1,1] = B[1,2] at pos 5
        
        U_total[:, 4:11] = U_s
        
        # Map V_s indices: V_s[j'*2+k', :] where j' in 0-1, k' in 0-1
        # V_s[0, :] -> B_strassen[0,0] -> B[0,1] -> pos 1
        # V_s[1, :] -> B_strassen[0,1] -> B[0,2] -> pos 2
        # V_s[2, :] -> B_strassen[1,0] -> B[1,1] -> pos 4
        # V_s[3, :] -> B_strassen[1,1] -> B[1,2] -> pos 5
        V_total[1, 4:11] = V_s[0, :]
        V_total[2, 4:11] = V_s[1, :]
        V_total[4, 4:11] = V_s[2, :]
        V_total[5, 4:11] = V_s[3, :]
        
        # Map W_s indices: W_s[k'*2+i', :] where k' in 0-1, i' in 0-1
        # W_s[0, :] -> C_strassen[0,0] -> C[0,1] -> pos 1
        # W_s[1, :] -> C_strassen[0,1] -> C[0,2] -> pos 2
        # W_s[2, :] -> C_strassen[1,0] -> C[1,1] -> pos 4
        # W_s[3, :] -> C_strassen[1,1] -> C[1,2] -> pos 5
        W_total[1, 4:11] = W_s[0, :]
        W_total[2, 4:11] = W_s[1, :]
        W_total[4, 4:11] = W_s[2, :]
        W_total[5, 4:11] = W_s[3, :]
        
        return U_total, V_total, W_total
    
    def build_3x2_2x6_factors():
        """
        Build factors for 3x2 * 2x6.
        A is 3x2, B is 2x6, C is 6x3 (indexed as k*3+i)
        
        Split A into A_top (1x2) and A_bot (2x2)
        Split B into B_left (2x3) and B_right (2x3)
        
        C_top_left = A_top * B_left (1x2 * 2x3, rank 6)
        C_top_right = A_top * B_right (1x2 * 2x3, rank 6)
        C_bot_left = A_bot * B_left (2x2 * 2x3, rank 11)
        C_bot_right = A_bot * B_right (2x2 * 2x3, rank 11)
        Total: 34
        """
        U_total = np.zeros((6, 34))  # 3*2 = 6
        V_total = np.zeros((12, 34))  # 2*6 = 12
        W_total = np.zeros((18, 34))  # 6*3 = 18
        
        # A_top: rows 0, cols 0-1 -> A[0,j] at position 0*2+j = j
        # A_bot: rows 1-2, cols 0-1 -> A[i,j] at position i*2+j
        # B_left: rows 0-1, cols 0-2 -> B[j,k] at position j*6+k
        # B_right: rows 0-1, cols 3-5 -> B[j,k] at position j*6+k
        # C_top_left: C[k,0] for k in 0-2 -> position k*3+0
        # C_top_right: C[k,0] for k in 3-5 -> position k*3+0
        # C_bot_left: C[k,i] for i in 1-2, k in 0-2 -> position k*3+i
        # C_bot_right: C[k,i] for i in 1-2, k in 3-5 -> position k*3+i
        
        col = 0
        
        # 1. A_top * B_left (1x2 * 2x3, rank 6)
        # A_top[i,j] = A[0,j] at pos j
        # B_left[j,k] = B[j,k] at pos j*6+k
        # C[k,0] at pos k*3+0
        for j in range(2):
            for k in range(3):
                U_total[j, col] = 1
                V_total[j*6+k, col] = 1
                W_total[k*3+0, col] = 1
                col += 1
        
        # 2. A_top * B_right (1x2 * 2x3, rank 6)
        # B_right[j,k] = B[j,k+3] at pos j*6+(k+3)
        for j in range(2):
            for k in range(3):
                U_total[j, col] = 1
                V_total[j*6+(k+3), col] = 1
                W_total[(k+3)*3+0, col] = 1
                col += 1
        
        # 3. A_bot * B_left (2x2 * 2x3, rank 11)
        U_bl, V_bl, W_bl = build_2x2_2x3_factors()
        # A_bot[i,j] = A[i+1,j] at pos (i+1)*2+j
        # B_left[j,k] = B[j,k] at pos j*6+k
        # C_bot_left[k,i] = C[k,i+1] at pos k*3+(i+1)
        for r in range(11):
            for a in range(4):
                if U_bl[a, r] != 0:
                    U_total[(a//2+1)*2+(a%2), col+r] = U_bl[a, r]
            for b in range(6):
                if V_bl[b, r] != 0:
                    V_total[b, col+r] = V_bl[b, r]
            for c in range(6):
                if W_bl[c, r] != 0:
                    # c is index in 2x3 result, which is k*2+i
                    k = c // 2
                    i = c % 2
                    W_total[k*3+(i+1), col+r] = W_bl[c, r]
        col += 11
        
        # 4. A_bot * B_right (2x2 * 2x3, rank 11)
        U_br, V_br, W_br = build_2x2_2x3_factors()
        # B_right[j,k] = B[j,k+3] at pos j*6+(k+3)
        # C_bot_right[k,i] = C[k+3,i+1] at pos (k+3)*3+(i+1)
        for r in range(11):
            for a in range(4):
                if U_br[a, r] != 0:
                    U_total[(a//2+1)*2+(a%2), col+r] = U_br[a, r]
            for b in range(6):
                if V_br[b, r] != 0:
                    # b is index in 2x3, which is j*3+k
                    j = b // 3
                    k = b % 3
                    V_total[j*6+(k+3), col+r] = V_br[b, r]
            for c in range(6):
                if W_br[c, r] != 0:
                    # c is index in 2x3 result, which is k*2+i
                    k = c // 2
                    i = c % 2
                    W_total[(k+3)*3+(i+1), col+r] = W_br[c, r]
        col += 11
        
        return U_total, V_total, W_total
    
    # Build the full 3x4 * 4x6 decomposition
    # Split A (3x4) into A_left (3x2) and A_right (3x2)
    # Split B (4x6) into B_top (2x6) and B_bottom (2x6)
    
    U_left, V_left, W_left = build_3x2_2x6_factors()
    U_right, V_right, W_right = build_3x2_2x6_factors()
    
    U_full = np.zeros((12, 68))
    V_full = np.zeros((24, 68))
    W_full = np.zeros((18, 68))
    
    # Left part: A_left[i,j] = A[i,j] at pos i*4+j
    # B_top[j,k] = B[j,k] at pos j*6+k
    # C[k,i] at pos k*3+i
    for r in range(34):
        for a in range(6):
            if U_left[a, r] != 0:
                # a is index in 3x2, which is i*2+j
                i = a // 2
                j = a % 2
                U_full[i*4+j, r] = U_left[a, r]
        for b in range(12):
            if V_left[b, r] != 0:
                # b is index in 2x6, which is j*6+k
                V_full[b, r] = V_left[b, r]
        for c in range(18):
            if W_left[c, r] != 0:
                W_full[c, r] = W_left[c, r]
    
    # Right part: A_right[i,j] = A[i,j+2] at pos i*4+(j+2)
    # B_bottom[j,k] = B[j+2,k] at pos (j+2)*6+k
    for r in range(34):
        for a in range(6):
            if U_right[a, r] != 0:
                # a is index in 3x2, which is i*2+j
                i = a // 2
                j = a % 2
                U_full[i*4+(j+2), 34+r] = U_right[a, r]
        for b in range(12):
            if V_right[b, r] != 0:
                # b is index in 2x6, which is j*6+k
                j = b // 6
                k = b % 6
                V_full[(j+2)*6+k, 34+r] = V_right[b, r]
        for c in range(18):
            if W_right[c, r] != 0:
                W_full[c, 34+r] = W_right[c, r]
    
    # Verify the 68-rank solution
    if verify(U_full, V_full, W_full):
        best_rank = 68
        best_U, best_V, best_W = U_full, V_full, W_full
    
    # If the 68-rank solution doesn't verify, fall back to schoolbook
    if not verify(best_U, best_V, best_W):
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = 72
    
    return {"U": to_integer_list(best_U), "V": to_integer_list(best_V), "W": to_integer_list(best_W)}
# EVOLVE-BLOCK-END
