"""Program for 2x4 by 4x8 matrix multiplication tensor decomposition."""
import time
import sys

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0  # seconds, with safety margin from 150s limit

def _check_deadline():
    """Return True if we should stop and return best result."""
    return (time.time() - _START_TIME) > (_DEADLINE - 5.0)

def construct():
    """
    Construct a rank-56 decomposition for 2x4 by 4x8 matrix multiplication.
    Uses 8 independent 2x2 Strassen decompositions.
    """
    # Dimensions
    n, m, p = DIMENSIONS  # A: (n, m) = (2,4), B: (m, p) = (4,8), C: (n, p) = (2,8)
    
    # We split into 8 2x2 block products:
    # A is 2x4, so A[i, j] for i in {0,1}, j in {0,1,2,3}
    # B is 4x8, so B[j, k] for j in {0,1,2,3}, k in {0,...,7}
    # C is 2x8, so C[i, k] for i in {0,1}, k in {0,...,7}
    
    # The 8 2x2 products come from:
    # For each i in {0,1} (row of A and C):
    #   For each column pair (c1, c2) in {(0,1), (2,3)} for A's columns:
    #     For each column pair (d1, d2) in {(0,1), (2,3), (4,5), (6,7)} for B's columns:
    #       But that's 2 * 2 * 4 = 16, not 8.
    
    # Let me reconsider. The natural 2x2 block decomposition:
    # Split A into two 2x2 blocks: A_left (cols 0-1) and A_right (cols 2-3)
    # Split B into four 2x2 column blocks: B0 (cols 0-1), B1 (cols 2-3), B2 (cols 4-5), B3 (cols 6-7)
    # 
    # C = A * B = A_left * B_top + A_right * B_bottom
    # where B_top = [B0, B1] (rows 0-1 of B) and B_bottom = [B2, B3] (rows 2-3 of B)
    #
    # So: C[:, 0:2] = A_left[:, 0:2] * B[0:2, 0:2] + A_right[:, 0:2] * B[2:4, 0:2]
    #     C[:, 2:4] = A_left[:, 0:2] * B[0:2, 2:4] + A_right[:, 0:2] * B[2:4, 2:4]
    #     C[:, 4:6] = A_left[:, 0:2] * B[0:2, 4:6] + A_right[:, 0:2] * B[2:4, 4:6]
    #     C[:, 6:8] = A_left[:, 0:2] * B[0:2, 6:8] + A_right[:, 0:2] * B[2:4, 6:8]
    
    # That gives 8 2x2 matrix products:
    # For each of 4 column pairs of B (d in {0,1,2,3}):
    #   Product 1: A_left (2x2) * B[0:2, 2d:2d+2] (2x2)
    #   Product 2: A_right (2x2) * B[2:4, 2d:2d+2] (2x2)
    
    # Each 2x2 product uses 7 Strassen multiplications.
    # Total rank = 8 * 7 = 56.
    
    if _check_deadline():
        # Return a trivial but valid rank-64 decomposition as fallback
        n_fb, m_fb, p_fb = DIMENSIONS
        rank_fb = n_fb * m_fb * p_fb
        U_fb = [[0.0] * rank_fb for _ in range(n_fb * m_fb)]
        V_fb = [[0.0] * rank_fb for _ in range(m_fb * p_fb)]
        W_fb = [[0.0] * rank_fb for _ in range(p_fb * n_fb)]
        for i in range(n_fb):
            for j in range(m_fb):
                for k in range(p_fb):
                    r = i * m_fb * p_fb + j * p_fb + k
                    U_fb[i * m_fb + j][r] = 1.0
                    V_fb[j * p_fb + k][r] = 1.0
                    W_fb[k * n_fb + i][r] = 1.0
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]  # shape (8, 56)
    V = [[0.0] * rank for _ in range(m * p)]  # shape (32, 56)
    W = [[0.0] * rank for _ in range(p * n)]  # shape (16, 56)
    
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each of the 8 block products, we add 7 Strassen terms.
    # We'll define a helper that adds one 2x2 Strassen decomposition.
    
    def add_strassen_2x2(base_rank, A_idx, B_idx, C_idx):
        """
        Add a 2x2 Strassen decomposition starting at base_rank.
        
        A_idx: function (r, c) -> row index in U for A[r,c]
        B_idx: function (r, c) -> row index in V for B[r,c]
        C_idx: function (r, c) -> row index in W for C[r,c]
        
        Returns the next available rank index.
        """
        for s in range(7):
            col = base_rank + s
            # Determine which Strassen product this is
            # We'll handle all 7 in one go
            pass
        return base_rank + 7
    
    # Let's define the 8 block products explicitly
    # Block product b (0 to 7):
    #   b = 4*d + h where d in {0,1,2,3} is B column pair index, h in {0,1} is A half
    #   h=0: A_left = A[:, 0:2], B_rows = 0:2
    #   h=1: A_right = A[:, 2:4], B_rows = 2:4
    
    for d in range(4):  # B column pair: 0->(0,1), 1->(2,3), 2->(4,5), 3->(6,7)
        for h in range(2):  # h=0: A left half, B top rows; h=1: A right half, B bottom rows
            base = (d * 2 + h) * 7
            if h == 0:
                # A block: A[i, 0:2] for i in {0,1}
                # B block: B[0:2, 2d:2d+2]
                # C block: C[i, 2d:2d+2] for i in {0,1}
                def a_idx(r, c):
                    # A[r, c] where c is 0 or 1, actual col is c
                    return r * m + c
                def b_idx(r, c):
                    # B[r, c] where r is 0 or 1, actual row is r; c is 0 or 1, actual col is 2d+c
                    return r * p + (2 * d + c)
                def c_idx(r, c):
                    # C[r, c] where c is 0 or 1, actual col is 2d+c
                    return (2 * d + c) * n + r
            else:
                # A block: A[i, 2:4] for i in {0,1}
                # B block: B[2:4, 2d:2d+2]
                # C block: C[i, 2d:2d+2] for i in {0,1}
                def a_idx(r, c):
                    # A[r, 2+c]
                    return r * m + (2 + c)
                def b_idx(r, c):
                    # B[2+r, 2d+c]
                    return (2 + r) * p + (2 * d + c)
                def c_idx(r, c):
                    # C[r, 2d+c]
                    return (2 * d + c) * n + r
            
            # Now add the 7 Strassen products
            # M1 = (A00 + A11) * (B00 + B11)
            col = base + 0
            for r in range(2):
                for c in range(2):
                    if (r, c) == (0, 0) or (r, c) == (1, 1):
                        U[a_idx(r, c)][col] += 1.0
                        V[b_idx(r, c)][col] += 1.0
            # W for M1: C00 += M1, C11 += M1
            W[c_idx(0, 0)][col] += 1.0
            W[c_idx(1, 1)][col] += 1.0
            
            # M2 = (A10 + A11) * B00
            col = base + 1
            for (r, c, sign) in [(1, 0, 1.0), (1, 1, 1.0)]:
                U[a_idx(r, c)][col] += sign
            V[b_idx(0, 0)][col] += 1.0
            # W for M2: C10 += M2, C11 -= M2
            W[c_idx(1, 0)][col] += 1.0
            W[c_idx(1, 1)][col] -= 1.0
            
            # M3 = A00 * (B01 - B11)
            col = base + 2
            U[a_idx(0, 0)][col] += 1.0
            V[b_idx(0, 1)][col] += 1.0
            V[b_idx(1, 1)][col] -= 1.0
            # W for M3: C01 += M3, C11 += M3
            W[c_idx(0, 1)][col] += 1.0
            W[c_idx(1, 1)][col] += 1.0
            
            # M4 = A11 * (B10 - B00)
            col = base + 3
            U[a_idx(1, 1)][col] += 1.0
            V[b_idx(1, 0)][col] += 1.0
            V[b_idx(0, 0)][col] -= 1.0
            # W for M4: C00 += M4, C10 += M4
            W[c_idx(0, 0)][col] += 1.0
            W[c_idx(1, 0)][col] += 1.0
            
            # M5 = (A00 + A01) * B11
            col = base + 4
            for (r, c, sign) in [(0, 0, 1.0), (0, 1, 1.0)]:
                U[a_idx(r, c)][col] += sign
            V[b_idx(1, 1)][col] += 1.0
            # W for M5: C00 -= M5, C01 += M5
            W[c_idx(0, 0)][col] -= 1.0
            W[c_idx(0, 1)][col] += 1.0
            
            # M6 = (A10 - A00) * (B00 + B01)
            col = base + 5
            U[a_idx(1, 0)][col] += 1.0
            U[a_idx(0, 0)][col] -= 1.0
            V[b_idx(0, 0)][col] += 1.0
            V[b_idx(0, 1)][col] += 1.0
            # W for M6: C11 += M6
            W[c_idx(1, 1)][col] += 1.0
            
            # M7 = (A01 - A11) * (B10 + B11)
            col = base + 6
            U[a_idx(0, 1)][col] += 1.0
            U[a_idx(1, 1)][col] -= 1.0
            V[b_idx(1, 0)][col] += 1.0
            V[b_idx(1, 1)][col] += 1.0
            # W for M7: C00 += M7
            W[c_idx(0, 0)][col] += 1.0
    
    return {"U": U, "V": V, "W": W}
