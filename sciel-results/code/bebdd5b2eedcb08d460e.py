"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def verify_full(U, V, W):
        """Verify the tensor decomposition is exact by checking all entries."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        R = U.shape[1]
        # Build the reconstructed tensor
        T_rec = np.zeros((nm, mp, pn))
        for r in range(R):
            T_rec += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        # Build the target tensor
        T_target = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        return np.allclose(T_rec, T_target, atol=1e-9)
    
    def naive_decomp():
        """Naive rank-16 decomposition."""
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def strassen_2x2_block(V_indices, W_indices, base_r):
        """
        Generate 7 Strassen terms for a 2x2*2x2 block.
        V_indices: [v00, v01, v10, v11] - V row indices for b00, b01, b10, b11
        W_indices: [w00, w01, w10, w11] - W row indices for c00, c01, c10, c11
        base_r: starting rank index
        Returns list of (u_vec, v_vec, w_vec) tuples where each is a dict {row: value}
        """
        terms = []
        
        # Standard Strassen for C = A*B where A=[[a00,a01],[a10,a11]], B=[[b00,b01],[b10,b11]]
        # C00 = a00*b00 + a01*b10
        # C01 = a00*b01 + a01*b11
        # C10 = a10*b00 + a11*b10
        # C11 = a10*b01 + a11*b11
        
        # Using the correct Strassen decomposition:
        # M1 = (a00 + a11)(b00 + b11)
        # M2 = (a10 + a11)(b00)
        # M3 = a00(b10 - b11)
        # M4 = a11(b01 - b00)
        # M5 = (a00 + a01)(b10)
        # M6 = (a10 - a00)(b00 + b01)
        # M7 = (a01 - a11)(b10 + b11)
        
        # C00 = M1 - M4 + M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Let me verify these formulas numerically first.
        # Actually, let me use a different well-known correct set:
        
        # From the literature, one correct Strassen decomposition is:
        # M1 = (a00 + a11)(b00 + b11)
        # M2 = (a10 + a11)(b00)
        # M3 = a00(b10 - b11)
        # M4 = a11(b01 - b00)
        # M5 = (a00 + a01)(b10)
        # M6 = (a10 - a00)(b00 + b01)
        # M7 = (a01 - a11)(b10 + b11)
        
        # C00 = M1 - M4 + M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Let me verify C00 = M1 - M4 + M5:
        # M1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
        # M4 = a11(b01-b00) = a11*b01 - a11*b00
        # M5 = (a00+a01)*b10 = a00*b10 + a01*b10
        # M1 - M4 + M5 = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a11*b01 + a11*b00 + a00*b10 + a01*b10
        # = a00*b00 + a00*b11 + 2*a11*b00 + a11*b11 - a11*b01 + a00*b10 + a01*b10
        # Expected: a00*b00 + a01*b10
        # This is NOT matching. There's an error in the formula.
        
        # Let me try another well-known correct Strassen:
        # M1 = (a11 - a01)(b01 + b11)
        # M2 = (a00 + a11)(b00 + b11)
        # M3 = (a10 + a11)(b10 + b11)
        # M4 = a00(b01 - b11)
        # M5 = a11(b00 - b10)
        # M6 = (a00 + a01)b10
        # M7 = (a10 + a11)b01
        # C00 = M1 + M4 + M5 - M7
        # C01 = M6 + M7
        # C10 = M3 - M5 - M1
        # C11 = M2 - M3 - M4 + M1
        
        # Verify C00 = M1 + M4 + M5 - M7:
        # M1 = (a11-a01)(b01+b11) = a11*b01 + a11*b11 - a01*b01 - a01*b11
        # M4 = a00(b01-b11) = a00*b01 - a00*b11
        # M5 = a11(b00-b10) = a11*b00 - a11*b10
        # M7 = (a10+a11)*b01 = a10*b01 + a11*b01
        # M1 + M4 + M5 - M7 = a11*b01 + a11*b11 - a01*b01 - a01*b11 + a00*b01 - a00*b11 + a11*b00 - a11*b10 - a10*b01 - a11*b01
        # = a00*b01 - a00*b11 + a11*b11 - a01*b01 - a01*b11 + a11*b00 - a11*b10 - a10*b01
        # Expected: a00*b00 + a01*b10
        # Still not matching. I clearly have the convention wrong.
        
        # The issue is that different sources use different conventions for A and B.
        # Let me just use a NUMERICAL approach to find the correct coefficients.
        
        # Actually, let me just directly construct the rank-14 by using the naive
        # decomposition and then verifying it works. Then I'll try to reduce the rank.
        
        # For now, let me return the naive rank-16 and a verified rank-14.
        
        # Let me try a completely different approach: just use the naive rank-16
        # and verify it, then try to find a better decomposition.
        
        # Actually, the simplest correct approach is:
        # For each output c0k = a00*b0k + a01*b1k and c1k = a10*b0k + a11*b1k
        # We can write this as 4 terms per k, giving 16 total.
        # Or we can use Strassen to reduce to 7 per 2x2 block, giving 14 total.
        
        # Let me just return the naive decomposition for now and verify it.
        # The key insight is that the naive rank-16 is always correct.
        
        return None  # Will be replaced by the actual construction below
    
    def construct_rank14():
        """
        Construct the rank-14 decomposition using two independent Strassen blocks.
        Block 1: B columns 0,1 (V indices 0,1,4,5; W indices 0,1,2,3)
        Block 2: B columns 2,3 (V indices 2,3,6,7; W indices 4,5,6,7)
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # I'll use a verified Strassen decomposition.
        # Let me define the 7 Strassen products explicitly and verify numerically.
        
        # For a 2x2*2x2 multiplication with A=[[a00,a01],[a10,a11]], B=[[b00,b01],[b10,b11]]:
        # The 7 products are:
        # P1 = (a00 + a11)(b00 + b11)
        # P2 = (a10 + a11)(b00)
        # P3 = a00(b10 - b11)
        # P4 = a11(b01 - b00)
        # P5 = (a00 + a01)(b10)
        # P6 = (a10 - a00)(b00 + b01)
        # P7 = (a01 - a11)(b10 + b11)
        
        # And the outputs are:
        # c00 = P1 - P4 + P5
        # c01 = P3 + P5
        # c10 = P2 + P4
        # c11 = P1 - P2 + P3 + P7
        
        # Let me verify this numerically with specific values.
        # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        # A*B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        
        # P1 = (1+4)(5+8) = 5*13 = 65
        # P2 = (3+4)(5) = 7*5 = 35
        # P3 = 1*(7-8) = 1*(-1) = -1
        # P4 = 4*(6-5) = 4*1 = 4
        # P5 = (1+2)*7 = 3*7 = 21
        # P6 = (3-1)(5+6) = 2*11 = 22
        # P7 = (2-4)(7+8) = (-2)*15 = -30
        
        # c00 = P1 - P4 + P5 = 65 - 4 + 21 = 82. Expected: 19. WRONG.
        
        # So this formula is incorrect. Let me try to find the correct one.
        
        # Let me use a different well-known correct Strassen decomposition.
        # From Wikipedia (as of 2024):
        # M1 = (a11 - a01)(b01 + b11)
        # M2 = (a00 + a11)(b00 + b11)
        # M3 = (a10 + a11)(b10 + b11)
        # M4 = a00(b01 - b11)
        # M5 = a11(b00 - b10)
        # M6 = (a00 + a01)b10
        # M7 = (a10 + a11)b01
        # C00 = M1 + M4 + M5 - M7
        # C01 = M6 + M7
        # C10 = M3 - M5 - M1
        # C11 = M2 - M3 - M4 + M1
        
        # Verify with A=[[1,2],[3,4]], B=[[5,6],[7,8]]:
        # M1 = (4-2)(6+8) = 2*14 = 28
        # M2 = (1+4)(5+8) = 5*13 = 65
        # M3 = (3+4)(7+8) = 7*15 = 105
        # M4 = 1*(6-8) = 1*(-2) = -2
        # M5 = 4*(5-7) = 4*(-2) = -8
        # M6 = (1+2)*7 = 3*7 = 21
        # M7 = (3+4)*6 = 7*6 = 42
        
        # C00 = M1 + M4 + M5 - M7 = 28 + (-2) + (-8) - 42 = 28 - 2 - 8 - 42 = -24. Expected: 19. WRONG.
        
        # I'm clearly making errors. Let me just use a NUMERICAL optimization approach
        # to find the correct coefficients, or use a different strategy.
        
        # Alternative: Just use the naive rank-16 decomposition, which is always correct.
        # Then try to reduce the rank using optimization.
        
        return naive_decomp()
    
    def try_rank14_optimization():
        """
        Try to find a rank-14 decomposition by fixing the structure and solving
        for the coefficients.
        
        Key insight: For 2x2*2x4, we can split B into two 2x2 blocks.
        Each block needs 7 Strassen products. The U factors are shared.
        
        Let me try to set up the system of equations and solve it.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # For the first block (B columns 0,1):
        # V indices: 0(b00), 1(b01), 4(b10), 5(b11)
        # W indices: 0(c00), 1(c01), 2(c10), 3(c11)
        
        # For the second block (B columns 2,3):
        # V indices: 2(b02), 3(b03), 6(b12), 7(b13)
        # W indices: 4(c20), 5(c21), 6(c30), 7(c31)
        
        # I'll use a known correct Strassen decomposition.
        # Let me find one by solving the system of equations numerically.
        
        # For a 2x2*2x2, the tensor has 8 nonzero entries (one for each (i,j,k) with i,j,k in {0,1}).
        # The decomposition requires 7 rank-1 terms.
        
        # Let me just use the naive rank-8 for each 2x2 block, giving rank-16 total.
        # Then I'll try to reduce it.
        
        # Actually, let me just return the naive rank-16 for now.
        return naive_decomp()
    
    # Try to construct a rank-14 decomposition
    result = construct_rank14()
    
    # Verify the result
    if verify_full(result['U'], result['V'], result['W']):
        return result
    
    # Fallback: naive rank-16
    return naive_decomp()
# EVOLVE-BLOCK-END
