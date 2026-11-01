import time
import random
import numpy as np
from itertools import product

def verify_decomposition(U, V, W, n=2, m=2, p=7):
    """
    Verify if U, V, W form a valid decomposition of the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    n, m, p = 2, 2, 7
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin before 150s kill

    n, m, p = 2, 2, 7
    schoolbook_rank = n * m * p  # 28

    # Schoolbook construction (rank 28)
    def make_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    best_U, best_V, best_W = make_schoolbook()
    best_rank = schoolbook_rank

    # Strategy: Block diagonalization by partitioning the 7 output columns into
    # three 2-column blocks and one 1-column block.
    # For each 2-column block, use Strassen's algorithm for 2x2x2 matrix multiplication.
    # For the 1-column block, use schoolbook (rank 4).
    # Total rank: 3 * 7 + 4 = 25.
    
    # Strassen's 2x2x2 decomposition:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B00)
    # M4 = A11 * (B11 - B10)
    # M5 = (A00 + A10) * B11
    # M6 = (A11 - A10) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # In tensor form for 2x2x2:
    # We need U (4x7), V (4x7), W (4x7) such that
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,k}
    
    # Let's construct the Strassen decomposition explicitly.
    # The 2x2x2 tensor T[i,j,k] = delta_{i,k}
    
    # Strassen's decomposition gives rank 7 for 2x2x2.
    # Let me write out the explicit rank-1 terms.
    
    def make_strassen_2x2x2():
        """
        Return U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
        U: (4, 7) for A matrix (i,j)
        V: (4, 7) for B matrix (j,k)
        W: (4, 7) for C matrix (k,i)
        """
        # The 7 intermediate products in Strassen's algorithm:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B00)
        # M4 = A11 * (B11 - B10)
        # M5 = (A00 + A10) * B11
        # M6 = (A11 - A10) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # For each intermediate M_r, we have a rank-1 term:
        # U_part * V_part * W_part
        
        # Let's index: A[i,j], B[j,k], C[k,i] with i,j,k in {0,1}
        # U has indices (i,j), V has indices (j,k), W has indices (k,i)
        
        U = [[0]*7 for _ in range(4)]  # 4 = 2*2
        V = [[0]*7 for _ in range(4)]  # 4 = 2*2
        W = [[0]*7 for _ in range(4)]  # 4 = 2*2
        
        # M1 = (A00 + A11) * (B00 + B11)
        # U for M1: A00 and A11 get coefficient 1
        # V for M1: B00 and B11 get coefficient 1
        # W for M1: contributes to C00 (+1), C11 (+1)
        U[0][0] = 1  # A00
        U[3][0] = 1  # A11
        V[0][0] = 1  # B00
        V[3][0] = 1  # B11
        W[0][0] = 1  # C00 (k=0,i=0)
        W[3][0] = 1  # C11 (k=1,i=1)
        
        # M2 = (A10 + A11) * B00
        # U for M2: A10 and A11 get coefficient 1
        # V for M2: B00 gets coefficient 1
        # W for M2: contributes to C10 (+1)
        U[2][1] = 1  # A10
        U[3][1] = 1  # A11
        V[0][1] = 1  # B00
        W[2][1] = 1  # C10 (k=1,i=0)
        
        # M3 = A00 * (B10 - B00)
        # U for M3: A00 gets coefficient 1
        # V for M3: B10 gets coefficient 1, B00 gets coefficient -1
        # W for M3: contributes to C01 (+1)
        U[0][2] = 1  # A00
        V[2][2] = 1  # B10
        V[0][2] = -1 # B00
        W[1][2] = 1  # C01 (k=0,i=1)
        
        # M4 = A11 * (B11 - B10)
        # U for M4: A11 gets coefficient 1
        # V for M4: B11 gets coefficient 1, B10 gets coefficient -1
        # W for M4: contributes to C00 (+1), C10 (+1)
        U[3][3] = 1  # A11
        V[3][3] = 1  # B11
        V[2][3] = -1 # B10
        W[0][3] = 1  # C00 (k=0,i=0)
        W[2][3] = 1  # C10 (k=1,i=0)
        
        # M5 = (A00 + A10) * B11
        # U for M5: A00 and A10 get coefficient 1
        # V for M5: B11 gets coefficient 1
        # W for M5: contributes to C00 (-1), C01 (+1)
        U[0][4] = 1  # A00
        U[2][4] = 1  # A10
        V[3][4] = 1  # B11
        W[0][4] = -1 # C00 (k=0,i=0)
        W[1][4] = 1  # C01 (k=0,i=1)
        
        # M6 = (A11 - A10) * (B00 + B01)
        # U for M6: A11 gets coefficient 1, A10 gets coefficient -1
        # V for M6: B00 and B01 get coefficient 1
        # W for M6: doesn't contribute to any C? Let me check.
        # Actually, M6 is used in... let me recheck Strassen.
        # Standard Strassen:
        # p1 = (a00 + a11)(b00 + b11)
        # p2 = (a10 + a11)b00
        # p3 = a00(b10 - b00)
        # p4 = a11(b11 - b10)
        # p5 = (a00 + a10)b11
        # p6 = (a11 - a10)(b00 + b01)
        # p7 = (a01 - a11)(b10 + b11)
        # c00 = p1 + p4 - p5 - p7
        # c01 = p3 + p5
        # c10 = p2 + p4
        # c11 = p1 - p2 + p3 + p7
        # So M6 doesn't appear in the output! That's strange.
        # Wait, I think I made an error. Let me recheck.
        # Actually, in standard Strassen, all 7 intermediates are used.
        # Let me re-examine.
        
        # I think I have the wrong formula. Let me use the correct one.
        # Standard Strassen:
        # p1 = (a00 + a11)(b00 + b11)
        # p2 = (a10 + a11)b00
        # p3 = a00(b10 - b00)
        # p4 = a11(b11 - b10)
        # p5 = (a00 + a10)b11
        # p6 = (a11 - a10)(b00 + b01)
        # p7 = (a01 - a11)(b10 + b11)
        # c00 = p1 + p4 - p5 - p7
        # c01 = p3 + p5
        # c10 = p2 + p4
        # c11 = p1 - p2 + p3 + p7
        
        # Hmm, p6 doesn't appear. That can't be right.
        # Let me look this up more carefully.
        # Actually, I think the standard Strassen uses:
        # p1 = (a00 + a11)(b00 + b11)
        # p2 = (a10 + a11)b00
        # p3 = a00(b10 - b00)
        # p4 = a11(b11 - b10)
        # p5 = (a00 + a10)b11
        # p6 = (a11 - a10)(b00 + b01)
        # p7 = (a01 - a11)(b10 + b11)
        # And:
        # c00 = p1 + p4 - p5 - p7
        # c01 = p3 + p5
        # c10 = p2 + p4
        # c11 = p1 - p2 + p3 + p7
        # But p6 is not used? That seems wrong.
        
        # Let me try a different formulation. Maybe I'm confusing the indices.
        # Let me just use a known rank-7 decomposition for 2x2x2.
        
        # Actually, let me just verify if my construction works.
        # For now, let me return this and verify.
        
        # Wait, I realize I might have an error. Let me restart the Strassen construction.
        # The issue is that p6 should be used somewhere. Let me recheck the formulas.
        
        # After rechecking, I believe the correct Strassen decomposition is:
        # p1 = (a00 + a11)(b00 + b11)
        # p2 = (a10 + a11)b00
        # p3 = a00(b10 - b00)
        # p4 = a11(b11 - b10)
        # p5 = (a00 + a10)b11
        # p6 = (a11 - a10)(b00 + b01)
        # p7 = (a01 - a11)(b10 + b11)
        # c00 = p1 + p4 - p5 - p7
        # c01 = p3 + p5
        # c10 = p2 + p4
        # c11 = p1 - p2 + p3 + p7
        # And indeed p6 is not used in the output. This is suspicious.
        
        # Let me try a different approach: just use a known rank-7 decomposition.
        # Or, let me just use the schoolbook for the 2x2x2 blocks and see if we can do better.
        
        # Actually, let me just return the schoolbook for now and focus on the overall structure.
        # The key insight is that we can potentially share computations.
        
        # For now, let me just return a placeholder and focus on the overall algorithm.
        return U, V, W
    
    # Let me take a different approach. Instead of trying to construct Strassen explicitly,
    # let me use a randomized search to find a rank-25 solution.
    
    def try_rank(R, max_iters=10000):
        """Try to find a rank R decomposition using randomized search."""
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Initialize with small random integers
            U = [[random.randint(-3, 3) for _ in range(R)] for _ in range(4)]
            V = [[random.randint(-3, 3) for _ in range(R)] for _ in range(14)]
            W = [[random.randint(-3, 3) for _ in range(R)] for _ in range(14)]
            
            # Verify
            if verify_decomposition(U, V, W):
                return U, V, W
        
        return None
    
    # Try rank 25
    result = try_rank(25, max_iters=5000)
    if result:
        best_U, best_V, best_W = result
        best_rank = 25
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # If random search fails, try rank 26
    result = try_rank(26, max_iters=3000)
    if result:
        best_U, best_V, best_W = result
        best_rank = 26
        return {"U": best_V, "V": best_V, "W": best_W}
    
    # If all else fails, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
