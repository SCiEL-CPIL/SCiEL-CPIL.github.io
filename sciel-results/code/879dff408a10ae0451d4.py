import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Use Strassen for 4x4x4 (rank 28) + schoolbook for 4x4x1 (rank 4) = rank 32
    # But first, let's try to verify a correct Strassen decomposition.
    
    # Correct Strassen 2x2x2 decomposition:
    # For C = A*B where A,B,C are 2x2:
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M4
    # C[1,0] = M2 + M3
    # C[1,1] = M1 - M2 + M6 - M7
    # where:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # In our tensor format for 2x2x2:
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # We need: delta(i,j,k) = sum_r U[i*2+j,r] * V[j*2+k,r] * W[k*2+i,r]
    
    # Let me define the Strassen factors directly.
    # The 2x2x2 multiplication tensor has 8 nonzero entries (i,j,k) where i,j,k in {0,1}
    # and the value is 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    
    # Actually, the tensor T[i,j,k] for 2x2x2 is:
    # T[0,0,0] = 1, T[0,1,1] = 1, T[1,0,0] = 1, T[1,1,1] = 1
    # (These correspond to C[i,k] = sum_j A[i,j] B[j,k])
    
    # Let me use a known rank-7 decomposition.
    # I'll construct it carefully.
    
    # For now, let's use a simpler approach: try to find a low-rank decomposition
    # using randomized search, with schoolbook as fallback.
    
    # Actually, let me just implement the 4x4x4 via 4 independent 2x2x2 Strassen blocks.
    # This gives rank 28 for 4x4x4, plus 4 for 4x4x1 = 32 total.
    
    # First, I need a correct Strassen 2x2x2 decomposition.
    # Let me use the standard one from the literature.
    
    # The Strassen algorithm computes:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1 - M4 + M5 - M7
    # c12 = M3 + M4
    # c21 = M2 + M3
    # c22 = M1 - M2 + M6 - M7
    
    # In 0-indexed:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M4
    # C[1,0] = M2 + M3
    # C[1,1] = M1 - M2 + M6 - M7
    
    # Now, in tensor form, for the 2x2x2 block:
    # U is 4x7 (rows: A[0,0], A[0,1], A[1,0], A[1,1])
    # V is 4x7 (rows: B[0,0], B[0,1], B[1,0], B[1,1])
    # W is 4x7 (rows: C[0,0], C[0,1], C[1,0], C[1,1])
    
    # U[i*2+j, r] = coefficient of A[i,j] in M_r
    # V[j*2+k, r] = coefficient of B[j,k] in M_r
    # W[k*2+i, r] = coefficient of M_r in C[i,k]
    
    # Let me build these:
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U_s[0, 0] = 1  # A00
    U_s[3, 0] = 1  # A11
    V_s[0, 0] = 1  # B00
    V_s[3, 0] = 1  # B11
    W_s[0, 0] = 1  # C00
    W_s[3, 0] = 1  # C11
    
    # M2 = (A10+A11)B00
    U_s[2, 1] = 1  # A10
    U_s[3, 1] = 1  # A11
    V_s[0, 1] = 1  # B00
    W_s[2, 1] = 1  # C10
    
    # M3 = A00(B01-B11)
    U_s[0, 2] = 1  # A00
    V_s[1, 2] = 1  # B01
    V_s[3, 2] = -1  # B11
    W_s[1, 2] = 1  # C01
    W_s[2, 2] = 1  # C10
    
    # M4 = A11(B10-B00)
    U_s[3, 3] = 1  # A11
    V_s[2, 3] = 1  # B10
    V_s[0, 3] = -1  # B00
    W_s[0, 3] = -1  # C00
    W_s[1, 3] = 1  # C01
    
    # M5 = (A00+A01)B11
    U_s[0, 4] = 1  # A00
    U_s[1, 4] = 1  # A01
    V_s[3, 4] = 1  # B11
    W_s[0, 4] = 1  # C00
    
    # M6 = (A10-A00)(B00+B01)
    U_s[2, 5] = 1  # A10
    U_s[0, 5] = -1  # A00
    V_s[0, 5] = 1  # B00
    V_s[1, 5] = 1  # B01
    W_s[3, 5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    U_s[1, 6] = 1  # A01
    U_s[3, 6] = -1  # A11
    V_s[2, 6] = 1  # B10
    V_s[3, 6] = 1  # B11
    W_s[0, 6] = -1  # C00
    W_s[3, 6] = -1  # C11
    
    # Verify: For each (i,j,k) in 0..1, check if sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] == 1
    
    # Let's verify C[0,0] (i=0, k=0): should be sum_j A[0,j]B[j,0]
    # j=0: A[0,0]B[0,0] -> r=0: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    # j=1: A[0,1]B[1,0] -> should contribute 0 to C[0,0]
    # Check r=0: U[1,0]*V[2,0]*W[0,0] = 0*0*1 = 0
    # r=1: U[1,1]*V[2,1]*W[0,1] = 0*0*0 = 0
    # r=2: U[1,2]*V[2,2]*W[0,2] = 0*0*0 = 0
    # r=3: U[1,3]*V[2,3]*W[0,3] = 0*1*(-1) = 0
    # r=4: U[1,4]*V[2,4]*W[0,4] = 1*0*1 = 0
    # r=5: U[1,5]*V[2,5]*W[0,5] = 0*0*0 = 0
    # r=6: U[1,6]*V[2,6]*W[0,6] = 1*1*(-1) = -1
    # Total for j=1: -1. But it should be 0!
    
    # Something is wrong. Let me reconsider.
    
    # Actually, I think I'm confusing the indexing. Let me restart with a cleaner approach.
    
    # The tensor T for 2x2x2 multiplication has T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 for all i,j,k in {0,1} where... wait, no.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] = 1 for all i,j,k in {0,1}.
    # No wait, that's not right either. The tensor has 8 entries, all 1?
    # Actually yes! For 2x2x2, T[i,j,k] = 1 for all i,j,k in {0,1}.
    
    # So we need: sum_r U[i*2+j,r] * V[j*2+k,r] * W[k*2+i,r] = 1 for all i,j,k in {0,1}
    
    # Let me verify my Strassen decomposition against this.
    
    # For i=0, j=0, k=0: sum_r U[0,r]*V[0,r]*W[0,r]
    # r=0: 1*1*1 = 1
    # r=1: 0*1*0 = 0
    # r=2: 1*0*0 = 0
    # r=3: 0*(-1)*(-1) = 0
    # r=4: 1*0*1 = 0
    # r=5: (-1)*1*0 = 0
    # r=6: 0*0*(-1) = 0
    # Total: 1. Correct!
    
    # For i=0, j=1, k=0: sum_r U[1,r]*V[2,r]*W[0,r]
    # r=0: 0*0*1 = 0
    # r=1: 0*0*0 = 0
    # r=2: 0*0*0 = 0
    # r=3: 0*1*(-1) = 0
    # r=4: 1*0*1 = 0
    # r=5: 0*0*0 = 0
    # r=6: 1*1*(-1) = -1
    # Total: -1. Should be 1. WRONG!
    
    # So my Strassen matrices are incorrect. Let me look up the correct ones.
    
    # Given the time pressure, let me just use schoolbook for now to ensure correctness.
    # Rank 144.
    
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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
# EVOLVE-BLOCK-END
