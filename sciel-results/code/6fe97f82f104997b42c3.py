"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 118.0  # Safety margin before 150s kill
    n, m, p = DIMENSIONS  # 3, 4, 4

    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Build the rank-7 decomposition for 2x2 * 2x2 -> 2x2
    # The multiplication tensor for 2x2x2 has rank 7.
    # We use a known explicit decomposition.
    #
    # For A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]], C = A*B
    # C[0,0] = a0*b0 + a1*b2
    # C[0,1] = a0*b1 + a1*b3
    # C[1,0] = a2*b0 + a3*b2
    # C[1,1] = a2*b1 + a3*b3
    #
    # Rank-7 decomposition (one verified form):
    # m1 = a0 * (b0 + b1)
    # m2 = a1 * (b2 + b3)
    # m3 = a2 * (b0 + b3)
    # m4 = a3 * (b1 + b2)
    # m5 = a0 * b3
    # m6 = a3 * b0
    # m7 = (a1 - a2) * (b2 - b1)
    #
    # C[0,0] = m1 + m2 - m5 - m7
    # C[0,1] = m1 + m2 - m6 - m7  -- wait, let me verify carefully
    #
    # Let me use a different, cleaner formulation. I'll use the one from
    # Bini, Capovani, Lotti, Romani (1979) or a verified variant.
    #
    # Actually, let me just use the standard 7-term decomposition:
    #
    # Term 0: u=[1,0,0,0], v=[1,0,1,0], w=[1,0,0,0]  -> a0*(b0+b2) -> C[0,0]
    #   Wait, I need to be consistent with indexing.
    #
    # Let me define: A indices (i,j) with i=row, j=col; B indices (j,k); C indices (i,k)
    # For 2x2: i in {0,1}, j in {0,1}, k in {0,1}
    #
    # U vector for A: index = i*2+j
    # V vector for B: index = j*2+k
    # W vector for C: index = k*2+i  (per the task convention W[k*n+i])
    #
    # So for 2x2x2, W index = k*2+i
    #
    # Rank-7 decomposition terms (u, v, w) where each is a 4-element vector:
    #
    # I'll use this verified decomposition:
    # t0: u=[1,0,0,0], v=[1,1,0,0], w=[1,0,0,0]
    #     = a00 * (b00 + b01) -> contributes to C[0,0] (w[0]=C[0*2+0]=C[0,0])
    # t1: u=[0,1,0,0], v=[0,0,1,1], w=[1,0,0,0]
    #     = a01 * (b10 + b11) -> contributes to C[0,0]
    # t2: u=[0,0,1,0], v=[1,0,0,1], w=[0,0,1,0]
    #     = a10 * (b00 + b11) -> contributes to C[1,0] (w[2]=C[0*2+1]... wait)
    #
    # W index = k*n + i = k*2 + i for 2x2
    # C[i,k] is at W index k*2+i
    # C[0,0] -> W[0], C[0,1] -> W[2], C[1,0] -> W[1], C[1,1] -> W[3]
    #
    # So:
    # t0: u=[1,0,0,0], v=[1,1,0,0], w=[1,0,0,0]
    #     a00*(b00+b01) -> C[0,0]
    # t1: u=[0,1,0,0], v=[0,0,1,1], w=[1,0,0,0]
    #     a01*(b10+b11) -> C[0,0]
    # t2: u=[0,0,1,0], v=[1,0,0,1], w=[0,1,0,0]
    #     a10*(b00+b11) -> C[1,0] (w[1]=C[0*2+1]=C[1,0])
    # t3: u=[0,0,0,1], v=[0,1,1,0], w=[0,1,0,0]
    #     a11*(b01+b10) -> C[1,0]
    # t4: u=[1,0,0,0], v=[0,0,0,1], w=[1,0,0,0]
    #     a00*b11 -> C[0,0]
    # t5: u=[0,0,0,1], v=[1,0,0,0], w=[0,1,0,0]
    #     a11*b00 -> C[1,0]
    # t6: u=[0,1,0,0], v=[0,0,1,0], w=[0,0,1,0]
    #     a01*b10 -> C[1,1] (w[2]=C[1*2+0]=C[1,0])... 
    #
    # I'm getting confused with the indexing. Let me be very explicit.
    #
    # A[i,j] where i in {0,1}, j in {0,1}. U index = i*2+j.
    # B[j,k] where j in {0,1}, k in {0,1}. V index = j*2+k.
    # C[i,k] where i in {0,1}, k in {0,1}. W index = k*2+i (per task: W[k*n+i]).
    #
    # C[0,0]: W index 0*2+0 = 0
    # C[0,1]: W index 1*2+0 = 2
    # C[1,0]: W index 0*2+1 = 1
    # C[1,1]: W index 1*2+1 = 3
    #
    # So W = [C[0,0], C[1,0], C[0,1], C[1,1]]
    #
    # Now the rank-7 decomposition. Let me use a known correct one.
    #
    # After careful thought, here's a verified rank-7 decomposition:
    #
    # m1 = a00 * (b00 + b01)
    # m2 = a01 * (b10 + b11)
    # m3 = a10 * (b00 + b11)
    # m4 = a11 * (b01 + b10)
    # m5 = a00 * b11
    # m6 = a11 * b00
    # m7 = (a01 - a10) * (b10 - b01)
    #
    # C[0,0] = a00*b00 + a01*b10 = m1 + m2 - m5 - m7
    #   m1 + m2 - m5 - m7 = a00*b00 + a00*b01 + a01*b10 + a01*b11 - a00*b11 - (a01*b10 - a01*b01 - a10*b10 + a10*b01)
    #   = a00*b00 + a00*b01 + a01*b10 + a01*b11 - a00*b11 - a01*b10 + a01*b01 + a10*b10 - a10*b01
    #   = a00*b00 + a00*b01 + a01*b11 - a00*b11 + a01*b01 + a10*b10 - a10*b01
    #   This should equal a00*b00 + a01*b10. It doesn't simplify correctly.
    #
    # Let me try a different approach. I'll use the fact that the 2x2x2 tensor
    # can be decomposed as follows (from literature):
    #
    # The 7 terms are:
    # 1. a00*b00 -> C[0,0]
    # 2. a01*b10 -> C[0,0]
    # 3. a00*b01 -> C[0,1]
    # 4. a01*b11 -> C[0,1]
    # 5. a10*b00 -> C[1,0]
    # 6. a11*b10 -> C[1,0]
    # 7. a10*b01 -> C[1,1]
    # 8. a11*b11 -> C[1,1]
    #
    # To get rank 7, we combine two of these into one term using a linear combination.
    #
    # A standard way: combine terms 2 and 7:
    # Instead of a01*b10 and a10*b01, use:
    # (a01 + a10) * (b10 + b01) and then subtract the extra terms.
    #
    # Actually, the cleanest approach for this task: since we need exact integer/half-integer
    # entries, let me just implement the schoolbook for the full 3x4x4 but try to find
    # a structured low-rank decomposition.
    #
    # Given the complexity and time constraints, let me implement a careful structured
    # decomposition using the rank-7 2x2x2 building block.
    #
    # For 3x4 * 4x4:
    # Split i (rows of A, cols of C): i in {0,1} (top) and i=2 (bottom)
    # Split k (cols of B, cols of C): k in {0,1} (left) and k in {2,3} (right)
    # Split j (cols of A, rows of B): j in {0,1} (left) and j in {2,3} (right)
    #
    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    #   = (2x2 * 2x2) + (2x2 * 2x2) -> rank 7 + 7 = 14
    # C[0:2, 2:4] = A[0:2, 0:2]*B[0:2, 2:4] + A[0:2, 2:4]*B[2:4, 2:4]
    #   = (2x2 * 2x2) + (2x2 * 2x2) -> rank 7 + 7 = 14
    # C[2, 0:2] = A[2, 0:2]*B[0:2, 0:2] + A[2, 2:4]*B[2:4, 0:2]
    #   = (1x2 * 2x2) + (1x2 * 2x2) -> rank 2 + 2 = 4
    # C[2, 2:4] = A[2, 0:2]*B[0:2, 2:4] + A[2, 2:4]*B[2:4, 2:4]
    #   = (1x2 * 2x2) + (1x2 * 2x2) -> rank 2 + 2 = 4
    #
    # Total rank: 14 + 14 + 4 + 4 = 36
    #
    # This is better than 48! Let me implement this.

    # Define the rank-7 decomposition for 2x2 * 2x2 -> 2x2
    # Returns lists of (u, v, w) where u, v, w are length-4 lists
    # u index: i*2+j for A[i,j]
    # v index: j*2+k for B[j,k]
    # w index: k*2+i for C[i,k] (task convention)
    
    def rank7_2x2():
        """Return 7 terms (u, v, w) for 2x2x2 multiplication."""
        # I'll use a verified decomposition.
        # Let me construct it from the 8 schoolbook terms and merge.
        #
        # Schoolbook terms (u, v, w):
        # (i,j,k) -> u has 1 at i*2+j, v has 1 at j*2+k, w has 1 at k*2+i
        #
        # (0,0,0): u=[1,0,0,0], v=[1,0,0,0], w=[1,0,0,0]
        # (0,0,1): u=[1,0,0,0], v=[0,1,0,0], w=[0,0,1,0]
        # (0,1,0): u=[0,1,0,0], v=[2,0,0,0], w=[1,0,0,0]
        # (0,1,1): u=[0,1,0,0], v=[3,0,0,0], w=[0,0,1,0]
        # (1,0,0): u=[0,0,1,0], v=[1,0,0,0], w=[0,1,0,0]
        # (1,0,1): u=[0,0,1,0], v=[0,1,0,0], w=[0,0,0,1]
        # (1,1,0): u=[0,0,0,1], v=[2,0,0,0], w=[0,1,0,0]
        # (1,1,1): u=[0,0,0,1], v=[3,0,0,0], w=[0,0,0,1]
        #
        # To get rank 7, I need to find 7 vectors that span the same tensor.
        #
        # A known rank-7 decomposition (using the fact that the tensor has a
        # linear dependency):
        #
        # Let me use this explicit form:
        terms = []
        # Term 0: a00 * (b00 + b01) -> C[0,0]
        terms.append(([1,0,0,0], [1,1,0,0], [1,0,0,0]))
        # Term 1: a01 * (b10 + b11) -> C[0,0]  
        terms.append(([0,1,0,0], [0,0,1,1], [1,0,0,0]))
        # Term 2: a10 * (b00 + b11) -> C[1,0]
        terms.append(([0,0,1,0], [1,0,0,1], [0,1,0,0]))
        # Term 3: a11 * (b01 + b10) -> C[1,0]
        terms.append(([0,0,0,1], [0,1,1,0], [0,1,0,0]))
        # Term 4: a00 * b11 -> C[0,0]
        terms.append(([1,0,0,0], [0,0,0,1], [1,0,0,0]))
        # Term 5: a11 * b00 -> C[1,0]
        terms.append(([0,0,0,1], [1,0,0,0], [0,1,0,0]))
        # Term 6: (a01 - a10) * (b10 - b01) -> C[0,0] and C[1,1]
        terms.append(([0,1,-1,0], [0,0,1,-1], [1,0,0,1]))
        
        # Verify:
        # C[0,0] should be a00*b00 + a01*b10
        # Sum of contributions to w[0] (C[0,0]):
        # Term 0: 1 * (a00) * (b00+b01) -> a00*b00 + a00*b01
        # Term 1: 1 * (a01) * (b10+b11) -> a01*b10 + a01*b11
        # Term 4: 1 * (a00) * (b11) -> a00*b11
        # Term 6: 1 * (a01-a10) * (b10-b01) -> a01*b10 - a01*b01 - a10*b10 + a10*b01
        # Total: a00*b00 + a00*b01 + a01*b10 + a01*b11 + a00*b11 + a01*b10 - a01*b01 - a10*b10 + a10*b01
        #        = a00*b00 + a00*b01 + 2*a01*b10 + a01*b11 + a00*b11 - a01*b01 - a10*b10 + a10*b01
        # This should equal a00*b00 + a01*b10. It doesn't.
        
        # This decomposition is incorrect. Let me use a different approach.
        #
        # Given the difficulty of getting the exact rank-7 decomposition right
        # in this time, let me just return the schoolbook solution.
        # The structured approach would give rank 36, but I need to get the
        # rank-7 decomposition exactly right first.
        
        # For now, return empty to signal we should use schoolbook
        return []
    
    # Since getting the exact rank-7 decomposition right is complex,
    # and the schoolbook gives rank 48, let me just return schoolbook.
    # A future iteration can implement the rank-36 structured decomposition.
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
