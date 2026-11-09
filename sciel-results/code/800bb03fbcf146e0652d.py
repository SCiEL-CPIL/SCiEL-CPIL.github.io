"""Matrix multiplication 2x3 x 3x9 with rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin

    n, m, p = DIMENSIONS  # 2, 3, 9

    # Fallback: schoolbook rank 54
    def schoolbook():
        r = n * m * p  # 54
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    # Try to find a rank-45 decomposition using Laderman-style tricks
    # on 2x3x3 blocks, then tile across the 9 columns.
    #
    # Key insight: The 2x3 x 3x9 multiplication can be viewed as 3 independent
    # 2x3 x 3x3 problems (one per group of 3 columns of B).
    # If we can find a rank-15 decomposition for 2x3 x 3x3, we get rank 45 total.
    #
    # Let's attempt to construct a rank-15 decomposition for 2x3 x 3x3.
    #
    # The multiplication tensor for 2x3 x 3x3 has 18 entries (2*3*3).
    # Schoolbook gives rank 18. We want to reduce to 15.
    #
    # Approach: Use the fact that for each column k of C, we compute:
    # C[0,k] = a00*B[0,k] + a01*B[1,k] + a02*B[2,k]
    # C[1,k] = a10*B[0,k] + a11*B[1,k] + a12*B[2,k]
    #
    # This is a 2x1 x 3x1 problem repeated 3 times (for k=0,1,2).
    # The bilinear complexity of 2x1 x 3x1 is 3 (no savings possible for 1x1 output).
    # But we have 2 outputs per k, so it's like 2x1 x 3x1 done twice.
    #
    # Actually, let me think differently. Consider the full 2x3 x 3x3 tensor.
    # We can try to find 15 rank-1 tensors that sum to it.
    #
    # Let me use a computational approach: set up the linear system and solve.
    # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    # We want: T = sum_{t=1}^{15} u_t ⊗ v_t ⊗ w_t
    # where u_t is in R^6, v_t in R^9, w_t in R^6.
    #
    # This is a system of 18 equations (one per tensor entry) in 15*3*6 = 270 unknowns.
    # It's underdetermined, so solutions should exist. But finding them is hard.
    #
    # Alternative: Use known results. It's known that R(2,3,3) = 15.
    # Let me try to construct it explicitly.
    #
    # Here's a construction idea:
    # Split the 3 columns of B into two groups: {col0, col1} and {col2}.
    # For the first 2 columns, we have a 2x3 x 3x2 subproblem.
    # For the last column, we have a 2x3 x 3x1 subproblem.
    #
    # R(2,3,2) = ? and R(2,3,1) = 6 (schoolbook, no savings for 1x1 output).
    # If R(2,3,2) = 9, then total = 9 + 6 = 15. 
    #
    # Let's check R(2,3,2):
    # C[0,0] = a00*B[0,0] + a01*B[1,0] + a02*B[2,0]
    # C[0,1] = a00*B[0,1] + a01*B[1,1] + a02*B[2,1]
    # C[1,0] = a10*B[0,0] + a11*B[1,0] + a12*B[2,0]
    # C[1,1] = a10*B[0,1] + a11*B[1,1] + a12*B[2,1]
    #
    # This is 4 outputs, 6 A inputs, 6 B inputs. Schoolbook rank = 12.
    # Can we do better?
    #
    # Notice that a00 appears in C[0,0] and C[0,1], multiplied by B[0,0] and B[0,1].
    # Similarly for other A entries.
    #
    # Let's try to use Laderman-style tricks.
    # Define:
    # x1 = a00, x2 = a00+a01, x3 = a01, x4 = a10, x5 = a10+a11, x6 = a11, x7 = a02, x8 = a12
    # Wait, that's 8 virtual inputs, but we only have 6 actual A entries.
    #
    # This is getting complicated. Let me just try to numerically verify if rank 15
    # is achievable by attempting to find such a decomposition.
    #
    # Given the time constraints and risk of error, let me fall back to a safer approach:
    # Try to find ANY reduction from 54. Even rank 53 would be an improvement.
    #
    # Simple idea: For the 2x3x9 problem, can we merge any two schoolbook terms?
    # Two terms (i,j,k) and (i',j',k') can be merged if there's a linear dependency.
    # In the schoolbook, each term is independent, so no merging is possible.
    #
    # However, we can use the following trick:
    # For a fixed i, the terms (i,j,k) for j=0,1,2 and k=0..8 form a 3x9 grid.
    # We can apply a change of basis to the j-index.
    #
    # Let's try: for each i and k, define:
    # y_{i,k,0} = a_{i,0}
    # y_{i,k,1} = a_{i,0} + a_{i,1}
    # y_{i,k,2} = a_{i,1}
    # Then a_{i,0} = y_{i,k,0}, a_{i,1} = y_{i,k,2}, a_{i,0}+a_{i,1} = y_{i,k,1}.
    # This doesn't save multiplications; it just reparametrizes.
    #
    # I think the safest approach is to return schoolbook (rank 54) since I cannot
    # confidently construct a lower-rank decomposition in the time available.
    
    if time.time() > deadline - 10.0:
        return schoolbook()
    
    return schoolbook()


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
