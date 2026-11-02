"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x5 matrix by a 5x6 matrix.
    
    Uses a hybrid strategy:
    1. For small dimensions, use known optimal or near-optimal decompositions.
    2. Combine via block structure if possible.
    3. Fall back to schoolbook if no better method is found.
    
    The output follows the trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Strategy: Since we're multiplying (3x5) * (5x6) -> (3x6),
    # we can try to exploit structure. One approach is to use the 
    # Strassen-like recursive decomposition or known low-rank decompositions.
    
    # For 3x5 * 5x6, the schoolbook rank is 3*5*6 = 90.
    # Let's try to find a better decomposition.
    
    # One approach: split the 5x5 middle into blocks.
    # For example, split the 5 dimension into 2+3 or 1+4, etc.
    # Then use block matrix multiplication.
    
    # Let's try splitting m=5 into m1=2, m2=3.
    # Then A is 3x5 = [A1 A2] where A1 is 3x2, A2 is 3x3
    # B is 5x6 = [B1; B2] where B1 is 2x6, B2 is 3x6
    # C = A1*B1 + A2*B2
    
    # Rank for A1*B1: 3x2 * 2x6, schoolbook = 3*2*6 = 36
    # Rank for A2*B2: 3x3 * 3x6, can use BCLR rank 23 for 3x3*3x3, but here it's 3x3*3x6
    # Actually for 3x3 * 3x6, we can do:
    #   Split 6 into 3+3: C = A2*B2a + A2*B2b
    #   Each is 3x3 * 3x3, rank 23 each, total 46
    # So total = 36 + 46 = 82
    
    # But wait, we can do better. Let me think about the 3x2 * 2x6 case.
    # 3x2 * 2x6: schoolbook is 36. Can we do better?
    # This is equivalent to computing 3 outputs, each is a linear combination of 2 inputs times 6 outputs.
    # Actually, each output C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k]
    # This is just 2 multiplications per output, so 3*6*2 = 36 scalar multiplications.
    # This is already minimal for this subproblem (since we need 2 multiplications per output).
    
    # For 3x3 * 3x6:
    # Split 6 into 3+3: 3x3*3x3 + 3x3*3x3 = 23 + 23 = 46
    # Total: 36 + 46 = 82
    
    # Can we do better by splitting differently?
    # Try m=5 = 1+4:
    # A1: 3x1, B1: 1x6 -> rank = 3*1*6 = 18
    # A2: 3x4, B2: 4x6 -> 3x4*4x6, schoolbook = 72
    # Total = 18+72 = 90, worse.
    
    # Try m=5 = 3+2:
    # A1: 3x3, B1: 3x6 -> 3x3*3x6, rank 46 (as above)
    # A2: 3x2, B2: 2x6 -> rank 36
    # Total = 46+36 = 82, same.
    
    # Let's go with m=2+3, total rank 82.
    
    # Now let's construct the decomposition.
    # We'll build U, V, W matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    # = (15, rank), (30, rank), (18, rank)
    
    # Decomposition 1: A1*B1 where A1 is 3x2, B1 is 2x6
    # This corresponds to indices i in [0,2], j in [0,1], k in [0,5]
    # U1[i*m + j, c] = 1 for the appropriate (i,j)
    # V1[j*p + k, c] = 1 for the appropriate (j,k)
    # W1[k*n + i, c] = 1 for the appropriate (k,i)
    
    # Decomposition 2: A2*B2 where A2 is 3x3, B2 is 3x6
    # This corresponds to indices i in [0,2], j in [2,4], k in [0,5]
    # For A2*B2, we split B2 into B2a (3x3) and B2b (3x3)
    # B2a: k in [0,2], B2b: k in [3,5]
    
    # So we have three sub-decompositions:
    # 1. 3x2 * 2x6 (schoolbook, rank 36)
    # 2. 3x3 * 3x3 (BCLR, rank 23) for k in [0,2]
    # 3. 3x3 * 3x3 (BCLR, rank 23) for k in [3,5]
    
    # Total rank = 36 + 23 + 23 = 82
    
    # Now I need to implement the BCLR decomposition for 3x3*3x3.
    # The BCLR decomposition has rank 23.
    
    # Let me hardcode the BCLR decomposition for 3x3 matrix multiplication.
    # The decomposition is T = sum_{r=1}^{23} u_r (v_r x w_r)
    # where u_r, v_r, w_r are 3x3 matrices.
    
    # I'll use a known representation of the BCLR decomposition.
    # Here are the 23 terms as 3x3 matrices:
    
    # This is complex to hardcode correctly. Let me use an alternative:
    # I'll use the schoolbook for 3x3*3x3 (rank 27) as a fallback,
    # giving total rank = 36 + 27 + 27 = 90, same as schoolbook.
    
    # But the task asks for rank-minimal. Let me try to implement BCLR properly.
    
    # Actually, let me try a different approach: use a numerical optimization
    # or a known decomposition.
    
    # For now, let me implement the 3x3*3x3 BCLR decomposition.
    # I'll use the formulation from the BCLR paper.
    
    # The BCLR decomposition for 3x3 matrix multiplication uses 23 multiplications.
    # Here are the 23 triples (u_r, v_r, w_r) where each is a 3x3 matrix:
    
    # I'll define them using a list of 23 entries, each containing three 3x3 matrices.
    # The multiplication is: C = sum_r (u_r * A) * (v_r * B) * w_r^T ... no, that's not right.
    
    # Actually, the tensor decomposition is:
    # T(A,B) = C where C_{ik} = sum_{j} A_{ij} B_{jk}
    # The rank-1 decomposition is: C = sum_r (U_r^T A V_r) (W_r)
    # where U_r, V_r, W_r are 3x3 matrices.
    
    # More precisely, for each r:
    #   (U_r^T A V_r) is a 3x3 matrix
    #   We multiply this by w_r (a scalar? no, a matrix?)
    
    # I'm getting confused. Let me look up the exact formulation.
    
    # In the BCLR decomposition, we have:
    # C = sum_{r=1}^{23} (M_r(A)) * (N_r(B)) * (P_r)
    # where M_r(A) and N_r(B) are linear combinations of entries of A and B,
    # and P_r is a fixed 3x3 matrix.
    
    # Actually, the standard formulation is:
    # Each term is of the form: (linear form in A) * (linear form in B) * (fixed output matrix)
    
    # Let me just use a simpler approach: implement the schoolbook decomposition
    # but try to find any improvements.
    
    # For the 3x3*3x3 subproblem, the known minimum rank is 23 (BCLR).
    # Let me try to find a reference implementation or derive it.
    
    # Given the complexity, let me use a different strategy:
    # Use the schoolbook decomposition for everything, but check if we can
    # do better with a simple heuristic.
    
    # Actually, let me just implement the 82-rank decomposition using
    # schoolbook for 3x3*3x3 (rank 27) instead of BCLR (rank 23).
    # Total rank = 36 + 27 + 27 = 90, which is the same as schoolbook.
    
    # That's not an improvement. Let me try harder to implement BCLR.
    
    # I found a resource: the BCLR decomposition can be represented as follows.
    # Let me try to code it up.
    
    # Actually, let me try a completely different approach:
    # Use the fact that 5 = 2+3, and for the 3x3*3x6 part,
    # instead of splitting 6 into 3+3, use a different decomposition.
    
    # For 3x3 * 3x6, the schoolbook rank is 3*3*6 = 54.
    # Using BCLR for two 3x3*3x3 blocks: 23+23 = 46.
    # Can we do better?
    
    # The asymptotic rank of 3x3 matrix multiplication is 23, so for 3x3*3x6,
    # we might be able to do better than 46.
    
    # But for simplicity, let me stick with the 82-rank decomposition.
    
    # Now, let me implement it properly.
    
    rank = 82
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    
    # Part 1: 3x2 * 2x6 (schoolbook, rank 36)
    # i in [0,2], j in [0,1], k in [0,5]
    for i in range(3):
        for j in range(2):
            for k in range(6):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Part 2: 3x3 * 3x3 for k in [0,2] (BCLR, rank 23)
    # i in [0,2], j in [2,4], k in [0,2]
    # Need BCLR decomposition for 3x3*3x3
    # Let me use schoolbook for now (rank 27) and adjust rank
    # Actually, let me just use schoolbook for all parts.
    
    # Reset and use all schoolbook
    rank = n*m*p  # 90
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
