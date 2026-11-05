"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n=2, m=4, p=7):
    """Check that the factor matrices exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # The multiplication tensor T[i,j,k] = 1 if we compute (AB)[i,k] += A[i,j]*B[j,k]
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We check that for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    # And for all other combinations, it should be 0 (but the tensor is only defined on valid indices)
    # Actually, the tensor T[i,j,k] = 1 for all valid i,j,k (it's the identity in the sense that
    # C[i,k] = sum_j A[i,j] B[j,k])
    # The constraint is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And for indices where the "cross term" doesn't correspond to a valid multiplication, it should be 0.
    # But actually, the standard formulation is:
    # C[i,k] = sum_j sum_c U[i*m+j,c] * A[i,j] * V[j*p+k,c] * B[j,k] * W[k*n+i,c]
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    # And sum_c U[i*m+j',c] * V[j'*p+k,c] * W[k*n+i,c] = 0 for j' != j (when fixing i,k)
    # More precisely, for fixed i,k: sum_c U[i*m+j,c] * W[k*n+i,c] * V[j*p+k,c] must be 1 if j is the column
    # and 0 otherwise. Wait, let me think more carefully.
    
    # The bilinear form: C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_k' V[j*p+k',c] * B[j,k']) * W[k*n+i,c]
    # This doesn't quite work. Let me use the standard tensor decomposition formulation.
    
    # Matrix multiplication tensor: T_{i,j,k} where C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor has entries T[i,j,k] = delta (it's 1 for all i,j,k in the "multiplication" sense)
    # Rank-r decomposition: T[i,j,k] = sum_{c=1}^r U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # So for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's verify this.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :]) * W[k*n+i, :]
                # Actually it's element-wise product then sum
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _strassen_decompose_2x2():
    """Return Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Returns U, V, W each of shape (4, 7) for the 2x2 case.
    """
    # Strassen's 7 multiplications for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    # Indices: A[i,j] with i,j in {0,1}, B[j,k] with j,k in {0,1}
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    # Let's build the factor matrices
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # c=0: m1 = (a+d)(e+h)
    # A part: a+d -> U[0,0]=1 (a=A[0,0]), U[1,0]=1 (d=A[1,1])
    # B part: e+h -> V[0,0]=1 (e=B[0,0]), V[3,0]=1 (h=B[1,1])
    # C part: contributes to C[0,0] and C[1,1]
    # C[0,0] += m1, C[1,1] += m1
    # W[k*2+i, c]: for C[0,0]: k=0,i=0 -> W[0,0]=1; for C[1,1]: k=1,i=1 -> W[3,0]=1
    U[0, 0] = 1; U[1, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # c=1: m2 = (c+d)e
    # A part: c+d -> U[2,1]=1 (c=A[1,0]), U[1,1]=1 (d=A[1,1])
    # B part: e -> V[0,1]=1 (e=B[0,0])
    # C part: C[1,0] += m2
    # W[k*2+i, c]: for C[1,0]: k=0,i=1 -> W[2,1]=1
    U[2, 1] = 1; U[1, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # c=2: m3 = a(f-h)
    # A part: a -> U[0,2]=1
    # B part: f-h -> V[1,2]=1 (f=B[0,1]), V[3,2]=-1 (h=B[1,1])
    # C part: C[0,1] += m3
    # W[k*2+i, c]: for C[0,1]: k=1,i=0 -> W[1,2]=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # c=3: m4 = d(g-e)
    # A part: d -> U[1,3]=1
    # B part: g-e -> V[2,3]=1 (g=B[1,0]), V[0,3]=-1 (e=B[0,0])
    # C part: C[0,0] += m4, C[1,0] += m4
    # W[0,3]=1 (C[0,0]), W[2,3]=1 (C[1,0])
    U[1, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # c=4: m5 = (a+b)h
    # A part: a+b -> U[0,4]=1, U[3,4]=1 (b=A[0,1])
    # B part: h -> V[3,4]=1
    # C part: C[0,0] -= m5, C[0,1] += m5
    # W[0,4]=-1 (C[0,0]), W[1,4]=1 (C[0,1])
    U[0, 4] = 1; U[3, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # c=5: m6 = (c-a)(e+f)
    # A part: c-a -> U[2,5]=1 (c), U[0,5]=-1 (a)
    # B part: e+f -> V[0,5]=1 (e), V[1,5]=1 (f)
    # C part: C[1,1] += m6
    # W[3,5]=1 (C[1,1])
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # c=6: m7 = (b-d)(g+h)
    # A part: b-d -> U[3,6]=1 (b), U[1,6]=-1 (d)
    # B part: g+h -> V[2,6]=1 (g), V[3,6]=1 (h)
    # C part: C[0,0] += m7
    # W[0,6]=1 (C[0,0])
    U[3, 6] = 1; U[1, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _combine_decompositions(decomps, dims):
    """Combine multiple rank-1 or block decompositions into a full decomposition.
    Each decomps entry is (U_block, V_block, W_block, i_offset, j_offset, k_offset)
    where the block covers A[i_offset:i_offset+ni, j_offset:j_offset+mj] * B[j_offset:j_offset+mj, k_offset:k_offset+pk]
    """
    # This is complex; let's just build the full thing directly.
    pass


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = 2, 4, 7
    
    # Strategy: Use Strassen for 2x2 blocks and combine with direct terms.
    # 2x4 * 4x7 = (2x2 * 2x2) blocks
    # A is 2x4, B is 4x7
    # We can split B into two 4x3.5... no, let's think differently.
    
    # Alternative: Split the middle dimension.
    # A is 2x4, B is 4x7.
    # We can write C = A * B where A = [A1, A2] (2x2 each) and B = [B1; B2] (2x7 each)
    # C = A1*B1 + A2*B2
    # Each A_i is 2x2, B_i is 2x7
    # For each pair, we need to multiply 2x2 by 2x7.
    # 2x2 * 2x7: We can use Strassen-like approach.
    # Actually, let's just use the known result that 2x2 * 2xn has rank at most 7*ceil(n/2) or similar.
    
    # Let me try a different approach: direct construction with rank reduction.
    # The schoolbook rank is 2*4*7 = 56.
    # We know from the cell findings that rank 50 is achievable.
    
    # Let's try to construct a rank-50 decomposition.
    # Idea: Use 2x2 Strassen blocks.
    # Split A into two 2x2 blocks: A = [A_left, A_right]
    # Split B into two 2x7 blocks: B = [B_top; B_bottom]
    # C = A_left * B_top + A_right * B_bottom
    
    # For 2x2 * 2x7:
    # We can further split B_top into B_top1 (2x3) and B_top2 (2x4)
    # This gets complicated. Let me just try a random search with validation.
    
    # Actually, let me try a smarter constructive approach.
    # For 2x4 * 4x7, we can use the fact that the tensor rank of 2x4x4x7 matrix mult is known.
    # Let's try to find a rank-50 decomposition by modifying the schoolbook.
    
    # Schoolbook: 56 terms, one for each (i,j,k).
    # We can potentially merge some terms.
    
    # Let me try a different strategy: use numpy to search for a low-rank decomposition.
    # The tensor T[i,j,k] = 1 for all i,j,k (in the multiplication sense).
    # We want to find U (8 x r), V (28 x r), W (14 x r) such that
    # sum_c U[i*4+j, c] * V[j*7+k, c] * W[k*2+i, c] = 1 for all i,j,k.
    
    # This is a system of 56 equations in r*50 unknowns (8r + 28r + 14r = 50r).
    # For r=50, we have 2500 unknowns and 56 equations - very underdetermined.
    # But the equations are nonlinear (bilinear/trilinear).
    
    # Let me try a greedy approach: start with schoolbook and try to reduce rank.
    
    # Actually, let me just return the schoolbook decomposition for now and try to improve.
    # But we know rank 50 is achievable, so let me try to construct it.
    
    # Known approach: For 2x2 * 2x2, Strassen gives rank 7 instead of 8.
    # For 2x4 * 4x7, we can think of it as:
    # C = [A1 A2] * [B1; B2] where A1,A2 are 2x2, B1,B2 are 2x7
    # C = A1*B1 + A2*B2
    # Each A_i * B_i is 2x2 * 2x7.
    
    # For 2x2 * 2x7:
    # Let B_i = [b1 b2 b3 b4 b5 b6 b7] (2x7)
    # We can split B_i into [B_i1 B_i2] where B_i1 is 2x3 and B_i2 is 2x4
    # A_i * B_i1: 2x2 * 2x3
    # A_i * B_i2: 2x2 * 2x4
    # This is getting too complex. Let me just do a numerical search.
    
    # Try to find a rank-r decomposition using iterative methods.
    # For r from 50 down to 40:
    #   Initialize U, V, W randomly
    #   Use ALS or similar to minimize the error
    #   Check if exact
    
    # But this is hard to make exact. Let me try a constructive approach.
    
    # Actually, let me just return the schoolbook for now.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
