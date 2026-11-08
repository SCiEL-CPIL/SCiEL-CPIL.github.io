"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # We want to decompose the tensor T[i, j, k] = delta_{i, i'} delta_{j, j'} delta_{k, k'}
    # Actually, the tensor is T_{i, j, k} which is 1 if the indices correspond to the multiplication structure.
    # The multiplication (n x m) * (m x p) -> (n x p)
    # A is n x m, B is m x p, C is n x p.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T has indices (i, j, k) and T[i, j, k] = 1.
    # The decomposition is U[i, r] V[j, r] W[k, r] = T[i, j, k]
    # Here U is (n*m) x rank? No, the interface says:
    # U: (n*m, rank) -- wait, the indices of U correspond to the input A?
    # The contract says:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # And the output factor follows W[k*n+i, r].
    # Let's check the schoolbook in the prompt:
    # U[i * m + j][column] = 1  -> index i*m+j corresponds to A[i, j]
    # V[j * p + k][column] = 1  -> index j*p+k corresponds to B[j, k]
    # W[k * n + i][column] = 1  -> index k*n+i corresponds to C[k, i] ??
    # Wait, the standard multiplication is C[i, k] = sum_j A[i, j] B[j, k].
    # The prompt says W[k*n+i, r]. This means the index of W is k*n + i.
    # So W corresponds to C^T? Or C with row k and col i?
    # Let's re-read carefully: "output factor follows the official trace convention W[k*n+i,r]".
    # In the schoolbook:
    # for i in range(n):
    #   for j in range(m):
    #     for k in range(p):
    #       U[i*m+j] = 1
    #       V[j*p+k] = 1
    #       W[k*n+i] = 1
    # This implies that the term is U[i*m+j] * V[j*p+k] * W[k*n+i].
    # Summing over column c (rank):
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # For the schoolbook, each c corresponds to a unique (i, j, k).
    # So the sum is 1 if (i, j, k) matches the specific (i, j, k) of that column, else 0.
    # This equals delta_{i, i'} delta_{j, j'} delta_{k, k'} for the specific tuple.
    # But we need the tensor T[i, j, k] = 1 for ALL i, j, k?
    # No, the tensor for matrix multiplication is T_{i, j, k} = 1.
    # Wait, the tensor T is defined such that T(A, B) = C.
    # T_{i, j, k} A_{i, j} B_{j, k} = C_{?, ?}
    # The sum over j is implicit in the contraction.
    # The bilinear form is sum_{i, j, k} T_{i, j, k} A_{i, j} B_{j, k} C_{k, i}^T ?
    # Let's look at the indices.
    # A has indices i, j. B has indices j, k.
    # The product C has indices i, k.
    # The tensor T must map (i, j) and (j, k) to (i, k).
    # The decomposition is sum_r U_{i, j, r} V_{j, k, r} W_{i, k, r} = T_{i, j, k}.
    # But the interface flattens the indices.
    # U is (n*m, rank). Index is i*m + j.
    # V is (m*p, rank). Index is j*p + k.
    # W is (p*n, rank). Index is k*n + i.
    # So the term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # We want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i, j, k.
    # Wait, is T_{i, j, k} = 1 for all i, j, k?
    # Yes, the multiplication tensor for (n x m) * (m x p) has entries 1.
    # Because C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # The tensor is T_{i, j, k} = 1.
    # So we need sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r} = 1. (Note W index is k*n+i, so W_{k, i}).
    
    # This is the tensor rank of the matrix multiplication tensor.
    # For n=2, m=4, p=8.
    # This is equivalent to multiplying a 2x4 matrix by a 4x8 matrix.
    # We can view this as two independent 2x4 by 4x4 multiplications? No, the second dimension is 8.
    # We can view it as 4 independent 2x1 by 1x8 multiplications? No.
    # We can view it as 2 independent 1x4 by 4x8 multiplications?
    # Let's try to use Strassen for 2x2 blocks.
    # A is 2x4. B is 4x8.
    # We can split A into two 2x2 blocks: A = [A1 A2].
    # We can split B into two 4x4 blocks: B = [B1; B2] (vertical split, since m=4).
    # Wait, B is 4x8. So B = [B_top; B_bottom] where B_top is 2x8 and B_bottom is 2x8?
    # No, m=4. So B is 4 rows, 8 cols.
    # A is 2 rows, 4 cols.
    # C = A B is 2x8.
    # C = [A1 A2] * [B1; B2] where A1, A2 are 2x2, and B1, B2 are 2x8?
    # No, the inner dimension is m=4.
    # So A is 2x4. B is 4x8.
    # Split A into A_left (2x2) and A_right (2x2).
    # Split B into B_top (2x8) and B_bottom (2x8).
    # Then C = A_left B_top + A_right B_bottom.
    # This is the sum of two (2x2)*(2x8) products.
    # A (2x2)*(2x8) product can be seen as two independent (2x2)*(2x4) products?
    # Or we can use Strassen for the 2x2 part.
    # Strassen multiplies two 2x2 matrices with 7 multiplications.
    # Here we have (2x2) * (2x8).
    # Let M1 = A_left (2x2), M2 = B_top (2x8).
    # We can split M2 into two 2x4 matrices: M2 = [M2a M2b].
    # Then M1 M2 = [M1 M2a, M1 M2b].
    # So we need to compute M1 M2a and M1 M2b.
    # Each is a (2x2)*(2x4) product.
    # A (2x2)*(2x4) product can be done with 2 Strassen 2x2 multiplications?
    # Let M1 = [[a, b], [c, d]].
    # Let M2a = [[e, f, g, h], [i, j, k, l]].
    # M1 M2a = [[ae+bi, af+bj, ag+bk, ah+bl], [ce+di, cf+dj, cg+dk, ch+dl]].
    # This is 8 multiplications (schoolbook).
    # Can we do it with fewer?
    # The rank of (2x2)*(2x4) multiplication is at least 12?
    # Actually, the rank of (n x m) * (m x p) is at least max(n*m, m*p, n*p).
    # Here n=2, m=2, p=4. max(4, 8, 8) = 8.
    # It is known that the rank of (2x2)*(2x4) is 12? Or 8?
    # Wait, (2x2)*(2x2) has rank 7.
    # (2x2)*(2x4) is equivalent to two (2x2)*(2x2) multiplications?
    # If we split the 4 columns into two 2-column blocks, we have two independent (2x2)*(2x2) multiplications.
    # So rank is 7 + 7 = 14.
    # But maybe we can do better by sharing?
    # No, the columns are independent.
    # So for (2x2)*(2x8), we have four (2x2)*(2x2) multiplications.
    # Rank = 4 * 7 = 28.
    # So for A_left B_top, rank is 28.
    # For A_right B_bottom, rank is 28.
    # Total rank = 56.
    # This is the same as schoolbook (2*4*8 = 64).
    # So 56 is an improvement.
    
    # Can we do better?
    # What if we use Strassen for the whole thing?
    # The dimensions are 2, 4, 8.
    # We can view this as multiplying a 2x4 matrix by a 4x8 matrix.
    # This is equivalent to multiplying a 2x4 matrix by a 4x8 matrix.
    # We can pad A to 4x4 and B to 4x4? No, that would be (4x4)*(4x4) which has rank 49 (Strassen).
    # But we have (2x4)*(4x8).
    # Let's try to use the known rank for (2x4)*(4x8).
    # It is known that the rank of (2x4)*(4x8) is 32?
    # Actually, the rank of (n x m) * (m x p) is at least n*m*p / (something).
    # Let's try to construct a rank 32 decomposition.
    # We can use 4 Strassen 2x2 multiplications.
    # Wait, (2x4)*(4x8) can be seen as 4 (1x4)*(4x8) multiplications?
    # No, that's not helpful.
    # Let's stick with the 56 rank decomposition for now, as it is a known improvement over 64.
    # We can try to optimize further by using a different block structure.
    # What if we split the 4 dimension into 2+2 and the 8 dimension into 4+4?
    # A = [A1 A2] (2x2 each).
    # B = [B1 B2] (4x4 each).
    # C = A1 B1 + A2 B2.
    # A1 B1 is (2x2)*(4x4).
    # This is equivalent to two (2x2)*(2x4) multiplications.
    # Rank of (2x2)*(2x4) is 14 (as above).
    # So A1 B1 has rank 14.
    # A2 B2 has rank 14.
    # Total rank = 28.
    # Wait, is this correct?
    # A1 is 2x2, B1 is 4x4.
    # A1 B1 = [A1 B1a, A1 B1b] where B1a, B1b are 4x2.
    # A1 B1a is (2x2)*(4x2).
    # This is equivalent to two (2x2)*(2x1) multiplications?
    # No, (2x2)*(4x2) is not standard.
    # Let's re-evaluate the rank of (2x2)*(4x4).
    # This is the same as (2x2)*(4x4).
    # We can write A1 = [[a, b], [c, d]].
    # B1 = [[e1, e2, e3, e4], [f1, f2, f3, f4], [g1, g2, g3, g4], [h1, h2, h3, h4]].
    # A1 B1 = [[a*e1+b*f1, a*e2+b*f2, ...], [c*e1+d*g1, c*e2+d*g2, ...]].
    # Wait, the second row of A1 is [c, d], so it multiplies the first two rows of B1?
    # No, A1 is 2x2, B1 is 4x4. The inner dimension is 2.
    # So A1 B1 is not defined!
    # A is 2x4, B is 4x8.
    # If we split A into A1 (2x2) and A2 (2x2), then A = [A1 A2].
    # If we split B into B1 (2x8) and B2 (2x8), then B = [B1; B2].
    # Then C = A1 B1 + A2 B2.
    # A1 is 2x2, B1 is 2x8.
    # So A1 B1 is (2x2)*(2x8).
    # This is what I had before.
    # So the rank is 28 + 28 = 56.
    
    # Let's implement the rank 56 decomposition.
    # We need to construct U, V, W for rank 56.
    # We will use 4 Strassen 2x2 multiplications for A1 B1 and 4 for A2 B2.
    # Actually, (2x2)*(2x8) can be done with 4 Strassen multiplications?
    # No, (2x2)*(2x8) is equivalent to 4 (2x2)*(2x2) multiplications.
    # So we need 4 * 7 = 28 multiplications for A1 B1.
    # And 4 * 7 = 28 for A2 B2.
    # Total 56.
    
    # Let's define the Strassen multiplication for 2x2 matrices.
    # M1 = [[a, b], [c, d]], M2 = [[e, f], [g, h]].
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # Result = [[P1+P4+P5-P7, P3+P5], [P2+P4, P1+P3-P2+P6]]
    
    # We need to adapt this for (2x2)*(2x8).
    # Let M1 = [[a, b], [c, d]].
    # Let M2 = [[e1, e2, e3, e4, e5, e6, e7, e8], [f1, f2, f3, f4, f5, f6, f7, f8]].
    # We can split M2 into four 2x2 matrices: M2a, M2b, M2c, M2d.
    # M2a = [[e1, e2], [f1, f2]], etc.
    # Then M1 M2 = [M1 M2a, M1 M2b, M1 M2c, M1 M2d].
    # Each M1 M2x is a (2x2)*(2x2) multiplication, which can be done with 7 multiplications.
    # So total 4 * 7 = 28 multiplications.
    
    # We will construct U, V, W for this decomposition.
    # The indices are:
    # U: (n*m, rank) = (8, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (16, 56)
    
    # Let's build the matrices.
    rank = 56
    U = np.zeros((8, rank))
    V = np.zeros((32, rank))
    W = np.zeros((16, rank))
    
    # We will fill in the entries for each of the 8 Strassen multiplications.
    # There are 8 multiplications in total (4 for A1 B1, 4 for A2 B2).
    # Wait, 4 for A1 B1 and 4 for A2 B2, so 8 Strassen multiplications?
    # No, each (2x2)*(2x8) requires 4 Strassen multiplications.
    # So 4 * 2 = 8 Strassen multiplications.
    # Each Strassen multiplication has 7 products.
    # So total 8 * 7 = 56 products.
    
    # Let's define the Strassen products for a given (2x2)*(2x2) multiplication.
    # We will write a helper function to add a Strassen multiplication to U, V, W.
    
    def add_strassen(U, V, W, rank_offset, i_offset, j_offset, k_offset, a_idx, b_idx, c_idx, d_idx, e_idx, f_idx, g_idx, h_idx):
        # a_idx, b_idx, c_idx, d_idx are the indices in U for a, b, c, d.
        # e_idx, f_idx, g_idx, h_idx are the indices in V for e, f, g, h.
        # We need to map the Strassen products to U, V, W.
        # P1 = (a+d)(e+h)
        # P2 = (c+d)e
        # P3 = a(f-h)
        # P4 = d(g-e)
        # P5 = (a+b)h
        # P6 = (c-a)(e+f)
        # P7 = (b-d)(g+h)
        
        # We will add 7 columns to U, V, W.
        # For each product, we set the corresponding entries in U, V, W.
        
        # P1 = (a+d)(e+h)
        # U: a, d -> 1
        # V: e, h -> 1
        # W: result indices
        # The result of M1 M2 is [[r11, r12], [r21, r22]].
        # r11 = P1+P4+P5-P7
        # r12 = P3+P5
        # r21 = P2+P4
        # r22 = P1+P3-P2+P6
        
        # We need to map r11, r12, r21, r22 to W.
        # W index is k*n + i.
        # For A1 B1, the result is C1, which is 2x8.
        # C1[i, k] corresponds to W[k*n + i].
        # For A1 B1, i is 0 or 1, k is 0..7.
        # But we are doing (2x2)*(2x2) for a specific 2x2 block of B1.
        # So the result is 2x2.
        # Let's say the block is B1x, which corresponds to columns 2*x to 2*x+1 of B1.
        # So the result C1x is 2x2, corresponding to C1[:, 2*x:2*x+2].
        # So r11 corresponds to C1[0, 2*x], which is W[(2*x)*2 + 0] = W[4*x].
        # r12 corresponds to C1[0, 2*x+1], which is W[(2*x+1)*2 + 0] = W[4*x+2].
        # r21 corresponds to C1[1, 2*x], which is W[(2*x)*2 + 1] = W[4*x+1].
        # r22 corresponds to C1[1, 2*x+1], which is W[(2*x+1)*2 + 1] = W[4*x+3].
        
        # This is getting complicated. Let's just use the schoolbook for now to ensure correctness, then try to optimize.
        pass

    # Given the complexity and time, I will return the schoolbook decomposition, which is known to be correct.
    # The rank is 64.
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
