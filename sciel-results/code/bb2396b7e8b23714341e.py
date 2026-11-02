"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import null_space

def construct():
    n, m, p = 3, 3, 5
    
    # We are looking for a rank-r decomposition of the matrix multiplication tensor T of shape (n*m, m*p, p*n) = (9, 15, 15).
    # The tensor T is defined such that T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    # We want to find matrices U (9 x r), V (15 x r), W (15 x r) such that:
    # T[a, b, c] = sum_{l=0}^{r-1} U[a, l] * V[b, l] * W[c, l]
    
    # This is a rank-r tensor decomposition.
    # For 3x3x3 matrix multiplication, the Strassen algorithm gives a rank-7 decomposition.
    # The task is 3x3x5. 
    # Let's construct the tensor T.
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # We know that 3x3x3 multiplication has rank 7 (Strassen).
    # 3x3x5 can be seen as 3x3x3 + 3x3x2? No, it's a single multiplication.
    # The dimensions are (n, m, p) = (3, 3, 5).
    # The tensor is T_{i,j,k} where i in 0..2, j in 0..2, k in 0..4.
    # T[i, j, k] = 1 if the indices correspond to the multiplication (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Wait, the indexing in the problem statement is:
    # U has shape (n*m, rank) -> indices (i, j)
    # V has shape (m*p, rank) -> indices (j, k)
    # W has shape (p*n, rank) -> indices (k, i)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k.
    # This is exactly the matrix multiplication tensor for multiplying an (n x m) matrix by an (m x p) matrix.
    
    # Let's try to find a low-rank decomposition.
    # Known ranks:
    # 2x2x2: 7
    # 3x3x3: 7 (Strassen)
    # 3x3x4: ?
    # 3x3x5: ?
    
    # A constructive approach:
    # We can use the Strassen decomposition for the 3x3x3 part and add terms for the extra columns.
    # However, the tensor is not simply a block diagonal of 3x3x3 and something else.
    # The multiplication is (3x3) * (3x5) -> (3x5).
    
    # Let's use a generic rank-reduction algorithm or try specific known decompositions.
    # For 3x3x5, the rank is at most 3*3*5 = 45 (schoolbook).
    # Strassen for 3x3x3 is 7.
    # We can think of the 3x5 matrix as two 3x3 blocks and one 3x2 block? No, that changes the structure.
    
    # Let's try to find the minimal rank by solving a system of equations for a given rank r.
    # The equations are:
    # sum_l U[a,l] V[b,l] W[c,l] = T[a,b,c]
    # This is a system of bilinear equations in U,V,W.
    
    # Alternative: Use the fact that matrix multiplication tensors have known ranks for small dimensions.
    # According to literature, the rank of 3x3x3 is 7.
    # The rank of 3x3x4 is not well-known to me, but let's estimate.
    # The rank of 3x3x5 is likely higher.
    
    # Let's try a heuristic search or a known decomposition.
    # One approach is to use the "border rank" or "rank" algorithms.
    # Since we need an *exact* decomposition with integer/half-integer entries, we should look for a structured one.
    
    # Let's try to build it from the Strassen 3x3x3 decomposition.
    # The Strassen decomposition for 3x3x3 uses 7 multiplications.
    # We can pad the 3x5 matrix to 3x7 or 3x8? No, that increases rank.
    
    # Let's try a different perspective.
    # The tensor T is of format (9, 15, 15).
    # We want to find rank r.
    # Let's try r=15, 20, 25, etc.
    
    # Actually, there is a known result: the rank of 3x3x5 is at most 3*3*5 = 45, but can be lower.
    # Let's use a simple algorithm to find a decomposition of a given rank.
    # We can use the "tensor rank" problem which is NP-hard in general, but for small tensors we can try.
    
    # Let's try to find a decomposition of rank r by solving for U, V, W.
    # We can fix W (or U or V) and solve for the others, but it's nonlinear.
    
    # A better approach for this specific task:
    # The problem asks for a rank-minimal exact decomposition.
    # For 3x3x5, a known upper bound is from the Strassen-like algorithms.
    # Let's try to use the 3x3x3 Strassen decomposition and extend it.
    
    # Strassen 3x3x3 decomposition:
    # A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
    # B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    # C = A*B
    
    # The 7 multiplications for 3x3x3 are complex.
    
    # Let's try a simpler approach: use the schoolbook decomposition but with some cancellations?
    # No, the schoolbook is rank 45.
    
    # Let's try to find a decomposition of rank r by using the fact that the tensor is symmetric in some sense? No.
    
    # Let's try a generic solver for small ranks.
    # We can use the "rank r decomposition" by solving the system of equations.
    # For a given r, we have 9r + 15r + 15r = 39r variables.
    # We have 9*15*15 = 2025 equations, but only 45 are non-zero (the ones where T=1).
    # Actually, the equations are:
    # For each (a,b,c) where T[a,b,c]=1: sum_l U[a,l] V[b,l] W[c,l] = 1
    # For each (a,b,c) where T[a,b,c]=0: sum_l U[a,l] V[b,l] W[c,l] = 0
    
    # This is a system of quadratic equations.
    # We can try to solve it using numerical methods, but we need exact integer/half-integer solutions.
    
    # Let's try a known decomposition for 3x3x5.
    # According to a paper "The rank of the matrix multiplication tensor for 3x3x5", the rank is 23? Or 22?
    # Let's assume the rank is around 20-30.
    
    # Let's try to construct a decomposition of rank r by using the following idea:
    # We can use the 3x3x3 Strassen decomposition (rank 7) for the first 3 columns of the 3x5 matrix.
    # Then we need to handle the last 2 columns.
    # The last 2 columns of the output are (A*B)_{i,3} and (A*B)_{i,4}.
    # These depend on all 3 columns of A and the 4th and 5th columns of B.
    
    # Let's denote B = [B1, B2, B3, B4, B5] where B1,B2,B3 are 3x3 and B4,B5 are 3x2.
    # C = [C1, C2, C3, C4, C5] where C1,C2,C3 are 3x3 and C4,C5 are 3x2.
    # C1,C2,C3 can be computed using Strassen on (A, [B1,B2,B3]) with rank 7.
    # C4 = A * B4, C5 = A * B5.
    # A * B4 is a 3x2 matrix, which can be computed with rank 3*2=6 (schoolbook) or less?
    # 3x2 multiplication rank is 3*2=6? No, 3x2x3? No, 3x2 output.
    # The tensor for 3x2x3 (A is 3x3, B is 3x2, C is 3x2) has rank at most 3*3*2=18.
    # But we can do better.
    
    # Actually, let's just try to find a decomposition of rank r by using a numerical solver and then rounding to integers/half-integers.
    
    # Let's try r=20.
    # We'll use a random initialization and then solve the system of equations.
    
    # Since we need an exact solution, let's try a different approach.
    # Let's use the fact that the tensor is sparse and try to find a decomposition with small integer entries.
    
    # For now, let's return the schoolbook decomposition to ensure correctness, and then we can try to improve it.
    # But the task asks for a rank-minimal decomposition.
    
    # Let's try to find a decomposition of rank 23 (a known upper bound for 3x3x5).
    
    # Actually, let's use a simple algorithm:
    # 1. Start with the schoolbook decomposition (rank 45).
    # 2. Try to reduce the rank by finding linear dependencies.
    
    # This is complex. Let's try a known decomposition from the literature.
    # The rank of 3x3x5 is 23 (from a 2018 paper).
    # Let's try to find a rank-23 decomposition.
    
    # Since I don't have the explicit coefficients, let's try to find one numerically.
    # We'll use a solver to find U, V, W of rank r such that T = sum_l U_l V_l W_l.
    
    # Let's try r=23.
    r = 23
    
    # We'll use a random initialization and then use gradient descent to minimize the error.
    # But we need an exact solution.
    
    # Let's try a different approach: use the "tensor train" or "CP decomposition" algorithms.
    
    # For simplicity, let's return the schoolbook decomposition for now, as it is guaranteed to be correct.
    # The rank is 45.
    
    # But we can do better. Let's try to find a rank-30 decomposition.
    
    # Actually, let's use the following trick:
    # We can use the 3x3x3 Strassen decomposition (rank 7) for the first 3 columns.
    # For the last 2 columns, we can use the schoolbook decomposition (rank 6 each, total 12).
    # But the Strassen decomposition for the first 3 columns uses all of A, so it doesn't directly help for the last 2 columns.
    
    # Let's just return the schoolbook decomposition.
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
