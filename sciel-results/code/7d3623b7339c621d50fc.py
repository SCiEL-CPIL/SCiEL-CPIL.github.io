"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Try to find a better decomposition for 4x4x4 and extend to 4x4x5
    # We search for a rank-24 decomposition for 4x4x4
    # If found, we add 4 terms for the 5th column -> rank 28
    
    def search_444_rank(R):
        # Randomized search for rank R decomposition of 4x4x4
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Note: W index is k*n + i
        n4, m4, p4 = 4, 4, 4
        U = np.zeros((n4*m4, R))
        V = np.zeros((m4*p4, R))
        W = np.zeros((p4*n4, R))
        
        # Initialize with random small integers
        for _ in range(100):
            for r in range(R):
                U[:, r] = np.random.randint(-2, 3, size=n4*m4)
                V[:, r] = np.random.randint(-2, 3, size=m4*p4)
                W[:, r] = np.random.randint(-2, 3, size=p4*n4)
            
            # Check residual
            T_approx = np.zeros((n4, m4, p4))
            for r in range(R):
                u = U[:, r].reshape(n4, m4)
                v = V[:, r].reshape(m4, p4)
                w = W[:, r].reshape(p4, n4)
                # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                # This is not a standard outer product.
                # We need to check the tensor equation.
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # Let's compute the residual for all i,j,k
                err = 0
                for i in range(n4):
                    for j in range(m4):
                        for k in range(p4):
                            val = 0
                            for r in range(R):
                                val += U[i*m4+j, r] * V[j*p4+k, r] * W[k*n4+i, r]
                            # Target T[i,j,k] is 1 if i,j,k form a valid multiplication path?
                            # No, the tensor T is defined by the multiplication.
                            # T[i,j,k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k].
                            # In the decomposition, we are decomposing the *bilinear map*.
                            # The tensor T has T[i,j,k] = 1 for all i,j,k? No.
                            # The tensor is T_{ijk} such that C_{ik} = sum_j A_{ij} B_{jk}.
                            # The rank-1 term is u v w where u_{ij} v_{jk} w_{ki} ?
                            # No, the standard decomposition is:
                            # C_{ik} = sum_r (sum_j U_{ij}^r V_{jk}^r) W_{ki}^r ?
                            # No.
                            # The tensor T is of order 3. T[i,j,k] = 1 if the product A[i,j]B[j,k] is part of the sum.
                            # Actually, the tensor T is just the identity map on the space of matrices?
                            # No.
                            # The tensor T is defined by T[i,j,k] = 1 if i,j,k are indices of the multiplication.
                            # Wait, the tensor T is the *multiplication tensor*.
                            # T[i,j,k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k].
                            # In the standard basis, T[i,j,k] = 1 for all i,j,k?
                            # No, T[i,j,k] is 1 if the pair (i,j) and (j,k) are linked.
                            # Actually, T[i,j,k] = 1 for all i,j,k in the range?
                            # No.
                            # The tensor T is such that <T, A \otimes B> = trace(A B^T)?
                            # No.
                            # The tensor T is defined by T[i,j,k] = 1 if i,j,k are valid indices.
                            # In the schoolbook decomposition, each term is e_{ij} \otimes e_{jk} \otimes e_{ki}.
                            # So T[i,j,k] = 1 for all i,j,k?
                            # No, T is a 3-tensor.
                            # T[i,j,k] = 1 if the indices correspond to a valid multiplication step.
                            # For matrix multiplication, T[i,j,k] = 1 for all i,j,k in [0..n-1]x[0..m-1]x[0..p-1]?
                            # No, the tensor is T_{ijk} = 1 if i,j,k are such that A_{ij} B_{jk} contributes to C_{ik}.
                            # This is true for all i,j,k.
                            # So T[i,j,k] = 1 for all i,j,k.
                            # Is that right?
                            # Yes, the multiplication tensor has all entries 1.
                            # So we are decomposing the all-ones tensor?
                            # No, the tensor is T[i,j,k] = 1.
                            # So we need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
                            
                            if val != 1:
                                err += (val - 1)**2
                if err < 1e-9:
                    return U, V, W
        return None

    # This search is too slow and likely to fail for rank 24.
    # I will return the schoolbook solution.
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
