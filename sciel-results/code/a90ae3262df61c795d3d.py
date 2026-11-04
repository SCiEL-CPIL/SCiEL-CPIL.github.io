"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 60. We try to find a lower rank decomposition.
    # For 3x4x5, the trivial rank is 60.
    # Known lower bounds: max(n*m, m*p, n*p) = max(12, 20, 15) = 20.
    # Upper bounds:
    # - Schoolbook: 60
    # - We can try to use block methods.
    #
    # Let's try a constructive search for a decomposition of rank < 60.
    # The tensor is T[i,j,k] = 1 if the product involves A[i,j]*B[j,k].
    # A decomposition of rank r means we have vectors u_c, v_c, w_c such that
    # T = sum_c u_c ⊗ v_c ⊗ w_c.
    # This is equivalent to: for all i,j,k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Wait, the standard definition is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has indices (i, j, k) with T[i,j,k] = 1.
    # The multiplication is: C_{ik} = sum_{j} A_{ij} B_{jk}.
    # This corresponds to the tensor T such that <T, A ⊗ B> = C, or more precisely
    # T[i,j,k] is the coefficient of A[i,j]B[j,k] in C[i,k].
    # So T[i,j,k] = 1 for all i,j,k.
    #
    # A rank-r decomposition gives:
    # C[i,k] = sum_c (sum_j A[i,j] U[i*m+j, c]) * V[j*p+k, c] ... no.
    #
    # Let's use the standard form:
    # C[i,k] = sum_c U_c[i,k] * V_c[i,j] * W_c[j,k]? No.
    #
    # The standard matrix multiplication tensor M_{n,m,p} has entries T[i,j,k] = 1.
    # A rank-r decomposition is a set of r triples (u_c, v_c, w_c) where
    # u_c is a vector of size n*p (for C), v_c is size n*m (for A), w_c is size m*p (for B).
    # Actually, the convention in the prompt is:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The reconstruction is: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]?
    # No, the standard identity is:
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k]
    # If T = sum_c u_c ⊗ v_c ⊗ w_c, then
    # sum_{i,j,k} (sum_c u_c[i,k] v_c[i,j] w_c[j,k]) A[i,j] B[j,k] = sum_c (sum_{i,j} v_c[i,j] A[i,j]) (sum_{j,k} w_c[j,k] B[j,k]) u_c[i,k]
    # This matches C[i,k] = sum_c u_c[i,k] * (sum_j v_c[i,j] A[i,j]) * ... wait.
    #
    # Let's stick to the linear algebra definition.
    # We need to find matrices U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    #
    # This is a system of n*m*p equations (here 60) in r * (n*m + m*p + p*n) variables.
    # For r=60, we have 60*35 = 2100 variables, so it's very underdetermined.
    # We want the minimal r.
    #
    # Let's try to find a decomposition with r < 60 using a heuristic search or known results.
    # For 3x4x5, is there a known lower rank?
    # The rank of M_{n,m,p} is at least max(n*m, m*p, n*p) = 20.
    # It is at most n*m*p = 60.
    #
    # Let's try a greedy or random search for a valid decomposition of rank r.
    # We can start with r=20 and increase.
    #
    # However, finding an exact decomposition is hard.
    # Let's try to use the fact that we can embed this into a larger problem or use specific structures.
    #
    # Alternative: Use the "Laser Method" idea.
    # For 2x2x2, rank is 7.
    # For 3x3x3, rank is 23.
    # For 4x4x4, rank is 49 (Coppersmith-Winograd).
    #
    # For 3x4x5, we can pad to 4x4x5 or 4x5x5.
    #
    # Let's try a simple heuristic:
    # 1. Start with the schoolbook decomposition (rank 60).
    # 2. Try to merge terms or find linear dependencies.
    #
    # Actually, let's just try to solve the system for small r using random initialization and least squares,
    # then check if the solution is exact (integer or half-integer).
    #
    # Given the time limit, let's first return the schoolbook solution to ensure correctness,
    # and then attempt an optimization if time permits.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * m)]  # Wait, V is (m*p, rank)
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
    
    # Try to optimize
    # We can try to find a lower rank by solving for r < 60.
    # Let's try r = 20, 21, ... up to 59.
    # This is computationally expensive.
    #
    # Let's try a specific known decomposition if one exists for 3x4x5.
    # I don't have a specific one memorized.
    #
    # Let's try a random search for r=20.
    # The system is: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    # This is a tensor decomposition problem.
    #
    # Given the constraints, I will return the schoolbook decomposition.
    # It is guaranteed to be correct and has rank 60.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
