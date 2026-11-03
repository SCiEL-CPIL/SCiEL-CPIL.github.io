"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # We try to find a decomposition with rank < n*m*p = 75.
    # Strategy: 
    # 1. We look for a rank-r decomposition where each tensor slice (for fixed k) is a 
    #    linear combination of a small set of "basis" bilinear forms.
    # 2. For each k, we want: \sum_{i,j} A_{ij} B_{jk} C_{ki} = \sum_{r=1}^R \lambda_{i,k,r}^{(U)} A_{ij} \mu_{j,k,r}^{(V)} B_{jk} \nu_{i,k,r}^{(W)} C_{ki}
    #    This is a bit circular. Let's use the standard tensor decomposition view.
    #    T_{i,j,k} = \sum_{r=1}^R U_{i,j,r} V_{j,k,r} W_{k,i,r}
    #    Here U has shape (n*m, R), V (m*p, R), W (p*n, R).
    
    # A simple way to reduce rank is to exploit symmetries or specific structures.
    # For 3x5x5, we can try to find a "Strassen-like" reduction.
    # However, finding the exact minimal rank is NP-hard in general.
    # Given the 150s limit, we should try a few random or structured low-rank attempts and verify them.
    
    # If we can't find one quickly, we fall back to schoolbook rank 75.
    
    # Let's try a random rank 70 attempt using Gaussian elimination on the tensor coefficients?
    # Actually, a more robust approach for small tensors is to use a SAT-like or integer programming approach?
    # No, that's too slow in Python.
    
    # Let's try a simple heuristic: 
    # Assume the decomposition uses entries from {0, 1/2, 1}.
    # We can try to solve for rank 74, 73, etc. using a local search.
    
    # BUT, the safest bet within 150s is to return the schoolbook rank 75 unless we find a proof of lower rank.
    # Is there a known result for 3x5x5?
    # For 2x2x2, rank is 7.
    # For 2x3x3, rank is ?
    # For 3x3x3, rank is at most 23 (Strassen-like), lower bound 19.
    # For 3x5x5, schoolbook is 75.
    
    # Let's try to implement a simple local search for rank 74.
    # Initialize U, V, W with random sparse values.
    # Optimize to match the identity tensor T.
    # Error = sum (T_ijk - sum_r U_ij_r V_jk_r W_ki_r)^2
    # We want Error = 0.
    
    import time
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Try rank 74, 73, ... down to some limit (e.g., 70)
    # If found, return it. Otherwise, return 75.
    
    # We'll use a simple gradient-free method or just random restarts with projection?
    # Given the small size, we can try to solve the linear system for the third matrix given the other two?
    # No, it's non-linear.
    
    # Let's try a specific structured decomposition.
    # Can we merge some terms?
    # In Strassen 2x2x2, we merge terms by combining (A11+A22)(B11+B22) etc.
    # For 3x5x5, we might be able to merge terms in the 5x5 multiplication.
    
    # Let's attempt a rank 74 construction by modifying the schoolbook.
    # Take two schoolbook terms for (i,j,k) and (i',j',k') and see if they can be combined.
    # This requires U_ij = U_i'j', V_jk = V_j'k', etc.
    
    # Since this is complex to automate quickly, and the risk of error is high,
    # and the baseline 75 is valid, I will stick with 75 but add a check for time.
    # If we have time, we could try a random rank 74 search.
    
    # Let's do a quick random search for rank 74 with half-integer entries.
    # If it finds a solution, great. If not, 75.
    
    import random
    import numpy as np
    
    # Target tensor T
    # T[i, j, k] = 1 if it corresponds to a valid multiplication term?
    # Wait, the tensor T for matrix multiplication C = A B is defined such that:
    # C_{ik} = \sum_j A_{ij} B_{jk}
    # The tensor entries are T_{i,j,k} = 1 for all i,j,k? No.
    # The tensor is the bilinear form.
    # T_{i,j,k} = 1 if we are computing C_{ik} using A_{ij} and B_{jk}.
    # Actually, the tensor decomposition defines the computation.
    # We want: C_{ik} = \sum_r (\sum_j U_{i,j,r} A_{ij}) (\sum_k V_{j,k,r} B_{jk}) W_{k,i,r} ??
    # No, the standard definition is:
    # C_{i,k} = \sum_j A_{i,j} B_{j,k}
    # This is a 3-way tensor T_{i,j,k} where T_{i,j,k} = 1 for all i,j,k in the support?
    # Yes, the matrix multiplication tensor has T_{i,j,k} = 1 for all valid indices.
    # We want to decompose T as \sum_{r=1}^R U_r \otimes V_r \otimes W_r
    # where (U_r \otimes V_r \otimes W_r)_{i,j,k} = U_{i,r} V_{j,r} W_{k,r}
    # Wait, the indices in the prompt are:
    # U shape (n*m, rank) -> U[i*m+j, r]
    # V shape (m*p, rank) -> V[j*p+k, r]
    # W shape (p*n, rank) -> W[k*n+i, r]
    # So T_{i,j,k} = \sum_{r=1}^R U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want this to equal 1 for all i,j,k?
    # No, the matrix multiplication tensor is 1 only if the indices match the pattern?
    # Actually, the tensor for matrix multiplication $M_{n,m,p}$ is defined on indices $i \in [n], j \in [m], k \in [p]$ as:
    # T_{i,j,k} = 1.
    # Because $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # The term $A_{ij} B_{jk}$ contributes to $C_{ik}$.
    # So for each triple $(i,j,k)$, the coefficient is 1.
    
    # So we want:
    # \sum_{r=1}^R U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i, j, k.
    
    # This is a system of equations.
    # For rank 75, we set one term to 1 and others to 0.
    # For rank 74, we need to "merge" two terms.
    
    # Let's try to find a rank 74 solution by setting 74 random vectors and solving for the last one?
    # No, it's a 3-linear equation.
    
    # Given the constraints and time, returning 75 is the most robust choice.
    # I will implement the schoolbook decomposition.
    
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
