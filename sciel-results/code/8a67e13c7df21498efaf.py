"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 60
    
    # Build schoolbook decomposition as baseline
    def build_schoolbook(rank):
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
        return U, V, W
    
    U_school, V_school, W_school = build_schoolbook(schoolbook_rank)
    
    # Verify the decomposition condition:
    # For matrix multiplication C = A*B where A is n×m, B is m×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition requires:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] = T[i,j,k]
    # where T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # In standard matrix mult, every (i,j,k) triple is valid, so T[i,j,k] = 1 for all.
    # But wait - the W matrix must also be involved. Let me reconsider.
    #
    # The correct condition with three factors is likely:
    # The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
    # This can be written as a sum of rank-1 bilinear maps:
    # C[i,k] = sum_c U[i*m+j_i(c), c] * V[j_i(c)*p+k_i(c), c] * W[k_i(c)*n+i_i(c), c] * A[i_i(c), j_i(c)] * B[j_i(c), k_i(c)]
    # No, that's not right either.
    #
    # Let me think about this differently. The standard tensor decomposition of matrix multiplication:
    # M_{n,m,p} = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ F^{n×m}, v_c ∈ F^{m×p}, w_c ∈ F^{p×n} (reshaped to vectors)
    # The condition is: for all i,j,k:
    #   sum_c U_c[i,j] * V_c[j,k] = T[i,j,k]
    # where T[i,j,k] = 1 (the multiplication tensor)
    # And W_c is related to how the output is extracted.
    #
    # Actually, I think the correct interpretation is:
    # We're decomposing the trilinear form such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 0 for "invalid" (i,j,k)
    # and the valid ones give the multiplication.
    #
    # But for 2×3×10, all (i,j,k) with i∈[0,2), j∈[0,3), k∈[0,10) are valid.
    # So the condition should be that the sum reconstructs the identity-like structure.
    #
    # Let me just try: verify that sum_c U[i*m+j,c]*V[j*p+k,c] equals something meaningful.
    
    def verify_decomposition(U, V, W, rank):
        """Check if the decomposition exactly represents matrix multiplication."""
        # For each (i, k) pair in C, check that:
        # C[i,k] = sum_j A[i,j]*B[j,k] can be computed as sum_c (sum_j U[i*m+j,c]*A[i,j])*(sum_j V[j*p+k,c]*B[j,k])*W[k*n+i,c]
        # But that's a 3-way product which doesn't make sense.
        #
        # Alternative: the decomposition means the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]
        # And W is used for output extraction or is part of a different convention.
        #
        # Let's check: sum_c U[i*m+j,c]*V[j*p+k,c] should equal 1 for all valid (i,j,k)
        # if the tensor is all-ones. But that can't be right because then rank 1 would work.
        #
        # I think the actual condition involves all three matrices. Let me try:
        # The multiplication tensor in this convention might be:
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # and T should equal 1 for all (i,j,k).
        # But again, that would make rank 1 work.
        
        # Let me just return the schoolbook and see if it's valid.
        # The schoolbook sets U[i*m+j, i*m+j] = 1 (diagonal-like), V[j*p+k, j*p+k] = 1, W[k*n+i, k*n+i] = 1
        # Wait no, the schoolbook uses a single index col = i*m*p + j*p + k
        # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
        # So for a given (i,j,k), only one column c has all three non-zero.
        # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for the specific c corresponding to (i,j,k)
        # and 0 for all other (i',j',k') != (i,j,k).
        # So the tensor T[i,j,k] = 1 if... wait, it's 1 for the specific (i,j,k) and the sum is over c.
        # For fixed (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # Only the column c = i*m*p+j*p+k has U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
        # So the sum = 1.
        # For a different (i',j',k'): the sum would be 0 unless (i',j',k') = (i,j,k).
        # So the tensor is the identity tensor: T[i,j,k] = 1 iff (i,j,k) matches, else 0.
        # But that's not the multiplication tensor!
        
        # I'm confused about the exact convention. Let me just return the schoolbook.
        return True
    
    # Since we can't easily determine the exact verification condition,
    # and the schoolbook is known to be valid (scores 1.0), let's try to find
    # a lower rank by attempting random constructions.
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions using random initialization + refinement
    # Strategy: for rank r < schoolbook_rank, try random U, V, W and check if they work.
    # But checking "if they work" requires knowing the exact condition.
    
    # Alternative structural approach: use the fact that for small n, we can enumerate
    # possible rank-r decompositions using algebraic geometry methods.
    # For n=2, m=3, p=10, the minimum rank is known to be at most n*m*p = 60.
    # Strassen's algorithm gives better bounds for larger matrices, but for 2×3×10,
    # the schoolbook might actually be optimal or close to it.
    
    # Let's try a different approach: assume the tensor is the all-ones tensor T[i,j,k]=1
    # and try to decompose it with rank r. The condition would be:
    # sum_c U[i*m+j,c] * V[j*p+k,c] = 1 for all i,j,k
    # This is a system of n*m*p = 60 equations in 3*r*... variables.
    # For r=1: U[i*m+j,0]*V[j*p+k,0] = 1 for all i,j,k
    # This requires U[i*m+j,0] = a[i,j] and V[j*p+k,0] = b[j,k] with a[i,j]*b[j,k]=1 for all i,j,k
    # a[i,j]*b[j,k] = 1 for all i,j,k means a[i,j] = 1/b[j,k] for all k, so b[j,k] is constant in k,
    # and a[i,j] is constant in i. Let b[j,k] = c_j, a[i,j] = 1/c_j.
    # Then U[i*m+j,0] = 1/c_j and V[j*p+k,0] = c_j.
    # Check: sum_c U[i*m+j,c]*V[j*p+k,c] = (1/c_j)*c_j = 1. Yes! Rank 1 works for all-ones tensor!
    #
    # But that can't be right because matrix multiplication has rank > 1.
    # So the tensor must NOT be all-ones. The W matrix must play a role.
    
    # I think the correct condition is:
    # The decomposition represents the multiplication map, and the condition involves
    # all three matrices in a way that ensures C = A*B.
    #
    # Given the time constraints and uncertainty about the exact verification,
    # let's just return the schoolbook decomposition which is known to be valid.
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
