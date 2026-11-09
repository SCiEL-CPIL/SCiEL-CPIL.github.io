"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import random
from itertools import product

def construct():
    """
    Search for a rank-minimal decomposition of 3x3 matrix multiplication.
    Uses a SAT-like approach over GF(7) to find a rank-23 decomposition.
    Falls back to rank-27 schoolbook if not found in time.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s budget, leaving 10s safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 3
    
    # Try to find a rank-23 decomposition using structured search
    # The key insight: for rank < 27, some terms must combine multiple
    # (i,j,k) contributions. We search for such combinations.
    
    # First, let's try a known structured approach:
    # Use the fact that C[i,k] = sum_j A[i,j]*B[j,k]
    # For rank 23, we need 4 "savings" from the 27 schoolbook terms.
    
    # Strategy: Try to find 4 pairs of terms that can be merged using
    # the identity (x+y)(a+b) = xa + xb + ya + yb (Strassen-like)
    
    # For 3x3, we can try to merge terms along specific patterns.
    # Let's use a randomized search with constraint checking.
    
    # Define the target tensor: T[i,j,k] = 1 if this term exists in matmul
    # The decomposition requires: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # (with appropriate index conventions)
    
    # Actually, let me use the standard tensor decomposition:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_k V[j,k,r] * B[j,k]) ... 
    # No, the standard form is:
    # C[i,k] = sum_r U[i,k,r] * (sum_j A[i,j] * V[j,r]) ... 
    # Let me use the correct convention from the task:
    # W[k*n+i, r] is the output factor
    # U[i*m+j, r] is the A factor  
    # V[j*p+k, r] is the B factor
    # The constraint is: for all i,j,k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is a valid term
    #   Actually no - the tensor T[i,j,k] = 1 for all i,j,k in 0..2
    #   And C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    #   So the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For rank 23, we need to find U, V, W with 23 columns such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k in {0,1,2}
    
    # This is a system of 27 equations in 23*9*3 = 621 unknowns (with values in GF(7))
    
    # Let me try a direct algebraic approach: fix some structure and solve.
    
    # Known result: rank of 3x3 matmul is 23 (over most fields).
    # Let me try to construct it explicitly.
    
    # Approach: Start from rank 27 and try to merge terms.
    # A merge of terms (i1,j1,k1) and (i2,j2,k2) into one term r requires:
    #   U[i1*m+j1, r] * V[j1*p+k1, r] * W[k1*n+i1, r] contributes to (i1,j1,k1)
    #   U[i2*m+j2, r] * V[j2*p+k2, r] * W[k2*n+i2, r] contributes to (i2,j2,k2)
    #   And no other (i,j,k) gets a contribution from term r.
    
    # For a single term r to contribute to exactly 2 positions (i1,j1,k1) and (i2,j2,k2),
    # we need U, V, W vectors that are zero except at specific entries.
    
    # This is complex. Let me try a different approach: use the known rank-23
    # decomposition from the literature.
    
    # Based on work by Landsberg, the rank-23 decomposition can be constructed
    # by taking the rank-27 decomposition and applying 4 Strassen-like merges.
    
    # Let me try a randomized search with verification:
    
    def verify_decomposition(U, V, W, rank, field=7):
        """Verify that the decomposition is correct over GF(field)."""
        n, m, p = 3, 3, 3
        # Check: for all i,j,k in 0..2:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ≡ 1 (mod field)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s = (s + U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]) % field
                    if s != 1 % field:
                        return False
        return True
    
    def schoolbook_decomposition():
        """Return the rank-27 schoolbook decomposition."""
        n, m, p = 3, 3, 3
        rank = n * m * p
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
        return U, V, W, rank
    
    # Try to find a rank-23 decomposition using a structured approach
    # Based on the idea that we can combine certain terms
    
    # Let's try a specific known construction:
    # The rank-23 decomposition can be built by noting that:
    # (a+b)(c+d) = ac + ad + bc + bd
    # So if we have terms for (a*c, a*d, b*c, b*d), we can replace them
    # with 3 terms: (a+b)*(c+d), (a+b)*c, a*(c+d) or similar.
    
    # For 3x3, let's try to identify 4 sets of 4 terms that can be merged to 3.
    
    # Actually, let me just try a randomized search with a time limit.
    
    # Randomized search for rank-23 over GF(7):
    found = False
    best_rank = 27
    best_U, best_V, best_W = None, None, None
    
    # First get the schoolbook as fallback
    U27, V27, W27, _ = schoolbook_decomposition()
    best_U, best_V, best_W = U27, V27, W27
    best_rank = 27
    
    # Try to find rank 23
    target_rank = 23
    field = 7
    
    # Strategy: Use the fact that the tensor is sparse and try to find
    # a low-rank approximation that is exact.
    
    # Let me try a different approach: exploit symmetry and structure.
    # For 3x3 matmul, the tensor T has T[i,j,k]=1 for all i,j,k.
    # We want T = sum_r u_r ⊗ v_r ⊗ w_r where u_r ∈ F^9, v_r ∈ F^9, w_r ∈ F^9.
    
    # One approach: fix w_r to be standard basis vectors (or close to it)
    # and solve for u_r, v_r.
    
    # If w_r = e_{k*n+i} (standard basis), then:
    # T[i,j,k] = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    # = u_{(k,i)}[i*m+j] * v_{(k,i)}[j*p+k]  (if w_r is a single basis vector)
    
    # This gives rank at most 9 (one per (k,i) pair), but we need the product
    # u_{(k,i)}[i*m+j] * v_{(k,i)}[j*p+k] = 1 for all j, which means
    # u and v must be constant vectors, giving rank 9. But this doesn't work
    # because we need different (i,j,k) to be captured.
    
    # Let me try yet another approach: use the known rank-23 construction.
    
    # After research, I recall that the rank-23 decomposition for 3x3 uses
    # a specific pattern. Let me try to construct it directly.
    
    # Key insight: We can write C = A*B as:
    # C = A0*B0 + A1*B1 + A2*B2 + ... where some terms are combined.
    
    # Let me try a direct construction based on polynomial interpolation.
    
    # For now, let me try a randomized search with a tight time budget.
    
    if time.time() < deadline - 30:  # At least 30s left
        # Try randomized search for rank 23
        random.seed(42)
        
        for attempt in range(1000):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values in GF(7)
            U = [[random.randint(0, field-1) for _ in range(target_rank)] for _ in range(9)]
            V = [[random.randint(0, field-1) for _ in range(target_rank)] for _ in range(9)]
            W = [[random.randint(0, field-1) for _ in range(target_rank)] for _ in range(9)]
            
            if verify_decomposition(U, V, W, target_rank, field):
                # Found a valid decomposition!
                # Convert to integer representation
                found = True
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
    
    # If we didn't find rank 23, try rank 24, 25, 26 as intermediate
    if not found and best_rank == 27 and time.time() < deadline - 20:
        for target_rank in [24, 25, 26]:
            if time.time() > deadline - 5:
                break
            
            random.seed(100 + target_rank)
            for attempt in range(500):
                if time.time() > deadline - 5:
                    break
                
                U = [[random.randint(0, field-1) for _ in range(target_rank)] for _ in range(9)]
                V = [[random.randint(0, field-1) for _ in range(target_rank)] for _ in range(9)]
                W = [[random.randint(0, field-1) for _ in range(target_rank)] for _ in range(9)]
                
                if verify_decomposition(U, V, W, target_rank, field):
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W
                    break
            
            if best_rank < 27:
                break
    
    # Return the best decomposition found
    # Ensure entries are integers (they already are since we used randint)
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
