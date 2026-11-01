"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # The key insight: 2x2 * 2x14 can be viewed as 2x2 matrix multiplication
    # over the ring of 2x2 matrices, by grouping the 14 columns into 7 blocks of 2.
    # A (2x2) times B (2x14) = A * [B0 | B1 | ... | B6] where each Bi is 2x2.
    # This is equivalent to multiplying a 2x2 matrix (over M_2(R)) by another
    # 2x2 matrix (over M_2(R)) in a block sense, but actually it's just:
    # C = A * B where A is 2x2 and B is 2x14.
    # 
    # Actually, let's think differently. The tensor for (2,2,14) matrix mult has:
    # T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,13}
    # 
    # The rank of the (2,2,p) matrix multiplication tensor:
    # For p=2: rank 7 (Strassen)
    # For p=14: we can use 7 independent Strassen decompositions -> rank 49
    # 
    # Can we do better? The lower bound is max(n*m, m*p, p*n) = max(4, 28, 28) = 28
    # 
    # Let's try to find a rank < 49 decomposition.
    # 
    # Strategy: Use the structure that n=2, m=2 are small.
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    #
    # For fixed (i,j), we need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all k
    #
    # Let's try a structured approach with rank 48, 47, etc.
    # First, verify the rank-49 Strassen solution works.
    
    r49 = _construct_strassen_49(n, m, p)
    if r49 is not None:
        # Try to improve
        best_result = r49
        best_rank = 49
        
        # Try rank reduction from 48 down
        for target_rank in range(48, 27, -1):
            if time.time() > deadline - 15:
                break
            result = _try_structured_rank(n, m, p, target_rank)
            if result is not None:
                best_result = result
                best_rank = target_rank
                break
        
        return best_result
    
    # Fallback: schoolbook rank-56
    return _construct_schoolbook(n, m, p)


def _construct_strassen_49(n, m, p):
    """
    Construct a rank-49 decomposition using Strassen's algorithm on 7 blocks of 2x2x2.
    """
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for pair_idx in range(7):
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        base_col = pair_idx * 7

        # Strassen's 7 products for 2x2 matrix multiplication
        # M1 = (A00+A11)(B00+B11)
        U[0][base_col + 0] = 1
        U[3][base_col + 0] = 1
        V[k1][base_col + 0] = 1
        V[p + k2][base_col + 0] = 1
        W[k1 * n + 0][base_col + 0] = 1
        W[k2 * n + 1][base_col + 0] = 1

        # M2 = A11(B10-B00)
        U[2][base_col + 1] = 1
        U[3][base_col + 1] = 1
        V[k1][base_col + 1] = 1
        W[k1 * n + 1][base_col + 1] = 1
        W[k2 * n + 1][base_col + 1] = -1

        # M3 = A00(B01-B11)
        U[0][base_col + 2] = 1
        V[k2][base_col + 2] = 1
        V[p + k2][base_col + 2] = -1
        W[k2 * n + 0][base_col + 2] = 1
        W[k2 * n + 1][base_col + 2] = 1

        # M4 = A11(A00+A11) -- wait, this is M4 = (A00+A11)(B00+B11) already used
        # Actually Strassen: M4 = A11(B10-B00) is M2, let me recheck
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)B11
        # M6 = (A01-A11)(B10+B11)
        # M7 = (A00-A10)(B00+B10)
        
        # C00 = M1+M4-M2-M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M3+M5-M6
        
        # Let me redo with correct Strassen:
        # For block b (columns k1, k2):
        # A is 2x2: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[2], A[1,1]=U[3]
        # B block is 2x2: B[0,0]=V[k1], B[0,1]=V[k2], B[1,0]=V[p+k1], B[1,1]=V[p+k2]
        # C block is 2x2: C[0,0]=W[k1*n+0], C[0,1]=W[k1*n+1], C[1,0]=W[k2*n+0], C[1,1]=W[k2*n+1]
        
        # M1 = (A00+A11)(B00+B11): U[0]=1,U[3]=1; V[k1]=1,V[p+k2]=1; W[k1*n+0]=1,W[k2*n+1]=1
        U[0][base_col + 0] = 1
        U[3][base_col + 0] = 1
        V[k1][base_col + 0] = 1
        V[p + k2][base_col + 0] = 1
        W[k1 * n + 0][base_col + 0] = 1
        W[k2 * n + 1][base_col + 0] = 1

        # M2 = (A10+A11)B00: U[2]=1,U[3]=1; V[k1]=1; W[k1*n+1]=1,W[k2*n+1]=-1
        # Wait: C10 += M2, C11 -= M2? No.
        # C00 = M1 + M4 - M2 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 - M6
        #
        # M2 = (A10+A11)*B00: contributes to C10 and C00
        # So: W for C10 gets +1, W for C00 gets -1
        # C10 is W[k2*n+0], C00 is W[k1*n+0]
        U[2][base_col + 1] = 1
        U[3][base_col + 1] = 1
        V[k1][base_col + 1] = 1
        W[k2 * n + 0][base_col + 1] = 1   # C10 += M2
        W[k1 * n + 0][base_col + 1] = -1  # C00 -= M2

        # M3 = A00*(B01-B11): U[0]=1; V[k2]=1, V[p+k2]=-1; W[k1*n+1]=1, W[k2*n+1]=-1
        # C01 += M3, C11 -= M3
        # C01 is W[k1*n+1], C11 is W[k2*n+1]
        U[0][base_col + 2] = 1
        V[k2][base_col + 2] = 1
        V[p + k2][base_col + 2] = -1
        W[k1 * n + 1][base_col + 2] = 1   # C01 += M3
        W[k2 * n + 1][base_col + 2] = -1  # C11 -= M3

        # M4 = A11*(B10-B00): U[3]=1; V[p+k1]=1, V[k1]=-1; W[k1*n+0]=1, W[k2*n+0]=1
        # C00 += M4, C10 += M4
        U[3][base_col + 3] = 1
        V[p + k1][base_col + 3] = 1
        V[k1][base_col + 3] = -1
        W[k1 * n + 0][base_col + 3] = 1   # C00 += M4
        W[k2 * n + 0][base_col + 3] = 1   # C10 += M4

        # M5 = (A00+A10)*B11: U[0]=1,U[2]=1; V[p+k2]=1; W[k1*n+1]=1, W[k2*n+1]=1
        # C01 += M5, C11 += M5
        U[0][base_col + 4] = 1
        U[2][base_col + 4] = 1
        V[p + k2][base_col + 4] = 1
        W[k1 * n + 1][base_col + 4] = 1   # C01 += M5
        W[k2 * n + 1][base_col + 4] = 1   # C11 += M5

        # M6 = (A01-A11)*(B10+B11): U[1]=1,U[3]=-1; V[p+k1]=1,V[p+k2]=1; W[k2*n+1]=-1
        # C11 -= M6
        U[1][base_col + 5] = 1
        U[3][base_col + 5] = -1
        V[p + k1][base_col + 5] = 1
        V[p + k2][base_col + 5] = 1
        W[k2 * n + 1][base_col + 5] = -1  # C11 -= M6

        # M7 = (A00-A10)*(B00+B10): U[0]=1,U[2]=-1; V[k1]=1,V[p+k1]=1; W[k1*n+0]=-1
        # C00 -= M7
        U[0][base_col + 6] = 1
        U[2][base_col + 6] = -1
        V[k1][base_col + 6] = 1
        V[p + k1][base_col + 6] = 1
        W[k1 * n + 0][base_col + 6] = -1  # C00 -= M7

    if _verify_decomposition(U, V, W, n, m, p, num_tests=20):
        return {"U": U, "V": V, "W": W}
    else:
        return None


def _verify_decomposition(U, V, W, n, m, p, num_tests=20):
    """Verify that the decomposition correctly computes matrix multiplication."""
    for _ in range(num_tests):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor_val = 0.0
                    for r in range(len(U[0])):
                        tensor_val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(tensor_val - 1.0) > 1e-10:
                        return False

    return True


def _construct_schoolbook(n, m, p):
    """Schoolbook rank-n*m*p decomposition."""
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_structured_rank(n, m, p, target_rank):
    """
    Try to find a rank-target_rank decomposition.
    
    Key structural insight: The (2,2,14) matrix multiplication tensor can be seen as
    4 vectors (one per (i,j) pair) in R^14, each of which must be representable as
    sum_r U[i,j,r] * (V[j,*,r] * W[*,i,r]) where the product is elementwise in k.
    
    For each (i,j), define f_{i,j}(k) = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    This is a system of bilinear equations. Let's try a parametric approach.
    
    Approach: Assume a block-cyclic structure where the 14 k-values are grouped
    and terms are shared across groups with different coefficients.
    
    Alternative: Use the fact that we only need 4 independent "directions" in U-space.
    Let U[i,j,r] take values from a small set.
    """
    rank = target_rank
    if rank < 28:
        return None
    
    # Try a structured approach: use rank-7 Strassen per block but with sharing
    # between adjacent blocks
    
    # Method: Start from rank-49 and try to merge terms
    # For each pair of blocks, try to combine their 7 Strassen terms into fewer
    
    import random
    random.seed(42)
    
    # Try different elimination patterns
    for attempt in range(50):
        if time.time() > 140:
            return None
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strategy: Use a global parameterization
        # For each rank term r, assign it to a specific (block, strassen_slot) or shared
        # 
        # Try: for rank 48, merge two Strassen terms from different blocks
        # The merged term must satisfy the tensor equations for both blocks
        
        # Simpler: try random sparse patterns
        # Each term r has a "type" determining which (i,j,k) entries it affects
        
        # Let's try a different approach: use the algebraic structure
        # T[i,j,k] = 1 for all i,j,k
        # = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        # For n=m=2, i,j in {0,1}
        # Fix j=0: T[i,0,k] = sum_r U[i,0,r] V[0,k,r] W[k,i,r] = 1
        # Fix j=1: T[i,1,k] = sum_r U[i,1,r] V[1,k,r] W[k,i,r] = 1
        
        # This is 4 * p = 56 equations with rank * (4 + 2p + 2p) = rank * (4+28+28) = 60*rank unknowns
        # But it's trilinear, so we need to be careful
        
        # Let's try a specific construction for lower rank
        # 
        # Idea: Use the fact that for the (2,2,2) case, rank 7 is optimal.
        # For (2,2,14), if we could share terms between the 7 blocks, we'd get lower rank.
        # 
        # Consider: what if all 7 blocks use the same Strassen structure but with
        # different V and W entries? That's already what we have (rank 49).
        # 
        # To share, we need some terms to contribute to multiple blocks.
        # A term r with U[i,j,r] != 0, V[j,k,r] != 0 for k in block a, and W[k,i,r] != 0
        # for k in block a can only contribute to block a (since V[j,k,r]=0 for other blocks).
        # 
        # Unless V[j,k,r] is non-zero for k in multiple blocks!
        # Then one term can contribute to multiple blocks.
        
        # Let's try: have some terms where V[j,k,r] is non-zero for k spanning multiple blocks
        # This requires the term to contribute to C[i,j,k] for k in multiple blocks
        
        # For a term to contribute to block b (columns 2b, 2b+1):
        # V[j, 2b, r] and V[j, 2b+1, r] must be set appropriately
        # W[2b*n+i, r] and W[(2b+1)*n+i, r] must be set appropriately
        
        # If a term has V[j,k,r] non-zero for all k, it's a "global" term
        # But then it must satisfy: for all k, sum of such terms = 1 (after accounting for others)
        
        # Let's try: rank = 7*7 - s where s terms are shared
        # A shared term replaces 2 (or more) block-specific terms
        
        # For now, try a numerical approach: set up the equations and solve
        # This is a system of trilinear equations, which is hard
        # 
        # Alternative: use the fact that we can set up a linear system for each (i,j)
        # given fixed V and W, solve for U, etc. (alternating minimization)
        
        # Initialize with Strassen-49 pattern but with some terms merged
        # Pick two blocks and try to merge one term from each
        
        block1 = attempt % 7
        block2 = (attempt + 1) % 7
        if block1 == block2:
            block2 = (block2 + 1) % 7
        
        # Start with rank-49 and zero out 2 terms, then try to fix
        U49 = [[0] * 49 for _ in range(n * m)]
        V49 = [[0] * 49 for _ in range(m * p)]
        W49 = [[0] * 49 for _ in range(p * n)]
        
        # Fill in Strassen-49
        for pair_idx in range(7):
            k1 = 2 * pair_idx
            k2 = 2 * pair_idx + 1
            base_col = pair_idx * 7
            
            U49[0][base_col + 0] = 1
            U49[3][base_col + 0] = 1
            V49[k1][base_col + 0] = 1
            V49[p + k2][base_col + 0] = 1
            W49[k1 * n + 0][base_col + 0] = 1
            W49[k2 * n + 1][base_col + 0] = 1
            
            U49[2][base_col + 1] = 1
            U49[3][base_col + 1] = 1
            V49[k1][base_col + 1] = 1
            W49[k2 * n + 0][base_col + 1] = 1
            W49[k1 * n + 0][base_col + 1] = -1
            
            U49[0][base_col + 2] = 1
            V49[k2][base_col + 2] = 1
            V49[p + k2][base_col + 2] = -1
            W49[k1 * n + 1][base_col + 2] = 1
            W49[k2 * n + 1][base_col + 2] = -1
            
            U49[3][base_col + 3] = 1
            V49[p + k1][base_col + 3] = 1
            V49[k1][base_col + 3] = -1
            W49[k1 * n + 0][base_col + 3] = 1
            W49[k2 * n + 0][base_col + 3] = 1
            
            U49[0][base_col + 4] = 1
            U49[2][base_col + 4] = 1
            V49[p + k2][base_col + 4] = 1
            W49[k1 * n + 1][base_col + 4] = 1
            W49[k2 * n + 1][base_col + 4] = 1
            
            U49[1][base_col + 5] = 1
            U49[3][base_col + 5] = -1
            V49[p + k1][base_col + 5] = 1
            V49[p + k2][base_col + 5] = 1
            W49[k2 * n + 1][base_col + 5] = -1
            
            U49[0][base_col + 6] = 1
            U49[2][base_col + 6] = -1
            V49[k1][base_col + 6] = 1
            V49[p + k1][base_col + 6] = 1
            W49[k1 * n + 0][base_col + 6] = -1
        
        # Create a rank-48 version by removing one term and trying to compensate
        # Remove term at position block1*7 + 6 (the last term of block1)
        # and try to adjust the remaining terms
        
        num_remove = 49 - rank
        if num_remove <= 0:
            continue
        
        # Remove the last num_remove terms from random blocks
        terms_to_remove = []
        for r in range(num_remove):
            b = (block1 + r) % 7
            terms_to_remove.append(b * 7 + 6)
        
        # Zero out those terms
        for t in terms_to_remove:
            for i in range(n * m):
                U49[i][t] = 0
            for j in range(m * p):
                V49[j][t] = 0
            for k in range(p * n):
                W49[k][t] = 0
        
        # Now try to fix by adjusting other terms
        # This is a hard problem; try a simple heuristic
        # 
        # For each affected (i,j,k), the missing contribution must be made up by other terms
        # 
        # Let's just check if the remaining terms still work (they won't in general)
        if _verify_decomposition(U49, V49, W49, n, m, p, num_tests=5):
            # Trim to target_rank
            U_trim = [row[:rank] for row in U49]
            V_trim = [row[:rank] for row in V49]
            W_trim = [row[:rank] for row in W49]
            return {"U": U_trim, "V": V_trim, "W": W_trim}
    
    return None
# EVOLVE-BLOCK-END
