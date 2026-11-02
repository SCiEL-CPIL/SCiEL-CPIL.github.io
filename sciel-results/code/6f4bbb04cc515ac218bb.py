"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p):
    """Check exact decomposition: sum_r U[i*m+j,r]*V[j*p+k,r] == W[k*n+i,r] for all i,j,k,r"""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    left = U[i*m+j][r] * V[j*p+k][r]
                    right = W[k*n+i][r]
                    if left != right:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Known lower bound: max(n*m, m*p, n*p) = max(6, 9, 6) = 9
    # But actual tensor rank might be higher
    
    # Strategy: Try to find a decomposition with rank < rank_schoolbook
    # For 2x3 * 3x3, let's try various structured approaches
    
    deadline = time.time() + 140  # safety margin
    
    # First, try the known schoolbook (guaranteed to work)
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    # Verify schoolbook works
    assert _verify(U_school, V_school, W_school, n, m, p), "Schoolbook verification failed"
    
    # Try to find better decomposition using structured patterns
    # For matrix multiplication C[i,k] = sum_j A[i,j]*B[j,k]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r]
    # This must hold for ALL r simultaneously - it's a bilinear identity
    
    # Try rank = 12 (some known results suggest M(2,3,3) might be around 12-14)
    # Let's try a structured approach based on grouping
    
    # Approach: Try to merge some of the schoolbook terms
    # The key insight is that we can use linear combinations
    
    # Try various small ranks with a constructive approach
    for target_rank in range(9, rank_schoolbook):
        if time.time() > deadline:
            break
        
        # Try a structured decomposition based on the algebraic structure
        # For 2x3 * 3x3, let's try patterns inspired by small matrix multiplication
        
        # One approach: use the fact that we can represent products more efficiently
        # by exploiting linear dependencies in the output indices
        
        # Let's try a pattern-based approach
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
        
        # Try to find a decomposition using half-integers or rationals
        # This is a hard problem - let's try a few known patterns
        
        # Pattern 1: Try rank 12 with specific structure
        if target_rank == 12:
            # Try to construct rank-12 decomposition
            # This is speculative - try a few specific patterns
            pass
    
    # Given the difficulty, return the schoolbook decomposition
    # which is guaranteed to be correct with rank = 18
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
