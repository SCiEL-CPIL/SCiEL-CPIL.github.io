"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Strassen-Winograd recursive decomposition
    # Embed 3x5x6 into 4x8x8 and use rank-7 2x2x2 sub-decompositions
    # This should give rank around 4*8*8/2 = 128, but with Strassen we can do better
    # Actually, let's use a direct construction approach
    
    # For 3x5x6, we can try to find a decomposition with rank < 90
    # Use the fact that we can pad to 4x8x8 and apply Strassen recursively
    
    # Direct construction using known low-rank decompositions
    # For small matrices, we can try explicit constructions
    
    # Try to construct a rank-89 decomposition by modifying one factor
    # The key insight: if we can find linear dependencies in the tensor, we can reduce rank
    
    # Let's try a systematic approach: start with schoolbook and look for
    # linear combinations that preserve the tensor
    
    # For 3x5x6, the tensor has 3*5*6 = 90 entries, each equal to 1
    # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    
    # Try rank 89: can we merge two terms?
    # If U[i*m+j,0] = U[i*m+j,1] = 1 and V[j*p+k,0] = V[j*p+k,1] = 1 and W[k*n+i,0] = W[k*n+i,1] = 1
    # then term0 + term1 = 2, not 1. So we need to adjust.
    
    # Let's try: set W[k*n+i,0] = 1, W[k*n+i,1] = 0 for all (k,i), but modify U and V
    # This doesn't work directly.
    
    # Alternative: use the fact that for fixed (i,j), the function of k is constant
    # So we might be able to reduce the rank in the k-direction
    
    # Let's try a different approach: use known results for small matrix multiplication
    # For 3x3x3, the minimum rank is 23 (Strassen-like)
    # For 3x5x6, we can embed into 3x6x6 and use known decompositions
    
    # Actually, let's just try to find a valid decomposition with rank < 90 using
    # a more clever construction
    
    # Pad to 4x8x8 and use Strassen's algorithm recursively
    # Strassen for 2x2x2 uses rank 7 instead of 8
    # For 4x4x4: rank = 4 * 7 = 28 (vs 64 schoolbook)
    # For 4x8x8: we can decompose into blocks
    
    # Let's construct explicit factors for a 4x8x8 Strassen decomposition
    # Then project down to 3x5x6
    
    # Strassen 2x2x2 factors (rank 7):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # For 4x4x4, we can partition into 2x2 blocks and apply Strassen to each
    # This gives rank 4 * 7 = 28
    
    # For 4x8x8, we can partition the 8x8 into 2x2 blocks of 4x4, giving 4 blocks
    # Each 4x4x4 uses rank 28, total rank = 4 * 28 = 112
    # But we can do better by partitioning differently
    
    # Actually, let's just try to find a valid decomposition with rank < 90
    # using a search approach with the structure in mind
    
    # Try ranks from 89 down to 80
    for target_rank in range(89, 79, -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations with structured patterns
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            # Initialize with schoolbook pattern and perturb
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Fill first 89 columns with schoolbook pattern
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        col = i * m * p + j * p + k
                        if col < target_rank:
                            U[i * m + j][col] = 1
                            V[j * p + k][col] = 1
                            W[k * n + i][col] = 1
            
            # For the remaining columns (if any), try to create linear dependencies
            if target_rank < 90:
                # We need to merge (90 - target_rank) pairs
                num_merges = 90 - target_rank
                for merge_idx in range(num_merges):
                    # Choose a random pair of columns to merge
                    col_a = random.randint(0, 88)
                    col_b = random.randint(0, 88)
                    if col_a == col_b:
                        continue
                    
                    # Try to set W values to create a linear dependency
                    # If we set W[*, col_b] = -W[*, col_a], and adjust U and V
                    # This is complex, so let's try a different approach
                    
                    # Instead, try to find a column that is a linear combination
                    # of others in the tensor sense
                    
                    # For now, just try random small perturbations
                    for idx in range(n * m):
                        if random.random() < 0.3:
                            U[idx][col_b] = random.choice([-1, 0, 1])
                    for idx in range(m * p):
                        if random.random() < 0.3:
                            V[idx][col_b] = random.choice([-1, 0, 1])
                    for idx in range(p * n):
                        if random.random() < 0.3:
                            W[idx][col_b] = random.choice([-1, 0, 1])
            
            # Check if this decomposition is exact
            is_valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(target_rank):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                        if abs(s - 1) > 1e-9:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                if not is_valid:
                    break
            
            if is_valid:
                # Convert to exact fractions
                U_frac = [[Fraction(x).limit_denominator(1000) for x in row] for row in U]
                V_frac = [[Fraction(x).limit_denominator(1000) for x in row] for row in V]
                W_frac = [[Fraction(x).limit_denominator(1000) for x in row] for row in W]
                
                # Verify with fractions
                is_valid_frac = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = Fraction(0)
                            for c in range(target_rank):
                                s += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                            if s != 1:
                                is_valid_frac = False
                                break
                        if not is_valid_frac:
                            break
                    if not is_valid_frac:
                        break
                
                if is_valid_frac:
                    best_rank = target_rank
                    best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                    break  # Found valid decomposition at this rank, try lower
    
    if time.time() > deadline - 2:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
