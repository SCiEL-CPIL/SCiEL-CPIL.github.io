"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x6 matrix.
    
    The matrix multiplication tensor T has entries T[i,j,k] for i in [0,n), j in [0,m), k in [0,p)
    where the output is C[i,k] = sum_j A[i,j]*B[j,k].
    
    The factor matrices are:
    - U: shape (n*m, rank) - input A factor
    - V: shape (m*p, rank) - input B factor  
    - W: shape (p*n, rank) - output C factor (with trace convention W[k*n+i, r])
    
    The decomposition satisfies: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For 2x2x6, the schoolbook rank is 2*2*6 = 24.
    We'll try to find a lower rank decomposition.
    
    Strategy: Use the fact that 2x2x6 can be viewed as three 2x2x2 blocks.
    Each 2x2x2 multiplication has rank 7 (Strassen). But we might be able to share
    some rank-1 terms across the three blocks.
    
    Let's try rank 20 first, then 19, etc.
    """
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    def check_decomposition(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0])
        # For each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    def schoolbook_decomposition():
        """Return the schoolbook decomposition of rank n*m*p."""
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
        return {"U": U, "V": V, "W": W}
    
    def try_strassen_based_decomposition(target_rank):
        """
        Try to construct a decomposition using Strassen-like ideas.
        
        The 2x2x6 tensor can be seen as three 2x2x2 tensors:
        Block 0: k=0,1 (columns 0-1 of B)
        Block 1: k=2,3 (columns 2-3 of B)  
        Block 2: k=4,5 (columns 4-5 of B)
        
        Each 2x2x2 has Strassen rank 7. Naive sum is 21.
        We try to find linear dependencies to reduce below 21.
        
        For now, let's just use the naive sum of three Strassen decompositions = rank 21.
        """
        # Strassen decomposition for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 + M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3
        
        # For each block, we have 7 rank-1 terms.
        # Total = 3 * 7 = 21
        
        if target_rank < 21:
            return None
        
        # Build 21-term decomposition
        rank = 21
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen coefficients for 2x2x2
        # For each rank-1 term r, we have:
        # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        
        # Let's define the Strassen decomposition explicitly
        # Term 0: (A00+A11)(B00+B11) -> contributes to C00, C11
        # Term 1: (A10+A11)B00 -> contributes to C10
        # Term 2: A00(B10-B11) -> contributes to C01
        # Term 3: A11(B01-B10) -> contributes to C10
        # Term 4: (A00+A01)B11 -> contributes to C00, C01
        # Term 5: (A10-A00)(B00+B01) -> contributes to C10
        # Term 6: (A01-A11)(B10+B11) -> contributes to C11
        
        # For a 2x2x2 block with output columns k0, k1:
        # A indices: (0,0), (0,1), (1,0), (1,1) -> row indices in U: 0, 1, 2, 3
        # B indices: (0,0), (0,1), (1,0), (1,1) -> col indices in V: 0, 1, 2, 3 (relative to block)
        # C indices: (0,0), (0,1), (1,0), (1,1) -> row indices in W: k0*n+0, k0*n+1, k1*n+0, k1*n+1
        
        strassen_terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # Each is a dict mapping index to coefficient
            # U index: i*m+j, V index: j*p+k, W index: k*n+i
            {
                'U': {(0,0): 1, (1,1): 1},
                'V': {(0,0): 1, (1,1): 1},
                'W': {(0,0): 1, (1,1): 1}
            },
            {
                'U': {(1,0): 1, (1,1): 1},
                'V': {(0,0): 1},
                'W': {(1,0): 1}
            },
            {
                'U': {(0,0): 1},
                'V': {(1,0): 1, (1,1): -1},
                'W': {(0,1): 1}
            },
            {
                'U': {(1,1): 1},
                'V': {(0,1): 1, (1,0): -1},
                'W': {(1,0): 1}
            },
            {
                'U': {(0,0): 1, (0,1): 1},
                'V': {(1,1): 1},
                'W': {(0,0): 1, (0,1): 1}
            },
            {
                'U': {(1,0): 1, (0,0): -1},
                'V': {(0,0): 1, (0,1): 1},
                'W': {(1,0): 1}
            },
            {
                'U': {(0,1): 1, (1,1): -1},
                'V': {(1,0): 1, (1,1): 1},
                'W': {(1,1): 1}
            },
        ]
        
        # For each of the 3 blocks, add 7 Strassen terms
        for block in range(3):
            k_offset = block * 2
            for term_idx in range(7):
                r = block * 7 + term_idx
                term = strassen_terms[term_idx]
                
                for (i, j), coeff in term['U'].items():
                    U[i * m + j][r] += coeff
                
                for (j, k_rel), coeff in term['V'].items():
                    k_abs = k_offset + k_rel
                    V[j * p + k_abs][r] += coeff
                
                for (k_rel, i), coeff in term['W'].items():
                    k_abs = k_offset + k_rel
                    W[k_abs * n + i][r] += coeff
        
        # Verify
        if check_decomposition(U, V, W):
            return {"U": U, "V": V, "W": W}
        
        return None
    
    # First, get the schoolbook decomposition as a fallback
    result = schoolbook_decomposition()
    best_rank = n * m * p  # 24
    
    # Try to find lower rank decompositions
    # Try rank 21 (sum of three Strassen)
    if time.time() < deadline - 10:
        decomp = try_strassen_based_decomposition(21)
        if decomp is not None:
            result = decomp
            best_rank = 21
    
    # Try to search for even lower rank using random initialization
    # This is a heuristic search - may not find anything better
    if time.time() < deadline - 30 and best_rank > 20:
        import random
        random.seed(42)
        
        def random_search(rank, num_trials=100):
            """Try random rank-1 term combinations."""
            for trial in range(num_trials):
                if time.time() > deadline - 15:
                    break
                
                U = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(rank)] for _ in range(n*m)]
                V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(rank)] for _ in range(m*p)]
                W = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(rank)] for _ in range(p*n)]
                
                if check_decomposition(U, V, W):
                    return {"U": U, "V": V, "W": W}
            
            return None
        
        for rank in range(20, best_rank):
            if time.time() > deadline - 15:
                break
            decomp = random_search(rank, num_trials=50)
            if decomp is not None:
                result = decomp
                best_rank = rank
                break
    
    return result
# EVOLVE-BLOCK-END
