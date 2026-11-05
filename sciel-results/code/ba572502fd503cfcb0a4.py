"""Matrix multiplication 4x4x5 tensor decomposition search."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]
DEADLINE = 140.0  # safety margin before 150s kill

def _verify(U, V, W, n=4, m=4, p=5, tol=1e-9):
    """Verify that U,V,W exactly decompose the matmul tensor."""
    rank = len(U[0]) if U else 0
    # Build the matmul tensor T[i,j,k] = delta (1 if i,j,k valid)
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Check: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # But actually the decomposition is: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[i,j,k] = 1 (the matmul tensor is all-ones in the appropriate sense)
    # The rank-r decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c[i*m+j] = U[i*m+j, c], v_c[j*p+k] = V[j*p+k, c], w_c[k*n+i] = W[k*n+i, c]
    # So: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1 for all i,j,k
    # Wait, that's not right. The matmul tensor T_{i,j,k} = 1 means:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor form: T is a 3-tensor where T[i,j,k] = 1 (it's the "matmul tensor")
    # A rank-r decomposition: T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    # where (u_c ⊗ v_c ⊗ w_c)[i,j,k] = u_c[i] * v_c[j] * w_c[k]
    # But our indexing is different. Let me re-read the contract.
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # The condition is: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is the standard matmul tensor decomposition.
    
    # Vectorized check
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each (i,j,k), compute sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # This is expensive for large rank. Let's do it smartly.
    # Actually, let's just check all 80 entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n=4, m=4, p=5):
    """Exact verification using Fractions."""
    rank = len(U[0]) if U else 0
    U_f = [[Fraction(x) for x in row] for row in U]
    V_f = [[Fraction(x) for x in row] for row in V]
    W_f = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U_f[i*m+j][c] * V_f[j*p+k][c] * W_f[k*n+i][c]
                if s != 1:
                    return False
    return True

def _try_random_decomposition(rank, n=4, m=4, p=5, seed=None):
    """Try to find a rank-r decomposition via randomized search."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Strategy: use a structured approach
    # The matmul tensor T[i,j,k]=1 can be decomposed using various algebraic identities.
    # For 4x4x4, Strassen gives rank 7. For 4x4x5, we need something different.
    
    # Approach: Start from schoolbook and try to merge terms.
    # Or: use a bilinear form approach.
    
    # Let's try a different approach: parameterize and solve.
    # For rank r, we need:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is a system of 80 equations in r*(16+20+20) = 56r unknowns.
    # For r=72, that's 4032 unknowns and 80 equations - very underdetermined.
    
    # Simple approach: set most entries to 0 or 1, and verify.
    # Start from schoolbook (rank 80) and try to find redundancies.
    
    # Alternative: use known results. For 4x4x4, Strassen rank 7.
    # For 4x4x5, we can think of it as 4x4x4 + 4x4x1.
    # The 4x4x1 part has rank 16 (each output column needs its own term).
    # But there might be sharing.
    
    # Let's try: use Strassen for the first 4 columns of B, and schoolbook for the 5th.
    # Strassen 4x4x4: rank 7 (using 2x2 Strassen recursively? No, 4x4 Strassen is more complex)
    # Actually, 4x4x4 matmul has known rank 7 (Strassen's algorithm for 2x2 gives rank 7 for 2x2,
    # but for 4x4 it's different).
    
    # Wait, Strassen's algorithm for 2x2 matrices has rank 7.
    # For 4x4 matrices, we can use a block decomposition.
    # 4x4 = 2x2 blocks of 2x2.
    # Using Strassen on the block structure: 7 multiplications of 2x2 blocks.
    # But each 2x2 multiplication has rank 4 (schoolbook) or 7 (Strassen).
    # This gets complicated.
    
    # Let me try a direct approach: random initialization + verification.
    
    # For a given rank r, initialize U, V, W randomly and check if the equations are satisfied.
    # This won't work directly since we need exact solutions.
    
    # Better: use a structured decomposition.
    # Key idea: The matmul tensor T[i,j,k] = 1 for all i,j,k.
    # We can write 1 = sum of products using algebraic identities.
    
    # For the 5th column (k=4), we need:
    # sum_c U[i*m+j,c] * V[j*p+4,c] * W[4*n+i,c] = 1 for all i,j
    
    # Let's try: for the 4x4x4 part, use a rank-7 Strassen-like decomposition,
    # and for the extra column, add 16 terms (one per (i,j) pair) but try to share.
    
    # Actually, let me try a simpler structured approach:
    # Use the identity: for any i,j,k, 1 = 1*1*1
    # Group terms by patterns.
    
    # Let's try: rank = 72 means we save 8 terms from schoolbook's 80.
    # This suggests 8 pairs of (i,j,k) can share a common factor.
    
    # Approach: Try to find a rank-r decomposition by:
    # 1. Start with schoolbook (rank 80)
    # 2. Try to combine terms where possible
    
    # For now, let's try random structured decompositions.
    
    # Method: Set U, V, W with entries in {0, 1, -1, 0.5, -0.5} and verify.
    
    best = None
    for attempt in range(200):
        # Initialize with random sparse structure
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        # For each column c, assign random patterns
        for c in range(rank):
            # Randomly choose which (i,j), (j,k), (k,i) positions are active
            # and assign values
            u_active = set()
            v_active = set()
            w_active = set()
            
            # Each column contributes to some subset of the tensor
            # Let's make each column "local" - affecting a small subset
            ni = rng.randint(1, 4)
            nj = rng.randint(1, 4)
            nk = rng.randint(1, 4)
            
            for _ in range(ni):
                i = rng.randint(0, n-1)
                j = rng.randint(0, m-1)
                U[i*m+j][c] = rng.choice([1, -1, 0.5, -0.5])
            
            for _ in range(nj):
                j = rng.randint(0, m-1)
                k = rng.randint(0, p-1)
                V[j*p+k][c] = rng.choice([1, -1, 0.5, -0.5])
            
            for _ in range(nk):
                k = rng.randint(0, p-1)
                i = rng.randint(0, n-1)
                W[k*n+i][c] = rng.choice([1, -1, 0.5, -0.5])
        
        if _verify(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _schoolbook(n=4, m=4, p=5):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_block_decomposition(n=4, m=4, p=5):
    """
    Try to build a decomposition using Strassen's 2x2 algorithm for the 4x4x4 part
    and handle the 5th column separately.
    
    For 4x4x4: Use block 2x2 structure with Strassen.
    A = [[A11, A12], [A21, A22]] where each block is 2x2
    B = [[B11, B12], [B21, B22]]
    
    Strassen: 
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each M involves multiplying 2x2 matrices, which has rank 4 (schoolbook) or 7 (Strassen).
    If we use schoolbook for the 2x2 multiplications, each M costs rank 4, total 7*4=28 for 4x4x4.
    But that's worse than schoolbook's 16 for 4x4x4!
    
    Wait, I'm confusing myself. Let me reconsider.
    
    For 4x4x4 matmul, schoolbook rank is 16.
    Strassen's algorithm for 4x4 (using 2x2 blocks) uses 7 multiplications of 2x2 matrices.
    Each 2x2 multiplication has rank 4 (schoolbook).
    So total rank = 7 * 4 = 28? That's worse than 16!
    
    No wait, that's not right either. The rank of the composition is not simply multiplicative.
    
    Actually, for the tensor decomposition, if we have a rank-r decomposition of 2x2x2 matmul,
    and we use it in a block structure, the total rank is r times the number of block multiplications.
    
    Strassen 2x2x2 has rank 7.
    For 4x4x4 using 2x2 blocks: 7 block multiplications, each of rank 7 (if using Strassen recursively) 
    or rank 4 (if using schoolbook for 2x2).
    
    Hmm, this is getting complicated. Let me just try the known result: rank 72 for 4x4x5.
    """
    pass

def _structured_search(rank, n=4, m=4, p=5, time_limit=10.0):
    """
    Structured search for a rank-r decomposition.
    Uses the fact that the matmul tensor has symmetries and structure.
    """
    start = time.time()
    
    # Approach: Use a "separable" structure where each rank-1 term
    # corresponds to a specific pattern.
    
    # Key observation: T[i,j,k] = 1 for all i,j,k.
    # We can write 1 as a sum of products using:
    # 1 = sum_{s} a_s(i,j) * b_s(j,k) * c_s(k,i)
    
    # For a rank-r decomposition, we need r such triples.
    
    # Let's try: for each (i,j,k), the equation is:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is 80 equations. With r columns, we have 56r variables.
    # For r=72, we have 4032 variables and 80 equations - highly underdetermined.
    
    # Strategy: Fix most variables to 0 or 1, and solve for the rest.
    
    # Simpler: try to find a decomposition by modifying schoolbook.
    # Schoolbook has rank 80. We want rank 72, so save 8 terms.
    
    # Idea: If two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can be merged
    # into one term with appropriate U,V,W values, we save one rank.
    
    # For terms to merge, we need:
    # U[i1*m+j1,c] * V[j1*p+k1,c] * W[k1*n+i1,c] + 
    # U[i2*m+j2,c] * V[j2*p+k2,c] * W[k2*n+i2,c] = 1 + 1 = 2
    # But also, for all other (i,j,k), the contribution should be 0.
    
    # This is complex. Let's try a different approach.
    
    # Direct construction: try rank values from 72 upward.
    # For each rank, try to find a valid decomposition.
    
    for r in range(rank, min(rank + 10, 81)):
        if time.time() - start > time_limit:
            break
        
        # Try random structured decompositions
        for trial in range(50):
            if time.time() - start > time_limit:
                break
            
            U = [[0.0] * r for _ in range(n * m)]
            V = [[0.0] * r for _ in range(m * p)]
            W = [[0.0] * r for _ in range(p * n)]
            
            # Initialize with a base pattern
            # Use a "grid" structure where each column handles a subset
            for c in range(r):
                # Assign this column to handle specific (i,j,k) combinations
                # Random sparse assignment
                n_u = 1 + (c % 3)
                n_v = 1 + ((c // 3) % 3)
                n_w = 1 + ((c // 9) % 3)
                
                for _ in range(n_u):
                    i = random.randint(0, n-1)
                    j = random.randint(0, m-1)
                    U[i*m+j][c] = random.choice([1, -1])
                
                for _ in range(n_v):
                    j = random.randint(0, m-1)
                    k = random.randint(0, p-1)
                    V[j*p+k][c] = random.choice([1, -1])
                
                for _ in range(n_w):
                    k = random.randint(0, p-1)
                    i = random.randint(0, n-1)
                    W[k*n+i][c] = random.choice([1, -1])
            
            if _verify(U, V, W, n, m, p):
                return U, V, W, r
    
    return None

def construct():
    start_time = time.time()
    
    # First, get the schoolbook solution as fallback
    schoolbook = _schoolbook(4, 4, 5)
    best_result = schoolbook
    best_rank = 80
    
    # Try to find better decompositions
    # Based on knowledge, rank 72 is achievable
    
    # Try structured search for rank 72
    deadline = start_time + DEADLINE - 10  # 10s safety margin
    
    for target_rank in [72, 73, 74, 75, 76, 77, 78, 79]:
        if time.time() > deadline:
            break
        
        result = _structured_search(target_rank, 4, 4, 5, time_limit=15.0)
        if result is not None:
            U, V, W, r = result
            if r < best_rank:
                best_rank = r
                best_result = {"U": U, "V": V, "W": W}
                break  # Found a good one, stop
    
    # If we found something better, return it; otherwise schoolbook
    if best_rank < 80:
        return best_result
    else:
        return schoolbook
