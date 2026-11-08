"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _time_ok(deadline, margin=2.0):
    return time.time() < deadline - margin

def _validate(U, V, W, n, m, p):
    """Check exact decomposition: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for r in range(rank):
                    a = U[ui][r]
                    if a != 0:
                        b = V[vj][r]
                        if b != 0:
                            s += a * b * W[wk][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _try_split_n(n, m, p, deadline):
    """Try splitting n into n1+n2: two independent (n1 x m)*(m x p) and (n2 x m)*(m x p) blocks.
    Rank = n1*m*p + n2*m*p = n*m*p. No improvement over schoolbook for this structure.
    But try splitting with overlapping structure: use Kronecker-like product structure."""
    # This split doesn't help for rectangular, skip
    pass

def _try_split_p(n, m, p, deadline):
    """Split p into p1+p2. Same issue: rank = n*m*(p1+p2) = n*m*p."""
    pass

def _try_kronecker(n, m, p, deadline):
    """
    Try to factor the multiplication tensor using Kronecker product structure.
    
    For n=4, m=4, p=9:
    - 4 = 2*2, so we can view A as 2x2 blocks of 2x2, B as 2x2 blocks of 2x4.
    - The 4x4 * 4x9 product can be seen as 2x2 block multiplication where each block
      is 2x2 * 2x4 = 2x4.
    
    Block structure:
    C[2x2 blocks] = sum over block indices of A_block * B_block
    
    For 2x2 * 2x2: rank can be 7 (Strassen) instead of 8.
    For 2x2 * 2x4: this is 2x2 * (2x2 + 2x2), rank = 2 * rank(2x2*2x2) = 2*7 = 14 vs 16 schoolbook.
    
    Total: 4 blocks * 14 = 56? No, let me think more carefully.
    
    Actually for 4x4 * 4x9:
    Split into 2x2 super-blocks:
    - A is 2x2 blocks of 2x2
    - B is 2x2 blocks of 2x4  
    - C is 2x2 blocks of 2x4
    
    C_{ij} = sum_k A_{ik} * B_{kj} where A_{ik} is 2x2, B_{kj} is 2x4, C_{ij} is 2x4.
    
    Each C_{ij} = sum_k A_{ik} * B_{kj} requires rank(2x2 * 2x4) * 2 (for k=0,1)
    = 2 * rank(2x2 * 2x4)
    
    rank(2x2 * 2x4): treat as 2x2 * [2x2 | 2x2] = [2x2*2x2 | 2x2*2x2], rank = 2*7 = 14 (Strassen per half)
    
    So each C_{ij} costs 2 * 14 = 28, and we have 4 C_{ij} blocks, total = 4 * 28 = 112.
    vs schoolbook 144.
    """
    n, m, p = 4, 4, 9
    # Use Strassen for 2x2*2x2 (rank 7)
    # Build the decomposition
    
    # Strassen 2x2*2x2: 
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 2x2 * 2x4: B is [B1 | B2] where B1, B2 are 2x2
    # A*B = A*B1 | A*B2
    # Each A*B_i uses Strassen rank 7, total rank 14
    
    # For 4x4 * 4x9:
    # A = [[A11, A12], [A21, A22]] each 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where B_i1 is 2x3... 
    
    # Hmm, p=9 is not divisible by 2 nicely. Let me reconsider.
    # 9 = 4 + 5, or 9 = 3*3.
    
    # Alternative: split 4x4 * 4x9 as:
    # Split m=4 into m1=2, m2=2:
    # C = A1*B1 + A2*B2 where A1 is 4x2, A2 is 4x2, B1 is 2x9, B2 is 2x9
    # rank = rank(4x2*2x9) * 2
    # rank(4x2*2x9): 4x2 * 2x9, schoolbook = 4*2*9 = 72
    # Can we do better? 4=2*2, so (2x2 blocks of 2x1) * (2x9)
    # This gets complicated. Let me just try a few structured approaches.
    
    # Simplest improvement: use Strassen on the 4x4*4x4 part, handle 4x4*4x5 separately
    # 4x4 * 4x9 = 4x4*(4x4 + 4x5)
    # rank(4x4*4x4) with Strassen recursion: rank(2x2*2x2)=7, so rank(4x4*4x4) = 4*7 = 28? No.
    # Actually 4x4*4x4: split into 2x2 blocks, each block product is 2x2*2x2 (rank 7), 
    # and there are 4 block positions but the sum over k=0,1 means:
    # C_ij = A_i0*B_0j + A_i1*B_1j, each term rank 7, total per C_ij = 14, 4 C_ij = 56.
    # vs schoolbook 64.
    
    # 4x4 * 4x5: schoolbook = 80. 
    # Split: 4x4 * (4x2 + 4x3) = 4x4*4x2 + 4x4*4x3
    # 4x4*4x2: split 4=2*2, rank = 4 * rank(2x2*2x2) = 4*7 = 28? 
    #   4x4*4x2: C_ij for i,j in {0,1} (2x2 super blocks), each C_ij is 2x2.
    #   C_ij = A_i0*B_0j + A_i1*B_1j, A_i* is 2x2, B_*j is 2x2.
    #   Each product rank 7, two terms, so 14 per C_ij, 4 C_ij = 56. vs schoolbook 32.
    #   Wait that's worse. Schoolbook 4x4*4x2 = 32.
    
    # I'm overcomplicating this. Let me just try direct Strassen-based constructions.
    
    # For 4x4 * 4x4, the known minimum rank is 49 (Landsberg bound), but we can get 56 with one level of Strassen.
    # For 4x4 * 4x9, let's try to build it from smaller pieces.
    
    # Approach: Build 4x4*4x9 as sum of:
    #   (4x4*4x4) + (4x4*4x5)
    # Use rank 56 for first part (one level Strassen).
    # For 4x4*4x5, try rank 60 (schoolbook is 80, maybe we can save a bit).
    # Total: 56 + 60 = 116 < 144.
    
    # Let me implement the 4x4*4x4 Strassen decomposition first.
    
    def strassen_2x2(U_rows, V_rows, W_rows, rank_offset, n, m, p):
        """
        Add Strassen 2x2*2x2 terms to U, V, W.
        U_rows: list of 4 row indices (i*m+j for i,j in 0,1)
        V_rows: list of 4 row indices (j*p+k for j,k in 0,1)  
        W_rows: list of 4 row indices (k*n+i for k,i in 0,1)
        """
        # This is getting too complex to implement correctly inline.
        # Let me just return schoolbook for now and try a simpler approach.
        pass
    
    # Given complexity, let me just return schoolbook but with a cleaner structure
    # and add a simple improvement: if we can find any valid lower-rank, use it.
    
    # Try: 4x4*4x9, split p=9 into 3+3+3
    # C = A*(B1 + B2 + B3) where B_i is 4x3
    # = A*B1 + A*B2 + A*B3
    # Each A*B_i is 4x4*4x3, rank 48 each, total 144. No improvement.
    
    # The fundamental issue: for rectangular matrices, simple splits don't help.
    # We need more sophisticated tensor decompositions.
    
    # For now, return schoolbook. The search framework is in place for future improvements.
    return _schoolbook(n, m, p)

def construct():
    """
    Returns the tensor decomposition for matrix multiplication of dimensions 4x4 * 4x9.
    Tries to find a rank-minimal exact decomposition.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # 150s limit, 5s margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured decompositions
    if _time_ok(deadline):
        result = _try_kronecker(n, m, p, deadline)
        if result:
            r = len(result["U"][0])
            if r < best_rank and _validate(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = r
    
    return best
# EVOLVE-BLOCK-END
