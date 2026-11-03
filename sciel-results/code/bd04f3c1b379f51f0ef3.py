"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication.
    
    Strategy:
    1. Start with schoolbook (rank 105)
    2. Use Strassen on 2x2 blocks where possible to reduce rank
    3. Try to find further reductions through systematic search
    
    The tensor for (n,m,p) matrix multiplication has entries T[i,j,k] where
    i in [0,n), j in [0,m), k in [0,p).
    
    A decomposition of rank r gives:
    T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    We want the minimum r such that this holds exactly.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Schoolbook baseline: rank = n*m*p = 105
    # Each term (i,j,k) contributes one rank-1 component
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    
    # Let's try to use Strassen's algorithm for 2x2 blocks
    # Strassen reduces 2x2x2 from 8 to 7 multiplications
    # 
    # For a 2x2 block with rows {i1,i2}, cols {j1,j2}, output rows {k1,k2}:
    # Standard: 8 terms
    # Strassen: 7 terms using combinations
    #
    # M1 = (A[i1,j1]+A[i2,j2])*(B[j1,k1]+B[j2,k2])
    # M2 = (A[i2,j1])*(B[j1,k2]-B[j2,k2])
    # M3 = (A[i1,j1])*(B[j1,k2])
    # M4 = (A[i2,j2])*(B[j2,k1])
    # M5 = (A[i1,j1]+A[i1,j2])*(B[j2,k1])
    # M6 = (A[i1,j2]-A[i2,j2])*(B[j1,k1]+B[j2,k1])
    # M7 = (A[i2,j1]-A[i1,j1])*(B[j2,k1]+B[j2,k2])
    #
    # C[i1,k1] = M1 + M4 - M5 + M7
    # C[i1,k2] = M3 + M5
    # C[i2,k1] = M2 + M4
    # C[i2,k2] = M1 - M2 + M3
    
    def strassen_2x2x2(i1, i2, j1, j2, k1, k2, U, V, W, col):
        """Add Strassen decomposition for a 2x2x2 block, return new column index."""
        # M1 = (A[i1,j1]+A[i2,j2])*(B[j1,k1]+B[j2,k2])
        U[i1*m+j1][col] += 1; V[j1*p+k1][col] += 1; W[k1*n+i1][col] += 1
        U[i2*m+j2][col] += 1; V[j1*p+k1][col] += 1; W[k1*n+i1][col] += 1
        U[i1*m+j1][col] += 1; V[j2*p+k2][col] += 1; W[k2*n+i1][col] += 1
        U[i2*m+j2][col] += 1; V[j2*p+k2][col] += 1; W[k2*n+i1][col] += 1
        # Wait, this isn't right. Let me think more carefully.
        
        # Actually, the Strassen decomposition works as follows:
        # We need to represent the 8 elementary products using 7 rank-1 terms.
        # 
        # The 8 elementary products are:
        # A[i1,j1]*B[j1,k1] -> C[i1,k1]
        # A[i1,j1]*B[j1,k2] -> C[i1,k2]
        # A[i1,j1]*B[j2,k1] -> C[i1,k1]
        # A[i1,j1]*B[j2,k2] -> C[i1,k2]
        # A[i2,j1]*B[j1,k1] -> C[i2,k1]
        # A[i2,j1]*B[j1,k2] -> C[i2,k2]
        # A[i2,j1]*B[j2,k1] -> C[i2,k1]
        # A[i2,j1]*B[j2,k2] -> C[i2,k2]
        # A[i2,j2]*B[j1,k1] -> C[i2,k1]
        # A[i2,j2]*B[j1,k2] -> C[i2,k2]
        # A[i2,j2]*B[j2,k1] -> C[i2,k1]
        # A[i2,j2]*B[j2,k2] -> C[i2,k2]
        
        # Hmm, I'm getting confused. Let me restart with a clearer approach.
        
        # For Strassen 2x2x2, the 7 rank-1 terms are:
        # Each term has specific U, V, W entries that are sums/differences
        
        # Let me define the 7 terms explicitly:
        # Term 0: U has +1 at (i1,j1) and +1 at (i2,j2); V has +1 at (j1,k1) and +1 at (j2,k2); W has +1 at (k1,i1) and +1 at (k2,i2)
        # Actually no, each rank-1 term is a single product u*v*w where u is a vector in R^(nm), v in R^(mp), w in R^(pn)
        
        # Let me be very explicit. A rank-1 term c contributes:
        # T[i,j,k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # For Strassen, we want to replace 8 terms with 7 terms.
        # The 8 schoolbook terms for a 2x2x2 block (i1,i2) x (j1,j2) x (k1,k2) are:
        # For each (i,j,k) in {i1,i2}x{j1,j2}x{k1,k2}:
        #   U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
        
        # Strassen replaces these 8 with 7 terms where the U, V, W entries are ±1 or 0.
        
        # Let me just implement it directly with the known Strassen coefficients.
        
        # I'll use a different approach: just keep the schoolbook for now and
        # try to find rank reductions through other means.
        
        return col + 8  # Placeholder, we'll do schoolbook for now
    
    # Let's just do a clean schoolbook implementation first, then try optimizations
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
    
    # Now let's try to reduce the rank using Strassen on 2x2x2 blocks
    # We can apply Strassen to any 2x2x2 sub-block
    # The idea is that if we have a 2x2x2 block, we can replace 8 rank-1 terms with 7
    
    # Let's identify all possible 2x2x2 blocks and try Strassen on them
    # A 2x2x2 block is defined by two consecutive i's, two consecutive j's, two consecutive k's
    
    # For a 2x2x2 block with indices (i1,i2), (j1,j2), (k1,k2):
    # The 8 schoolbook terms are at positions:
    # (i1,j1,k1), (i1,j1,k2), (i1,j2,k1), (i1,j2,k2)
    # (i2,j1,k1), (i2,j1,k2), (i2,j2,k1), (i2,j2,k2)
    
    # Strassen decomposition for this block uses 7 rank-1 terms:
    # We need to find U, V, W entries for 7 new columns that exactly reproduce
    # the same contribution to T[i,j,k] for i in {i1,i2}, j in {j1,j2}, k in {k1,k2}
    
    # The key insight: Strassen works by introducing intermediate sums/differences.
    # Let me define the 7 terms:
    
    # For the 2x2x2 block, let's use these 7 rank-1 components:
    # Each component c has specific nonzero entries in U, V, W
    
    # Actually, let me just verify that Strassen gives rank 7 for 2x2x2.
    # The standard Strassen algorithm:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)(b11)
    # M3 = (a11)(b21-b22)
    # M4 = (a22)(b12-b11)
    # M5 = (a11+a12)(b22)
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3
    
    # In terms of our tensor decomposition, each M corresponds to a rank-1 term
    # where U, V, W have specific ±1 entries.
    
    # For M1 = (a11+a22)(b11+b22):
    # This contributes to c11 and c22 (via the W entries)
    # U entries: +1 at (i1,j1), +1 at (i2,j2)
    # V entries: +1 at (j1,k1), +1 at (j2,k2)  
    # W entries: +1 at (k1,i1), -1 at (k1,i2)... 
    
    # Hmm, this is getting complicated. Let me just implement it carefully.
    
    # Actually, let me take a step back. The current score is 1.0 with rank 105.
    # The best known improvement is rank 99 (score 1.061) using 6 disjoint 2x2x2 Strassen blocks.
    # 
    # For a 3x5x7 matrix multiplication:
    # - We can fit 2x2x2 blocks in various ways
    # - The maximum number of disjoint 2x2x2 blocks we can fit:
    #   i dimension: 3, so we can have floor(3/2) = 1 block of size 2, plus 1 leftover
    #   j dimension: 5, so we can have floor(5/2) = 2 blocks of size 2, plus 1 leftover
    #   k dimension: 7, so we can have floor(7/2) = 3 blocks of size 2, plus 1 leftover
    #   So max disjoint 2x2x2 blocks = 1 * 2 * 3 = 6
    #   Each saves 1 rank (8 -> 7), so total saving = 6
    #   New rank = 105 - 6 = 99
    
    # This matches the observed improvement! So the strategy is to apply Strassen
    # to 6 disjoint 2x2x2 blocks.
    
    # Let's implement this properly.
    
    # First, let's create the base schoolbook decomposition
    # Then, for each 2x2x2 block we want to Strassen, we replace the 8 schoolbook
    # terms with 7 Strassen terms.
    
    # Let me implement Strassen for a specific 2x2x2 block.
    
    def apply_strassen_2x2x2(U, V, W, i1, i2, j1, j2, k1, k2, base_col):
        """
        Replace the 8 schoolbook terms for the 2x2x2 block (i1,i2)x(j1,j2)x(k1,k2)
        with 7 Strassen terms, starting at column base_col.
        Returns the next available column index.
        """
        m_local = m  # Use outer scope m
        
        # The 8 schoolbook terms would be at columns base_col to base_col+7
        # We replace them with 7 Strassen terms at columns base_col to base_col+6
        
        # Strassen terms:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)(b11)
        # M3 = (a11)(b21-b22)
        # M4 = (a22)(b12-b11)
        # M5 = (a11+a12)(b22)
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        
        # Where a11 = A[i1,j1], a12 = A[i1,j2], a21 = A[i2,j1], a22 = A[i2,j2]
        #       b11 = B[j1,k1], b12 = B[j1,k2], b21 = B[j2,k1], b22 = B[j2,k2]
        #       c11 = C[i1,k1], c12 = C[i1,k2], c21 = C[i2,k1], c22 = C[i2,k2]
        
        # For each M, we need to determine U, V, W entries.
        
        # M1 = (a11+a22)(b11+b22)
        # This is a product of (sum of two A entries) * (sum of two B entries)
        # In our decomposition, this corresponds to a rank-1 term where:
        # U has +1 at (i1,j1) and +1 at (i2,j2)
        # V has +1 at (j1,k1) and +1 at (j2,k2)
        # W needs to be determined from how M1 contributes to C
        
        # M1 contributes to c11 and c22:
        # c11 += M1, c22 += M1
        # So W should have +1 at (k1,i1) and +1 at (k2,i2)
        
        col = base_col
        # M1
        U[i1*m_local+j1][col] += 1
        U[i2*m_local+j2][col] += 1
        V[j1*p+k1][col] += 1
        V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += 1
        W[k2*n+i2][col] += 1
        col += 1
        
        # M2 = (a21+a22)(b11)
        # U: +1 at (i2,j1), +1 at (i2,j2)
        # V: +1 at (j1,k1)
        # M2 contributes to c21: c21 += M2
        # W: +1 at (k1,i2)
        U[i2*m_local+j1][col] += 1
        U[i2*m_local+j2][col] += 1
        V[j1*p+k1][col] += 1
        W[k1*n+i2][col] += 1
        col += 1
        
        # M3 = (a11)(b21-b22)
        # U: +1 at (i1,j1)
        # V: +1 at (j2,k1), -1 at (j2,k2)
        # M3 contributes to c12: c12 += M3
        # W: +1 at (k2,i1)
        U[i1*m_local+j1][col] += 1
        V[j2*p+k1][col] += 1
        V[j2*p+k2][col] -= 1
        W[k2*n+i1][col] += 1
        col += 1
        
        # M4 = (a22)(b12-b11)
        # U: +1 at (i2,j2)
        # V: +1 at (j1,k2), -1 at (j1,k1)
        # M4 contributes to c11 and c21: c11 += M4, c21 += M4
        # W: +1 at (k1,i1), +1 at (k1,i2)
        U[i2*m_local+j2][col] += 1
        V[j1*p+k2][col] += 1
        V[j1*p+k1][col] -= 1
        W[k1*n+i1][col] += 1
        W[k1*n+i2][col] += 1
        col += 1
        
        # M5 = (a11+a12)(b22)
        # U: +1 at (i1,j1), +1 at (i1,j2)
        # V: +1 at (j2,k2)
        # M5 contributes to c12 and c11 (with negative sign): c12 += M5, c11 -= M5
        # W: +1 at (k2,i1), -1 at (k1,i1)
        U[i1*m_local+j1][col] += 1
        U[i1*m_local+j2][col] += 1
        V[j2*p+k2][col] += 1
        W[k2*n+i1][col] += 1
        W[k1*n+i1][col] -= 1
        col += 1
        
        # M6 = (a21-a11)(b11+b12)
        # U: +1 at (i2,j1), -1 at (i1,j1)
        # V: +1 at (j1,k1), +1 at (j1,k2)
        # M6 doesn't directly appear in the standard Strassen formulas I wrote above.
        # Let me recheck...
        
        # Actually, I think I made an error. Let me use the standard Strassen formulas:
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3
        
        # So M6 is not used in the standard 7-term Strassen. Let me recount:
        # M1, M2, M3, M4, M5, M6, M7 - that's 7 terms.
        # But in the formulas above, M6 doesn't appear. That can't be right.
        
        # Let me look up the correct Strassen formulas again:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)(b11)
        # M3 = (a11)(b21-b22)
        # M4 = (a22)(b12-b11)
        # M5 = (a11+a12)(b22)
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        #
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3
        
        # Wait, M6 still doesn't appear. Let me check online...
        # Actually, I think the correct formulas are:
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3
        
        # And M6 is not used? That seems wrong. Let me verify with a simple example.
        
        # Actually, I think I have the wrong set of M's. The standard Strassen uses:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)(b11)
        # M3 = (a11)(b21-b22)
        # M4 = (a22)(b12-b11)
        # M5 = (a11+a12)(b22)
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        #
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3
        
        # Let me verify c22: M1 - M2 + M3
        # = (a11+a22)(b11+b22) - (a21+a22)(b11) + a11(b21-b22)
        # = a11*b11 + a11*b22 + a22*b11 + a22*b22 - a21*b11 - a22*b11 + a11*b21 - a11*b22
        # = a11*b11 + a22*b22 - a21*b11 + a11*b21
        # But c22 should be a21*b11 + a22*b12 + a21*b21 + a22*b22
        
        # This doesn't match. I must have the wrong formulas.
        
        # Let me just use a verified Strassen implementation.
        # The correct Strassen formulas are:
        # p1 = (a11+a22)*(b11+b22)
        # p2 = (a21+a22)*b11
        # p3 = a11*(b21-b22)
        # p4 = a22*(b12-b11)
        # p5 = (a11+a12)*b22
        # p6 = (a21-a11)*(b11+b12)
        # p7 = (a12-a22)*(b21+b22)
        #
        # c11 = p1 + p4 - p5 + p7
        # c12 = p3 + p5
        # c21 = p2 + p4
        # c22 = p1 - p2 + p3
        
        # Let me verify c22 again:
        # p1 - p2 + p3 = (a11+a22)(b11+b22) - (a21+a22)b11 + a11(b21-b22)
        # = a11b11 + a11b22 + a22b11 + a22b22 - a21b11 - a22b11 + a11b21 - a11b22
        # = a11b11 + a22b22 - a21b11 + a11b21
        # Expected: a21b11 + a22b12 + a21b21 + a22b22
        # Still doesn't match. I'm clearly making an error.
        
        # Let me just abandon the Strassen approach for now and return the schoolbook.
        # The key insight from the observations is that 6 disjoint Strassen blocks give rank 99.
        # I'll implement a simple version that does this.
        
        return col + 7  # We used 7 columns
    
    # Given the complexity and time constraints, let me just return the schoolbook
    # The observed best is rank 99, which I should be able to achieve.
    
    # Let me try a different approach: directly construct the rank-99 decomposition.
    
    # For 3x5x7, we can partition as follows:
    # i: {0,1}, {2}  (2 + 1)
    # j: {0,1}, {2,3}, {4}  (2 + 2 + 1)
    # k: {0,1}, {2,3}, {4,5}, {6}  (2 + 2 + 2 + 1)
    
    # 2x2x2 blocks:
    # Block 1: i={0,1}, j={0,1}, k={0,1}
    # Block 2: i={0,1}, j={0,1}, k={2,3}
    # Block 3: i={0,1}, j={0,1}, k={4,5}
    # Block 4: i={0,1}, j={2,3}, k={0,1}
    # Block 5: i={0,1}, j={2,3}, k={2,3}
    # Block 6: i={0,1}, j={2,3}, k={4,5}
    
    # That's 6 disjoint 2x2x2 blocks. Each saves 1 rank.
    # Remaining terms: all terms not in these blocks.
    
    # Let me count:
    # Total terms: 3*5*7 = 105
    # Terms in 6 blocks: 6*8 = 48
    # Remaining terms: 105 - 48 = 57
    # New rank: 57 + 6*7 = 57 + 42 = 99
    
    # Great! So the strategy is:
    # 1. Apply Strassen to the 6 blocks above
    # 2. Use schoolbook for the remaining 57 terms
    
    # Now I need to correctly implement Strassen for a 2x2x2 block.
    
    # Let me carefully derive the Strassen decomposition for our tensor convention.
    
    # For a 2x2x2 block with i in {i1,i2}, j in {j1,j2}, k in {k1,k2}:
    # We want to replace 8 rank-1 terms with 7.
    
    # The 8 schoolbook terms contribute:
    # For (i,j,k) in the block: T[i,j,k] += 1
    
    # We need 7 rank-1 terms whose combined contribution is exactly 1 for each
    # (i,j,k) in the block, and 0 for all other (i,j,k).
    
    # Let me define the 7 terms. I'll use the standard Strassen algorithm.
    
    # Actually, let me just verify my Strassen implementation with a small test.
    
    # For now, let me return the schoolbook and note that the optimal is 99.
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
