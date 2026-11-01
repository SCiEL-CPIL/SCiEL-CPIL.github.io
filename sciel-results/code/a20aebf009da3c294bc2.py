"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if val != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    def make_schoolbook():
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
        return U, V, W, rank
    
    U_school, V_school, W_school, rank_school = make_schoolbook()
    
    # Try to find a lower rank decomposition
    # For 2x2 * 2xp, the optimal rank is known to be 2p+2 = 32 for p >= 2
    # Construction using Strassen-like sharing:
    #
    # For each column k, we need:
    #   c0k = a00*b0k + a01*b1k
    #   c1k = a10*b0k + a11*b1k
    #
    # Define shared linear combinations:
    #   s0 = a00 + a10
    #   s1 = a01 + a11
    #   d0 = a00 - a10
    #   d1 = a01 - a11
    #
    # Then:
    #   a00 = (s0 + d0)/2
    #   a10 = (s0 - d0)/2
    #   a01 = (s1 + d1)/2
    #   a11 = (s1 - d1)/2
    #
    # Products per column k:
    #   P0k = s0 * b0k
    #   P1k = s1 * b1k
    #   P2k = d0 * b0k
    #   P3k = d1 * b1k
    #
    # That's 4p products, which is the same as schoolbook.
    #
    # Better construction for 2p+2:
    # Use 2 shared products that combine across columns, plus 2(p-1) column-specific products
    #
    # Actually, the correct 2p+2 construction for 2x2 * 2xp:
    # 
    # Let's use the following approach:
    # For the 2x2 case (p=2), rank is 7 (Strassen).
    # For general p, we can extend this.
    #
    # Construction:
    #   P0 = (a00 + a10) * (b00 + b10)  -- no, this mixes rows and columns
    #
    # Let me try a different approach. For 2x2 * 2xp:
    # We can write:
    #   C = A * B where A is 2x2 and B is 2xp
    #
    # Split B into B[:, 0:2] and B[:, 2:p]
    # For the first 2 columns, use Strassen (7 products)
    # For the remaining p-2 columns, use 2 products each
    # Total: 7 + 2(p-2) = 2p + 3
    #
    # Hmm, that gives 2p+3, not 2p+2.
    #
    # Let me try another construction:
    # For each column k, use:
    #   Qk = (a00 + a10) * b0k + (a01 + a11) * b1k  -- this is not a single product
    #
    # I think I need to be more careful. Let me use a known result:
    # The rank of matrix multiplication <2,2,p> is 2p+2 for p >= 2.
    #
    # One valid construction:
    #   For k = 0, ..., p-1:
    #     P_{2k}   = (a00 + a10) * b0k
    #     P_{2k+1} = (a01 + a11) * b1k
    #   Plus 2 additional products:
    #     P_{2p}   = (a00 - a10) * (b00 - b01 + b02 - ... + (-1)^{p-1} b0_{p-1})
    #     P_{2p+1} = (a01 - a11) * (b10 - b11 + b12 - ... + (-1)^{p-1} b1_{p-1})
    #
    # This doesn't seem right either. Let me just implement the schoolbook 
    # and return it, since deriving the exact 2p+2 construction is complex.
    
    # Return schoolbook as it's guaranteed correct
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
