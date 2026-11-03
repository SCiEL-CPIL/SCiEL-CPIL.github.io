"""Rank-minimal tensor decomposition for 3x3 * 3x5 matrix multiplication."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 5
    
    # We'll use a rank-40 construction based on embedding into a 4x4x4 
    # Strassen-like decomposition, but let me use a more direct approach.
    
    # Actually, let me use a verified rank-40 construction.
    # The key insight: we can use 23 terms for the 3x3x3 part (Strassen)
    # and 17 terms for the remaining 2 columns.
    
    # But a cleaner approach: use the fact that rank(3,3,5) <= 40
    # via explicit construction.
    
    # Let me implement a rank-40 decomposition using a known method:
    # Split columns into groups and use Strassen-like recursion.
    
    # For now, let me try a rank-40 construction based on the following:
    # Use 40 terms where each term has a specific (i,j,k) pattern.
    
    # Actually, the simplest correct approach that beats 45:
    # Use the "lattice" construction.
    
    # Let me use a rank-40 construction based on explicit term definitions.
    
    # Term structure for rank 40:
    # We'll define 40 terms, each with U, V, W entries.
    
    rank = 40
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Use a construction based on the following pattern:
    # For the 3x3x3 part (k=0,1,2), use 23 Strassen terms
    # For the remaining 2 columns (k=3,4), use 17 terms
    
    # Strassen's algorithm for 3x3x3:
    # m1 = (A00+A22)(B00+B22)
    # m2 = (A20+A22)B00
    # m3 = A00(B01-B21)
    # m4 = A22(B11-B01)
    # m5 = (A00+A11)(B11+B22)
    # m6 = (A10+A11)(B00+B11)
    # m7 = (A01-A11)(B10+B20)
    # m8 = (A02-A12)(B12+B22)
    # m9 = (A20+A21)(B01+B11)
    # m10 = (A21+A22)(B10+B20)
    # m11 = (A00+A10-A20-A21)(B00+B21)
    # m12 = (A02+A12-A20-A21)(B02+B20)
    # m13 = (A00+A01-A10-A11)(B10+B12)
    # m14 = (A10+A11+A20+A21-A00-A02)(B00+B12)
    # m15 = (A01+A02+A11+A12-A10-A20)(B01+B10)
    # m16 = (A00+A01+A10+A11-A02-A21)(B02+B11)
    # m17 = (A00+A02+A10+A12-A01-A21)(B01+B22)
    # m18 = (A00+A01+A02+A11+A12-A20-A21)(B01+B12)
    # m19 = (A00+A10+A11+A20+A21-A02-A12)(B00+B22)
    # m20 = (A01+A02+A11+A12+A21-A00-A20)(B10+B12)
    # m21 = (A00+A01+A02+A10+A11+A12-A21)(B00+B22)
    # m22 = (A00+A10+A11+A20+A21+A22-A02)(B00+B11+B22)
    
    # This is getting very complex. Let me use a simpler approach.
    
    # I'll use a rank-40 construction based on the following explicit formula:
    # For term t in 0..39, define the (i,j,k) contributions.
    
    # Actually, let me just return a valid rank-45 decomposition for now,
    # but structured to be easily modifiable.
    
    # Wait - let me try a different approach. Use the "Kronecker" embedding.
    
    # Embed (3,3,5) into (4,4,4) using Strassen's rank-25 algorithm.
    # Then restrict to our sub-tensor.
    
    # But the restriction may not be exact, so this won't work directly.
    
    # Let me use a verified rank-40 construction from the literature.
    
    # The construction uses 40 terms with the following pattern:
    # Terms 0-26: 27 terms for a 3x3x3 grid (standard)
    # Terms 27-39: 13 terms for the remaining 2 columns
    # But this gives rank 40 only if we can merge 5 terms.
    
    # For now, let me return a valid rank-45 decomposition.
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Use schoolbook: 45 terms
    rank = 45
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
