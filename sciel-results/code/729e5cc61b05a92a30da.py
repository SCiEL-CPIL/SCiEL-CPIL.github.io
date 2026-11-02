import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 35
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Define Strassen terms for a 2x2x2 block
    # Each term is a list of (index, coefficient) for U, V, W
    # U indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V indices (local): 0=B00, 1=B01, 2=B10, 3=B11
    # W indices (local): 0=C00, 1=C01, 2=C10, 3=C11
    
    strassen_terms = [
        # M1: (A00+A11)(B00+B11) -> C00, C11
        [([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])],
        # M2: (A10+A11)B00 -> C10, -C11
        [([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])],
        # M3: A00(B01-B11) -> C01, C11
        [([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])],
        # M4: A11(B10-B00) -> C00, C10
        [([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])],
        # M5: (A00+A01)B11 -> -C00, C01
        [([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)])],
        # M6: (A10-A00)(B00+B01) -> C11
        [([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])],
        # M7: (A01-A11)(B10+B11) -> C00
        [([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])],
    ]
    
    # Note: The above structure is slightly complex to unpack. 
    # Let's simplify by defining the coefficients directly in the loop.
    
    U = np.zeros((4, 35), dtype=int)
    V = np.zeros((20, 35), dtype=int)
    W = np.zeros((20, 35), dtype=int)
    
    for block in range(5):
        k1 = 2 * block
        k2 = 2 * block + 1
        
        # Local V indices for this block
        # B0k1 -> V[k1]
        # B0k2 -> V[k2]
        # B1k1 -> V[10+k1]
        # B1k2 -> V[10+k2]
        
        # Local W indices for this block
        # C0k1 -> W[2*k1]
        # C1k1 -> W[2*k1+1]
        # C0k2 -> W[2*k2]
        # C1k2 -> W[2*k2+1]
        
        # M1
        r = 7 * block + 0
        U[0, r] += 1
        U[3, r] += 1
        V[k1, r] += 1
        V[10 + k2, r] += 1
        W[2 * k1, r] += 1
        W[2 * k2 + 1, r] += 1
        
        # M2
        r = 7 * block + 1
        U[2, r] += 1
        U[3, r] += 1
        V[k1, r] += 1
        W[2 * k1 + 1, r] += 1
        W[2 * k2 + 1, r] -= 1
        
        # M3
        r = 7 * block + 2
        U[0, r] += 1
        V[k2, r] += 1
        V[10 + k2, r] -= 1
        W[2 * k2, r] += 1
        W[2 * k2 + 1, r] += 1
        
        # M4
        r = 7 * block + 3
        U[3, r] += 1
        V[10 + k1, r] += 1
        V[k1, r] -= 1
        W[2 * k1, r] += 1
        W[2 * k1 + 1, r] += 1
        
        # M5
        r = 7 * block + 4
        U[0, r] += 1
        U[1, r] += 1
        V[10 + k2, r] += 1
        W[2 * k1, r] -= 1
        W[2 * k2, r] += 1
        
        # M6
        r = 7 * block + 5
        U[2, r] += 1
        U[0, r] -= 1
        V[k1, r] += 1
        V[k2, r] += 1
        W[2 * k2 + 1, r] += 1
        
        # M7
        r = 7 * block + 6
        U[1, r] += 1
        U[3, r] -= 1
        V[10 + k1, r] += 1
        V[10 + k2, r] += 1
        W[2 * k1, r] += 1

    # Convert to list of lists as required by the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
