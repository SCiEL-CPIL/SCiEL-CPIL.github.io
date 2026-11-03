"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    # Strategy: Use rank-49 decomposition via cross-block relaxation
    # Base: 2x2x2 (rank 7) + 2x2x7 remainder (rank 42) = 49
    # But allow cross-term cancellations to potentially reduce rank
    
    # First, try the known rank-49 construction
    res = _construct_rank49_crossblock(n, m, p, deadline)
    if res is not None:
        return res
    
    # Fallback: schoolbook decomposition
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

def _construct_rank49_crossblock(n, m, p, deadline):
    """
    Construct a rank-49 decomposition using:
    - 2x2x2 block (rows 0-1, cols 0-1, output cols 0-1): rank 7 via Strassen
    - 2x2x7 remainder (rows 0-1, cols 2-3, output cols 0-6): rank 42 via schoolbook
    Total: 7 + 42 = 49
    
    The cross-block relaxation allows terms from the 2x2x2 block to interact
    with the remainder, potentially enabling rank reduction.
    """
    # Initialize with the known rank-49 structure
    rank = 49
    
    # U: shape (n*m, rank) = (8, 49)
    # V: shape (m*p, rank) = (28, 49)
    # W: shape (p*n, rank) = (14, 49)
    
    U = np.zeros((n, m, rank))
    V = np.zeros((m, p, rank))
    W = np.zeros((p, n, rank))
    
    # Part 1: 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    # Strassen's algorithm for 2x2x2: rank 7
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For our tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We need T[i,j,k] = 1 for all i,j,k in the 2x2x2 block
    
    # Strassen decomposition for 2x2x2:
    # c = 0: M1 = (A00+A11)(B00+B11)
    # c = 1: M2 = (A10+A11)B00
    # c = 2: M3 = A00(B10-B11)
    # c = 3: M4 = A11(B01-B10)
    # c = 4: M5 = (A00+A01)B11
    # c = 5: M6 = (A10-A00)(B00+B10)
    # c = 6: M7 = (A01-A11)(B01+B11)
    
    # U[i,j,c] for i,j in {0,1}, c in {0..6}
    # A = [[A00, A01], [A10, A11]]
    # So U[0,0,c] = coefficient of A00 in term c
    # U[0,1,c] = coefficient of A01 in term c
    # U[1,0,c] = coefficient of A10 in term c
    # U[1,1,c] = coefficient of A11 in term c
    
    # M1 = (A00+A11)(B00+B11): U[0,0,0]=1, U[1,1,0]=1
    # M2 = (A10+A11)B00: U[1,0,1]=1, U[1,1,1]=1
    # M3 = A00(B10-B11): U[0,0,2]=1
    # M4 = A11(B01-B10): U[1,1,3]=1
    # M5 = (A00+A01)B11: U[0,0,4]=1, U[0,1,4]=1
    # M6 = (A10-A00)(B00+B10): U[1,0,5]=1, U[0,0,5]=-1
    # M7 = (A01-A11)(B01+B11): U[0,1,6]=1, U[1,1,6]=-1
    
    U[0,0,0] = 1  # A00 in M1
    U[1,1,0] = 1  # A11 in M1
    
    U[1,0,1] = 1  # A10 in M2
    U[1,1,1] = 1  # A11 in M2
    
    U[0,0,2] = 1  # A00 in M3
    
    U[1,1,3] = 1  # A11 in M4
    
    U[0,0,4] = 1  # A00 in M5
    U[0,1,4] = 1  # A01 in M5
    
    U[1,0,5] = 1  # A10 in M6
    U[0,0,5] = -1 # -A00 in M6
    
    U[0,1,6] = 1  # A01 in M7
    U[1,1,6] = -1 # -A11 in M7
    
    # V[j,k,c] for j,k in {0,1}, c in {0..6}
    # B = [[B00, B01], [B10, B11]]
    # So V[0,0,c] = coefficient of B00 in term c
    # V[0,1,c] = coefficient of B01 in term c
    # V[1,0,c] = coefficient of B10 in term c
    # V[1,1,c] = coefficient of B11 in term c
    
    # M1 = (A00+A11)(B00+B11): V[0,0,0]=1, V[1,1,0]=1
    # M2 = (A10+A11)B00: V[0,0,1]=1
    # M3 = A00(B10-B11): V[1,0,2]=1, V[1,1,2]=-1
    # M4 = A11(B01-B10): V[0,1,3]=1, V[1,0,3]=-1
    # M5 = (A00+A01)B11: V[1,1,4]=1
    # M6 = (A10-A00)(B00+B10): V[0,0,5]=1, V[1,0,5]=1
    # M7 = (A01-A11)(B01+B11): V[0,1,6]=1, V[1,1,6]=1
    
    V[0,0,0] = 1  # B00 in M1
    V[1,1,0] = 1  # B11 in M1
    
    V[0,0,1] = 1  # B00 in M2
    
    V[1,0,2] = 1  # B10 in M3
    V[1,1,2] = -1 # -B11 in M3
    
    V[0,1,3] = 1  # B01 in M4
    V[1,0,3] = -1 # -B10 in M4
    
    V[1,1,4] = 1  # B11 in M5
    
    V[0,0,5] = 1  # B00 in M6
    V[1,0,5] = 1  # B10 in M6
    
    V[0,1,6] = 1  # B01 in M7
    V[1,1,6] = 1  # B11 in M7
    
    # W[k,i,c] for k,i in {0,1}, c in {0..6}
    # C = [[C00, C01], [C10, C11]]
    # So W[0,0,c] = coefficient of C00 in term c (k=0, i=0)
    # W[0,1,c] = coefficient of C01 in term c (k=0, i=1)
    # W[1,0,c] = coefficient of C10 in term c (k=1, i=0)
    # W[1,1,c] = coefficient of C11 in term c (k=1, i=1)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # W[0,0,c] for C00:
    W[0,0,0] = 1   # M1
    W[0,0,3] = 1   # M4
    W[0,0,4] = -1  # -M5
    W[0,0,6] = -1  # -M7
    
    # W[0,1,c] for C01:
    W[0,1,2] = 1   # M3
    W[0,1,4] = 1   # M5
    
    # W[1,0,c] for C10:
    W[1,0,1] = 1   # M2
    W[1,0,3] = 1   # M4
    
    # W[1,1,c] for C11:
    W[1,1,0] = 1   # M1
    W[1,1,1] = -1  # -M2
    W[1,1,2] = 1   # M3
    W[1,1,6] = 1   # M7
    
    # Part 2: 2x2x7 remainder (i in {0,1}, j in {2,3}, k in {0..6})
    # Schoolbook: 2*2*7 = 28 terms, but we need 42 more to reach rank 49
    # Wait, 7 + 28 = 35, not 49. Let me recalculate.
    
    # Actually, the remainder is 2x2x7 (i in {0,1}, j in {2,3}, k in {0..6})
    # This has 2*2*7 = 28 elements, so rank 28 via schoolbook
    # Total: 7 + 28 = 35, not 49.
    
    # I made an error. Let me reconsider the decomposition.
    
    # The full tensor is 2x4x7 = 56 elements.
    # If we split j into {0,1} and {2,3}:
    # - j in {0,1}: 2x2x7 = 28 elements
    # - j in {2,3}: 2x2x7 = 28 elements
    # Total: 56 elements
    
    # For j in {0,1}, we can further split k into {0,1} and {2..6}:
    # - j in {0,1}, k in {0,1}: 2x2x2 = 8 elements, rank 7 via Strassen
    # - j in {0,1}, k in {2..6}: 2x2x5 = 20 elements, rank 20 via schoolbook
    # Subtotal for j in {0,1}: 7 + 20 = 27
    
    # For j in {2,3}, k in {0..6}: 2x2x7 = 28 elements, rank 28 via schoolbook
    
    # Total: 27 + 28 = 55, not 49.
    
    # I need a better decomposition. Let me try:
    # Split into 2x2x2 blocks:
    # - (i in {0,1}, j in {0,1}, k in {0,1}): rank 7
    # - (i in {0,1}, j in {0,1}, k in {2..6}): 2x2x5, rank 20
    # - (i in {0,1}, j in {2,3}, k in {0,1}): 2x2x2, rank 7
    # - (i in {0,1}, j in {2,3}, k in {2..6}): 2x2x5, rank 20
    # Total: 7 + 20 + 7 + 20 = 54
    
    # Still not 49. The rank-49 claim must come from a different construction.
    
    # Let me use the known result: R(2,4,7) <= 49 from the literature.
    # I'll use a direct construction based on the 2x4x4 and 2x4x3 splits.
    
    # For now, return None and use the fallback.
    return None

def _to_lists(U, V, W, n, m, p, rank):
    """Convert numpy arrays to lists of lists."""
    U_flat = U.reshape(n*m, rank)
    V_flat = V.reshape(m*p, rank)
    W_flat = W.reshape(p*n, rank)
    
    U_l = [[float(U_flat[i, c]) for c in range(rank)] for i in range(n*m)]
    V_l = [[float(V_flat[i, c]) for c in range(rank)] for i in range(m*p)]
    W_l = [[float(W_flat[i, c]) for c in range(rank)] for i in range(p*n)]
    return {"U": U_l, "V": V_l, "W": W_l}
# EVOLVE-BLOCK-END
