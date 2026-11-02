"""
Tensor decomposition for 2x2 matrix multiplication.
Implements Strassen's algorithm with rank 7.
"""
import numpy as np

DIMENSIONS = [2, 2, 2]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    for the tensor decomposition of 2x2 matrix multiplication.
    Shapes:
    U: (n*m, rank) = (4, 7)
    V: (m*p, rank) = (4, 7)
    W: (p*n, rank) = (4, 7)
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Indices for U: U[i*m + j, r] corresponds to A[i, j]
    # Indices for V: V[j*p + k, r] corresponds to B[j, k]
    # Indices for W: W[k*n + i, r] corresponds to C[i, k] (Note: task specifies W[k*n+i, r] is the output factor for C[i,k] or similar trace convention. 
    # Standard convention: C[i,k] = sum_j A[i,j] B[j,k].
    # The prompt says "output factor follows the official trace convention W[k*n+i,r]". 
    # Usually, the tensor T_{ijk} = A_{ij} B_{jk}. The output index is usually k (col) and i (row).
    # Let's stick to the standard Strassen mapping which is well-tested.
    
    # Strassen's 7 products:
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    
    # Reconstruction:
    # C00 = P1 + P4 - P5 + P7
    # C10 = P2 + P4
    # C01 = P3 + P5
    # C11 = P1 - P2 + P3 + P6
    
    # Map to indices:
    # U row indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V row indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W row indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1) 
    # Wait, the prompt says W[k*n+i, r]. 
    # If n=2, indices are:
    # k=0, i=0 -> 0*2+0 = 0 -> C[0,0]
    # k=0, i=1 -> 0*2+1 = 1 -> C[1,0] ? No, usually W index maps to the output matrix index.
    # Let's check the seed program's W mapping.
    # Seed:
    # W[0] (k=0,i=0) is M[0,0]
    # W[1] (k=0,i=1) is M[1,0]
    # W[2] (k=1,i=0) is M[0,1]
    # W[3] (k=1,i=1) is M[1,1]
    # This implies W index `idx` corresponds to output element C[i, k] where idx = k*n + i.
    # So:
    # idx 0: k=0, i=0 -> C[0,0]
    # idx 1: k=0, i=1 -> C[1,0]
    # idx 2: k=1, i=0 -> C[0,1]
    # idx 3: k=1, i=1 -> C[1,1]
    
    # Let's populate U, V, W based on Strassen formulas.
    
    # P1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    
    # P2 = (A10 + A11) * B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    
    # P3 = A00 * (B01 - B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # B11
    
    # P4 = A11 * (B10 - B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    
    # P5 = (A00 + A01) * B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    
    # P6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    
    # P7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    
    # Now populate W based on C[i,k] reconstruction.
    # C[0,0] (idx 0) = P1 + P4 - P5 + P7
    W[0, 0] = 1
    W[0, 3] = 1
    W[0, 4] = -1
    W[0, 6] = 1
    
    # C[1,0] (idx 1) = P2 + P4
    W[1, 1] = 1
    W[1, 3] = 1
    
    # C[0,1] (idx 2) = P3 + P5
    W[2, 2] = 1
    W[2, 4] = 1
    
    # C[1,1] (idx 3) = P1 - P2 + P3 + P6
    W[3, 0] = 1
    W[3, 1] = -1
    W[3, 2] = 1
    W[3, 5] = 1
    
    # Convert to lists to match expected output format if strict types are required, 
    # though numpy arrays are often accepted. The seed returned lists of lists.
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
