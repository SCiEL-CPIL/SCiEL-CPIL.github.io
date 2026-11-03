"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    rank = 10
    
    # Initialize tensors with zeros
    # U: (n*m, rank) = (6, 10)
    # V: (m*p, rank) = (9, 10)
    # W: (p*n, rank) = (6, 10)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Indices:
    # A: i*m + j (0..5)
    # B: j*p + k (0..8)
    # C: k*n + i (0..5)
    
    # --- Strassen's Algorithm for A[0:2, 0:2] * B[0:2, 0:2] ---
    # Terms 0-6
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1  # C[0,0]
    W[3, 0] = 1  # C[1,1]
    
    # M2 = (A10 + A11) * B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1  # C[1,0]
    W[3, 1] = -1 # C[1,1]
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1  # C[0,1]
    W[3, 2] = 1  # C[1,1]
    
    # M4 = A11 * (B10 - B00)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C[0,0]
    W[1, 3] = 1  # C[1,0]
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[2, 4] = 1  # C[0,1]
    W[0, 4] = -1 # C[0,0]
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1  # C[0,0]
    
    # --- Tail Term: A[:, 2] * B[2, 0:2] -> C[:, 0:2] ---
    # Term 7
    # A02: idx 2, A12: idx 5
    # B20: idx 6, B21: idx 7
    # C00: idx 0, C10: idx 1, C01: idx 2, C11: idx 3
    U[2, 7] = 1
    U[5, 7] = 1
    V[6, 7] = 1
    V[7, 7] = 1
    W[0, 7] = 1
    W[1, 7] = 1
    W[2, 7] = 1
    W[3, 7] = 1
    
    # --- Tail Terms: A[:, 0:2] * B[0:2, 2] + A[:, 2] * B[2, 2] -> C[:, 2] ---
    # Term 8 for C[0, 2]
    # A00: 0, A01: 1, A02: 2
    # B02: 2, B12: 5, B22: 8
    # C02: idx 4
    U[0, 8] = 1
    U[1, 8] = 1
    U[2, 8] = 1
    V[2, 8] = 1
    V[5, 8] = 1
    V[8, 8] = 1
    W[4, 8] = 1
    
    # Term 9 for C[1, 2]
    # A10: 3, A11: 4, A12: 5
    # B02: 2, B12: 5, B22: 8
    # C12: idx 5
    U[3, 9] = 1
    U[4, 9] = 1
    U[5, 9] = 1
    V[2, 9] = 1
    V[5, 9] = 1
    V[8, 9] = 1
    W[5, 9] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
