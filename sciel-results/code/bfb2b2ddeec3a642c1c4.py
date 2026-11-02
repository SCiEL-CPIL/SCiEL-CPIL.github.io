import numpy as np

def construct():
    n, m, p = 2, 2, 13
    
    # Total rank
    num_strassen_blocks = p // 2
    num_standard_blocks = p % 2
    total_rank = num_strassen_blocks * 7 + num_standard_blocks * 4
    
    # Dimensions
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    U = np.zeros((dim_u, total_rank))
    V = np.zeros((dim_v, total_rank))
    W = np.zeros((dim_w, total_rank))
    
    col_idx = 0
    
    # Helper to add a Strassen block for B cols (k, k+1) and C cols (k, k+1)
    def add_strassen_block(k):
        nonlocal col_idx
        
        # B indices for this block:
        # B[0, k] -> 0*13 + k
        # B[1, k] -> 1*13 + k
        # B[0, k+1] -> 0*13 + (k+1)
        # B[1, k+1] -> 1*13 + (k+1)
        
        # C indices for this block:
        # C[0, k] -> 0*13 + k
        # C[1, k] -> 1*13 + k
        # C[0, k+1] -> 0*13 + (k+1)
        # C[1, k+1] -> 1*13 + (k+1)
        
        # A indices:
        # A[0,0] -> 0
        # A[0,1] -> 1
        # A[1,0] -> 2
        # A[1,1] -> 3
        
        # Strassen coefficients
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B01-B11)
        # P4 = A11(B10-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # W coefficients (Output)
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # We will fill columns col_idx to col_idx+6
        
        # P1
        U[0, col_idx] = 1
        U[3, col_idx] = 1
        V[0*13 + k, col_idx] = 1
        V[1*13 + k, col_idx] = 1
        W[0*13 + k, col_idx] = 1
        W[1*13 + (k+1), col_idx] = 1
        col_idx += 1
        
        # P2
        U[2, col_idx] = 1
        U[3, col_idx] = 1
        V[0*13 + k, col_idx] = 1
        W[1*13 + k, col_idx] = 1
        W[1*13 + (k+1), col_idx] = -1
        col_idx += 1
        
        # P3
        U[0, col_idx] = 1
        V[0*13 + (k+1), col_idx] = 1
        V[1*13 + (k+1), col_idx] = -1
        W[0*13 + (k+1), col_idx] = 1
        W[1*13 + (k+1), col_idx] = 1
        col_idx += 1
        
        # P4
        U[3, col_idx] = 1
        V[1*13 + k, col_idx] = 1
        V[0*13 + k, col_idx] = -1
        W[0*13 + k, col_idx] = 1
        W[1*13 + k, col_idx] = 1
        col_idx += 1
        
        # P5
        U[0, col_idx] = 1
        U[1, col_idx] = 1
        V[1*13 + (k+1), col_idx] = 1
        W[0*13 + k, col_idx] = -1
        W[0*13 + (k+1), col_idx] = 1
        col_idx += 1
        
        # P6
        U[2, col_idx] = 1
        U[0, col_idx] = -1
        V[0*13 + k, col_idx] = 1
        V[0*13 + (k+1), col_idx] = 1
        W[1*13 + (k+1), col_idx] = 1
        col_idx += 1
        
        # P7
        U[1, col_idx] = 1
        U[3, col_idx] = -1
        V[1*13 + k, col_idx] = 1
        V[1*13 + (k+1), col_idx] = 1
        W[0*13 + k, col_idx] = 1
        W[1*13 + (k+1), col_idx] = 1
        col_idx += 1

    # Helper to add a standard block for B col k and C col k
    def add_standard_block(k):
        nonlocal col_idx
        
        # Q1 = A00 B0
        U[0, col_idx] = 1
        V[0*13 + k, col_idx] = 1
        W[0*13 + k, col_idx] = 1
        col_idx += 1
        
        # Q2 = A01 B1
        U[1, col_idx] = 1
        V[1*13 + k, col_idx] = 1
        W[0*13 + k, col_idx] = 1
        col_idx += 1
        
        # Q3 = A10 B0
        U[2, col_idx] = 1
        V[0*13 + k, col_idx] = 1
        W[1*13 + k, col_idx] = 1
        col_idx += 1
        
        # Q4 = A11 B1
        U[3, col_idx] = 1
        V[1*13 + k, col_idx] = 1
        W[1*13 + k, col_idx] = 1
        col_idx += 1

    for k in range(num_strassen_blocks):
        add_strassen_block(2*k)
        
    if num_standard_blocks:
        add_standard_block(2*num_strassen_blocks)
        
    return {"U": U, "V": V, "W": W}
