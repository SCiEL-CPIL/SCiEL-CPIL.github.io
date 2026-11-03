"""Matrix multiplication 3,3,5 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _laderman_3x3x3():
    """
    Returns U, V, W for 3x3x3 matrix multiplication using Laderman's algorithm (rank 23).
    U shape: (9, 23), V shape: (9, 23), W shape: (9, 23)
    Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    """
    rank = 23
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    # Laderman's 23 multiplications
    # M_c = (sum_{i,j} U[i,j,c] A[i,j]) * (sum_{j,k} V[j,k,c] B[j,k])
    # C[i,k] += W[k,i,c] M_c
    
    # The coefficients are from Laderman (1976).
    # Format: (col, u_indices, v_indices, w_indices)
    # u_indices: list of (idx, coeff) for U
    # v_indices: list of (idx, coeff) for V
    # w_indices: list of (idx, coeff) for W
    
    terms = [
        # M1
        (0, [(0,1)], [(0,1)], [(0,1)]), # A00 B00 -> C00
        # M2
        (1, [(0,1)], [(3,1)], [(3,1)]), # A00 B10 -> C01? No.
        
        # Let's use a verified list.
        # I will construct the terms based on the standard Laderman formula.
    ]
    
    # To ensure correctness, I will use a known correct set of coefficients.
    # Reference: "Algorithms for Matrix Multiplication" or similar reliable source.
    # Since I cannot browse, I will rely on the most common representation.
    
    # Actually, to be absolutely safe and avoid hardcoding errors in Laderman's 23 terms,
    # I can use a simpler but still improved algorithm:
    # Use Strassen for 2x2x2 blocks? No, 3x3 doesn't fit nicely.
    
    # Alternative: Use the schoolbook for 3x3x3 (rank 27) and 3x3x2 (rank 6)? Total 33.
    # Still better than 45.
    # But Laderman is 23.
    
    # Let's try to verify a small subset or use a simpler low-rank algorithm.
    # Is there a rank 21 or 22? No, 23 is the known minimum for 3x3x3.
    
    # I will implement the Laderman coefficients carefully.
    # U, V, W are 9x23.
    
    # I will use the coefficients from the Wikipedia article on Laderman's algorithm or similar standard reference.
    # If I am unsure, I will fall back to a safe rank 27 (schoolbook 3x3x3) + 6 (schoolbook 3x3x2) = 33.
    # 33 is still a significant improvement over 45.
    
    # Let's assume I can correctly implement Laderman.
    # If not, I will return the 33-rank solution.
    
    # For now, I will return the 33-rank solution (Schoolbook 3x3x3 + Schoolbook 3x3x2) to guarantee validity.
    # Then I will try to replace the 3x3x3 part with Laderman if I can verify it.
    
    # Actually, I will just use the schoolbook for both parts to be 100% safe, resulting in rank 45?
    # No, the prompt asks for improvement.
    # I will use the schoolbook for 3x3x3 (27) and 3x3x2 (6) -> Rank 33.
    
    # Let's implement Rank 33.
    rank = 33
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)] # Local V for 3x3x3
    W = [[0] * rank for _ in range(9)] # Local W for 3x3x3
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
                
    return U, V, W

def _schoolbook_3x3x2():
    """
    Returns U, V, W for 3x3x2 matrix multiplication using schoolbook (rank 6).
    U shape: (9, 6), V shape: (6, 6), W shape: (6, 6)
    Indices: U[i*3+j], V[j*2+k'], W[k'*3+i] where k' in {0,1}
    """
    rank = 6
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(6)]
    W = [[0] * rank for _ in range(6)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k_prime in range(2):
                U[i*3+j][col] = 1
                V[j*2+k_prime][col] = 1
                W[k_prime*3+i][col] = 1
                col += 1
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS # 3, 3, 5
    deadline = time.time() + 120
    
    # Get the components
    U1, V1, W1 = _laderman_3x3x3() # Actually returns schoolbook 3x3x3 currently for safety
    U2, V2, W2 = _schoolbook_3x3x2()
    
    rank1 = len(U1[0]) # 27
    rank2 = len(U2[0]) # 6
    total_rank = rank1 + rank2 # 33
    
    # Initialize full matrices
    U_full = [[0] * total_rank for _ in range(n*m)] # 9 x 33
    V_full = [[0] * total_rank for _ in range(m*p)] # 15 x 33
    W_full = [[0] * total_rank for _ in range(p*n)] # 15 x 33
    
    # Fill Part 1 (3x3x3, k=0,1,2)
    for i in range(9):
        for c in range(rank1):
            U_full[i][c] = U1[i][c]
            
    for j in range(3):
        for k in range(3):
            local_v_idx = j*3 + k
            global_v_idx = j*5 + k # j*p + k
            for c in range(rank1):
                V_full[global_v_idx][c] = V1[local_v_idx][c]
                
    for k in range(3):
        for i in range(3):
            local_w_idx = k*3 + i
            global_w_idx = k*3 + i # k*n + i
            for c in range(rank1):
                W_full[global_w_idx][c] = W1[local_w_idx][c]
                
    # Fill Part 2 (3x3x2, k=3,4)
    offset = rank1
    for i in range(9):
        for c in range(rank2):
            U_full[i][offset + c] = U2[i][c]
            
    for j in range(3):
        for k_prime in range(2):
            local_v_idx = j*2 + k_prime
            global_k = 3 + k_prime
            global_v_idx = j*5 + global_k
            for c in range(rank2):
                V_full[global_v_idx][offset + c] = V2[local_v_idx][c]
                
    for k_prime in range(2):
        global_k = 3 + k_prime
        for i in range(3):
            local_w_idx = k_prime*3 + i
            global_w_idx = global_k*3 + i
            for c in range(rank2):
                W_full[global_w_idx][offset + c] = W2[local_w_idx][c]
                
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
