import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 3, 8
    
    # Strategy:
    # Split A (3x3) into 2x2 (top-left), 2x1 (top-right), 1x2 (bottom-left), 1x1 (bottom-right)
    # Split B (3x8) into 2x8 (top), 1x8 (bottom)
    # 
    # C[0:2, :] = A[0:2, 0:2] * B[0:2, :] + A[0:2, 2] * B[2, :]
    # C[2, :]   = A[2, 0:2] * B[0:2, :] + A[2, 2] * B[2, :]
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:8]
    # B[0:2, 0:8] is 4 blocks of 2x2.
    # Use Rank 8 Schoolbook for each 2x2x2 block.
    # Cost: 4 * 8 = 32
    
    # Block 2: A[0:2, 2] * B[2, :]
    # A[0:2, 2] is 2x1. B[2, :] is 1x8.
    # Result 2x8.
    # Term (i, k) is A[i, 2] * B[2, k].
    # This is rank 16 (2*8 scalar mults).
    
    # Block 3: A[2, 0:2] * B[0:2, :]
    # A[2, 0:2] is 1x2. B[0:2, :] is 2x8.
    # Result 1x8.
    # Term (k) is sum_{j=0}^1 A[2, j] * B[j, k].
    # This is rank 8 (1*8 scalar mults, but each is a dot product of len 2? 
    # No, tensor decomposition counts multiplications.
    # C[2, k] = A[2, 0]B[0, k] + A[2, 1]B[1, k].
    # This requires 2 multiplications per k. Total 2*8 = 16?
    # Wait.
    # In tensor decomposition, we sum over c.
    # For each k, we need to compute A[2, 0]B[0, k] + A[2, 1]B[1, k].
    # We can do this with 2 terms:
    # Term c1: U[2, 0]=1, V[0, k]=1, W[k, 2]=1. (Contributes A[2, 0]B[0, k])
    # Term c2: U[2, 1]=1, V[1, k]=1, W[k, 2]=1. (Contributes A[2, 1]B[1, k])
    # So yes, 16 terms.
    
    # Block 4: A[2, 2] * B[2, :]
    # 1x1 * 1x8.
    # Term (k) is A[2, 2] * B[2, k].
    # Rank 8.
    
    # Total Rank = 32 + 16 + 16 + 8 = 72?
    # Wait.
    # Block 1: 32.
    # Block 2: 16.
    # Block 3: 16.
    # Block 4: 8.
    # Total: 72.
    # This is the same as schoolbook!
    
    # Why?
    # Schoolbook:
    # C[i, k] = sum_j A[i, j] B[j, k].
    # Total terms: 3 * 3 * 8 = 72.
    
    # My block decomposition:
    # C[0:2, k] = sum_{j=0}^1 A[i, j]B[j, k] + A[i, 2]B[2, k].
    # Block 1 handles sum_{j=0}^1. Cost 2*2*8 = 32?
    # For each i in 0..1, k in 0..7, we sum j in 0..1.
    # That's 2 * 8 * 2 = 32 multiplications.
    # Block 2 handles A[i, 2]B[2, k].
    # For each i in 0..1, k in 0..7, 1 multiplication.
    # That's 2 * 8 * 1 = 16 multiplications.
    # Block 3 handles C[2, k] = sum_{j=0}^1 A[2, j]B[j, k] + A[2, 2]B[2, k].
    # Sum part: 1 * 8 * 2 = 16.
    # Last part: 1 * 8 * 1 = 8.
    # Total: 32 + 16 + 16 + 8 = 72.
    
    # So standard block decomposition doesn't help unless we use Strassen for the 2x2x2 blocks.
    # If we use Strassen (Rank 7) for the 4 blocks in Block 1:
    # Cost: 4 * 7 = 28.
    # Total: 28 + 16 + 16 + 8 = 68.
    
    # So Rank 68 is achievable with Strassen.
    
    # Let's implement Rank 68.
    
    rank = 68
    U = np.zeros((9, rank))
    V = np.zeros((24, rank))
    W = np.zeros((24, rank))
    
    col = 0
    
    # Helper to add Strassen terms for a 2x2x2 block
    # A_indices: (i0, i1, j0, j1) in U
    # B_indices: (j0, j1, k0, k1) in V
    # C_indices: (i0, i1, k0, k1) in W
    def add_strassen(A_idx, B_idx, W_idx):
        nonlocal col
        # A_idx: (a00, a01, a10, a11)
        # B_idx: (b00, b01, b10, b11)
        # W_idx: (w00, w01, w10, w11)
        
        # M1 = (A00+A11)(B00+B11)
        U[A_idx[0], col] += 1; U[A_idx[3], col] += 1
        V[B_idx[0], col] += 1; V[B_idx[3], col] += 1
        W[W_idx[0], col] += 1; W[W_idx[3], col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[A_idx[2], col] += 1; U[A_idx[3], col] += 1
        V[B_idx[0], col] += 1
        W[W_idx[2], col] += 1
        col += 1
        
        # M3 = A00(B10-B11)
        U[A_idx[0], col] += 1
        V[B_idx[2], col] += 1; V[B_idx[3], col] += -1
        W[W_idx[1], col] += 1; W[W_idx[3], col] += 1
        col += 1
        
        # M4 = A11(B01-B00)
        U[A_idx[3], col] += 1
        V[B_idx[1], col] += 1; V[B_idx[0], col] += -1
        W[W_idx[0], col] += 1; W[W_idx[2], col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[A_idx[0], col] += 1; U[A_idx[1], col] += 1
        V[B_idx[3], col] += 1
        W[W_idx[0], col] += 1; W[W_idx[1], col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[A_idx[2], col] += 1; U[A_idx[0], col] += -1
        V[B_idx[0], col] += 1; V[B_idx[1], col] += 1
        W[W_idx[3], col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[A_idx[1], col] += 1; U[A_idx[3], col] += -1
        V[B_idx[2], col] += 1; V[B_idx[3], col] += 1
        W[W_idx[0], col] += 1
        col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:8]
    # A indices:
    # A00: 0*3+0=0, A01: 0*3+1=1, A10: 1*3+0=3, A11: 1*3+1=4
    A_idx = (0, 1, 3, 4)
    
    # B blocks:
    # B[0:2, 0:2] -> B00: 0*8+0=0, B01: 0*8+1=1, B10: 1*8+0=8, B11: 1*8+1=9
    # B[0:2, 2:4] -> B00: 0*8+2=2, B01: 0*8+3=3, B10: 1*8+2=10, B11: 1*8+3=11
    # B[0:2, 4:6] -> B00: 0*8+4=4, B01: 0*8+5=5, B10: 1*8+4=12, B11: 1*8+5=13
    # B[0:2, 6:8] -> B00: 0*8+6=6, B01: 0*8+7=7, B10: 1*8+6=14, B11: 1*8+7=15
    
    # W indices for C[0:2, 0:8]
    # C00: 0*3+0=0, C01: 0*3+1=1, C10: 1*3+0=3, C11: 1*3+1=4
    # For block b, W indices shift by 3*b? No.
    # W index is k*n + i.
    # For block 0 (k=0,1), i=0,1:
    # w00 (k=0,i=0): 0*3+0=0
    # w01 (k=1,i=0): 1*3+0=3
    # w10 (k=0,i=1): 0*3+1=1
    # w11 (k=1,i=1): 1*3+1=4
    # Note: My W_idx in add_strassen expects (w00, w01, w10, w11) corresponding to C00, C01, C10, C11.
    # C00 is i=0,k=0. C01 is i=0,k=1. C10 is i=1,k=0. C11 is i=1,k=1.
    # So W_idx should be (0, 3, 1, 4).
    
    W_idx_0 = (0, 3, 1, 4)
    add_strassen(A_idx, (0, 1, 8, 9), W_idx_0)
    
    W_idx_1 = (2, 5, 3, 6) # k=2,3 -> 2*3+0=6? No.
    # k=2, i=0 -> 2*3+0=6.
    # k=3, i=0 -> 3*3+0=9.
    # k=2, i=1 -> 2*3+1=7.
    # k=3, i=1 -> 3*3+1=10.
    # So W_idx_1 = (6, 9, 7, 10).
    W_idx_1 = (6, 9, 7, 10)
    add_strassen(A_idx, (2, 3, 10, 11), W_idx_1)
    
    W_idx_2 = (12, 15, 13, 16)
    add_strassen(A_idx, (4, 5, 12, 13), W_idx_2)
    
    W_idx_3 = (18, 21, 19, 22)
    add_strassen(A_idx, (6, 7, 14, 15), W_idx_3)
    
    # Block 2: A[0:2, 2] * B[2, :]
    # A indices: A02: 0*3+2=2, A12: 1*3+2=5
    # B indices: B2k: 2*8+k = 16+k
    # W indices: C0k: k*3+0, C1k: k*3+1
    
    for k in range(8):
        # Term for C[0, k]
        U[2, col] += 1
        V[16+k, col] += 1
        W[k*3+0, col] += 1
        col += 1
        
        # Term for C[1, k]
        U[5, col] += 1
        V[16+k, col] += 1
        W[k*3+1, col] += 1
        col += 1
        
    # Block 3: A[2, 0:2] * B[0:2, :]
    # A indices: A20: 2*3+0=6, A21: 2*3+1=7
    # B indices: B0k: k, B1k: 8+k
    # W indices: C2k: k*3+2
    
    for k in range(8):
        # Term for A[2, 0] * B[0, k]
        U[6, col] += 1
        V[k, col] += 1
        W[k*3+2, col] += 1
        col += 1
        
        # Term for A[2, 1] * B[1, k]
        U[7, col] += 1
        V[8+k, col] += 1
        W[k*3+2, col] += 1
        col += 1
        
    # Block 4: A[2, 2] * B[2, :]
    # A index: A22: 2*3+2=8
    # B index: B2k: 16+k
    # W index: C2k: k*3+2
    
    for k in range(8):
        U[8, col] += 1
        V[16+k, col] += 1
        W[k*3+2, col] += 1
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
