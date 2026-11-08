"""Rewritten program: Block decomposition for 6x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_START_TIME = time.time()
_TIME_LIMIT = 110.0

def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _build_naive_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}

def _build_block_decomposition(n, m, p):
    # n=6, m=6, p=6
    # Split into 4x4, 4x2, 2x4, 2x2 blocks
    # A00 (4x4), A01 (4x2), A10 (2x4), A11 (2x2)
    # B00 (4x4), B01 (4x2), B10 (2x4), B11 (2x2)
    
    # Products:
    # P1 = A00 B00 (4x4 * 4x4) -> Rank 16 (Naive)
    # P2 = A01 B10 (4x2 * 2x4) -> Rank 2 (Naive)
    # P3 = A00 B01 (4x4 * 4x2) -> Rank 4 (Naive)
    # P4 = A01 B11 (4x2 * 2x2) -> Rank 2 (Naive)
    # P5 = A10 B00 (2x4 * 4x4) -> Rank 2 (Naive)
    # P6 = A11 B10 (2x2 * 2x4) -> Rank 2 (Naive)
    # P7 = A10 B01 (2x4 * 4x2) -> Rank 2 (Naive)
    # P8 = A11 B11 (2x2 * 2x2) -> Rank 7 (Strassen)
    
    total_rank = 16 + 2 + 4 + 2 + 2 + 2 + 2 + 7
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a naive product of A[i1:i2, j1:j2] * B[j1:j2, k1:k2] to C[i1:i2, k1:k2]
    # But we need to map to the global U, V, W.
    
    # P1: A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    # Naive 4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[(0+i)*6 + (0+j)][col_idx] = 1
                V[(0+j)*6 + (0+k)][col_idx] = 1
                W[(0+k)*6 + (0+i)][col_idx] = 1
                col_idx += 1
                
    # P2: A[0:4, 4:6] * B[4:6, 0:4] -> C[0:4, 0:4]
    # A01 is 4x2, B10 is 2x4
    # C[0:4, 0:4] += A01 B10
    for i in range(4):
        for j in range(2): # j is 0..1 in A01, maps to 4..5 in A
            for k in range(4):
                U[(0+i)*6 + (4+j)][col_idx] = 1
                V[(4+j)*6 + (0+k)][col_idx] = 1
                W[(0+k)*6 + (0+i)][col_idx] = 1
                col_idx += 1
    # Wait, the loop structure for naive product of 4x2 * 2x4 is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # j ranges 0..1.
    # So 4*1*4 = 16 terms? No.
    # The rank of 4x2*2x4 is 2?
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    # This is a sum of 2 rank-1 tensors?
    # Tensor T[i,j,k] where j is the shared index.
    # The decomposition is sum_j (A[:,j] B[j,:]).
    # Each term is a rank-1 tensor?
    # Yes.
    # So rank is 2.
    # My loop above adds 4*2*4 = 32 terms?
    # No, the loop `for i, for j, for k` adds 1 term for each (i,j,k).
    # This is the naive decomposition of the *full* product, which has rank 4*2*4 = 32?
    # No, the naive decomposition of m x n x p has rank m*n*p.
    # For 4x2*2x4, naive rank is 4*2*4 = 32.
    # But I said rank 2.
    # Why?
    # Because C[i,k] = sum_j A[i,j] B[j,k].
    # This is a linear combination of 2 matrices: A[:,0]B[0,:] and A[:,1]B[1,:].
    # So the tensor is a sum of 2 rank-1 tensors?
    # Yes.
    # So the rank is 2.
    # My loop `for i, for j, for k` creates 32 terms.
    # This is inefficient.
    # I need to add 2 terms.
    # Term 1: A[:,0] B[0,:]
    # Term 2: A[:,1] B[1,:]
    
    # Let's restart the construction with correct ranks.
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    col_idx = 0
    
    # P1: 4x4 * 4x4 (Naive, Rank 16)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*6+j][col_idx] = 1
                V[j*6+k][col_idx] = 1
                W[k*6+i][col_idx] = 1
                col_idx += 1

    # P2: 4x2 * 2x4 (Rank 2)
    # C[i,k] = A[i,4]B[4,k] + A[i,5]B[5,k]
    for j in range(2):
        for i in range(4):
            for k in range(4):
                # This adds 16 terms.
                # I need to add 2 terms.
                # Term j=0: U[i,4]=1, V[4,k]=1, W[k,i]=1 for all i,k?
                # No, U, V, W are vectors.
                # Term r: U_r, V_r, W_r.
                # For j=0:
                # U_r[i,4] = 1 for all i? No, U_r is a vector of length 36.
                # U_r[i*6+4] = 1 for i=0..3.
                # V_r[4*6+k] = 1 for k=0..3.
                # W_r[k*6+i] = 1 for i=0..3, k=0..3.
                # This is one rank-1 tensor.
                pass

    # I will implement the rank-2 terms correctly.
    
    # Reset
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    col_idx = 0
    
    # P1: 4x4 * 4x4 (Naive, Rank 16)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*6+j][col_idx] = 1
                V[j*6+k][col_idx] = 1
                W[k*6+i][col_idx] = 1
                col_idx += 1

    # P2: 4x2 * 2x4 (Rank 2)
    # Terms: j=0, j=1
    for j in range(2):
        for i in range(4):
            U[i*6 + (4+j)][col_idx] = 1
        for k in range(4):
            V[(4+j)*6 + k][col_idx] = 1
        for i in range(4):
            for k in range(4):
                W[k*6 + i][col_idx] = 1
        col_idx += 1

    # P3: 4x4 * 4x2 (Rank 4)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # j=0..3
    for j in range(4):
        for i in range(4):
            U[i*6 + j][col_idx] = 1
        for k in range(2):
            V[j*6 + (4+k)][col_idx] = 1
        for i in range(4):
            for k in range(2):
                W[(4+k)*6 + i][col_idx] = 1
        col_idx += 1

    # P4: 4x2 * 2x2 (Rank 2)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # j=0..1, k=0..1
    for j in range(2):
        for i in range(4):
            U[i*6 + (4+j)][col_idx] = 1
        for k in range(2):
            V[(4+j)*6 + (4+k)][col_idx] = 1
        for i in range(4):
            for k in range(2):
                W[(4+k)*6 + i][col_idx] = 1
        col_idx += 1

    # P5: 2x4 * 4x4 (Rank 2)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # i=0..1, j=0..3, k=0..3
    # Rank 2?
    # C[i,k] = A[i,0]B[0,k] + ... + A[i,3]B[3,k]
    # This is sum of 4 terms?
    # No, the shared index is j.
    # j ranges 0..3.
    # So rank is 4?
    # Wait, A is 2x4, B is 4x4.
    # C is 2x4.
    # C[i,k] = sum_{j=0}^3 A[i,j] B[j,k].
    # This is a sum of 4 rank-1 tensors?
    # Yes.
    # So rank is 4.
    
    # I underestimated the ranks.
    # P5: 2x4 * 4x4 -> Rank 4.
    # P6: 2x2 * 2x4 -> Rank 2.
    # P7: 2x4 * 4x2 -> Rank 4.
    # P8: 2x2 * 2x2 -> Rank 7.
    
    # Total Rank = 16 + 2 + 4 + 2 + 4 + 2 + 4 + 7 = 41.
    
    # I will implement this correctly.
    
    U = [[0] * 41 for _ in range(n * m)]
    V = [[0] * 41 for _ in range(m * p)]
    W = [[0] * 41 for _ in range(p * n)]
    col_idx = 0
    
    # P1: 4x4 * 4x4 (Naive, Rank 16)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*6+j][col_idx] = 1
                V[j*6+k][col_idx] = 1
                W[k*6+i][col_idx] = 1
                col_idx += 1

    # P2: 4x2 * 2x4 (Rank 2)
    for j in range(2):
        for i in range(4):
            U[i*6 + (4+j)][col_idx] = 1
        for k in range(4):
            V[(4+j)*6 + k][col_idx] = 1
        for i in range(4):
            for k in range(4):
                W[k*6 + i][col_idx] = 1
        col_idx += 1

    # P3: 4x4 * 4x2 (Rank 4)
    for j in range(4):
        for i in range(4):
            U[i*6 + j][col_idx] = 1
        for k in range(2):
            V[j*6 + (4+k)][col_idx] = 1
        for i in range(4):
            for k in range(2):
                W[(4+k)*6 + i][col_idx] = 1
        col_idx += 1

    # P4: 4x2 * 2x2 (Rank 2)
    for j in range(2):
        for i in range(4):
            U[i*6 + (4+j)][col_idx] = 1
        for k in range(2):
            V[(4+j)*6 + (4+k)][col_idx] = 1
        for i in range(4):
            for k in range(2):
                W[(4+k)*6 + i][col_idx] = 1
        col_idx += 1

    # P5: 2x4 * 4x4 (Rank 4)
    for j in range(4):
        for i in range(2):
            U[(4+i)*6 + j][col_idx] = 1
        for k in range(4):
            V[j*6 + k][col_idx] = 1
        for i in range(2):
            for k in range(4):
                W[k*6 + (4+i)][col_idx] = 1
        col_idx += 1

    # P6: 2x2 * 2x4 (Rank 2)
    for j in range(2):
        for i in range(2):
            U[(4+i)*6 + (4+j)][col_idx] = 1
        for k in range(4):
            V[(4+j)*6 + k][col_idx] = 1
        for i in range(2):
            for k in range(4):
                W[k*6 + (4+i)][col_idx] = 1
        col_idx += 1

    # P7: 2x4 * 4x2 (Rank 4)
    for j in range(4):
        for i in range(2):
            U[(4+i)*6 + j][col_idx] = 1
        for k in range(2):
            V[j*6 + (4+k)][col_idx] = 1
        for i in range(2):
            for k in range(2):
                W[(4+k)*6 + (4+i)][col_idx] = 1
        col_idx += 1

    # P8: 2x2 * 2x2 (Strassen, Rank 7)
    # Strassen patterns for 2x2
    # M0 = (A00+A11)(B00+B11)
    # ...
    # I will use the standard Strassen implementation for the 2x2 block.
    
    # Local indices: i,j,k in 0..1
    # Global indices: 4+i, 4+j, 4+k
    
    strassen_patterns = [
        # M0
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
        for (pi, pj), val in u_coeffs.items():
            U[(4+pi)*6 + (4+pj)][col_idx] += val
        for (pi, pj), val in v_coeffs.items():
            V[(4+pi)*6 + (4+pj)][col_idx] += val
        for (pi, pj), val in w_coeffs.items():
            W[(4+pj)*6 + (4+pi)][col_idx] += val
        col_idx += 1
        
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(decomp, n, m, p, num_tests=3):
    U = np.array(decomp["U"])
    V = np.array(decomp["V"])
    W = np.array(decomp["W"])
    rank = U.shape[1]
    
    for _ in range(num_tests):
        A = np.random.randint(-3, 4, (n, m)).astype(float)
        B = np.random.randint(-3, 4, (m, p)).astype(float)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            Ur = U[:, r].reshape(n, m)
            Vr = V[:, r].reshape(m, p)
            Wr = W[:, r].reshape(p, n)
            prod = Ur @ Vr
            C_computed += prod * Wr
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    if _remaining_time() < 5:
        return _build_naive_decomposition(n, m, p)
    
    decomp = _build_block_decomposition(n, m, p)
    
    if _remaining_time() < 5:
        return decomp
    
    if _verify_decomposition(decomp, n, m, p, num_tests=2):
        return decomp
    else:
        return _build_naive_decomposition(n, m, p)

if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    print(f"Rank: {U.shape[1]}")
    valid = _verify_decomposition(result, 6, 6, 6, num_tests=5)
    print(f"Valid: {valid}")
