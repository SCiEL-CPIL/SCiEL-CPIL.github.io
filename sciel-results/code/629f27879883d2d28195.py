import numpy as np
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 2, 3, 4
    rank = 20
    
    U = [[0] * rank for _ in range(n * m)] # 6 x 20
    V = [[0] * rank for _ in range(m * p)] # 12 x 20
    W = [[0] * rank for _ in range(p * n)] # 8 x 20
    
    # Define the rank-10 scheme for 2x3x2
    # Local indices:
    # A: i in {0,1}, j in {0,1,2}
    # B: j in {0,1,2}, k in {0,1}
    # C: k in {0,1}, i in {0,1}
    
    # The 10 terms for one block:
    # t: U_indices (relative to block start in A), V_indices (relative to block start in B), W_indices (relative to block start in C), coeff
    # Note: U index is i*m + j. V index is j*p + k. W index is k*n + i.
    # In the local 2x3x2 context:
    # U local: i*3 + j
    # V local: j*2 + k
    # W local: k*2 + i
    
    # Term definitions: (U_coeffs, V_coeffs, W_coeffs)
    # Each term is a sum of products.
    # We will add contributions to U, V, W for each of the 10 terms.
    
    def add_block(k_offset):
        # k_offset is 0 for first block (cols 0,1 of B), 2 for second block (cols 2,3 of B)
        # The C matrix columns also shift by k_offset? No, C is 2x4.
        # C[k, i] corresponds to W[k*n + i].
        # In the local 2x3x2, C has 2 columns.
        # Local C col 0 corresponds to global C col k_offset.
        # Local C col 1 corresponds to global C col k_offset + 1.
        
        # So W index for local (k_loc, i) is (k_offset + k_loc)*n + i.
        
        # The 10 terms:
        # 1. a11 * b11
        # 2. a12 * b21
        # 3. a13 * b31
        # 4. a21 * b11
        # 5. a22 * b21
        # 6. a23 * b31
        # 7. (a11+a12+a13) * (b12+b22+b32)
        # 8. (a21+a22+a23) * (b12+b22+b32)
        # 9. (a11+a12) * (b12+b22)
        # 10. (a21+a22) * (b12+b22)
        
        # Let's map these to indices.
        # A indices (U):
        # a11 -> 0*3+0 = 0
        # a12 -> 0*3+1 = 1
        # a13 -> 0*3+2 = 2
        # a21 -> 1*3+0 = 3
        # a22 -> 1*3+1 = 4
        # a23 -> 1*3+2 = 5
        
        # B indices (V) for local k=0,1:
        # b11 (j=0, k=0) -> 0*2+0 = 0
        # b21 (j=1, k=0) -> 1*2+0 = 2
        # b31 (j=2, k=0) -> 2*2+0 = 4
        # b12 (j=0, k=1) -> 0*2+1 = 1
        # b22 (j=1, k=1) -> 1*2+1 = 3
        # b32 (j=2, k=1) -> 2*2+1 = 5
        
        # C indices (W) for local k=0,1:
        # C11 (k=0, i=0) -> 0*2+0 = 0
        # C21 (k=0, i=1) -> 0*2+1 = 1
        # C12 (k=1, i=0) -> 1*2+0 = 2
        # C22 (k=1, i=1) -> 1*2+1 = 3
        
        # Global V index = j*p + (k_offset + k_loc)
        # Global W index = (k_offset + k_loc)*n + i
        
        # Term 1: a11 * b11
        # U[0] += 1
        # V[0*p + k_offset] += 1
        # W[(k_offset+0)*n + 0] += 1
        U[0][0] += 1
        V[0][0] += 1
        W[k_offset * n + 0][0] += 1
        
        # Term 2: a12 * b21
        # U[1] += 1
        # V[1*p + k_offset] += 1
        # W[(k_offset+0)*n + 0] += 1
        U[1][1] += 1
        V[1 * p + k_offset][1] += 1
        W[k_offset * n + 0][1] += 1
        
        # Term 3: a13 * b31
        # U[2] += 1
        # V[2*p + k_offset] += 1
        # W[(k_offset+0)*n + 0] += 1
        U[2][2] += 1
        V[2 * p + k_offset][2] += 1
        W[k_offset * n + 0][2] += 1
        
        # Term 4: a21 * b11
        # U[3] += 1
        # V[0*p + k_offset] += 1
        # W[(k_offset+0)*n + 1] += 1
        U[3][3] += 1
        V[0][3] += 1
        W[k_offset * n + 1][3] += 1
        
        # Term 5: a22 * b21
        # U[4] += 1
        # V[1*p + k_offset] += 1
        # W[(k_offset+0)*n + 1] += 1
        U[4][4] += 1
        V[1 * p + k_offset][4] += 1
        W[k_offset * n + 1][4] += 1
        
        # Term 6: a23 * b31
        # U[5] += 1
        # V[2*p + k_offset] += 1
        # W[(k_offset+0)*n + 1] += 1
        U[5][5] += 1
        V[2 * p + k_offset][5] += 1
        W[k_offset * n + 1][5] += 1
        
        # Term 7: (a11+a12+a13) * (b12+b22+b32)
        # U[0]+=1, U[1]+=1, U[2]+=1
        # V[0*p + (k_offset+1)] += 1
        # V[1*p + (k_offset+1)] += 1
        # V[2*p + (k_offset+1)] += 1
        # W[(k_offset+1)*n + 0] += 1
        U[0][6] += 1
        U[1][6] += 1
        U[2][6] += 1
        V[0 * p + k_offset + 1][6] += 1
        V[1 * p + k_offset + 1][6] += 1
        V[2 * p + k_offset + 1][6] += 1
        W[(k_offset + 1) * n + 0][6] += 1
        
        # Term 8: (a21+a22+a23) * (b12+b22+b32)
        # U[3]+=1, U[4]+=1, U[5]+=1
        # V[0*p + (k_offset+1)] += 1
        # V[1*p + (k_offset+1)] += 1
        # V[2*p + (k_offset+1)] += 1
        # W[(k_offset+1)*n + 1] += 1
        U[3][7] += 1
        U[4][7] += 1
        U[5][7] += 1
        V[0 * p + k_offset + 1][7] += 1
        V[1 * p + k_offset + 1][7] += 1
        V[2 * p + k_offset + 1][7] += 1
        W[(k_offset + 1) * n + 1][7] += 1
        
        # Term 9: (a11+a12) * (b12+b22)
        # U[0]+=1, U[1]+=1
        # V[0*p + (k_offset+1)] += 1
        # V[1*p + (k_offset+1)] += 1
        # W[(k_offset+1)*n + 0] -= 1
        U[0][8] += 1
        U[1][8] += 1
        V[0 * p + k_offset + 1][8] += 1
        V[1 * p + k_offset + 1][8] += 1
        W[(k_offset + 1) * n + 0][8] -= 1
        
        # Term 10: (a21+a22) * (b12+b22)
        # U[3]+=1, U[4]+=1
        # V[0*p + (k_offset+1)] += 1
        # V[1*p + (k_offset+1)] += 1
        # W[(k_offset+1)*n + 1] -= 1
        U[3][9] += 1
        U[4][9] += 1
        V[0 * p + k_offset + 1][9] += 1
        V[1 * p + k_offset + 1][9] += 1
        W[(k_offset + 1) * n + 1][9] -= 1
        
    # Add first block (k_offset = 0, uses terms 0-9)
    # We need to shift the term indices for the second block.
    # The add_block function above hardcoded term indices 0-9.
    # We need to modify it to accept a base index or create two separate calls with different indices.
    # Let's refactor add_block to take a base index.
    
    # Reset matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_block_v2(k_offset, base_idx):
        # Term 1
        U[0][base_idx + 0] += 1
        V[0 * p + k_offset][base_idx + 0] += 1
        W[k_offset * n + 0][base_idx + 0] += 1
        
        # Term 2
        U[1][base_idx + 1] += 1
        V[1 * p + k_offset][base_idx + 1] += 1
        W[k_offset * n + 0][base_idx + 1] += 1
        
        # Term 3
        U[2][base_idx + 2] += 1
        V[2 * p + k_offset][base_idx + 2] += 1
        W[k_offset * n + 0][base_idx + 2] += 1
        
        # Term 4
        U[3][base_idx + 3] += 1
        V[0 * p + k_offset][base_idx + 3] += 1
        W[k_offset * n + 1][base_idx + 3] += 1
        
        # Term 5
        U[4][base_idx + 4] += 1
        V[1 * p + k_offset][base_idx + 4] += 1
        W[k_offset * n + 1][base_idx + 4] += 1
        
        # Term 6
        U[5][base_idx + 5] += 1
        V[2 * p + k_offset][base_idx + 5] += 1
        W[k_offset * n + 1][base_idx + 5] += 1
        
        # Term 7
        U[0][base_idx + 6] += 1
        U[1][base_idx + 6] += 1
        U[2][base_idx + 6] += 1
        V[0 * p + k_offset + 1][base_idx + 6] += 1
        V[1 * p + k_offset + 1][base_idx + 6] += 1
        V[2 * p + k_offset + 1][base_idx + 6] += 1
        W[(k_offset + 1) * n + 0][base_idx + 6] += 1
        
        # Term 8
        U[3][base_idx + 7] += 1
        U[4][base_idx + 7] += 1
        U[5][base_idx + 7] += 1
        V[0 * p + k_offset + 1][base_idx + 7] += 1
        V[1 * p + k_offset + 1][base_idx + 7] += 1
        V[2 * p + k_offset + 1][base_idx + 7] += 1
        W[(k_offset + 1) * n + 1][base_idx + 7] += 1
        
        # Term 9
        U[0][base_idx + 8] += 1
        U[1][base_idx + 8] += 1
        V[0 * p + k_offset + 1][base_idx + 8] += 1
        V[1 * p + k_offset + 1][base_idx + 8] += 1
        W[(k_offset + 1) * n + 0][base_idx + 8] -= 1
        
        # Term 10
        U[3][base_idx + 9] += 1
        U[4][base_idx + 9] += 1
        V[0 * p + k_offset + 1][base_idx + 9] += 1
        V[1 * p + k_offset + 1][base_idx + 9] += 1
        W[(k_offset + 1) * n + 1][base_idx + 9] -= 1

    # First block: k_offset = 0, base_idx = 0
    add_block_v2(0, 0)
    
    # Second block: k_offset = 2, base_idx = 10
    add_block_v2(2, 10)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    # Quick verification
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Random test
    A = np.random.rand(2, 3)
    B = np.random.rand(3, 4)
    C_expected = A @ B
    
    C_calc = np.zeros((2, 4))
    for r in range(20):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        prod = np.dot(u, A.ravel()) * np.dot(v, B.ravel())
        C_calc += prod * w.reshape(4, 2).T # W is k*n+i, so reshape to (4,2) then transpose to (2,4)
        
    # Wait, W index is k*n+i.
    # C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # My loop above:
    # u = U[:, r] is length 6. A.ravel() is length 6. Dot product is sum_{i,j} U[i,j] A[i,j]. Correct.
    # v = V[:, r] is length 12. B.ravel() is length 12. Dot product is sum_{j,k} V[j,k] B[j,k]. Correct.
    # w = W[:, r] is length 8.
    # C_calc[i, k] should add prod * W[k*n+i, r].
    # My code: C_calc += prod * w.reshape(4, 2).T
    # w.reshape(4, 2) has w[k*n+i] at index [k, i].
    # .T has index [i, k] with value w[k*n+i].
    # So C_calc[i, k] += prod * w[k*n+i]. Correct.
    
    error = np.max(np.abs(C_expected - C_calc))
    print(f"Max error: {error}")
    if error > 1e-9:
        print("Verification Failed")
    else:
        print("Verification Passed")
