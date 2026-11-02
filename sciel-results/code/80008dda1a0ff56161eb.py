import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = 3, 4, 7
    
    # We attempt a slice-and-dice construction.
    # The matrix multiplication tensor T_{i,j,k} for C = AB (3x4 * 4x7)
    # is decomposed into independent sub-tensors.
    # Since the middle dimension m=4 is even, we can split the sum over j into two halves:
    # C_{i,k} = sum_{j=0}^{1} A_{i,j} B_{j,k} + sum_{j=2}^{3} A_{i,j} B_{j,k}
    # 
    # Let's define two sub-problems:
    # 1. Multiplying a 3x2 matrix by a 2x7 matrix. (Indices i in 0..2, j in 0..1, k in 0..6)
    # 2. Multiplying a 3x2 matrix by a 2x7 matrix. (Indices i in 0..2, j in 2..3, k in 0..6)
    #
    # Wait, the "slice" is over the contraction dimension j.
    # The tensor is sum_j A_{i,j} B_{j,k}.
    # We can view this as the sum of two tensors:
    # T1_{i,j,k} = A_{i,j} B_{j,k} for j in {0,1}
    # T2_{i,j,k} = A_{i,j} B_{j,k} for j in {2,3}
    #
    # Each sub-tensor corresponds to matrix multiplication of size (3, 2, 7).
    # Let's find the minimal rank for (3, 2, 7) multiplication.
    # Schoolbook rank for (3, 2, 7) is 3*2*7 = 42.
    # Is there a better rank?
    # For (2, 2, 2), Strassen gives rank 7.
    # For (3, 2, 2), we can slice the 3 rows into 2+1?
    #   - 2x2x2 part: rank 7
    #   - 1x2x2 part: rank 2 (since C_{i,k} = sum_{j=0}^1 A_{i,j}B_{j,k} is just 2 terms, rank 2)
    #   Total rank for (3,2,2) = 7 + 2 = 9?
    #   Let's verify.
    #   The tensor for (3,2,2) has indices i in {0,1,2}, j in {0,1}, k in {0,1}.
    #   We can separate i=2 from i in {0,1}.
    #   For i in {0,1}, we have a 2x2x2 tensor. Rank 7 (Strassen).
    #   For i=2, we have a 1x2x2 tensor. This is just A_{2,0}B_{0,k} + A_{2,1}B_{1,k}.
    #   This requires rank 2 (two rank-1 terms: A_{2,0}B_{0,0}, A_{2,0}B_{0,1} etc? No.)
    #   The tensor T_{2,j,k} = A_{2,j} B_{j,k}.
    #   The terms are A_{2,0}B_{0,0}, A_{2,0}B_{0,1}, A_{2,1}B_{1,0}, A_{2,1}B_{1,1}.
    #   Can we do this in rank 2?
    #   U_{2, r} V_{j, r} W_{k, r}.
    #   If we use r=0: U_{2,0}=1, V_{0,0}=1, W_{0,0}=1, W_{1,0}=0. Term: A_{2,0}B_{0,0}.
    #   If we use r=1: U_{2,1}=1, V_{1,1}=1, W_{0,1}=0, W_{1,1}=1. Term: A_{2,1}B_{1,1}.
    #   We are missing A_{2,0}B_{0,1} and A_{2,1}B_{1,0}.
    #   So rank 2 is not enough for 1x2x2?
    #   Actually, the rank of 1x2x2 is 2?
    #   Let's check. The tensor is a 1x2x2 array.
    #   T_{0,0,0}=1, T_{0,0,1}=1, T_{0,1,0}=1, T_{0,1,1}=1.
    #   Wait, the indices are (i,j,k) but i is fixed at 0.
    #   The matrix of the tensor (flattened) is:
    #   j=0: [1, 1]
    #   j=1: [1, 1]
    #   This matrix has rank 1. So the tensor rank is 1?
    #   No, tensor rank is not matrix rank.
    #   T_{0, j, k} = delta_{j,0} * 1 + delta_{j,1} * 1? No.
    #   T_{0, j, k} = A_{0,j} B_{j,k}.
    #   The tensor entries are 1 if the indices match the multiplication pattern.
    #   For 1x2x2 multiplication:
    #   C_0 = A_{0,0}B_{0,0} + A_{0,1}B_{1,0}
    #   C_1 = A_{0,0}B_{0,1} + A_{0,1}B_{1,1}
    #   The tensor T_{i,j,k} is 1 if (i,j,k) is a valid term.
    #   Valid terms: (0,0,0), (0,1,0), (0,0,1), (0,1,1).
    #   So T is a 1x2x2 tensor with 1s at (0,0,0), (0,1,0), (0,0,1), (0,1,1).
    #   This is the outer product of vector [1,1] (for j) and [1,1] (for k)?
    #   T_{0,j,k} = 1 * 1 = 1 for all j,k.
    #   So T = u v^T w^T?
    #   T_{0,j,k} = U_{0,r} V_{j,r} W_{k,r}.
    #   If we set U_{0,0}=1, V_{j,0}=1, W_{k,0}=1, then T_{0,j,k}=1 for all j,k.
    #   This matches the tensor!
    #   So the rank of 1x2x2 is 1.
    #
    #   Let's re-evaluate (3,2,2).
    #   Split i into {0,1} and {2}.
    #   Part 1 (i in {0,1}): 2x2x2 tensor. Rank 7 (Strassen).
    #   Part 2 (i=2): 1x2x2 tensor. Rank 1.
    #   Total rank for (3,2,2) = 7 + 1 = 8.
    #
    #   Now, can we do better for (3,2,7)?
    #   We can slice the k dimension (7) into 2+2+2+1.
    #   (3,2,2) + (3,2,2) + (3,2,2) + (3,2,1)
    #   Rank of (3,2,2) is 8.
    #   Rank of (3,2,1)?
    #   Split i into {0,1} and {2}.
    #   Part 1 (i in {0,1}): 2x2x1 tensor.
    #   Valid terms: (0,0,0), (0,1,0), (1,0,0), (1,1,0).
    #   T_{i,j,0} = 1 for all i,j.
    #   This is outer product of [1,1] (i), [1,1] (j), [1] (k).
    #   Rank 1.
    #   Part 2 (i=2): 1x2x1 tensor.
    #   Valid terms: (2,0,0), (2,1,0).
    #   T_{2,j,0} = 1 for all j.
    #   Outer product of [1] (i), [1,1] (j), [1] (k).
    #   Rank 1.
    #   Total rank for (3,2,1) = 1 + 1 = 2.
    #
    #   So rank of (3,2,7) = 3 * rank(3,2,2) + rank(3,2,1) = 3*8 + 2 = 26.
    #
    #   Total rank for (3,4,7) = 2 * rank(3,2,7) = 2 * 26 = 52.
    #
    #   This is better than 84.
    #
    #   Let's verify the Strassen decomposition for 2x2x2.
    #   The 7 rank-1 terms for 2x2x2 are:
    #   M1 = (A11 - A12)(B21 + B22)
    #   M2 = (A11 + A12)B22
    #   M3 = A11(B11 - B12)
    #   M4 = A22(B22 - B11)
    #   M5 = (A12 + A22)(B11 + B12)
    #   M6 = (A11 - A21)(B12 + B22)
    #   M7 = (A21 + A11)(B21 - B11)
    #
    #   We need to map these to our U, V, W matrices.
    #   The tensor T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    #   Wait, the indices in the problem description are:
    #   U has shape (n*m, rank). Index: i*m + j.
    #   V has shape (m*p, rank). Index: j*p + k.
    #   W has shape (p*n, rank). Index: k*n + i.
    #
    #   Let's construct the 2x2x2 decomposition explicitly.
    #   i in {0,1}, j in {0,1}, k in {0,1}.
    #   U indices: 0*m, 1*m, 0*m+1, 1*m+1. (Assuming m=2 for this sub-problem, but in global m=4)
    #   We need to be careful with index mapping.
    
    # Let's implement the (3,2,2) decomposition first.
    # Indices: i in {0,1,2}, j in {0,1}, k in {0,1}.
    # We use 8 ranks.
    
    def get_3x2x2_decomp(offset_u, offset_v, offset_w):
        """
        Returns U, V, W contributions for 3x2x2 multiplication.
        offset_u: base index in U (i*m + j)
        offset_v: base index in V (j*p + k)
        offset_w: base index in W (k*n + i)
        """
        U = np.zeros((3, 2, 8)) # i, j, r
        V = np.zeros((2, 2, 8)) # j, k, r
        W = np.zeros((2, 3, 8)) # k, i, r
        
        # Strassen for i in {0,1}, j in {0,1}, k in {0,1}
        # M1
        U[0, 0, 0] = 1; U[0, 1, 0] = -1
        V[0, 0, 0] = 0; V[0, 1, 0] = 0
        V[1, 0, 0] = 1; V[1, 1, 0] = 1
        W[0, 0, 0] = 1; W[1, 0, 0] = 1
        # M2
        U[0, 0, 1] = 1; U[0, 1, 1] = 1
        V[0, 0, 1] = 0; V[0, 1, 1] = 0
        V[1, 0, 1] = 0; V[1, 1, 1] = 1
        W[0, 0, 1] = 1; W[1, 0, 1] = 0
        # M3
        U[0, 0, 2] = 1
        V[0, 0, 2] = 1; V[0, 1, 2] = -1
        W[0, 0, 2] = 1; W[1, 0, 2] = 0
        # M4
        U[1, 1, 3] = 1
        V[0, 0, 3] = -1; V[0, 1, 3] = 0
        V[1, 0, 3] = 0; V[1, 1, 3] = 1
        W[0, 1, 3] = 1; W[1, 1, 3] = 0
        # M5
        U[0, 1, 4] = 1; U[1, 1, 4] = 1
        V[0, 0, 4] = 1; V[0, 1, 4] = 1
        W[0, 0, 4] = 1; W[1, 1, 4] = 1
        # M6
        U[0, 0, 5] = 1; U[1, 0, 5] = -1
        V[0, 0, 5] = 0; V[0, 1, 5] = 1
        V[1, 0, 5] = 0; V[1, 1, 5] = 1
        W[0, 0, 5] = 1; W[1, 1, 5] = 1
        # M7
        U[0, 0, 6] = 1; U[1, 0, 6] = 1
        V[0, 0, 6] = 1; V[0, 1, 6] = 0
        V[1, 0, 6] = -1; V[1, 1, 6] = 0
        W[0, 0, 6] = 1; W[1, 1, 6] = -1
        
        # Rank 1 for i=2, j in {0,1}, k in {0,1}
        # T_{2,j,k} = 1 for all j,k
        U[2, 0, 7] = 1; U[2, 1, 7] = 1
        V[0, 0, 7] = 1; V[0, 1, 7] = 1
        V[1, 0, 7] = 1; V[1, 1, 7] = 1
        W[0, 2, 7] = 1; W[1, 2, 7] = 1
        
        return U, V, W

    def get_3x2x1_decomp(offset_u, offset_v, offset_w):
        """
        Returns U, V, W contributions for 3x2x1 multiplication.
        """
        U = np.zeros((3, 2, 2))
        V = np.zeros((2, 1, 2))
        W = np.zeros((1, 3, 2))
        
        # Rank 1 for i in {0,1}
        U[0, 0, 0] = 1; U[0, 1, 0] = 1
        U[1, 0, 0] = 1; U[1, 1, 0] = 1
        V[0, 0, 0] = 1; V[1, 0, 0] = 1
        W[0, 0, 0] = 1; W[0, 1, 0] = 1
        
        # Rank 1 for i=2
        U[2, 0, 1] = 1; U[2, 1, 1] = 1
        V[0, 0, 1] = 1; V[1, 0, 1] = 1
        W[0, 2, 1] = 1
        
        return U, V, W

    # Global Dimensions
    n, m, p = 3, 4, 7
    rank = 52
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # We split j into {0,1} and {2,3}
    # For each half, we split k into {0,1}, {2,3}, {4,5}, {6}
    
    # Half 1: j in {0,1}
    # k block 0: k in {0,1} -> 3x2x2 (Rank 8)
    # k block 1: k in {2,3} -> 3x2x2 (Rank 8)
    # k block 2: k in {4,5} -> 3x2x2 (Rank 8)
    # k block 3: k in {6}   -> 3x2x1 (Rank 2)
    
    # Half 2: j in {2,3}
    # k block 0: k in {0,1} -> 3x2x2 (Rank 8)
    # k block 1: k in {2,3} -> 3x2x2 (Rank 8)
    # k block 2: k in {4,5} -> 3x2x2 (Rank 8)
    # k block 3: k in {6}   -> 3x2x1 (Rank 2)
    
    current_rank = 0
    
    for j_start in [0, 2]:
        for k_start in [0, 2, 4, 6]:
            if k_start == 6:
                # 3x2x1
                u_sub, v_sub, w_sub = get_3x2x1_decomp(0, 0, 0)
                # Map indices
                for i in range(3):
                    for j in range(2):
                        U[i * m + (j_start + j), current_rank:current_rank + 2] = u_sub[i, j, :]
                for j in range(2):
                    for k in range(1):
                        V[(j_start + j) * p + (k_start + k), current_rank:current_rank + 2] = v_sub[j, k, :]
                for k in range(1):
                    for i in range(3):
                        W[(k_start + k) * n + i, current_rank:current_rank + 2] = w_sub[k, i, :]
                current_rank += 2
            else:
                # 3x2x2
                u_sub, v_sub, w_sub = get_3x2x2_decomp(0, 0, 0)
                for i in range(3):
                    for j in range(2):
                        U[i * m + (j_start + j), current_rank:current_rank + 8] = u_sub[i, j, :]
                for j in range(2):
                    for k in range(2):
                        V[(j_start + j) * p + (k_start + k), current_rank:current_rank + 8] = v_sub[j, k, :]
                for k in range(2):
                    for i in range(3):
                        W[(k_start + k) * n + i, current_rank:current_rank + 8] = w_sub[k, i, :]
                current_rank += 8
    
    # Verify
    if time.time() > deadline:
        pass # Should not happen, but just in case
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    n, m, p = 3, 4, 7
    rank = U.shape[1]
    
    # Check correctness
    # T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    # Should be 1 if (i,j,k) is a valid term, 0 otherwise.
    # Valid terms: all i,j,k.
    # Wait, the tensor for matrix multiplication is T_{i,j,k} = 1 for all i,j,k?
    # No. C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So for each (i,j,k), there is a term.
    # The tensor is T_{i,j,k} = 1 for all i,j,k in the ranges.
    # Let's check if our decomposition gives 1 for all i,j,k.
    
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-9:
                    errors += 1
                    print(f"Error at {i},{j},{k}: {val}")
    print(f"Errors: {errors}")
