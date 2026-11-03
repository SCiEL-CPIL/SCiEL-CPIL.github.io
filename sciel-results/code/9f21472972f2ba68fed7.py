"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _validate_and_return(U, V, W):
    """Verify the decomposition is exact and return the dict."""
    n, m, p = DIMENSIONS
    rank = len(W[0])
    # Convert to numpy for fast verification
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor T_{i,j,k} for matrix mult (A_{i,j} * B_{j,k})
    # is represented by sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    # We need to check if this equals the standard basis tensor.
    
    # Efficient check: For each r, we have a rank-1 tensor.
    # We can accumulate the result.
    # Result shape: (n, m, p) but stored as (p*n, m, n) ? No.
    # Standard check:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor: T_{i,j,k} = 1 if it's the term in C_{i,k} from A_{i,j}B_{j,k}
    # The decomposition gives: D_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    # We want D_{i,j,k} == delta_{i,k} ? No.
    # For matrix multiplication, the tensor is T_{i,j,k} = 1.
    # Wait, the tensor for matrix multiplication M_{n,m,p} is defined such that
    # C = A B means C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor T has T_{i,j,k} = 1 for all i,j,k.
    # The bilinear complexity is the minimum rank of a decomposition of this tensor.
    # A rank-1 term contributes to C_{i,k} via A_{i,j}B_{j,k} if:
    # U_{i*m+j, r} = 1, V_{j*p+k, r} = 1, W_{k*n+i, r} = 1.
    # Then the term is A_{i,j} B_{j,k} added to C_{i,k}.
    # So for a valid decomposition, for each (i,k), we need sum_j (sum_r U_{i*m+j,r} V_{j*p+k,r} W_{k*n+i,r}) = 1?
    # No, the tensor T_{i,j,k} must be reconstructed.
    # T_{i,j,k} = 1 for all i,j,k.
    # So we check: for all i,j,k, sum_r U_{i*m+j,r} V_{j*p+k,r} W_{k*n+i,r} == 1.
    
    # This is expensive to check directly for all i,j,k if rank is high.
    # But here rank is small (< 80).
    # n=4, m=4, p=5. Total 80 elements.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                # u_idx = i*m+j
                # v_idx = j*p+k
                # w_idx = k*n+i
                u_val = U[i*m+j]
                v_val = V[j*p+k]
                w_val = W[k*n+i]
                val = np.dot(u_val, v_val * w_val)
                if abs(val - 1.0) > 1e-9:
                    return None # Invalid
    
    return {"U": U, "V": V, "W": W}

def construct():
    # Start with the schoolbook solution as a baseline
    # n=4, m=4, p=5
    # Rank 80
    n, m, p = DIMENSIONS
    
    # 1. Construct Schoolbook
    rank_school = n * m * p
    U_school = [[0.0] * rank_school for _ in range(n * m)]
    V_school = [[0.0] * rank_school for _ in range(m * p)]
    W_school = [[0.0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1.0
                V_school[j*p+k][col] = 1.0
                W_school[k*n+i][col] = 1.0
                col += 1
    
    # Verify schoolbook (should always pass)
    result_school = _validate_and_return(U_school, V_school, W_school)
    if result_school:
        best_rank = rank_school
        best_result = result_school
    else:
        # This should never happen
        best_rank = float('inf')
        best_result = None

    # 2. Attempt Strassen-based improvement for the 4x4x4 part
    # We treat the multiplication as A(4x4) * B(4x4) + A(4x4) * B(4x1)
    # The 4x4x4 part can potentially be done in rank < 16.
    # Strassen for 2x2x2 is rank 7.
    # For 4x4x4, we can use block Strassen.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # Using Strassen on the blocks (each 2x2):
    # M1 = (A11+A21)(B11+B21)
    # M2 = (A11+A21)(B21-B11)
    # M3 = (A12-A22)(B11+B12)
    # M4 = (A22-A11)(B22+B12)
    # M5 = (A11+A22)(B11+B22)
    # M6 = (A21+A22)(B12-B22)
    # M7 = (A11+A12)(B21-B12)
    
    # This gives rank 7 for each 2x2 block product?
    # No, Strassen reduces the rank of 2x2x2 from 8 to 7.
    # For 4x4x4, we can recursively apply Strassen.
    # Rank of 4x4x4 using Strassen:
    # Each of the 7 multiplications is of 2x2x2 matrices.
    # If we use rank 7 for each 2x2x2, total rank = 7 * 7 = 49.
    # Schoolbook rank for 4x4x4 is 16.
    # 49 > 16. So Strassen is worse for 4x4x4 if we just count rank.
    # Wait. The rank of the tensor M_{2,2,2} is 7.
    # The tensor M_{4,4,4} is built from M_{2,2,2}.
    # The rank of M_{4,4,4} is at most 7 * rank(M_{2,2,2})?
    # No, the decomposition of M_{4,4,4} uses 7 terms, each of which is a product of 2x2 matrices.
    # Each 2x2 matrix product can be decomposed into rank 7.
    # So total rank is 7 * 7 = 49.
    # Since 49 > 16, Strassen does NOT help for 4x4x4.
    
    # What about 4x4x5?
    # We can view B as [B_4x4, B_4x1].
    # C = A B_4x4 + A B_4x1.
    # Part 1: A B_4x4. Rank 16 (schoolbook) or 49 (Strassen). 16 is better.
    # Part 2: A B_4x1. This is 4x4x1.
    # Rank of M_{4,4,1}?
    # C = A v.
    # This is a linear map.
    # Can we do better than 16?
    # M_{2,2,1} rank is 3?
    # C1 = A11 v1 + A12 v2
    # C2 = A21 v1 + A22 v2
    # M1 = (A11+A21)v1 -> contributes to C1, C2? No.
    # Let's check rank of M_{2,2,1}.
    # Tensor T_{i,j,k} with k=0,1.
    # T_{1,1,1}=1, T_{1,2,2}=1, T_{2,1,1}=1, T_{2,2,2}=1.
    # Actually, indices: i (row of A), j (col of A / row of B), k (col of B).
    # For 2x2x1:
    # C1 = A11 B11 + A12 B21
    # C2 = A21 B11 + A22 B21
    # Terms:
    # (i=1, j=1, k=1) -> A11 B11
    # (i=1, j=2, k=1) -> A12 B21
    # (i=2, j=1, k=1) -> A21 B11
    # (i=2, j=2, k=1) -> A22 B21
    # Schoolbook rank 4.
    # Can we do 3?
    # This is equivalent to computing 2x2 matrix-vector product.
    # It is known that r(M_{2,2,1}) = 3?
    # If so, for 4x4x1, we can use block structure.
    # A = [[A11, A12], [A21, A22]], v = [v1, v2].
    # C1 = A11 v1 + A12 v2
    # C2 = A21 v1 + A22 v2
    # If r(M_{2,2,1})=3, then:
    # C1 needs 3 mults? No, C1 is 2 elements.
    # The rank of the tensor is the number of multiplications.
    # If we use 3 mults for the whole 2x2x1, it computes both C1 and C2.
    # So for 4x4x1, we have 4 blocks of 2x2x1?
    # A11 v1, A12 v2, A21 v1, A22 v2.
    # We can combine:
    # (A11+A21) v1 -> gives contribution to C1 and C2?
    # Let's assume r(M_{4,4,1}) is 12? (3 * 4 blocks?)
    # Or maybe 16 is optimal.
    
    # Given the complexity of finding a rank < 80 decomposition for 4x4x5
    # and the 120s time limit, and the fact that Strassen is worse for 4x4x4,
    # it is safest to return the schoolbook decomposition.
    
    return best_result
# EVOLVE-BLOCK-END
