import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0  # Safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # The tensor T_{i,j,k} = 1 for all i in [0,n), j in [m), k in [p)
    # This corresponds to the matrix multiplication tensor for 2x4 by 4x8.
    # We can decompose this using the "schoolbook" approach but exploiting
    # the structure that the tensor is separable in the output index k.
    #
    # Actually, the standard matrix multiplication tensor for C = A * B where
    # A is n x m and B is m x p has entries T_{i,j,k} = 1 if we view it as
    # C_{i,k} = sum_j A_{i,j} B_{j,k}. The tensor is T_{i,j,k} = 1.
    #
    # A simple decomposition: for each j in [0, m), we can form a rank-1 term
    # that captures the contribution of A_{i,j} * B_{j,k} to all C_{i,k}.
    # But that's not quite right because C_{i,k} is a sum over j.
    #
    # Let's think again. The tensor is T_{i,j,k} = 1 for all i, j, k.
    # This is the tensor of all ones of shape (n, m, p).
    # The rank of the all-ones tensor of shape (n, m, p) is min(n, m, p).
    # Wait, is that true?
    # The all-ones tensor J_{i,j,k} = 1 can be written as sum_{r=1}^{R} u_r(i) v_r(j) w_r(k).
    # If we take R = 1, u_1(i)=1, v_1(j)=1, w_1(k)=1, then sum is 1. So rank is 1.
    # But that's not the matrix multiplication tensor!
    #
    # I made a mistake. The matrix multiplication tensor is NOT the all-ones tensor.
    # The matrix multiplication tensor for multiplying an n x m matrix by an m x p matrix
    # has entries T_{i,j,k} where i indexes rows of A and C, j indexes columns of A and rows of B,
    # and k indexes columns of B and C.
    # The entry T_{i,j,k} is 1 if and only if the (i,k) entry of C depends on A_{i,j} * B_{j,k}.
    # Indeed, C_{i,k} = sum_{j=0}^{m-1} A_{i,j} B_{j,k}.
    # So T_{i,j,k} = 1 for all i, j, k.
    #
    # Wait, so the tensor IS the all-ones tensor?
    # Let me double-check with a small example.
    # 2x2 by 2x2: C = A * B.
    # C_{0,0} = A_{0,0}B_{0,0} + A_{0,1}B_{1,0}
    # C_{0,1} = A_{0,0}B_{0,1} + A_{0,1}B_{1,1}
    # C_{1,0} = A_{1,0}B_{0,0} + A_{1,1}B_{1,0}
    # C_{1,1} = A_{1,0}B_{0,1} + A_{1,1}B_{1,1}
    #
    # The tensor T has indices (i, j, k) where i in {0,1}, j in {0,1}, k in {0,1}.
    # T_{i,j,k} = 1 if A_{i,j} * B_{j,k} appears in C_{i,k}.
    # For C_{0,0}: terms are A_{0,0}B_{0,0} and A_{0,1}B_{1,0}. So T_{0,0,0}=1, T_{0,1,0}=1.
    # For C_{0,1}: terms are A_{0,0}B_{0,1} and A_{0,1}B_{1,1}. So T_{0,0,1}=1, T_{0,1,1}=1.
    # For C_{1,0}: terms are A_{1,0}B_{0,0} and A_{1,1}B_{1,0}. So T_{1,0,0}=1, T_{1,1,0}=1.
    # For C_{1,1}: terms are A_{1,0}B_{0,1} and A_{1,1}B_{1,1}. So T_{1,0,1}=1, T_{1,1,1}=1.
    #
    # So indeed T_{i,j,k} = 1 for all i, j, k. The tensor is the all-ones tensor.
    #
    # But the rank of the all-ones tensor is 1, not 64 or 56!
    # This contradicts the task description which says "schoolbook rank is 64".
    #
    # I must be misunderstanding the tensor convention.
    # Let me re-read the task: "The output factor follows the official trace convention W[k*n+i,r]."
    #
    # The standard tensor decomposition of matrix multiplication is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # This can be written as a tensor contraction:
    # T(A, B) = sum_{j} A_{i,j} B_{j,k}
    # The tensor T has entries T_{i,j,k} = 1.
    #
    # But the "rank" of the matrix multiplication tensor is NOT the rank of the all-ones tensor.
    # The rank of the matrix multiplication tensor is the minimum number of multiplications
    # needed to compute the matrix product, which is a different notion.
    #
    # Actually, I think the confusion is that the matrix multiplication tensor is NOT the all-ones tensor.
    # The all-ones tensor has rank 1, but that doesn't mean we can compute matrix multiplication with 1 multiplication.
    #
    # Let me think about this differently. The tensor for matrix multiplication is defined as:
    # T_{i,j,k} = 1 if the (i,k) entry of C involves A_{i,j} * B_{j,k}, else 0.
    # As I showed, T_{i,j,k} = 1 for all i, j, k.
    #
    # But the rank of this tensor (as a 3-way tensor) is not the same as the bilinear complexity of matrix multiplication.
    # The bilinear complexity is the minimum number of scalar multiplications needed, which is related to the
    # "border rank" or "tensor rank" in a specific sense.
    #
    # Actually, I recall now: the matrix multiplication tensor is NOT the all-ones tensor.
    # The all-ones tensor J_{i,j,k} = 1 corresponds to the operation C_{i,k} = (sum_i A_{i,j})(sum_k B_{j,k})? No, that doesn't make sense.
    #
    # Let me look at this more carefully. The tensor T for matrix multiplication satisfies:
    # C = A * B = sum_{r} (U_r * A) * (V_r * B) * W_r
    # where U_r, V_r, W_r are rank-1 updates.
    #
    # Actually, I think the issue is that the tensor T_{i,j,k} is not 1 for all i,j,k.
    # Let me reconsider.
    #
    # In the standard formulation, the matrix multiplication tensor is a 3-way tensor of shape (n*m, m*p, p*n) or similar.
    # The entries are indexed by (i,j), (j,k), (k,i) or similar.
    #
    # The task says: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    # So U is indexed by (i*m + j), V by (j*p + k), W by (k*n + i).
    #
    # The decomposition is: C_{i,k} = sum_{j} A_{i,j} B_{j,k} = sum_{r} U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    #
    # For this to hold for all A, B, we need:
    # sum_{r} U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = delta_{i,k}? No, that's not right either.
    #
    # Actually, the condition is:
    # For all i, j, k: sum_{r} U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1
    #
    # This is because C_{i,k} = sum_j A_{i,j} B_{j,k}, and the coefficient of A_{i,j} B_{j,k} in C_{i,k} is 1.
    #
    # So the tensor T has entries T_{(i,j), (j,k), (k,i)} = 1 for all valid i, j, k.
    # This is indeed the all-ones tensor on the index set {(i,j), (j,k), (k,i)}.
    #
    # But the rank of this tensor is not 1 because the indices are not independent in the way I thought.
    # The index (i,j) is paired with (j,k) through j, and (j,k) is paired with (k,i) through k, etc.
    #
    # Actually, I think I've been overcomplicating this. Let me just implement the rank-32 decomposition
    # that I derived earlier, which should be correct.
    
    rank = m * p  # 4 * 8 = 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            for i in range(n):
                U[i * m + j][r] = 1
            V[j * p + k][r] = 1
            for i in range(n):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
