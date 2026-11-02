"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time

def _to_half_int(x):
    """Convert a Fraction to a float that is a half-integer (k/2 for integer k).
    Returns the float representation."""
    # x is a Fraction; check if 2*x is integer
    two_x = 2 * x
    if two_x.denominator == 1:
        return float(two_x.numerator) / 2.0
    else:
        # Should not happen if our construction is correct
        return float(x)

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For all i, k: sum_t U[i*m + j, t] * V[j*p + k, t] summed over j should equal W[k*n + i, t] ... 
    Actually the tensor decomposition means:
    M[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]  ... no.
    
    Standard bilinear decomposition:
    The multiplication tensor T has entries T[i,j,k] = delta_{i,k} (for the product).
    Actually, for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    A bilinear decomposition of rank r writes:
    C[i,k] = sum_{t=0}^{r-1} U[i,t] * V[k,t]  ... no, that's not right either.
    
    Let me re-read the task. The factor matrices are:
    U: (n*m, rank) - indexed by (i,j)
    V: (m*p, rank) - indexed by (j,k)
    W: (p*n, rank) - indexed by (k,i)
    
    The decomposition should satisfy:
    For the tensor T_{i,j,k} which represents the multiplication:
    T_{i,j,k} = sum_t U[(i,j), t] * V[(j,k), t] * W[(k,i), t]
    
    Wait, that's a 3-way decomposition. Let me think again.
    
    Actually, for matrix multiplication, the standard formulation is:
    C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    A bilinear algorithm uses r multiplications. Each multiplication computes:
    M_t = (sum_j u_{i,j}^t A_{i,j}) * (sum_j v_{j,k}^t B_{j,k})
    Then C_{i,k} = sum_t w_{i,k}^t M_t
    
    So U, V, W would be:
    U: indexed by (i,j) and t -> u_{i,j}^t
    V: indexed by (j,k) and t -> v_{j,k}^t  
    W: indexed by (i,k) and t -> w_{i,k}^t
    
    But the task says W has shape (p*n, rank) indexed by (k,i) following the trace convention W[k*n+i, r].
    So W is indexed by (k,i) which is the same as (i,k) just in different order.
    
    The condition is:
    For all i, k: sum_j delta_{i,k} ... no.
    
    The tensor for matrix multiplication [n,m,p] has entries:
    T_{i,j,k} = 1 if we're computing C_{i,k} from A_{i,j} and B_{j,k}, else 0.
    
    Actually, the multiplication tensor T has T_{i,j,k} = delta_{j,j} which is always 1 for valid indices.
    No, that's not right.
    
    Let me be precise. The matrix multiplication tensor M_{n,m,p} is a 3-tensor with:
    M[i,j,k] = 1 for all valid i, j, k.
    
    Wait no. The tensor represents the bilinear map (A, B) -> C where C = AB.
    The entries are: for each (i,j,k), the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    
    So the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    A rank-r decomposition writes:
    T[i,j,k] = sum_{t=0}^{r-1} U[i,j,t] * V[j,k,t] * W[k,i,t]
    
    Wait, but that would be a 3-tensor decomposition, not a bilinear one.
    
    Actually, I think the correct interpretation is:
    T[i,j,k] = sum_t U[(i,j), t] * V[(j,k), t] * W[(k,i), t]
    
    And we need this to equal 1 for all valid (i,j,k).
    
    Let me verify with the schoolbook: rank = n*m*p.
    U[i*m+j, t] = 1 if t = i*m*p + j*p + k, else 0
    V[j*p+k, t] = 1 if t = i*m*p + j*p + k, else 0
    W[k*n+i, t] = 1 if t = i*m*p + j*p + k, else 0
    
    Then sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1 when t = i*m*p+j*p+k, else 0.
    So the sum is 1. Correct.
    """
    rank = U.shape[1]
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U_arr[i*m+j, t] * V_arr[j*p+k, t] * W_arr[k*n+i, t]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try to find a better decomposition using structured approaches.
    # 
    # Key idea: We can use the fact that the tensor T[i,j,k] = 1 for all (i,j,k)
    # can be decomposed using circulant or other structured matrices.
    #
    # One approach: use the fact that we can write 1 = sum of products using
    # roots of unity or other structured patterns.
    #
    # Let's try a block-circulant approach:
    # For the j-index (size m=5), we can use a circulant matrix of size N >= 5.
    # The DFT of size N gives a decomposition where each "frequency" contributes
    # one rank term.
    #
    # Specifically, if we use N = 5 (since m=5 is prime), we can use the 5-point DFT.
    # But the entries involve 5th roots of unity, which are not rational.
    #
    # Alternative: use a real-valued construction.
    #
    # Let's try a different approach: use the fact that we can decompose the
    # multiplication using a combination of smaller multiplications.
    #
    # For 3x5x7, we can split the m=5 dimension into parts and use a
    # block structure.
    #
    # Actually, let's try a simple improvement: use the fact that for fixed i and k,
    # the sum over j is just a dot product. We can use the identity:
    # sum_j a_j * b_j = (1/2)(sum_j (a_j + b_j)^2 - sum_j a_j^2 - sum_j b_j^2)
    # But this increases the rank.
    #
    # Let's try a more systematic approach: search for a low-rank decomposition
    # using random initialization and gradient descent, but constrained to
    # half-integer values.
    #
    # Actually, let's try a known result: the rank of [3,5,7] is at most 3*5*7 = 105,
    # but we can do better.
    #
    # One known bound: rank([n,m,p]) <= n*m*p - min(n,m,p) + 1 for some cases.
    # But I'm not sure about the exact bound.
    #
    # Let's try a practical approach: use a randomized search for a low-rank
    # decomposition with half-integer entries.
    #
    # Since the problem is small, we can try to find a decomposition with rank
    # significantly less than 105.
    
    # Try rank values from a lower bound up to the schoolbook rank
    # The lower bound for rank([n,m,p]) is at least max(n*p, m*p, n*m) = max(21, 35, 15) = 35
    
    # Let's try to find a decomposition with rank around 50-80 using a structured approach.
    
    # Approach: Use a circulant embedding with N = 8 (next power of 2 >= 7)
    # The 8-point DFT has entries involving sqrt(2), which are not half-integers.
    # But we can use a real-valued circulant approach.
    
    # Let's try a different approach: use the fact that we can write the identity
    # matrix of size m as a sum of rank-1 matrices, and use this to reduce the rank.
    
    # For m=5, the identity matrix I_5 can be written as a sum of 5 rank-1 matrices
    # (the standard basis). But we can also write it as a sum of fewer rank-1 matrices
    # if we allow complex entries, but we need real half-integer entries.
    
    # Actually, the minimum number of rank-1 real matrices needed to sum to I_5 is 5,
    # since the rank of I_5 is 5.
    
    # So the schoolbook decomposition is optimal in this sense? No, because we're
    # not summing matrices, we're decomposing a tensor.
    
    # Let me try a different approach: use the Strassen-like idea for 2x2x2.
    # For 2x2x2, the rank is 7 instead of 8.
    
    # We can embed our 3x5x7 problem into a larger problem and use this.
    
    # Actually, let's just try to find a better decomposition using a heuristic search.
    
    # For now, return the schoolbook decomposition.
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
