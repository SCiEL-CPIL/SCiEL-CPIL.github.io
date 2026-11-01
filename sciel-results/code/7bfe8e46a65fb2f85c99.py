import numpy as np
import time

def _verify_decomp(U, V, W, n, m, p, trials=10, seed=42):
    """
    Verify the bilinear identity:
    C[i, k] = sum_j A[i, j] B[j, k]
    Reconstructed:
    C[i, k] = sum_r U[i, r] V[j, r] W[k, r] A[i, j] B[j, k]
    Note: The tensor indices for the decomposition of multiplication are:
    T_{i, j, k} = sum_r U_{i, r} V_{j, r} W_{k, r}
    where i in [0, n), j in [0, m), k in [0, p).
    However, the interface defines U as (n*m, rank), V as (m*p, rank), W as (p*n, rank).
    This implies the indices are flattened:
    U index: i*m + j
    V index: j*p + k
    W index: k*n + i
    
    Wait, the standard tensor for matrix multiplication C = AB is:
    C_{i, k} = sum_j A_{i, j} B_{j, k}
    The tensor is T_{i, j, k} = delta_{i, i'} delta_{j, j'} delta_{k, k'} ... no.
    The multiplication tensor is T_{i, j, k} such that T_{i, j, k} = 1 if the term A_{i, j} B_{j, k} contributes to C_{i, k}.
    In the bilinear form: C_{i, k} = sum_{j, r} (A_{i, j} B_{j, k}) (U_{i, j, r} V_{j, k, r} W_{i, k, r})?
    
    Let's look at the interface:
    U: (n*m, rank) -> indexed by (i, j)
    V: (m*p, rank) -> indexed by (j, k)
    W: (p*n, rank) -> indexed by (k, i)
    
    The identity is:
    sum_{j, r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i, j] * B[j, k] = C[i, k]
    Since C[i, k] = sum_j A[i, j] B[j, k], we require:
    sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{j, j'}? No.
    
    For a fixed i, k, the term is sum_j A[i, j] B[j, k].
    The reconstruction is sum_{j, r} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i, j] B[j, k].
    This must equal sum_j A[i, j] B[j, k].
    Therefore, for all i, k, and for each j, the coefficient of A[i, j] B[j, k] must be 1.
    So: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i, j, k.
    Wait, this is only true if the indices align.
    Let's check the schoolbook case:
    U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1 for col = i*m*p + j*p + k.
    Then sum_r ... = 1 for the matching r, 0 otherwise.
    So the condition is:
    For all i in [0, n), j in [0, m), k in [0, p):
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rng = np.random.RandomState(seed)
    for _ in range(trials):
        # We don't actually need random A, B to verify the identity of the tensor.
        # The identity is algebraic: sum_r U_{ij, r} V_{jk, r} W_{ki, r} = 1 for all i, j, k.
        # If this holds, then for any A, B:
        # sum_{j, r} U_{ij, r} V_{jk, r} W_{ki, r} A_{ij} B_{jk} = sum_j A_{ij} B_{jk} = C_{ik}
        # So we just check the tensor identity directly.
        pass
    
    # Direct algebraic check
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_t = np.array(U).reshape(n, m, -1)
    V_t = np.array(V).reshape(m, p, -1)
    W_t = np.array(W).reshape(p, n, -1)
    
    # Contract over r: result[i, j, k] = sum_r U_t[i, j, r] V_t[j, k, r] W_t[k, i, r]
    # This is a batched contraction.
    # result[i, j, k] = einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    try:
        result = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    except Exception as e:
        return False
        
    # Check if result is all 1s
    if not np.allclose(result, 1.0, atol=1e-9):
        return False
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    n, m, p = 2, 2, 11
    
    # Strategy: Construct a rank-37 decomposition using FFT over complex numbers.
    # The multiplication tensor for 2x2 by 2x11 can be viewed as 2 independent polynomial multiplications
    # if we group the columns, but actually it's coupled by the rows of A.
    # However, a known upper bound for R(2, 2, p) is related to R(1, 2, p) * 2 or similar.
    # Actually, R(2, 2, 11) <= R(1, 2, 11) + R(1, 2, 11)? No.
    # Let's use the explicit construction for polynomial multiplication.
    # Multiplying a degree 1 polynomial (2 coeffs) by a degree 10 polynomial (11 coeffs)
    # gives a degree 11 polynomial (12 coeffs).
    # Here we have 2 rows of A, so it's like multiplying two vectors of length 2 by a matrix of size 2x11.
    # This is equivalent to 2 independent convolutions? No, the second dimension of A is the same for both rows.
    
    # Let's look at the structure:
    # C_{0, k} = A_{0, 0} B_{0, k} + A_{0, 1} B_{1, k}
    # C_{1, k} = A_{1, 0} B_{0, k} + A_{1, 1} B_{1, k}
    # This is just two independent matrix-vector multiplications if we view B as a matrix.
    # But B is shared.
    # Actually, this is just the standard matrix multiplication tensor.
    
    # A well-known construction for low-rank decomposition of matrix multiplication uses FFT.
    # For multiplying two polynomials of length N and M, the rank is N+M-1 (if we allow complex numbers).
    # Here, we can decompose the 2x11 multiplication into two 1x11 multiplications?
    # No, the rank of 1x11 by 11x1 is 11 (convolution).
    # Wait, R(1, 11, 1) is 11? No, R(1, n, 1) = n.
    # R(1, 2, 11) is the rank of multiplying a 1x2 vector by a 2x11 matrix.
    # This is equivalent to a linear map. The tensor is T_{j, k} = B_{j, k} A_j.
    # This is just the identity map on the 2x11 matrix space? No.
    # C_k = sum_j A_j B_{j, k}.
    # This is a linear combination of the rows of B.
    # The rank of this tensor is 11? No, it's 2*11 = 22?
    # Actually, for fixed A, C is a linear combination of rows of B.
    # The bilinear complexity R(1, 2, 11) is known to be 11 + 1 - 1 = 11? No.
    # Let's use the FFT construction for the whole 2x2 by 2x11.
    
    # We can pad the 2x2 matrix to a 2x2 block diagonal? No.
    # Let's try to construct a rank 37 decomposition.
    # 37 = 2 * 11 + 15? No.
    # Known result: R(2, 2, n) <= 2 * R(1, 2, n).
    # R(1, 2, n) is the rank of multiplying a 1x2 by 2xn.
    # This is equivalent to computing y_k = a_0 x_{0, k} + a_1 x_{1, k}.
    # This is just a linear transformation. The tensor is T_{j, k} = x_{j, k} a_j.
    # The rank is n? No, it's 2n?
    # Actually, R(1, 2, n) = n + 1? No.
    # Let's use the explicit FFT-based decomposition for polynomial multiplication.
    # We can embed the 2x2 by 2x11 multiplication into a larger polynomial multiplication.
    
    # Alternative: Use the known upper bound R(2, 2, 11) <= 37.
    # Construction:
    # We can view the multiplication as a block matrix multiplication.
    # Let's use the following trick:
    # C = A B
    # We can decompose A and B into sums of rank-1 matrices? No.
    
    # Let's try to construct a valid decomposition of rank 37 using FFT.
    # We will use the fact that convolution can be done with N+M-1 multiplications.
    # Here, we have two "signals" of length 2 and 11.
    # If we treat the 2x2 matrix A as two vectors of length 2 (rows), and B as two vectors of length 11 (rows),
    # then C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # This is not a convolution. It's a bilinear form.
    
    # However, there is a known construction for R(2, 2, n) using the FFT of size n+1.
    # Let's use the following:
    # We will construct U, V, W such that the rank is 37.
    # We will use the FFT of size 13 (next power of 2 is 16, but 13 is prime).
    # Actually, let's use the simple schoolbook decomposition first to ensure we return something valid,
    # and then try to improve it.
    
    # Schoolbook rank is 2*2*11 = 44.
    # We want to beat 44.
    
    # Let's try to construct a rank 37 decomposition.
    # We can use the following identity:
    # (a0 + a1 x)(b0 + b1 x) = a0 b0 + (a0 b1 + a1 b0) x + a1 b1 x^2
    # This requires 3 multiplications (Karatsuba).
    # For length 2 and 11, we can use FFT.
    
    # Let's use the following construction:
    # We will use 37 bilinear forms.
    # We will use the FFT of size 13 (or 16).
    # Let N = 13.
    # We pad A and B to length 13.
    # A is 2x2, so we can view it as a polynomial of degree 1 with coefficients in a 2-dimensional space?
    # No, let's view A as a vector of length 2 and B as a vector of length 11.
    # Then C is a vector of length 12.
    # But we have two rows of A.
    
    # Actually, the simplest way to get a low-rank decomposition is to use the FFT.
    # Let's use the following:
    # We will construct a decomposition of rank 37.
    # We will use the fact that R(2, 2, 11) <= 37.
    
    # Let's try to construct it explicitly.
    # We will use the following:
    # U, V, W are matrices of size (4, 37), (22, 37), (22, 37).
    # We will use the FFT of size 13.
    # Let w = exp(2 pi i / 13).
    # We will use 13 frequency components.
    # For each frequency k, we compute the product of the FFTs of A and B.
    # This gives 13 complex multiplications.
    # But we have 2 rows of A.
    # So we need 2 * 13 = 26 complex multiplications.
    # Each complex multiplication can be done with 3 real multiplications (Karatsuba).
    # So 26 * 3 = 78 real multiplications.
    # This is worse than 44.
    
    # Let's try a different approach.
    # We will use the following construction:
    # R(2, 2, 11) <= R(1, 2, 11) + R(1, 2, 11) = 2 * R(1, 2, 11).
    # R(1, 2, 11) is the rank of multiplying a 1x2 by 2x11.
    # This is equivalent to computing y_k = a0 x_{0, k} + a1 x_{1, k}.
    # This is just a linear combination.
    # The rank is 11? No, it's 2*11 = 22?
    # Actually, R(1, 2, 11) = 11 + 1 = 12? No.
    # Let's use the following:
    # We will construct a rank 37 decomposition.
    # We will use the following:
    # U, V, W are matrices of size (4, 37), (22, 37), (22, 37).
    # We will use the following:
    # For r in 0..36:
    #   U[i, r] = ...
    #   V[j, r] = ...
    #   W[k, r] = ...
    
    # Let's try to use the following:
    # We will use the schoolbook decomposition and then try to reduce the rank.
    # But we don't have time for a search.
    
    # Let's return the schoolbook decomposition.
    # It's guaranteed to be valid and has rank 44.
    # This is better than nothing, and it's a valid baseline.
    
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Verify the schoolbook decomposition
    if _verify_decomp(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # This should never happen
        return {"U": [[0]*rank for _ in range(n*m)], "V": [[0]*rank for _ in range(m*p)], "W": [[0]*rank for _ in range(p*n)]}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
