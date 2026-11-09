import time
import sys

# Task: Matrix multiplication 3x4 by 4x5
# Dimensions: n=3, m=4, p=5
# Rank of schoolbook decomposition: n*m*p = 3*4*5 = 60

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    for the tensor decomposition of matrix multiplication.
    
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    The decomposition satisfies:
    (A * B)_{i,k} = sum_{r=0}^{rank-1} U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    Wait, the standard decomposition is:
    C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    Tensor T_{i,j,k} = sum_{r} U_{i,j,r} V_{j,k,r} W_{i,k,r} ? No.
    
    The contract says:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    And the trace convention is W[k*n+i, r].
    
    Let's verify the indices.
    Input A is n x m. Index (i, j) where i in [0, n), j in [0, m).
    Input B is m x p. Index (j, k) where j in [0, m), k in [0, p).
    Output C is n x p. Index (i, k).
    
    C_{i,k} = sum_{j=0}^{m-1} A_{i,j} B_{j,k}
    
    Tensor form:
    T_{i,j,k} is 1 if it contributes to the sum?
    Actually, the tensor for matrix multiplication is defined by its action on A and B.
    The decomposition is:
    C_{i,k} = sum_{r} (sum_{j} U_{i*m+j, r} A_{i,j}) (sum_{j} V_{j*p+k, r} B_{j,k}) * W_{k*n+i, r} ?
    No, the standard rank-r decomposition of the multiplication tensor T is:
    T = sum_{r=1}^R T^{(r)}
    where T^{(r)}_{i,j,k} = U_{i,j,r} V_{j,k,r} W_{i,k,r} is not quite right because the indices are coupled.
    
    Usually, it is written as:
    C_{i,k} = sum_{r} (A U_r)_{i} (V_r B)_{k} W_{i,k,r}
    where A U_r is a vector of length n, V_r B is a vector of length p.
    
    Let's look at the shapes:
    U is (n*m, R). This can be seen as a matrix U_{i,j,r}.
    V is (m*p, R). This can be seen as a matrix V_{j,k,r}.
    W is (p*n, R). This can be seen as a matrix W_{k,i,r}.
    
    The product is:
    C_{i,k} = sum_{r} W_{k*n+i, r} * (sum_{j} U_{i*m+j, r} A_{i,j}) * (sum_{j} V_{j*p+k, r} B_{j,k})
    
    For this to equal sum_{j} A_{i,j} B_{j,k} for all A, B,
    we need the bilinear form to match.
    
    In the schoolbook case, each term r corresponds to a specific (i, j, k) triple?
    No, schoolbook has R = n*m*p terms.
    Each term r corresponds to a unique j? No, to a unique (j, i, k)?
    
    Let's trace the seed code:
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    
    So for a specific r (column), there is a unique (i, j, k).
    Let r map to (i_r, j_r, k_r).
    Then U_{i*m+j, r} is 1 if i=i_r and j=j_r, else 0.
    V_{j*p+k, r} is 1 if j=j_r and k=k_r, else 0.
    W_{k*n+i, r} is 1 if k=k_r and i=i_r, else 0.
    
    The term in the sum for C_{i,k} is:
    W_{k*n+i, r} * (sum_{j'} U_{i*m+j', r} A_{i,j'}) * (sum_{j''} V_{j''*p+k, r} B_{j'',k})
    
    For a fixed r with indices (i_r, j_r, k_r):
    W_{k*n+i, r} is 1 only if k=k_r and i=i_r.
    If i != i_r or k != k_r, the term is 0.
    So C_{i,k} only gets contributions from r where i_r=i and k_r=k.
    
    For such an r, the first sum is:
    sum_{j'} U_{i*m+j', r} A_{i,j'}
    Since U_{i*m+j', r} is 1 only if j'=j_r, this sum is A_{i, j_r}.
    
    The second sum is:
    sum_{j''} V_{j''*p+k, r} B_{j'',k}
    Since V_{j''*p+k, r} is 1 only if j''=j_r, this sum is B_{j_r, k}.
    
    So the contribution is A_{i, j_r} B_{j_r, k}.
    
    Summing over all r where i_r=i and k_r=k:
    C_{i,k} = sum_{r: i_r=i, k_r=k} A_{i, j_r} B_{j_r, k}
    
    In the schoolbook loop, for each (i, j, k), there is exactly one r.
    So for a fixed (i, k), the r's are those with i_r=i and k_r=k.
    The j_r varies over all j in [0, m).
    So the sum is sum_{j=0}^{m-1} A_{i,j} B_{j,k}.
    
    This matches the definition of matrix multiplication.
    So the schoolbook decomposition is valid.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists as per seed, but numpy might be faster for initialization?
    # However, the output must be serializable or compatible with evaluator.
    # The seed returns lists of lists. I will stick to lists of lists to be safe,
    # or use numpy arrays if the evaluator accepts them.
    # "factor matrices of shapes ... with integer/half-integer entries"
    # Usually numpy arrays are fine. Let's use numpy for speed and cleanliness.
    
    import numpy as np
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    n, m, p = DIMENSIONS
    rank = n*m*p
    U = res["U"]
    V = res["V"]
    W = res["W"]
    
    # Random test
    import numpy as np
    np.random.seed(42)
    A = np.random.randint(0, 10, (n, m))
    B = np.random.randint(0, 10, (m, p))
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0
            for r in range(rank):
                # Term: W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
                # Optimized: precompute AU and VB? No, just direct sum for check.
                # Actually, the formula is:
                # C_{i,k} = sum_r W_{k*n+i, r} (U[:, r]^T A[:, i? No A is n x m])
                
                # Let's stick to the definition:
                # u_val = sum_{j} U[i*m+j, r] * A[i,j]
                # v_val = sum_{j} V[j*p+k, r] * B[j,k]
                # s += W[k*n+i, r] * u_val * v_val
                
                u_val = 0
                for j in range(m):
                    u_val += U[i*m+j, r] * A[i, j]
                v_val = 0
                for j in range(m):
                    v_val += V[j*p+k, r] * B[j, k]
                s += W[k*n+i, r] * u_val * v_val
            C_computed[i, k] = s
            
    if not np.allclose(C_computed, C_expected):
        print("Test Failed")
        print(C_computed)
        print(C_expected)
    else:
        print("Test Passed")
