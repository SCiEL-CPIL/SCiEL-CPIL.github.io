"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import itertools

# EVOLVE-BLOCK-START
def _is_half_int(x):
    """Check if x is a half-integer (multiple of 0.5)."""
    x2 = x * 2
    return abs(x2 - round(x2)) < 1e-10

def _make_rank_minimal():
    """
    Construct a rank-45 schoolbook decomposition.
    U: (n*m, rank) = (9, 45)
    V: (m*p, rank) = (15, 45)
    W: (p*n, rank) = (15, 45)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, max_iter=10000):
    """
    Verify that U, V, W form a valid tensor decomposition for matrix multiplication.
    Returns True if valid, False otherwise.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c]
    # We need T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B produce C[i,k]
    # Actually, the tensor for matrix multiplication is T[i,j,k] = 1 for all valid (i,j,k)
    # The decomposition is: C = sum_c U[:,c] * V[:,c] * W[:,c]
    # Where (U[:,c]) is a matrix, (V[:,c]) is a matrix, (W[:,c]) is a matrix
    # Actually, the standard form is:
    # C_{i,k} = sum_{j,c} A_{i,j} * B_{j,k} * T_{i,j,k,c}
    # No, the tensor decomposition is:
    # C_{i,k} = sum_c sum_j U_{i,j,c} * V_{j,k,c} * W_{i,k,c}
    # Wait, let me re-read the convention.
    
    # The convention is:
    # U has shape (n*m, rank), indexed by (i,j) -> i*m+j
    # V has shape (m*p, rank), indexed by (j,k) -> j*p+k
    # W has shape (p*n, rank), indexed by (k,i) -> k*n+i (trace convention W[k*n+i, r])
    
    # The decomposition says:
    # C_{i,k} = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
    # summed over j and c
    # No wait, that's not right either. Let me think again.
    
    # Standard tensor decomposition for matrix multiplication:
    # The matrix multiplication tensor T has entries T[i,j,k] = 1
    # A decomposition of rank r means we can write:
    # T[i,j,k] = sum_{c=0}^{r-1} a_{i,c} * b_{j,c} * c_{k,c}
    # But that's for the "simple" tensor. For matrix multiplication, the tensor is:
    # M[i,j,k] = delta_{i,j,k} where the multiplication is C[i,k] = sum_j A[i,j] B[j,k]
    
    # Actually, the correct formulation is:
    # C_{i,k} = sum_{j=0}^{m-1} sum_{c=0}^{rank-1} A_{i,j} * B_{j,k} * W_{k*n+i, c} * ... 
    # No. Let me look at this more carefully.
    
    # The tensor for multiplying n×m by m×p to get n×p is:
    # T_{i,j,k} where i∈[0,n), j∈[0,m), k∈[0,p)
    # And T_{i,j,k} = 1 for all i,j,k (since C_{i,k} = sum_j A_{i,j}B_{j,k})
    
    # A rank-r decomposition writes:
    # T_{i,j,k} = sum_{c=0}^{r-1} U_{(i,j),c} * V_{(j,k),c} * W_{(k,i),c}
    # Wait, that doesn't quite work dimensionally.
    
    # Let me reconsider. The standard way:
    # We want: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # This can be written as: C_{i,k} = sum_{j,c} A_{i,j} B_{j,k} delta_{i,k,j,c}
    # No...
    
    # OK here's the correct interpretation:
    # The matrix multiplication tensor is a 3-tensor T with indices (i, j, k) where:
    # i ranges over n (row of A, row of C)
    # j ranges over m (col of A, row of B)  
    # k ranges over p (col of B, col of C)
    # T[i,j,k] = 1
    # 
    # A rank-r decomposition means: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait no. U has n*m entries, V has m*p entries, W has p*n entries.
    # The product U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] summed over c should equal T[i,j,k] = 1?
    # But T[i,j,k] depends on three indices and the product also depends on three indices.
    # However, the sum is only over c, so for each (i,j,k) we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # for all i in [0,n), j in [0,m), k in [0,p).
    
    # Let's verify with schoolbook: U[i*m+j, c] = 1 if c = i*m+j then... no.
    # In schoolbook, for each (i,j,k), there's one column c where all three are 1.
    # So sum_c gives exactly 1 for each (i,j,k). Good.
    
    # So verification: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    
    # For efficiency, we can check a subset or use random testing.
    # Let's do exact check for all (i,j,k) since n*m*p = 45 which is small.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-10:
                    return False
    return True

def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds safety margin
    
    result = _make_rank_minimal()
    # Verify the schoolbook solution
    assert _verify(result["U"], result["V"], result["W"]), "Schoolbook verification failed"
    
    return result
# EVOLVE-BLOCK-END
