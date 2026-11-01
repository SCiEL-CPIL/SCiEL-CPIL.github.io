"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def _strassen_rank7():
    """
    Returns the standard Strassen rank-7 decomposition for 2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    U row: i*m + j (i in 0..1, j in 0..1) -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    V row: j*p + k (j in 0..1, k in 0..1) -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    W row: k*n + i (k in 0..1, i in 0..1) -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    Strassen 7 terms:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 - M4 + M5 - M7
    C01 = M3 + M4
    C10 = M2 + M5
    C11 = M1 - M2 + M3 - M6
    
    Let's map these to U, V, W.
    C[r, c] = sum_k U[r, k] * V[c, k] * W[k, k]? No, the tensor is T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] ?
    The interface says:
    U shape (n*m, rank) -> U[i*m+j, c]
    V shape (m*p, rank) -> V[j*p+k, c]
    W shape (p*n, rank) -> W[k*n+i, c]
    
    The product is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    This must equal the entry (i,k) of the matrix product A*B.
    A is n x m, B is m x p.
    (A*B)[i,k] = sum_j A[i,j] B[j,k].
    
    So the tensor T[i,j,k] in the decomposition corresponds to the coefficient of A[i,j] B[j,k] in the output C[i,k].
    Wait, the standard tensor for matrix multiplication M_{n,m,p} has entries T[i,j,k] such that
    (A B)[i,k] = sum_{j} A[i,j] B[j,k] is NOT a trilinear form in the same way.
    The bilinear map is (A, B) -> C.
    The tensor decomposition of rank r means:
    (A B)[i,k] = sum_{c=1}^r U[i,j,c] V[j,k,c] W[k,i,c] ??
    No, U, V, W are factor matrices.
    Usually, a rank-r decomposition of the multiplication tensor T (size n x m x p) is:
    T[i,j,k] = sum_{c=1}^r U[i,j,c] V[j,k,c] W[k,i,c] is incorrect notation for factors.
    
    Let's look at the interface again:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    The condition for exact decomposition is:
    For all i in [0,n), j in [0,m), k in [0,p):
    sum_{c=0}^{rank-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    (This assumes the tensor entries are 1 for the multiplication pattern? No.)
    
    The multiplication tensor T has T[i,j,k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    Actually, the tensor T is defined such that the bilinear map is <T, A ⊗ B> = C.
    T[i,j,k] is 1 for all i,j,k? No.
    The matrix multiplication tensor T_{n,m,p} has T[i,j,k] = 1 if the input indices (i,j) and (j,k) map to output (i,k).
    Actually, T[i,j,k] = 1 for all i,j,k in the standard basis?
    Let's check the schoolbook code:
    U[i*m+j][col] = 1
    V[j*p+k][col] = 1
    W[k*n+i][col] = 1
    This sets the sum to 1 for the specific (i,j,k) triple.
    This implies the target tensor T has T[i,j,k] = 1 for all i,j,k?
    Yes, because the bilinear form for matrix multiplication is:
    C[i,k] = sum_j A[i,j] B[j,k].
    If we view A and B as vectors, the coefficient of A[i,j] and B[j,k] in C[i,k] is 1.
    So the tensor T has T[i,j,k] = 1 for all i,j,k.
    
    So we need U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] to sum to 1 for all i,j,k.
    
    Strassen's algorithm:
    C00 = M1 - M4 + M5 - M7
    C01 = M3 + M4
    C10 = M2 + M5
    C11 = M1 - M2 + M3 - M6
    
    Let's verify the indices for W.
    W[k*n+i, c].
    For C00 (i=0, k=0): W[0*n+0, c] = W[0, c].
    For C01 (i=0, k=1): W[1*n+0, c] = W[1, c].
    For C10 (i=1, k=0): W[0*n+1, c] = W[2, c].
    For C11 (i=1, k=1): W[1*n+1, c] = W[3, c].
    
    Let's construct U, V, W for Strassen.
    U rows: (i,j) -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    V rows: (j,k) -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    W rows: (k,i) -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    M1 = (A00+A11)(B00+B11)
    U: A00 (0,0)=1, A11 (1,1)=1 -> U[0,0]=1, U[3,0]=1
    V: B00 (0,0)=1, B11 (1,1)=1 -> V[0,0]=1, V[3,0]=1
    W: Contributes to C00 and C11.
    C00 coeff +1 -> W[0,0]=1
    C11 coeff +1 -> W[3,0]=1
    
    M2 = (A10+A11)B00
    U: A10 (1,0)=1, A11 (1,1)=1 -> U[2,1]=1, U[3,1]=1
    V: B00 (0,0)=1 -> V[0,1]=1
    W: Contributes to C10 (+1) and C11 (-1).
    C10 (i=1, k=0) -> W[2,1]=1
    C11 (i=1, k=1) -> W[3,1]=-1
    
    M3 = A00(B10-B11)
    U: A00 (0,0)=1 -> U[0,2]=1
    V: B10 (1,0)=1, B11 (1,1)=-1 -> V[2,2]=1, V[3,2]=-1
    W: Contributes to C01 (+1) and C11 (-1).
    C01 (i=0, k=1) -> W[1,2]=1
    C11 (i=1, k=1) -> W[3,2]=-1
    
    M4 = A11(B01-B00)
    U: A11 (1,1)=1 -> U[3,3]=1
    V: B01 (0,1)=1, B00 (0,0)=-1 -> V[1,3]=1, V[0,3]=-1
    W: Contributes to C00 (-1) and C01 (+1).
    C00 (i=0, k=0) -> W[0,3]=-1
    C01 (i=0, k=1) -> W[1,3]=1
    
    M5 = (A00+A01)B11
    U: A00 (0,0)=1, A01 (0,1)=1 -> U[0,4]=1, U[1,4]=1
    V: B11 (1,1)=1 -> V[3,4]=1
    W: Contributes to C00 (+1) and C10 (+1).
    C00 (i=0, k=0) -> W[0,4]=1
    C10 (i=1, k=0) -> W[2,4]=1
    
    M6 = (A10-A00)(B00+B01)
    U: A10 (1,0)=1, A00 (0,0)=-1 -> U[2,5]=1, U[0,5]=-1
    V: B00 (0,0)=1, B01 (0,1)=1 -> V[0,6]=1, V[1,6]=1
    W: Contributes to C11 (-1).
    C11 (i=1, k=1) -> W[3,6]=-1
    
    M7 = (A01-A11)(B10+B11)
    U: A01 (0,1)=1, A11 (1,1)=-1 -> U[1,7]=1, U[3,7]=-1
    V: B10 (1,0)=1, B11 (1,1)=1 -> V[2,7]=1, V[3,7]=1
    W: Contributes to C00 (-1).
    C00 (i=0, k=0) -> W[0,7]=-1
    
    Let's assemble the matrices.
    U (4x7):
    Row 0 (A00): [1, 0, 1, 0, 1, -1, 0]
    Row 1 (A01): [0, 0, 0, 0, 1, 0, 1]
    Row 2 (A10): [0, 1, 0, 0, 0, 1, 0]
    Row 3 (A11): [1, 1, 0, 1, 0, 0, -1]
    
    V (4x7):
    Row 0 (B00): [1, 1, 0, -1, 0, 1, 0]
    Row 1 (B01): [0, 0, 0, 1, 0, 1, 0]
    Row 2 (B10): [0, 0, 1, 0, 0, 0, 1]
    Row 3 (B11): [1, 0, -1, 0, 1, 0, 1]
    
    W (4x7):
    Row 0 (C00): [1, 0, 0, -1, 1, 0, -1]
    Row 1 (C01): [0, 0, 1, 1, 0, 0, 0]
    Row 2 (C10): [0, 1, 0, 0, 1, 0, 0]
    Row 3 (C11): [1, -1, -1, 0, 0, -1, 0]
    
    Let's double check C00: M1 - M4 + M5 - M7
    W[0] should be [1, 0, 0, -1, 1, 0, -1].
    M1 (col 0): W[0,0]=1. Correct.
    M2 (col 1): W[0,1]=0. Correct.
    M3 (col 2): W[0,2]=0. Correct.
    M4 (col 3): W[0,3]=-1. Correct.
    M5 (col 4): W[0,4]=1. Correct.
    M6 (col 5): W[0,5]=0. Correct.
    M7 (col 6): W[0,6]=-1. Correct.
    
    C01: M3 + M4
    W[1] should be [0, 0, 1, 1, 0, 0, 0].
    M3 (col 2): W[1,2]=1. Correct.
    M4 (col 3): W[1,3]=1. Correct.
    Others 0. Correct.
    
    C10: M2 + M5
    W[2] should be [0, 1, 0, 0, 1, 0, 0].
    M2 (col 1): W[2,1]=1. Correct.
    M5 (col 4): W[2,4]=1. Correct.
    Others 0. Correct.
    
    C11: M1 - M2 + M3 - M6
    W[3] should be [1, -1, -1, 0, 0, -1, 0].
    M1 (col 0): W[3,0]=1. Correct.
    M2 (col 1): W[3,1]=-1. Correct.
    M3 (col 2): W[3,2]=-1. Correct.
    M4 (col 3): W[3,3]=0. Correct.
    M5 (col 4): W[3,4]=0. Correct.
    M6 (col 5): W[3,5]=-1. Correct.
    M7 (col 6): W[3,6]=0. Correct.
    
    This looks correct.
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, -1, 1, 0, -1],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [1, -1, -1, 0, 0, -1, 0]
    ]
    return {"U": U, "V": V, "W": W}

def _check_decomposition(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Target tensor T[i,j,k] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _search_rank6(time_limit=50):
    """
    Search for a rank 6 decomposition.
    This is extremely unlikely to succeed, but it's a different strategy.
    We use a small grid of values {-1, 0, 1} to keep it fast.
    """
    start_time = time.time()
    values = [-1, 0, 1]
    n, m, p = DIMENSIONS
    rank = 6
    
    # Random initialization and local search is too slow for 150s if not careful.
    # We'll just do a few random tries.
    import random
    for _ in range(1000):
        if time.time() - start_time > time_limit:
            break
        U = [[random.choice(values) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.choice(values) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.choice(values) for _ in range(rank)] for _ in range(p*n)]
        
        # Simple check
        if _check_decomposition(U, V, W):
            return {"U": U, "V": V, "W": W}
            
    return None

def construct():
    # Try to find rank 6 (impossible, but different strategy)
    # Use a small time budget to not risk the whole run
    result = _search_rank6(time_limit=10)
    if result is not None:
        return result
    
    # Fall back to known optimal rank 7
    return _strassen_rank7()
# EVOLVE-BLOCK-END
