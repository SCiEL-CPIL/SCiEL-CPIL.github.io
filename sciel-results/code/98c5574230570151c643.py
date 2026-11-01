import numpy as np

def construct():
    """
    Returns the optimal rank-7 Strassen decomposition for 2x2 matrix multiplication.
    The rank 7 is the theoretical minimum for 2x2 matrix multiplication.
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize matrices
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen decomposition coefficients
    # Input A indices: 0=a, 1=b, 2=c, 3=d (row-major)
    # Input B indices: 0=e, 1=f, 2=g, 3=h (row-major)
    # Output C indices: 0=c00, 1=c10, 2=c01, 3=c11 (row-major)
    # Note: The task specifies W[k*n+i, r] which corresponds to C[i, k] in standard math notation if we map k to row and i to col?
    # Let's check the task description: "output factor follows the official trace convention W[k*n+i,r]"
    # Standard matmul: C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T_{i, j, k} = delta_{i, k} ... no, T_{i, j, k} is 1 if i=k? No.
    # The tensor for matmul is T_{i, j, k} such that C_{i, k} = sum_j A_{i, j} B_{j, k} T_{i, j, k}?
    # Actually, the decomposition is $T_{i, j, k} = \sum_r U_{i, j, r} V_{j, k, r} W_{i, k, r}$.
    # The task says W[k*n+i, r]. This implies the index of W is flattened as `k*n + i`.
    # If n=2, indices are 0,1,2,3.
    # If i is row and k is col:
    # i=0, k=0 -> idx 0
    # i=1, k=0 -> idx 2 (if k*n+i) or 1 (if i*n+k)?
    # "W[k*n+i,r]" suggests index = k * n + i.
    # Let's assume standard row-major for A and B, and the specific W indexing given.
    # Usually, matrix C is indexed C[i, k].
    # If W index is k*n + i, then:
    # W[0] corresponds to k=0, i=0 -> C[0,0]
    # W[1] corresponds to k=0, i=1 -> C[1,0]
    # W[2] corresponds to k=1, i=0 -> C[0,1]
    # W[3] corresponds to k=1, i=1 -> C[1,1]
    
    # Let's verify Strassen's formulas with this W mapping.
    # C[0,0] = M1 + M4 - M5 + M7
    # C[1,0] = M2 + M4
    # C[0,1] = M3 + M5
    # C[1,1] = M1 - M2 + M3 + M6
    
    # U[i*m+j, r] where A is 2x2.
    # A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d.
    # U indices:
    # 0: a (i=0, j=0)
    # 1: b (i=0, j=1)
    # 2: c (i=1, j=0)
    # 3: d (i=1, j=1)
    
    # V[j*p+k, r] where B is 2x2.
    # B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h.
    # V indices:
    # 0: e (j=0, k=0)
    # 1: f (j=0, k=1)
    # 2: g (j=1, k=0)
    # 3: h (j=1, k=1)
    
    # W[k*n+i, r]
    # W indices:
    # 0: C[0,0] (k=0, i=0)
    # 1: C[1,0] (k=0, i=1)
    # 2: C[0,1] (k=1, i=0)
    # 3: C[1,1] (k=1, i=1)
    
    # Strassen Products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Fill U
    # M1: a+d -> U[0,0]=1, U[3,0]=1
    U[0,0] = 1; U[3,0] = 1
    # M2: c+d -> U[2,1]=1, U[3,1]=1
    U[2,1] = 1; U[3,1] = 1
    # M3: a -> U[0,2]=1
    U[0,2] = 1
    # M4: d -> U[3,3]=1
    U[3,3] = 1
    # M5: a+b -> U[0,4]=1, U[1,4]=1
    U[0,4] = 1; U[1,4] = 1
    # M6: c-a -> U[2,5]=1, U[0,5]=-1
    U[2,5] = 1; U[0,5] = -1
    # M7: b-d -> U[1,6]=1, U[3,6]=-1
    U[1,6] = 1; U[3,6] = -1
    
    # Fill V
    # M1: e+h -> V[0,0]=1, V[3,0]=1
    V[0,0] = 1; V[3,0] = 1
    # M2: e -> V[0,1]=1
    V[0,1] = 1
    # M3: f-h -> V[1,2]=1, V[3,2]=-1
    V[1,2] = 1; V[3,2] = -1
    # M4: g-e -> V[2,3]=1, V[0,3]=-1
    V[2,3] = 1; V[0,3] = -1
    # M5: h -> V[3,4]=1
    V[3,4] = 1
    # M6: e+f -> V[0,5]=1, V[1,5]=1
    V[0,5] = 1; V[1,5] = 1
    # M7: g+h -> V[2,6]=1, V[3,6]=1
    V[2,6] = 1; V[3,6] = 1
    
    # Fill W
    # C[0,0] (W[0]) = M1 + M4 - M5 + M7
    W[0,0] = 1; W[0,3] = 1; W[0,4] = -1; W[0,6] = 1
    # C[1,0] (W[1]) = M2 + M4
    W[1,1] = 1; W[1,3] = 1
    # C[0,1] (W[2]) = M3 + M5
    W[2,2] = 1; W[2,4] = 1
    # C[1,1] (W[3]) = M1 - M2 + M3 + M6
    W[3,0] = 1; W[3,1] = -1; W[3,2] = 1; W[3,5] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
