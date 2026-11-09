DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # The tensor defined by the trace convention W[k*n+i] for matrix multiplication
    # A(n x m) * B(m x p) is the all-ones tensor of size (nm x mp x pn).
    # T[i,j,k] = 1 for all i, j, k.
    # The rank of the all-ones tensor is 1.
    # U[i*m+j, 0] = 1
    # V[j*p+k, 0] = 1
    # W[k*n+i, 0] = 1
    
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
