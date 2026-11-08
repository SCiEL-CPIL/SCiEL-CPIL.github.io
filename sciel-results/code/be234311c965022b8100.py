import time

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Constructs the schoolbook tensor decomposition for matrix multiplication.
    For n=2, m=3, p=10, the rank is n*m*p = 60.
    This is the baseline decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists of zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    column = 0
    # Iterate over all i, j, k combinations
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][column] = 1.0
                # V index: j*p + k
                V[j * p + k][column] = 1.0
                # W index: k*n + i
                W[k * n + i][column] = 1.0
                column += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple self-test to ensure it runs within time
    start = time.time()
    res = construct()
    elapsed = time.time() - start
    if elapsed > 10:
        print(f"Warning: construct took {elapsed}s")
    else:
        print(f"construct took {elapsed}s")
