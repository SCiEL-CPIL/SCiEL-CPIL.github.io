import time

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Returns the schoolbook decomposition for matrix multiplication of dimensions 2x3 and 3x10.
    The rank is n*m*p = 60.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists for easy conversion to numpy if needed, 
    # but the interface expects standard Python structures or NumPy arrays.
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries".
    # The seed uses lists of lists. I will stick to lists of lists to match the seed's interface exactly.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U maps input A (n x m) to intermediate space
                # Index in U: i*m + j
                U[i * m + j][column] = 1
                
                # V maps input B (m x p) to intermediate space
                # Index in V: j*p + k
                V[j * p + k][column] = 1
                
                # W maps intermediate space to output C (n x p)
                # Index in W: k*n + i (trace convention W[k*n+i, r])
                W[k * n + i][column] = 1
                
                column += 1
                
                # Time check
                if time.time() > deadline:
                    # This loop is very fast (60 iterations), so this check is redundant but safe
                    break

    return {"U": U, "V": V, "W": W}
