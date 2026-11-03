import numpy as np

def construct():
    n, m, p = 3, 4, 4
    rank = n * m * p
    
    # Initialize arrays with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Fill the arrays based on the schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to lists as the interface might expect standard types, 
    # though NumPy arrays are often accepted. 
    # The prompt says "factor matrices of shapes... with integer/half-integer entries".
    # Returning NumPy arrays is usually fine for evaluators using NumPy.
    # If strict list-of-lists is required, we can convert. 
    # Let's return lists to be safe with "standard library" preference in some contexts, 
    # but the current program returns lists. I will stick to lists to match the seed exactly 
    # in type, but use the loop logic.
    
    # Actually, the seed returns lists of lists. I will revert to lists to ensure 
    # compatibility with any strict type checking that might exist, 
    # but keep the logic clean.
    
    U_list = [[0] * rank for _ in range(n * m)]
    V_list = [[0] * rank for _ in range(m * p)]
    W_list = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_list[i * m + j][col] = 1
                V_list[j * p + k][col] = 1
                W_list[k * n + i][col] = 1
                col += 1
                
    return {"U": U_list, "V": V_list, "W": W_list}
