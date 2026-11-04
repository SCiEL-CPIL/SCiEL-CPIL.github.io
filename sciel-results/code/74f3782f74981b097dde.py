import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor is the matrix multiplication tensor
    # T[i, j, k] = 1 for all i, j, k
    # Wait, if T is all-ones, rank is 1.
    # Let's assume the tensor is NOT all-ones.
    # Maybe the tensor is T[i, j, k] = 1 if i, j, k are distinct? No.
    # Let's assume the standard matrix multiplication tensor.
    # If rank is 1, we return rank 1.
    # But seed is 50.
    # I will try to find a rank < 50.
    
    # If the tensor is all-ones, rank 1 works.
    # U[i*m+j, 0] = 1
    # V[j*p+k, 0] = 1
    # W[k*n+i, 0] = 1
    # Sum = 1.
    # So rank 1 is a valid decomposition for the all-ones tensor.
    # If the task is indeed matrix multiplication, and the tensor is all-ones,
    # then rank 1 is the answer.
    # But the seed program uses rank 50.
    # This suggests the tensor is NOT all-ones.
    # What other tensor could it be?
    # Maybe T[i, j, k] = 1 only if j is the same for all? No.
    
    # Let's assume the seed program is correct and the tensor is NOT all-ones.
    # I will try to find a lower rank decomposition using a randomized approach.
    # If I fail, I return the schoolbook.
    
    # Schoolbook
    rank = n * m * p
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
