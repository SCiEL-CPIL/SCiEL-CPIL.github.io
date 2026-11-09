import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor M_{n,m,p} has entries:
    # T[i,j,k] = A[i,j] * B[j,k]
    # We need rank-r decomposition: sum_{r=1}^{R} U[i,j,r] * V[j,k,r] * W[k,i,r] = A[i,j]*B[j,k]
    # where W[k*n+i, r] is the convention
    
    # For 2x3x7, the schoolbook rank is 42.
    # Known results: R(2,3,7) is known to be at most some value.
    # The best known upper bound for R(2,3,7) from literature is around 30-35.
    # But for an exact construction, we need to be careful.
    
    # Let's try to find a better decomposition using the structure.
    # The tensor is (A ⊗ B) where A is 2x3 and B is 3x7.
    
    # Actually, for matrix multiplication, the tensor rank R(n,m,p) satisfies:
    # R(n,m,p) >= n*m*p / (n*m + m*p + p*n - 2) is a lower bound (not tight)
    
    # For small dimensions, we can use known results or try to construct.
    # The schoolbook gives 42. Can we do better?
    
    # Let me think about this differently. The bilinear map is:
    # (A, B) -> A*B
    # where A is 2x3 and B is 3x7.
    
    # The tensor T has components T_{i,j,k} for i in {0,1}, j in {0,1,2}, k in {0,...,6}
    # T_{i,j,k}(A,B) = A_{i,j} * B_{j,k}
    
    # For a rank-1 term: u_{i,j} * v_{j,k} * w_{k,i}
    # We need: sum_r u_{i,j,r} * v_{j,k,r} * w_{k,i,r} = A_{i,j} * B_{j,k}
    
    # This is equivalent to finding the tensor rank of the multiplication tensor.
    
    # Known: R(2,2,2) = 7 (Strassen)
    # R(2,2,3) is known to be around 12-13
    # R(2,3,7) - I'm not sure of the exact known value.
    
    # For now, let's stick with the schoolbook decomposition which is guaranteed to work.
    # The rank is 42.
    
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple self-test
    res = construct()
    print("Rank:", len(res['U'][0]))
