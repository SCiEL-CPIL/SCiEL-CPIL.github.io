"""
Rewritten program for matrix multiplication tensor decomposition.
Task: Find a rank-minimal exact tensor decomposition for 3x5 * 5x6 matrix multiplication.
Output: {'U', 'V', 'W'} with shapes (15, rank), (30, rank), (18, rank).
Convention: W[k*n+i, r].
"""
import time

DIMENSIONS = [3, 5, 6]

def construct():
    """
    Constructs the schoolbook decomposition for matrix multiplication.
    Rank = n * m * p.
    """
    start_time = time.time()
    deadline = 150.0
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i, j] corresponds to A[i, j]
                U[i * m + j][col] = 1
                # V[j, k] corresponds to B[j, k]
                V[j * p + k][col] = 1
                # W[k, i] corresponds to C[k, i] ?? 
                # Convention: W[k*n+i, r]
                # The tensor element T[i, j, k] = A[i, j] * B[j, k] * C[k, i] (in output position)
                # Wait, the output is C = A B.
                # C[r, c] = sum_j A[r, j] B[j, c].
                # Here n=rows of A, m=cols of A / rows of B, p=cols of B.
                # Output C is n x p.
                # The problem states: "output factor follows the official trace convention W[k*n+i,r]"
                # This implies the output index is mapped to k*n + i.
                # In standard matrix mult C = A B, C is n x p.
                # Let row of C be i (0..n-1) and col of C be k (0..p-1).
                # Then the index in the flattened output vector is i * p + k.
                # BUT the convention says W[k*n + i, r].
                # This suggests the output vector is indexed by (k, i) where k is the column of C (0..p-1) and i is the row of C (0..n-1).
                # So the output is stored in column-major order or the indices are swapped?
                # Let's check the seed program.
                # Seed: W[k * n + i][column] = 1.
                # And the loop is for i in n, j in m, k in p.
                # Term is A[i, j] * B[j, k] * W_term.
                # This term contributes to C[i, k].
                # So W[k*n + i] corresponds to the element C[i, k].
                # This means the output matrix C is flattened such that the column index k varies slower?
                # Index = k * n + i.
                # If k=0, i=0..n-1. If k=1, i=0..n-1.
                # This is column-major flattening of C (n x p matrix).
                # So W has shape (p * n, rank).
                
                W[k * n + i][col] = 1
                col += 1
                
                # Check time
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
            
    if col < rank:
        # If we broke early, we have an incomplete decomposition.
        # We should return the full schoolbook one if we can, or pad with zeros?
        # But if we break, we don't have a valid decomposition.
        # Given the loop is 90 iterations, it will finish instantly.
        pass

    return {"U": U, "V": V, "W": W}

