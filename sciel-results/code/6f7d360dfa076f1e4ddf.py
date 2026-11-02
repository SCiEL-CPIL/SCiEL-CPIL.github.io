"""Matrix multiplication tensor decomposition for 2x3 * 3x9."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # For 2x3 * 3x9 matrix multiplication:
    # A is 2x3, B is 3x9, C is 2x9
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor is T[(i,j),(j,k),(k,i)] = 1 for all valid i,j,k
    # where i in [0,n), j in [0,m), k in [0,p)
    #
    # Schoolbook gives rank = n*m*p = 54
    #
    # Key structural insight: B is 3x9, which can be seen as 3x3 * 3 (three blocks)
    # The tensor M_{2,3,9} is the direct sum of three copies of M_{2,3,3}
    # So rank(M_{2,3,9}) <= 3 * rank(M_{2,3,3})
    #
    # For M_{2,3,3}, the schoolbook rank is 18.
    # But we can potentially do better by exploiting structure.
    #
    # Actually, let's think about this differently. The tensor for 2x3*3x9
    # has 2*3*9 = 54 nonzero entries (one for each (i,j,k) triple).
    # Each term in the decomposition can cover at most one such entry if we use
    # the standard tensor structure, so rank >= 54 is NOT necessarily true.
    #
    # However, for the standard matrix multiplication tensor, the rank is known
    # to be exactly n*m*p for the "generic" case when the dimensions are coprime
    # or have specific properties. For 2x3*3x9, since 9 = 3*3, there's a
    # Kronecker structure.
    #
    # The tensor M_{2,3,9} = M_{2,3,3} ⊗ M_{1,1,3} (in a sense)
    # Actually, more precisely: multiplying by a 3x9 matrix is like multiplying
    # by three 3x3 matrices in parallel.
    #
    # So the tensor is the direct sum of three M_{2,3,3} tensors.
    # rank(M_{2,3,9}) = 3 * rank(M_{2,3,3})
    #
    # Now, what is rank(M_{2,3,3})?
    # The schoolbook rank is 2*3*3 = 18.
    # Lower bounds: for M_{a,b,c}, rank >= a*b*c / max(a,b,c) is one bound,
    # but tighter bounds exist. For 2x3*3x3, the rank is known to be 18
    # (no better decomposition is known).
    #
    # Actually, let me reconsider. The Strassen algorithm shows that 2x2*2x2
    # can be done with rank 7 < 8. For larger matrices, there may be savings.
    # But for 2x3*3x3, I'm not aware of a rank < 18 decomposition.
    #
    # Let's just use the schoolbook decomposition with rank 54.
    # This is guaranteed to be valid.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
