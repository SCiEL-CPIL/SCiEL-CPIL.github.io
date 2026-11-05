"""
DFT-based decomposition for 4x5 * 5x5 matrix multiplication.

We apply a DFT over Z_5 to the summation index k. This decomposes the
inner product into 5 independent diagonal multiplications in the frequency domain.

For each frequency w in {0, 1, 2, 3, 4}, we have a rank-4 tensor (one per row i of A).
Total rank = 5 * 4 = 20.

The DFT matrix F of size 5 has entries F[a][b] = omega^(a*b) where omega = e^(2*pi*i/5).
Since we need integer/half-integer entries, we work with the algebraic integer ring
Z[omega]. However, the task requires entries to be integer/half-integer.

Wait - the DFT matrix of size 5 involves 5th roots of unity, which are NOT integers or half-integers.
So this approach as stated won't give integer coefficients directly.

Let me reconsider. The directive says "integer coefficients derived from the DFT matrix of size 5".
This is problematic because the DFT of size 5 doesn't have integer entries.

However, there's a well-known result: the tensor rank of matrix multiplication can sometimes
be reduced using structured decompositions. But for 4x5 * 5x5, the schoolbook rank is 100.

Actually, let me re-read the directive more carefully. It says "yields a rank-20 exact decomposition 
with integer coefficients derived from the DFT matrix of size 5". 

This seems mathematically suspect because the DFT of size 5 involves complex numbers that are 
not integers. But let me think about whether there's a way to make this work.

One approach: use the fact that the 5th cyclotomic polynomial is x^4 + x^3 + x^2 + x + 1, 
and work in the ring Z[omega] where omega is a primitive 5th root of unity. But the entries 
would be in this ring, not in Z or (1/2)Z.

Given the constraint of integer/half-integer entries, the DFT approach over Z_5 doesn't directly 
apply. Let me think of another approach.

Actually, wait. Let me reconsider the problem. Perhaps the directive is pointing to a 
different kind of decomposition. Let me think about Strassen-like methods or other 
structured decompositions.

For now, let me implement what the directive asks for as best as possible, even if the 
entries might need to be complex. But the task says "integer/half-integer entries", so 
this is a problem.

Let me try a different interpretation: maybe we can use a real-valued transform that 
approximates the DFT, or perhaps there's a way to embed the DFT into a larger integer 
structure.

Actually, I think the most pragmatic approach given the constraints is to implement the 
schoolbook method but look for any possible rank reduction. The minimum known rank for 
4x5 * 5x5 is likely close to 100, and I don't see an obvious way to get to 20 with 
integer entries.

Let me just implement the schoolbook method correctly and see if there's any simple 
optimization.

Wait, I just realized: maybe the directive is aspirational or based on a misunderstanding. 
Let me implement the best I can with integer entries.

Actually, let me try one more idea: use the fact that 5 is prime and see if there's a 
combinatorial decomposition. But I don't see an obvious one.

Given the time constraint, let me implement the schoolbook method, which is guaranteed 
to work with integer entries.
"""

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    rank = n * m * p  # 100
    
    U = [[0] * rank for _ in range(n * m)]  # 20 x 100
    V = [[0] * rank for _ in range(m * p)]  # 25 x 100
    W = [[0] * rank for _ in range(p * n)]  # 20 x 100
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, col] = 1
                U[i * m + j][col] = 1
                # V[j*p + k, col] = 1
                V[j * p + k][col] = 1
                # W[k*n + i, col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
