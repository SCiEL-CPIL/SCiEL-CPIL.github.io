"""Matrix multiplication 2,2,13 tensor decomposition.

We exploit the fact that the 2x2 matrix multiplication tensor has rank 7 (Strassen),
and that for a 2 x (m*p) right factor, the tensor rank is m * rank(2x2 mult) = 2*7 = 14... 
but actually we need to be more careful.

For C = A * B where A is 2x2 and B is 2x13, we can think of B as 13 columns, each a 2x1 vector.
The multiplication of a 2x2 matrix by a 2x1 vector has tensor rank equal to the rank of the 
bilinear map (a,b) -> a*b where a is 2x2, b is 2x1.

Actually, let's think about this differently. The tensor for multiplying (n x m) * (m x p) 
has rank at most n*m*p. For n=2, m=2, p=13, that's 52.

But we can do better. The key insight: the multiplication C[i,k] = sum_j A[i,j] * B[j,k] 
can be viewed as: for each output position (i,k), we need A[i,0]*B[0,k] + A[i,1]*B[1,k].

This is a bilinear map from A (4 entries) and B (26 entries) to C (26 entries).

We can use the fact that the 2x2 matrix multiplication has Strassen rank 7. 
For each column k of B, we're doing a 2x2 times 2x1 multiplication. The rank of 
multiplying 2x2 by 2x1 is at most... let's think.

C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]

This is like computing two dot products of length 2. The bilinear map from (A[i,0], A[i,1]) 
and (B[0,k], B[1,k]) to C[i,k] has rank 2 (it's just a dot product). But across different i, 
the A factors are shared.

Actually, let me think about this more carefully using the tensor structure.

The multiplication tensor T_{i,j,k} for (2x2)*(2x13) has indices:
- i in {0,1} (row of A, row of C)
- j in {0,1} (col of A, row of B)  
- k in {0,...,12} (col of B, col of C)

T[i,j,k] = delta[i,k] for the appropriate... no.

Actually the tensor is: C[i,k] = sum_j A[i,j] B[j,k]

So the tensor has entries T[i,j,k,l] where l is... no, it's a 3-mode tensor in the 
A,B,C space. Let me define it properly.

The bilinear map: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]

This is a 3-tensor with modes (i,k) for C, (i,j) for A, (j,k) for B.
In the standard formulation, we have a tensor T of shape (n*p, n*m, m*p) where
T[i*p+k, i*m+j, j*p+k'] is 1 if k==k', else 0.

Wait, that's not right either. Let me use the standard definition.

For matrix multiplication (n x m) * (m x p), the tensor is T in R^{(n*p) x (n*m) x (m*p)}
where T[(i,k), (i,j), (j,k)] = 1 for all valid i,j,k (with the constraint that the same 
i,j,k appear in all three indices).

So T has rank at most n*m*p (schoolbook), and potentially less.

For n=2, m=2, p=13:
- C space: 2*13 = 26 dimensions, indexed by (i,k) where i in {0,1}, k in {0..12}
- A space: 2*2 = 4 dimensions, indexed by (i,j) where i in {0,1}, j in {0,1}
- B space: 2*13 = 26 dimensions, indexed by (j,k) where j in {0,1}, k in {0..12}

The tensor entry T[(i,k), (i',j'), (j'',k'')] = 1 if i==i'==... wait.

T[(i,k), (i_a, j_a), (j_b, k_b)] = 1 if i_a == i and j_a == j_b and k_b == k, else 0.

So for a fixed (i, j, k), there's one nonzero entry: T[(i,k), (i,j), (j,k)] = 1.

There are n*m*p = 2*2*13 = 52 such entries, each contributing a rank-1 tensor.

Now, can we do better? The key is that for a fixed (i,j), as k varies, we get 
T[(i,k), (i,j), (j,k)] = 1 for k=0..12. These are 13 rank-1 tensors that share the same 
(i,j) in the A-mode.

Similarly, for fixed (j,k), as i varies, we get 2 rank-1 tensors sharing (j,k) in B-mode.
For fixed (i,k), as j varies, we get 2 rank-1 tensors sharing (i,k) in C-mode.

The schoolbook gives rank 52. Can we combine some of these?

Consider fixing j and i. For j=0, i=0: we have terms T[(0,k), (0,0), (0,k)] for k=0..12.
These are 13 terms. In the A-mode, they all use coordinate (0,0). In the B-mode, they use 
(0,0), (0,1), ..., (0,12). In the C-mode, they use (0,0), (0,1), ..., (0,12).

This looks like a "block diagonal" structure. The rank of this sub-tensor (for fixed i=0, j=0) 
is 13 (it's essentially the identity on a 13-dim space in a specific sense).

Actually, I think the schoolbook rank of 52 might be close to optimal here, but let me think 
if there's a way to reduce it.

One approach: use the fact that for the 2x2 part, Strassen gives rank 7 instead of 4. 
But that's for square matrix multiplication. For 2x2 times 2x13, the structure is different.

Let me try a different approach. Consider that C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].

For i=0: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
For i=1: C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]

The bilinear map can be seen as: for each k, we compute two linear combinations.
The A-part has 4 parameters, the B-part has 26 parameters (13 for B[0,:] and 13 for B[1,:]).

Let me define:
u_k = B[0,k], v_k = B[1,k] for k=0..12
Then C[0,k] = a00*u_k + a01*v_k
     C[1,k] = a10*u_k + a11*v_k

This is like multiplying a 2x2 matrix A by a 2x13 matrix [u_0, u_1, ..., u_12; v_0, ..., v_12].

The rank of this bilinear map... I believe it's at most 4*13 = 52 (schoolbook), and possibly 
less. But I don't see an obvious Strassen-like improvement because the "inner dimension" is 2 
and the "outer dimension" is 13, which doesn't lend itself to the same kind of recursion.

However, we can use the following trick: express the 2x2 multiplication using 7 products 
(Strassen), and apply this to each "column block" appropriately. But since p=13 and m=2, 
the structure is a bit unusual.

Actually, let me just go with a careful schoolbook decomposition but check if any simplification 
is possible. Given the constraints, I'll output the schoolbook rank of 52.

Wait - let me reconsider. Is there a way to get rank less than 52?

The tensor is T[i,k, i',j', j'',k''] where it's 1 iff i=i', j'=j'', k=k''.

This is actually a very structured tensor. It's the "commutative" or "diagonal" type tensor.

I think for this specific problem, the rank is indeed n*m*p = 52, or possibly slightly less 
due to some linear dependencies, but it's hard to find them without explicit computation.

Let me just produce the schoolbook decomposition with rank 52.
"""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 52
    U = [[0] * rank for _ in range(n * m)]   # 4 x 52
    V = [[0] * rank for _ in range(m * p)]   # 26 x 52
    W = [[0] * rank for _ in range(p * n)]   # 26 x 52
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, col] = 1
                # V[j*p + k, col] = 1
                # W[k*n + i, col] = 1
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
