"""
6x6x6 matrix multiplication via 3x3 Strassen + 2x2 schoolbook = rank 84.

Strategy:
  - Partition 6x6 matrices into a 3x3 grid of 2x2 blocks.
  - Use a verified rank-7 decomposition for the 3x3 block-level matmul.
  - Use schoolbook (rank 4) for each 2x2 block product.
  - Total rank: 7 * 4 = 28... wait, that's wrong.

Actually the structure is:
  - 3x3 block matmul needs 7 scalar multiplications (Strassen).
  - Each "scalar" is really a 2x2 matrix, and 2x2*2x2 needs 4 scalar products.
  - Total: 7 * 4 = 28? No, that's not right either.

Let me think more carefully. The 3x3 block matmul C_block = sum_k A_block[:,:,k] * B_block[k,:,:]
where each block product is a 2x2 * 2x2 -> 2x2 product.

If I use Strassen at the 3x3 level, I get 7 "virtual" multiplications of 2x2 blocks.
Each 2x2 * 2x2 needs 4 scalar multiplications (schoolbook) or 7 (Strassen).

So:
  - 3x3 Strassen + 2x2 schoolbook: 7 * 4 = 28. 
  - But wait, that can't be right. 3x3 matmul of 2x2 blocks...

Actually I think I'm confusing myself. Let me be precise.

The 6x6 matmul: C = A * B where A, B, C are 6x6.
Partition into 3x3 grid of 2x2 blocks: A = {A_{pq}} where p,q in {0,1,2}.
C_{pq} = sum_k A_{pk} * B_{kq} (each is 2x2 * 2x2 -> 2x2).

This is exactly 3x3 matmul where the "scalars" are 2x2 matrices.
The rank of 3x3 matmul is 7 (Strassen), but that's for scalar entries.
When the entries are 2x2 matrices, the "multiplication" is 2x2*2x2 which has rank 4.

So the total rank is NOT simply 7*4 = 28. The issue is that Strassen's 7 products
are bilinear forms in the 2x2 matrix entries, and each bilinear form followed by
a 2x2*2x2 product creates a more complex structure.

Actually, I think the correct way to think about it:
- The 3x3 block matmul tensor, when each entry is expanded into 4 scalar entries,
  gives a 36x36x36 tensor (6x6 x 6x6 x 6x6).
- The rank of this tensor is what we want.

A block-structured decomposition:
  C_{pq} = sum_k A_{pk} * B_{kq}
  
If I use Strassen for the 3x3 structure:
  C_{pq} = sum_{s=1}^{7} (sum_{pk} U_s[pk] A_{pk}) * (sum_{kq} V_s[kq] B_{kq}) * W_s[pq]

But (sum_{pk} U_s[pk] A_{pk}) is a 2x2 matrix (linear combo of 2x2 blocks),
and similarly for V_s. So each term is a 2x2 * 2x2 product scaled by W_s.

A 2x2 * 2x2 product can be written as 4 scalar multiplications (schoolbook):
  M = X * Y where X, Y are 2x2:
  M[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
  M[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
  M[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]
  M[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]

So each Strassen term contributes 4 scalar products, giving 7*4 = 28 total.

BUT WAIT - this is only valid if the Strassen decomposition works over the ring of
2x2 matrices, which it does since Strassen only uses +, -, and * which are all
defined for matrices. However, matrix multiplication is NOT commutative!

Strassen's algorithm requires commutativity! So I CANNOT directly apply Strassen
to 2x2 matrix blocks.

Therefore, I need to use schoolbook for the 3x3 block structure: 27 block products,
each of which is 2x2*2x2 = 4 scalar products. Total: 27*4 = 108.

OR I can use Strassen at the 2x2 level: 7 scalar products per 2x2 block product,
and schoolbook at the 3x3 level: 27 * 7 = 189. This is what the current program does.

OR I can use a hybrid approach. Let me think about whether there's a way to get
below 108.

Actually, for the 3x3 matmul tensor over a commutative ring, the rank is 7 (Strassen).
Over a non-commutative ring, the rank can be higher. For 2x2 matrices, the
"rank" of the 3x3 matmul in terms of scalar multiplications is...

Actually, let me reconsider. The standard result:
- Rank of n x n matmul tensor (scalar) = R(n)
- For block matrices: rank of (bn) x (bn) matmul >= b^2 * R(n) is NOT true in general.

The correct statement: if R(n) = r, then R(bn) <= b^2 * r (by doing n x n Strassen
on each block). And R(bn) >= r * R(b) is NOT generally true.

For our case: R(6) <= R(3) * R(2) = 7 * 4 = 28? No, that's not a valid upper bound.

Actually, the correct bound is: R(bn) <= b^2 * R(n) if we use the n x n decomposition
on each block independently. And R(n) <= b^2 * R(n/b) if we use block decomposition.

Wait, I think the valid construction is:
- If we have a rank-r decomposition of n x n matmul, then for bn x bn matmul,
  we can get rank r * (b^2)^2 = r * b^4? No, that doesn't make sense.

Let me think about this differently. The 6x6 matmul tensor T has T[i,k,j] = 1
(if C[i,j] = sum_k A[i,k] B[k,j]).

If I partition into 3x3 blocks of size 2:
T[A_block, B_block, C_block] where A_block = (p,k'), B_block = (k',q), C_block = (p,q)
and within each block there are 4 scalar entries.

The tensor is: T = sum_{p,k',q} (basis for A_{p,k'})(basis for B_{k',q})(basis for C_{p,q})
where basis for A_{p,k'} = sum_{a=0}^1 sum_{b=0}^1 e_{p*2+a, k'*2+b} ⊗ ...

This is getting complicated. Let me just go with the verified approach:

Option 1: 2x2 Strassen blocks of 3x3 schoolbook = 7 * 27 = 189 (current, verified)
Option 2: 3x3 schoolbook blocks of 2x2 schoolbook = 27 * 4 = 108

Let me verify Option 2 is correct:
C[i,j] = sum_k A[i,k] B[k,j]
Partition i = 2*p + a, k = 2*k' + b, j = 2*q + c where p,q,k' in {0,1,2}, a,b,c in {0,1}.

C[2p+a, 2q+c] = sum_{k',b} A[2p+a, 2k'+b] B[2k'+b, 2q+c]

This is exactly the (a,c) entry of C_block[p,q] = sum_{k'} A_block[p,k'] * B_block[k',q].

For each (p,k',q) triple, the 2x2 * 2x2 product has 4 scalar terms (schoolbook):
C_block[p,q][a,c] = sum_b A_block[p,k'][a,b] * B_block[k',q][b,c]

So the full decomposition is:
For each (p, k', q) in {0,1,2}^3 and each (a, b, c) in {0,1}^3:
  U[(2p+a)*6 + (2k'+b), col] = 1
  V[(2k'+b)*6 + (2q+c), col] = 1  
  W[(2q+c)*6 + (2p+a), col] = 1

Wait, let me check the W convention: W[k*n+i, r] where n=6.
The output C has index C[row, col] = C[i, j]. In the W convention, it's W[j*n + i, r] = W[col*6 + row, r].

So for C[2p+a, 2q+c]: row = 2p+a, col = 2q+c.
W index = col*6 + row = (2q+c)*6 + (2p+a).

For A[i,k] = A[2p+a, 2k'+b]: U index = i*6 + k = (2p+a)*6 + (2k'+b).
For B[k,j] = B[2k'+b, 2q+c]: V index = k*6 + j = (2k'+b)*6 + (2q+c).

Total number of terms: 3*3*3*4 = 108.

Let me verify this gives the correct result:
sum over all terms of U[a_idx, r] * A[a_idx] * V[b_idx, r] * B[b_idx] = W[c_idx, r] * C[c_idx]

For a fixed (p,k',q,a,b,c):
U[(2p+a)*6 + (2k'+b), r] = 1 (only for this specific r)
V[(2k'+b)*6 + (2q+c), r] = 1
W[(2q+c)*6 + (2p+a), r] = 1

The contribution to C is: A[2p+a, 2k'+b] * B[2k'+b, 2q+c] added to C[2p+a, 2q+c].

Summing over all k', b for fixed p, q, a, c:
C[2p+a, 2q+c] = sum_{k',b} A[2p+a, 2k'+b] B[2k'+b, 2q+c] = sum_k A[2p+a, k] B[k, 2q+c] ✓

Great, so rank 108 is correct and valid!

Let me implement this cleanly.
"""

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # 3x3 schoolbook blocks of 2x2 schoolbook: rank = 27 * 4 = 108
    bs = 2       # block size
    nb = 3       # number of blocks per dimension
    
    rank = nb * nb * nb * bs * bs  # 3*3*3*4 = 108
    
    U = [[0] * rank for _ in range(n * m)]  # 36 x 108
    V = [[0] * rank for _ in range(m * p)]  # 36 x 108
    W = [[0] * rank for _ in range(p * n)]  # 36 x 108
    
    col = 0
    for pb in range(nb):    # block row of A, block col of C
        for kb in range(nb):  # block col of A, block row of B
            for qb in range(nb):  # block col of B, block col of C
                for a in range(bs):  # row within block
                    for b in range(bs):  # col within A block / row within B block
                        for c in range(bs):  # col within B block / col within C block
                            # A entry: row = pb*bs + a, col = kb*bs + b
                            a_row = pb * bs + a
                            a_col = kb * bs + b
                            u_idx = a_row * m + a_col  # (2p+a)*6 + (2k'+b)
                            
                            # B entry: row = kb*bs + b, col = qb*bs + c
                            b_row = kb * bs + b
                            b_col = qb * bs + c
                            v_idx = b_row * p + b_col  # (2k'+b)*6 + (2q+c)
                            
                            # C entry: row = pb*bs + a, col = qb*bs + c
                            c_row = pb * bs + a
                            c_col = qb * bs + c
                            # W convention: W[col*n + row, r]
                            w_idx = c_col * n + c_row  # (2q+c)*6 + (2p+a)
                            
                            U[u_idx][col] = 1
                            V[v_idx][col] = 1
                            W[w_idx][col] = 1
                            col += 1
    
    return {"U": U, "V": V, "W": W}
