"""Block-structured low-rank decomposition for 4x5x6 matrix multiplication.

Strategy:
- Use 2x2x2 Winograd (rank 7) blocks for all (i,j,k) where i,j,k are in even
  positions or covered by 2-wide blocks.
- For the remaining dimensions (j=4, k=4,5), use 1x2x2 and 2x2x2 blocks
  where possible, and schoolbook for the rest.

This should give rank = 2*2*2*7 + (remainder terms)
Actually let's compute:
- i: 4 -> 2 blocks of 2
- j: 5 -> 2 blocks of 2 (covering j=0,1 and j=2,3) + 1 single (j=4)
- k: 6 -> 3 blocks of 2 (covering k=0,1; k=2,3; k=4,5)

For the 2x2x2 blocks (i-block, j-block, k-block): 2*2*3 = 12 blocks * 7 = 84
For j=4 with i-blocks and k-blocks: 2*1*3 = 6 blocks. Each is 2x1x2.
  A 2x1x2 matmul: A is 2x1, B is 1x2, C is 2x2. This is just 2 scalar products
  for each column of C, so rank = 2*2 = 4 per block? No, it's 2x1x2 tensor.
  Actually 2x1x2: sum over j (only j=4), so C[i,k] = A[i,4]*B[4,k].
  This is a rank-2 tensor (outer product of A column and B row). But we need
  to decompose it into rank-1 terms. The tensor is A[:,4] * B[4,:].
  This is rank 1 if we treat it as a single term? No, C[i,k] = A[i,4]*B[4,k].
  This is the outer product of vector A[:,4] (length 2) and B[4,:] (length 2).
  As a bilinear form, this is rank 1: one term with U[i,4]=A[i,4], V[4,k]=B[4,k], W[k,i]=1.
  Wait, the decomposition is C[i,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c].
  For a fixed j=4, C[i,k] = U[i*5+4,c] * V[4*6+k,c] * W[k*4+i,c].
  If we set one term c with U[i*5+4,c]=1 for i in block, V[4*6+k,c]=1 for k in block, W[k*4+i,c]=1...
  But that gives C[i,k] += 1*1*1 = 1 for all i,k in the block, which is wrong.
  
  Actually for the 2x1x2 case (i in {2ib,2ib+1}, j=4, k in {2kb,2kb+1}):
  C[i,k] = A[i,4] * B[4,k]
  We can use 2 terms:
  Term 1: i=2ib, k=2kb: U[2ib*5+4]=1, V[4*6+2kb]=1, W[2kb*4+2ib]=1 -> contributes to C[2ib,2kb]
  Term 2: i=2ib+1, k=2kb+1: ... no that's not right.
  
  Actually the simplest is schoolbook for each (i,4,k) triple. That's 4*3=12 terms for j=4.
  But we can do better. For fixed j=4, the sub-problem is 4x1x6 -> 4x6.
  C[i,k] = A[i,4]*B[4,k]. This is a rank-1 matrix (outer product). 
  To decompose into rank-1 tensor terms: we need C[i,k] = sum_c U[i,4,c]*V[4,k,c]*W[k,i,c].
  If we set W[k,i,c] = 1 for all k,i in a block, and U[i,4,c] and V[4,k,c] appropriately...
  Actually, the tensor T[i,k] = A[i,4]*B[4,k] for i in I, k in K.
  This tensor has rank 1 (as a matrix). But in the tensor decomposition sense,
  we're looking at the bilinear complexity. The bilinear complexity of a rank-1 matrix
  multiplication (where one factor is a vector) is just the number of entries? No.
  
  Let me just use schoolbook for the remainder to be safe. The main gain is from
  the 2x2x2 Winograd blocks.

Total rank estimate:
- 2x2x2 blocks: 2 (i) * 2 (j) * 3 (k) = 12 blocks * 7 = 84
- Remainder: j=4 (all i,k): 4*6 = 24 schoolbook terms
- Total: 84 + 24 = 108

That's the same as before. Let me think if we can do better for the j=4 part.

Actually, for j=4, we have a 4x6 output where C[i,k] = A[i,4]*B[4,k].
This is just an outer product. The bilinear complexity of computing A[i,4]*B[4,k]
for all i,k is just 1 if we could do it as one multiplication, but we need separate
outputs for each (i,k). 

In tensor decomposition terms, the tensor for this sub-problem is:
T[i,k] = e_i^T * (a * b^T) * e_k = (e_i^T a) * (b^T e_k)
This is a rank-1 tensor in the (i,k) space, but our decomposition has the form
sum_c U[i,4,c] * V[4,k,c] * W[k,i,c].

If we set W[k,i,c] = delta_{k,i} or something... no.

Actually, the standard way: for each (i,k) pair, we need one term unless we can
share. The tensor T[i,k] = a_i * b_k. This can be written as sum over c of
U_i,c * V_k,c * W_k,i,c.

If we set W[k,i,c] = 1 for all, then we need U_i,c * V_k,c = a_i * b_k.
This is possible with rank 1: U_i,0 = a_i, V_k,0 = b_k, W_k,i,0 = 1.
But a_i and b_k are the input values, not constants! The decomposition must work
for ALL inputs, so U, V, W must be constant matrices.

So we can't use a_i and b_k in U and V. We need:
C[i,k] = sum_c U[i,4,c] * V[4,k,c] * W[k,i,c] = A[i,4] * B[4,k] for all A, B.

This means: for all A, B, sum_c U[i,4,c] * V[4,k,c] * W[k,i,c] = A[i,4] * B[4,k].

This is only possible if the map (A,B) -> C is a bilinear map, and we're looking
for a rank-1 decomposition of this bilinear map. The bilinear map here is
(A,B) -> A[i,4]*B[4,k] for each (i,k). 

The bilinear complexity of computing all C[i,k] = A[i,4]*B[4,k] is 24 (one per
output), because each output depends on a distinct pair (A[i,4], B[4,k]) and there's
no sharing possible since each output uses a different A entry and different B entry.

Wait, actually there IS sharing. A[i,4] is used in 6 outputs (one for each k),
and B[4,k] is used in 4 outputs (one for each i). But in a rank-1 term, we multiply
one A-entry (linear combo), one B-entry (linear combo), and one C-entry (linear combo).

For the sub-problem j=4: we have inputs A[i,4] for i=0..3 and B[4,k] for k=0..5.
Outputs C[i,k] for i=0..3, k=0..5.
C[i,k] = A[i,4] * B[4,k].

A rank-1 term: (linear combo of A's) * (linear combo of B's) * (linear combo of C's).
Let u_i = U[i,4,c], v_k = V[4,k,c], w_{k,i} = W[k,i,c].
Then the term contributes u_i * v_k * w_{k,i} to C[i,k].
We need sum_c u_i * v_k * w_{k,i} = A[i,4] * B[4,k] for all A, B.

This means: for each (i,k), sum_c u_i * v_k * w_{k,i} must equal A[i,4]*B[4,k].
Since this must hold for all A, B, we need:
- The coefficient of A[i,4]*B[4,k'] is 0 for k' != k
- The coefficient of A[i',4]*B[4,k] is 0 for i' != i
- The coefficient of A[i,4]*B[4,k] is 1

This forces w_{k,i,c} to be zero unless (i,k) is the specific pair, and then u_i * v_k * w_{k,i} = 1.
So each (i,k) pair needs its own term. Rank = 24 for j=4.

So total rank = 84 + 24 = 108. Same as before.

Let me just implement this cleanly and verify.
"""
import time
import sys

DIMENSIONS = [4, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0


def remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)


def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W, "rank": rank}


def _verify(U, V, W, rank):
    import random
    n, m, p = DIMENSIONS
    for _ in range(3):
        if remaining() <= 0:
            return False
        A = [[random.randint(-2, 2) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-2, 2) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0
                for j in range(m):
                    for c in range(rank):
                        u = U[i * m + j][c]
                        if u == 0:
                            continue
                        acc += u * W[k * n + i][c] * V[j * p + k][c]
                if acc != C[i][k]:
                    return False
    return True


def _winograd_2x2x2():
    """Returns 7 rank-1 terms for 2x2x2 matmul."""
    Uterms = []
    Vterms = []
    Wterms = []
    
    # p1=(a+d)(e+h)
    Uterms.append([1, 0, 0, 1])
    Vterms.append([1, 0, 0, 1])
    Wterms.append([1, 0, 0, 1])
    
    # p2=(c+d)e
    Uterms.append([0, 0, 1, 1])
    Vterms.append([1, 0, 0, 0])
    Wterms.append([0, 0, 1, -1])
    
    # p3=a(f-h)
    Uterms.append([1, 0, 0, 0])
    Vterms.append([0, 1, 0, -1])
    Wterms.append([0, 1, 0, 0])
    
    # p4=d(g-e)
    Uterms.append([0, 0, 0, 1])
    Vterms.append([-1, 0, 1, 0])
    Wterms.append([-1, 1, 1, 0])
    
    # p5=(a+b)g
    Uterms.append([1, 1, 0, 0])
    Vterms.append([0, 0, 1, 0])
    Wterms.append([1, 1, 0, 0])
    
    # p6=(c-a)(e+f)
    Uterms.append([-1, 0, 1, 0])
    Vterms.append([1, 1, 0, 0])
    Wterms.append([0, 0, -1, 1])
    
    # p7=(b-d)(f+h)
    Uterms.append([0, 1, 0, -1])
    Vterms.append([0, 1, 0, 1])
    Wterms.append([-1, 1, 1, 0])
    
    return Uterms, Vterms, Wterms


def _block_decomposition():
    n, m, p = DIMENSIONS
    terms_U = []
    terms_V = []
    terms_W = []
    
    Ut, Vt, Wt = _winograd_2x2x2()
    
    # 2x2x2 blocks: i in {0,1}x{2,3}, j in {0,1}x{2,3}, k in {0,1}x{2,3}x{4,5}
    for ib in range(2):
        for jb in range(2):
            for kb in range(3):
                for u, v, w in zip(Ut, Vt, Wt):
                    ufull = [0] * (n * m)
                    vfull = [0] * (m * p)
                    wfull = [0] * (p * n)
                    for a in range(2):
                        for b in range(2):
                            ui = (2 * ib + a) * m + (2 * jb + b)
                            ufull[ui] += u[a * 2 + b]
                    for b in range(2):
                        for c in range(2):
                            vi = (2 * jb + b) * p + (2 * kb + c)
                            vfull[vi] += v[b * 2 + c]
                    for c in range(2):
                        for a in range(2):
                            wi = (2 * kb + c) * n + (2 * ib + a)
                            wfull[wi] += w[c * 2 + a]
                    terms_U.append(ufull)
                    terms_V.append(vfull)
                    terms_W.append(wfull)
    
    # Remainder: j=4, all i, k
    for i in range(n):
        for k in range(p):
            ufull = [0] * (n * m)
            vfull = [0] * (m * p)
            wfull = [0] * (p * n)
            ufull[i * m + 4] = 1
            vfull[4 * p + k] = 1
            wfull[k * n + i] = 1
            terms_U.append(ufull)
            terms_V.append(vfull)
            terms_W.append(wfull)
    
    rank = len(terms_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c, (u, v, w) in enumerate(zip(terms_U, terms_V, terms_W)):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    return {"U": U, "V": V, "W": W, "rank": rank}


def construct():
    n, m, p = DIMENSIONS
    best = None
    
    try:
        if remaining() > 10:
            cand = _block_decomposition()
            if remaining() > 5 and _verify(cand["U"], cand["V"], cand["W"], cand["rank"]):
                best = cand
    except Exception:
        pass
    
    if best is None:
        best = _schoolbook()
    
    del best["rank"]
    return best


if __name__ == "__main__":
    r = construct()
    print("U shape:", len(r["U"]), "x", len(r["U"][0]))
    print("rank:", len(r["U"][0]))
