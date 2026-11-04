"""Block-based rank-reduced exact decomposition for 4x4 * 4x9 matrix multiplication.

Idea: Split the inner dimension j into two groups of 2, and the k dimension into groups.
For each (j-block, k-block) pair, multiply a 4x2 matrix by a 2x3 matrix (or 2x1).
- 4x2 * 2x3: rank 24 schoolbook. We can do better by writing 4x2 = [4x1; 4x1] structure... 
  Actually, let's just do direct schoolbook for the sub-blocks but count carefully.
  
Simpler and more reliable improvement: use the fact that 9 = 3+3+3.
For each of the 3 k-groups of size 3, we multiply A (4x4) by a (4x3) submatrix of B.
The product A*(4x3) gives a (4x3) block of C.
Schoolbook for 4x4 * 4x3 costs 4*4*3 = 48.
Total: 3 * 48 = 144. No improvement.

Better: split j into two groups of 2.
C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
Each term is (4x2) * (2x9) -> rank 4*2*9 = 72 each. Total 144. Still schoolbook.

The key is that simple block decomposition with schoolbook within blocks doesn't help.
We need genuine low-rank multiplications within subproblems.

Let's try: split i into two groups of 2, j into two groups of 2, k into three groups of 3.
That gives 2*2*3 = 12 subproblems of (2x2)*(2x3).
Each (2x2)*(2x3) has schoolbook rank 2*2*3=12.
Can we do (2x2)*(2x3) in less than 12?
(2x2)*(2x3) = [a b; c d] * [e f g; h i j]
Result: [ae+bh, af+bi, ag+bj; ce+dh, cf+di, cg+dj]
This is 6 outputs, each a sum of 2 products.
Total distinct bilinear forms needed: 
- For output (0,0): ae, bh
- For output (0,1): af, bi
- For output (0,2): ag, bj
- For output (1,0): ce, dh
- For output (1,1): cf, di
- For output (1,2): cg, dj

The products are: ae, bh, af, bi, ag, bj, ce, dh, cf, di, cg, dj
That's 12 distinct products. None are shared across outputs.
So schoolbook rank 12 is minimal for (2x2)*(2x3). No improvement there.

Let me try a different split. What about using Strassen on the 4x4 part and doing schoolbook for the 4x1 columns?
Write B = [B_4x4, B_4x4, B_4x1] (three blocks for the 9 columns, each... wait, 9 = 4+4+1? No, 9 = 4+4+1 = 9, yes).
Actually 9 = 4 + 4 + 1 works dimensionally.

C = A * B = [A * B1 | A * B2 | A * B3] where B1, B2 are 4x4 and B3 is 4x1.

If we could compute A*B1 and A*B2 each in rank <= 19 (Strassen's rank for 4x4), and A*B3 in rank 4 (schoolbook for 4x4*4x1 is 4*4*1=16, but can we do better? 4x4*4x1: C[i] = sum_j A[i,j] B[j]. This is just matrix-vector multiply. Rank is 16 (each output C[i] = sum_j A[i,j]*B[j], and all 16 products are distinct and needed). Actually, can we share? No, because B[j] appears in 4 different outputs (one per i), but A[i,j] is different for each i,j. The products are A[i,j]*B[j] for all i,j. These are 16 distinct bilinear forms. So rank 16.)

So if we had a rank-19 decomposition for 4x4*4x4, total would be 19+19+16 = 54.

But I don't have an explicit rank-19 decomposition readily available to code.

Alternative: let me just try the R=36 constructions more carefully and fix the verification to check ALL inconsistent indices, not just samples.

Actually, the current program's R=36 (n*p) construction:
r = i*p + k
U[i'*m+j, r] = 1 if i'=i
V[j'*p+k', r] = 1 if k'=k
W[k''*n+i'', r] = 1 if k''=k and i''=i

The issue: for a fixed (i,k), U has 1's in all rows a=i*m+j for j=0..3.
V has 1's in all rows b=j'*p+k for j'=0..3.
W has 1's only in row c=k*n+i.

So for consistent (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
= sum_{i',k'} [i'=i]*[k'=k]*[k''=k][i''=i] where r=(i',k')
= sum_{i',k'} [i'=i][k'=k][k=k][i=i] ... 

Wait, W[k*n+i, r] = 1 if k''=k_r and i''=i_r where r=(i_r,k_r).
So W[k*n+i, i'*p+k'] = 1 iff k'=k AND i'=i.

So for consistent (i,j,k):
sum_{i',k'} U[i*m+j, i'*p+k'] * V[j*p+k, i'*p+k'] * W[k*n+i, i'*p+k']
= sum_{i',k'} [i'=i] * [k'=k] * [k'=k and i'=i]
= sum_{i',k'} [i'=i][k'=k][k'=k][i'=i]
= 1 (only i'=i, k'=k contributes, and [i'=i][k'=k][k'=k][i'=i] = 1)

Wait that's correct! Let me re-check the inconsistent case.

Inconsistent (a,b,c): a=i1*m+j1, b=j2*p+k2, c=k3*n+i3
Sum over r=(i',k'):
U[a, r] = [i'=i1]
V[b, r] = [k'=k2]
W[c, r] = [k'=k3 and i'=i3]

Product = [i'=i1][k'=k2][k'=k3][i'=i3]

This is 1 iff i1=i3 and k2=k3.
If i1=i3 and k2=k3 but j1!=j2, the product is still 1, but T should be 0.

So yes, this construction fails. The problem is that U and V don't encode j.

What if we include j in r? Let r = j*n*p + i*p + k? That gives R = 4*4*9 = 144. Back to schoolbook.

Let me try: r = i*m + j... no.

What about a tensor-train or hierarchical approach?

Actually, let me try to just verify the current R=36 constructions with a FULL check (all inconsistent indices) and see if maybe I made an error in my analysis.

Let me also try: r indexes (j,k) with R=36, but with a smarter assignment:
U[i*m+j, j'*p+k'] = 1 if j'=j AND i = i0... no.

OK here's another idea: use the structure that T = I_n ⊗ I_m ⊗ I_p (tensor product of identity-like tensors). The rank of a tensor product T⊗T'⊗T'' is at most rank(T)*rank(T')*rank(T''), but this isn't tight.

For our tensor, T[i,j,k] = 1 iff indices consistent. This is essentially the "copy" tensor or something related.

Let me just code the full verification for the R=36 constructions and also try a few more ideas within the time budget.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110

    def build_schoolbook():
        R = n * m * p
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = build_schoolbook()

    def verify_full(U, V, W, R):
        """Verify exact decomposition by checking ALL indices."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Check all consistent indices
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_np[a], V_np[b] * W_np[c])
                    if abs(val - 1.0) > 1e-9:
                        return False

        # Check ALL inconsistent indices
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i1, j1 = divmod(a, m)
                    j2, k2 = divmod(b, p)
                    k3, i3 = divmod(c, n)
                    if j1 == j2 and i1 == i3 and k2 == k3:
                        continue
                    val = np.dot(U_np[a], V_np[b] * W_np[c])
                    if abs(val) > 1e-9:
                        return False
        return True

    # Try R = n*p = 36: r = i*p + k
    R1 = n * p
    U1 = [[0] * R1 for _ in range(n * m)]
    V1 = [[0] * R1 for _ in range(m * p)]
    W1 = [[0] * R1 for _ in range(p * n)]
    for i in range(n):
        for k in range(p):
            r = i * p + k
            for j in range(m):
                U1[i * m + j][r] = 1
            for j in range(m):
                V1[j * p + k][r] = 1
            W1[k * n + i][r] = 1
    if verify_full(U1, V1, W1, R1):
        return {"U": U1, "V": V1, "W": W1}

    # Try R = m*p = 36: r = j*p + k
    R2 = m * p
    U2 = [[0] * R2 for _ in range(n * m)]
    V2 = [[0] * R2 for _ in range(m * p)]
    W2 = [[0] * R2 for _ in range(p * n)]
    for j in range(m):
        for k in range(p):
            r = j * p + k
            for i in range(n):
                U2[i * m + j][r] = 1
            V2[j * p + k][r] = 1
            for i in range(n):
                W2[k * n + i][r] = 1
    if verify_full(U2, V2, W2, R2):
        return {"U": U2, "V": V2, "W": W2}

    # Try R = n*m = 16: r = i*m + j
    R3 = n * m
    U3 = [[0] * R3 for _ in range(n * m)]
    V3 = [[0] * R3 for _ in range(m * p)]
    W3 = [[0] * R3 for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U3[i * m + j][r] = 1
            for k in range(p):
                V3[j * p + k][r] = 1
            for k in range(p):
                W3[k * n + i][r] = 1
    if verify_full(U3, V3, W3, R3):
        return {"U": U3, "V": V3, "W": W3}

    # Try R = m*p = 36 with different structure: 
    # For each (j,k), U depends on i via a pattern
    # Actually try: r = j*4 + i ... no, R must be fixed.

    # Try a construction where R = 48: split into two groups of (2,4,9) or something
    # For now, try R = 36 with r = (i,j) mapped cleverly... R=16 done.

    # Try R = 36: r indexes (j, i) with j in range(4), i in range(4) -> R=16, already tried.
    # Try R = 72: maybe not helpful.

    # Let's try a random search for small R with small integer entries
    # This is unlikely to find anything but let's try R=120, 100, 90 with random +/-1,0
    import random
    random.seed(123)

    def try_random(R, n_trials=20):
        for trial in range(n_trials):
            if time.time() > deadline - 20:
                return None
            U_r = np.random.randint(0, 2, size=(n * m, R))
            V_r = np.random.randint(0, 2, size=(m * p, R))
            W_r = np.random.randint(0, 2, size=(p * n, R))
            # Quick check on a subset
            ok = True
            for _ in range(20):
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                a, b, c = i*m+j, j*p+k, k*n+i
                val = np.dot(U_r[a], V_r[b] * W_r[c])
                if abs(val - 1.0) > 1e-6:
                    ok = False
                    break
            if not ok:
                continue
            # Full verification
            if verify_full(U_r.tolist(), V_r.tolist(), W_r.tolist(), R):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
        return None

    for R_try in [120, 110, 100, 96, 90, 80]:
        if time.time() > deadline - 30:
            break
        result = try_random(R_try, n_trials=10)
        if result is not None:
            return {"U": result[0], "V": result[1], "W": result[2]}

    # Fallback to schoolbook
    if time.time() > deadline:
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    return {"U": U_sb, "V": V_sb, "W": W_sb}
