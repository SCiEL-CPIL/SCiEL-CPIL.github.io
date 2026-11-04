"""Direct approach: exploit the specific 2x3x8 structure with Strassen-style decomposition.

The key insight: multiply a 2x3 matrix by a 3x8 matrix.
We can view this as:
- Split the 3x8 matrix into columns. Group columns into pairs.
- Use Strassen-like tricks to reduce the rank below the schoolbook 48.

Let's think about the structure:
- A is 2x3, B is 3x8, C is 2x8
- C[i,k] = sum_{a=0}^{2} A[i,a] * B[a,k]

Direct decomposition: rank = 2*3*8 = 48

Can we do better? Let's try to use the fact that A only has 2 rows.
For each pair of columns (k, k+1) in B, we need:
C[i, k]   = A[i,0]*B[0,k]   + A[i,1]*B[1,k]   + A[i,2]*B[2,k]
C[i, k+1] = A[i,0]*B[0,k+1] + A[i,1]*B[1,k+1] + A[i,2]*B[2,k+1]

For a fixed pair of columns, this is like multiplying a 2x3 by a 3x2.
The tensor for 2x3 x 3x2 has rank... let's think.
Actually, we can precompute combinations.

Alternative: think of it as computing 8 dot products of length 3, but each involves the same A rows.
C[:,k] = A @ B[:,k] where A is 2x3 and B[:,k] is 3x1.

This is essentially 8 independent 2x3 @ 3x1 multiplications, but they share A.

Let's try a different grouping. Consider the 3x8 matrix B. We can split its 3 rows.
Actually, let's try to use bilinear tricks on the "middle" dimension of size 3.

For a fixed output column k, C[:,k] depends on A (2x3) and B[:,k] (3x1).
The map (A, B[:,k]) -> C[:,k] is bilinear in (A, B[:,k]).

Let's define the tensor T with indices (i, a, k) where i in {0,1}, a in {0,1,2}, k in {0..7}.
T[i,a,k] = 1 for all valid indices.

We want to write T as a sum of rank-1 tensors: T[i,a,k] = sum_r U[i,a,r] * V[a,k,r] * W[i,k,r]
Wait, the convention is U has shape (n*m, rank) = (6, rank), V has shape (m*p, rank) = (24, rank), W has shape (p*n, rank) = (16, rank).

The indices are:
U[i*m + a, r]
V[a*p + k, r]
W[k*n + i, r]

And we need sum_r U[i*m+a, r] * V[a*p+k, r] * W[k*n+i, r] = 1 for all i,a,k.

Let's try a direct construction. The schoolbook gives rank 48.
Can we reduce it? 

Consider combining terms. Notice that for a fixed a, the terms are:
sum_i sum_k U[i*m+a, r] * V[a*p+k, r] * W[k*n+i, r]

This doesn't immediately suggest a reduction. Let's try a small experiment with a 2x2x2 Strassen-like approach on subsets.

Actually, let's reconsider the problem. The dimensions are 2, 3, 8.
The schoolbook rank is 2*3*8 = 48.

Is there a known lower bound or a known decomposition with rank < 48?
For matrix multiplication, the rank of the 2x3x3 tensor is 7 (Strassen-like).
For 2x3x8, we can think of it as 4 blocks of 2x3x2.
If each 2x3x2 block has rank 5 or 6, then total rank would be 4*5=20 or 4*6=24.
But the blocks share the A matrix, so we can't just multiply independently.

Let's try a different approach. Use the fact that the middle dimension is 3.
We can express A as a sum of 3 rank-1 matrices, but that doesn't help directly.

Let's try to use a bilinear form trick. Define:
For each r, we choose U_r (6x1), V_r (24x1), W_r (16x1) such that the sum works.

Actually, let's just implement the schoolbook and see if we can verify it works, then try to optimize.
The current program already does this with rank 44. Let's see if we can do better.

Wait, the current program uses rank 44. How? It uses Strassen for 2x2 blocks.
Part 1: A1 (2x2) * B1 (2x8). B1 is split into 4 blocks of 2 columns. Each 2x2x2 block uses Strassen (rank 7). So 4*7 = 28.
Part 2: A2 (2x1) * B2 (1x8). This is 2*8 = 16.
Total: 28 + 16 = 44.

Can we do better? What if we use a different split?
What if we split A into 2x1 and 2x1 and 2x1? Then 3 blocks of 2x1x8. Each is 16, total 48. Worse.

What if we use a more clever decomposition for 2x3x8?
The tensor rank of 2x3x3 is 7. For 2x3x8, if we could decouple the 8 dimension, we'd get 4 * rank(2x3x2).
What is rank(2x3x2)? It's the rank of multiplying 2x3 by 3x2.
This is equivalent to the rank of the matrix multiplication tensor for these dimensions.
For 2x3x2, the schoolbook rank is 12. Can we do better?
Actually, 2x3x2 is the same as 2x2x3 by symmetry. The rank of 2x2x3 is 7 (same as 2x2x2 Strassen? No, 2x2x2 is 7, 2x2x3 is also 7 I think).

Let's assume rank(2x2x3) = 7. Then 4 * 7 = 28. But we need to account for the shared A.
The issue is that the 4 blocks of B share the same A. So we can't just use 4 independent Strassen decompositions.

Let's try a different strategy. Use a "block" Strassen where we treat the 8 columns as 4 pairs, and use a 2x2 Strassen on the column pairs, but with a 2x3 A.
This is getting complex. Let's just implement a clean schoolbook and a verified Strassen-based approach, and pick the best.

Actually, the current program already scores 1.0909 with rank 44. Let's see if we can get rank 42 or 40.

Another idea: use a decomposition based on the fact that 3 = 2 + 1.
Split A into A_left (2x2) and A_right (2x1).
C = A_left @ B_left + A_right @ B_right
where B_left is 2x8 (first 2 rows of B) and B_right is 1x8 (last row of B).

For A_left @ B_left: this is 2x2 @ 2x8. We can split B_left into 4 blocks of 2x2. Each block uses Strassen (rank 7). Total: 28.
For A_right @ B_right: this is 2x1 @ 1x8. Rank 16.
Total: 44. Same as before.

What if we use a different split for the 2x2 @ 2x8?
Instead of 4 independent Strassen, can we use a global decomposition?
The tensor for 2x2 @ 2x8 is the same as 4 copies of 2x2 @ 2x2, but with shared A.
So it's not simply 4*7.

Let's try a bilinear form approach. The tensor is T[i, a, k] for i in {0,1}, a in {0,1,2}, k in {0..7}.
We can write this as a sum over r of U[i,a,r] V[a,k,r] W[i,k,r].

Let's try to find a decomposition with rank 42.
Idea: use 6 rank-1 terms for the "a=2" part (instead of 8), by combining k values.
For a=2: we need U[i,2,r] V[2,k,r] W[i,k,r] to sum to 1 for all i,k.
If we use 6 terms, we need V[2,k,r] to have some structure.
For example, let r=0..5. Let V[2,0,r]=1, V[2,1,r]=1, V[2,2,r]=1, V[2,3,r]=1, V[2,4,r]=1, V[2,5,r]=1, and V[2,6,r]=V[2,7,r]=0.
Then we still need to cover k=6,7. So we need 8 terms for a=2. No gain.

What if we combine a=1 and a=2?
This is getting too speculative. Let's just implement a robust solution with rank 44 and verify it, then try to squeeze out 1-2 more.

Actually, let's try a completely different decomposition.
Use the fact that the matrix multiplication can be viewed as a bilinear form on the space of 2x3 matrices and 3x8 matrices.
The dimension of the space of 2x3 matrices is 6, and 3x8 is 24. The output is 16.
The rank is the minimum number of rank-1 bilinear forms needed.

Let's just go with the known good decomposition of rank 44 and ensure it's correct.
The current program does this. Let me rewrite it cleanly.
"""
import time

def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += u_row[c] * v_row[c] * w_row[c]
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2
    S_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i*2 + a]
                        v_val = S_V[r][a*2 + k]
                        w_val = S_W[r][k*2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: A1 (2x2) * B1 (2x8), split B1 into 4 blocks of 2 columns
    for block_idx in range(4):
        k0 = block_idx * 2
        k1 = k0 + 1
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val == 0:
                        continue
                    u_idx = li * m + la
                    U[u_idx][col] += u_val

            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val == 0:
                        continue
                    v_idx = la * p + (k0 + lk)
                    V[v_idx][col] += v_val

            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val == 0:
                        continue
                    w_idx = (k0 + lk) * n + li
                    W[w_idx][col] += w_val

            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A2 (2x1) * B2 (1x8)
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}

    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()
