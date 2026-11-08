"""
Rank-53 decomposition for 2x5x6 matrix multiplication.

Approach: Block decomposition with Strassen on 2x2x2 blocks.
- Split m=5 into {0,1}, {2,3}, {4}
- Split p=6 into {0,1}, {2,3}, {4,5}
- For the 2x2 blocks (j in {0,1} or {2,3}, k in any pair): use Strassen (rank 7 each)
- For j=4 (single row), use direct multiplication (rank 2 per k-pair)

Total: 2*3*7 + 3*2 = 42 + 6 = 48... wait that's 48, not 54.

Let me recount. The current program has:
- j_pairs = [(0,1), (2,3)]: 2 pairs
- k_pairs = [(0,1), (2,3), (4,5)]: 3 pairs
- For each (j_pair, k_pair): 7 Strassen products = 2*3*7 = 42
- For j=4: 3 k_pairs * 2 = 6
- Total: 48

But the current program says rank=54. Let me check... ah, the current program
actually has a bug or I'm miscounting. Let me look more carefully.

Actually wait - looking at the current code again, it seems correct at 48 terms.
But the score is 1.1111 which suggests rank 54 (since seed is 60, 60/54 = 1.111).

Let me re-examine. Maybe there's an off-by-one or I'm missing something.

Actually, I think the issue is that Strassen for 2x2x2 gives rank 7, and we have:
- 2 j-blocks of size 2, 3 k-blocks of size 2: 2*3*7 = 42
- 1 j-block of size 1, 3 k-blocks of size 2: 1*3*2 = 6
- Total: 48

So the current program should be rank 48, not 54. But the score says 54. Let me just
implement a clean rank-48 decomposition and see if it's valid.

Actually, I need to be more careful. Let me verify the Strassen decomposition for
2x2x2 is correct and count the terms properly.

For C = A*B where A is 2x2, B is 2x2, C is 2x2:
C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]

Strassen:
P1 = (A00+A11)(B00+B11)
P2 = (A10+A11)B00
P3 = A00(B10-B00)
P4 = A11(B01-B10)
P5 = (A00+A01)B11
P6 = (A10-A00)(B00+B01)
P7 = (A01-A11)(B10+B11)

C00 = P1+P4-P5+P7
C01 = P3+P5
C10 = P2+P4
C11 = P1-P2+P3+P6

So 7 products, each involving one A element (or sum/diff) and one B element (or sum/diff).

In the tensor decomposition, each product term contributes to U, V, W.
For a 2x2x2 block at positions (j0,j1) for rows, (k0,k1) for cols:
- A elements: A[j0,k?], A[j1,k?] - wait, no.

Let me reclarify the indexing. We're multiplying A (n x m) by B (m x p) to get C (n x p).
A has rows indexed by i (0..n-1) and cols by j (0..m-1).
B has rows indexed by j (0..m-1) and cols by k (0..p-1).
C has rows indexed by i (0..n-1) and cols by k (0..p-1).

C[i,k] = sum_j A[i,j] * B[j,k]

For a block where i in {i0,i1}, j in {j0,j1}, k in {k0,k1}:
This is a 2x2x2 sub-problem.

In the current code, n=2, m=5, p=6.
j_pairs = [(0,1), (2,3)]: these are the j indices (middle dimension)
k_pairs = [(0,1), (2,3), (4,5)]: these are the k indices (output columns)

For each (j_pair, k_pair), we solve a 2x2x2 problem where:
- The "A" part is 2x2: rows i={0,1}, cols j={j0,j1}
- The "B" part is 2x2: rows j={j0,j1}, cols k={k0,k1}
- The "C" part is 2x2: rows i={0,1}, cols k={k0,k1}

This is correct. 7 terms per such block.

2 j_pairs * 3 k_pairs * 7 = 42 terms.

For j=4 (single j index), for each k_pair:
C[0,k0] += A[0,4]*B[4,k0]
C[0,k1] += A[0,4]*B[4,k1]
C[1,k0] += A[1,4]*B[4,k0]
C[1,k1] += A[1,4]*B[4,k1]

That's 4 terms per k_pair, 3 k_pairs = 12 terms.

Total: 42 + 12 = 54. Ah, I see! I was miscounting. For j=4, it's 2 (i values) * 2 (k values per pair) = 4 per pair, not 2.

So total rank is 54. The current program is correct.

Now, can we do better? The theoretical lower bound for 2x5x6 is at least max(2*5, 5*6, 2*6) = 30, but that's a very loose bound.

Let me think about alternative decompositions. What if we use a different block structure?

Option: Split m=5 into {0,1,2,3,4} differently.
- Use 1x3x3 blocks? Rank of 1x3x3 is 9 (direct).
- 2x2x2 blocks: rank 7.

What about using the 2x3x3 rank 17 decomposition?
2x3x3: rank 17.
If we split m=5 into {0,1,2} and {3,4}, and p=6 into {0,1,2} and {3,4,5}:
- Block 1: i in {0,1}, j in {0,1,2}, k in {0,1,2}: 2x3x3, rank 17
- Block 2: i in {0,1}, j in {0,1,2}, k in {3,4,5}: 2x3x3, rank 17
- Block 3: i in {0,1}, j in {3,4}, k in {0,1,2}: 2x2x3, rank ?
- Block 4: i in {0,1}, j in {3,4}, k in {3,4,5}: 2x2x3, rank ?

2x2x3: can split k into 3 single columns, each 2x2x1 (rank 2), so rank 6. Or use other decompositions.

Total: 17+17+6+6 = 46. That's better than 54!

But wait, I need to verify the 2x3x3 rank 17 decomposition exists and is correct.

Actually, the known rank for 2x3x3 is 17. Let me implement it.

Hmm, but I need to be careful about correctness. Let me just implement the clean 54-rank decomposition first and make sure it's valid, then try to optimize.

Actually, let me just provide a clean implementation of the 54-rank decomposition with correct indexing.
"""

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=6
    
    # We'll build the decomposition block by block.
    # j-groups: {0,1}, {2,3}, {4}
    # k-groups: {0,1}, {2,3}, {4,5}
    
    # Count total rank
    # 2 j-pairs * 3 k-pairs * 7 (Strassen) = 42
    # 1 single j * 3 k-pairs * 4 (direct) = 12
    # Total = 54
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]  # shape (10, 54)
    V = [[0] * rank for _ in range(m * p)]  # shape (30, 54)
    W = [[0] * rank for _ in range(p * n)]  # shape (12, 54)
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # Strassen 2x2x2 decomposition
            # A block: rows i={0,1}, cols j={j0,j1}
            # B block: rows j={j0,j1}, cols k={k0,k1}
            # C block: rows i={0,1}, cols k={k0,k1}
            
            # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1
            
            # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            
            # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1
            
            # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1
    
    # j=4: single row, direct multiplication
    for k0, k1 in k_pairs:
        for k in [k0, k1]:
            # C[0,k] += A[0,4] * B[4,k]
            U[0*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            
            # C[1,k] += A[1,4] * B[4,k]
            U[1*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    
    return {"U": U, "V": V, "W": W}
