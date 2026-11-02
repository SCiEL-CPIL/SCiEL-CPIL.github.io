import time
import numpy as np
import itertools

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    # Dimensions: A is n x m, B is m x p, C is n x p
    # Indices: 
    # A[i, j] for i in [0, n), j in [0, m)
    # B[j, k] for j in [0, m), k in [0, p)
    # C[i, k] for i in [0, n), k in [0, p)
    
    # The tensor T has indices (i, k, j) such that C[i, k] = sum_j A[i, j] B[j, k]
    # In the decomposition:
    # C[i, k] = sum_r U[i, r] * V[k, r] * W[j, r]  -- Wait, standard form is:
    # C_{i,k} = sum_{r} U_{i,r} W_{k,r} V_{j,r} is not quite right for matrix mult.
    # Standard tensor decomposition for matmul (A B = C):
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # We want: C_{i,k} = sum_{r=1}^R U_{i,r} * V_{k,r} * W_{j,r} is WRONG.
    # The bilinear form is: C_{i,k} = sum_{r} U_{i,r} * W_{j,r} * V_{k,r}? No.
    # Let's stick to the interface definition:
    # U: (n*m, rank) -> U[i*m + j, r]
    # V: (m*p, rank) -> V[j*p + k, r]
    # W: (p*n, rank) -> W[k*n + i, r]
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # This implies the standard decomposition:
    # C[i, k] = sum_j A[i, j] B[j, k] = sum_r (sum_j U[i*m+j, r] * W[k*n+i, r]) * V[j*p+k, r] ?
    # No, that's not bilinear in A and B separately if W depends on i,k.
    
    # Let's re-read the standard tensor decomposition for matrix multiplication.
    # The tensor is T_{i, k, j} = delta_{i, k?} No.
    # T is the 3-tensor such that <A, T, B> = C.
    # Usually: C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
    # Decomposition: C_{i,k} = sum_{r=1}^R U_{i,r} * W_{k,r} * V_{j,r} is not correct dimensions.
    
    # Correct Standard Form:
    # We define matrices U, V, W of size (n*m) x R, (m*p) x R, (p*n) x R?
    # Actually, the standard decomposition of the matrix multiplication tensor $T_{i,j,k}$ (where $C_k = \sum_{i,j} T_{i,j,k} A_i B_j$)
    # is given by $T_{i,j,k} = \sum_{r=1}^R U_{i,r} V_{j,r} W_{k,r}$.
    # Here:
    # i indexes A (n*m elements)
    # j indexes B (m*p elements)
    # k indexes C (n*p elements)
    
    # The interface specifies:
    # U: (n*m, rank) -> U[i, r] where i is index of A
    # V: (m*p, rank) -> V[j, r] where j is index of B
    # W: (p*n, rank) -> W[k, r] where k is index of C
    
    # So the condition is:
    # For all i, j, k:
    # T_{i,j,k} = sum_{r} U[i, r] * V[j, r] * W[k, r]
    
    # What is T_{i,j,k}?
    # A is flattened as A[i] where i = i_row * m + i_col.
    # B is flattened as B[j] where j = j_row * p + j_col.
    # C is flattened as C[k] where k = k_row * n + k_col. (Task says W[k*n+i, r] -> index is k*n+i? 
    # "W[k*n+i,r]" suggests the index for W is `k*n + i`. 
    # In C[k], if C is n x p, standard row-major is `k_row * p + k_col`.
    # But the prompt says `W[k*n+i, r]`. This implies the index into W is `k*n + i`.
    # Let's assume the index `idx_W` corresponds to the pair `(k, i)` where `k` is the column index of C (0..p-1) and `i` is the row index of C (0..n-1).
    # So `idx_W = k * n + i`.
    # This matches the shape `(p*n, rank)`.
    
    # The tensor element T[idx_A, idx_B, idx_C] is 1 if and only if:
    # A's element at idx_A and B's element at idx_B contribute to C's element at idx_C.
    # Let idx_A = i * m + j  (row i, col j of A)
    # Let idx_B = j' * p + k (row j', col k of B)
    # Let idx_C = k * n + i' (col k, row i' of C? Or row i', col k?)
    # Standard MatMul: C[i', k] = sum_j A[i', j] B[j, k].
    # So, contribution exists if:
    # 1. Row of A (i) == Row of C (i')
    # 2. Col of A (j) == Row of B (j')
    # 3. Col of B (k) == Col of C (k)
    
    # So T[idx_A, idx_B, idx_C] = 1 iff:
    # i_A == i_C
    # j_A == j_B
    # k_B == k_C
    
    # Let's construct the "Schoolbook" decomposition first to be sure.
    # Rank = n * m * p
    # For each triple (i, j, k):
    #   idx_A = i * m + j
    #   idx_B = j * p + k
    #   idx_C = k * n + i
    #   Set U[idx_A, col] = 1
    #   Set V[idx_B, col] = 1
    #   Set W[idx_C, col] = 1
    
    # Strategy: 
    # 1. Start with Schoolbook (Rank 48).
    # 2. Use a "Merging" heuristic: 
    #    Try to find two terms r1, r2 such that we can replace them with a single term r_new 
    #    by exploiting linear dependencies in the columns of U, V, W.
    #    This is hard to do randomly.
    
    # Alternative Strategy: 
    # Use known algorithms for small matrices.
    # For 2x4 x 4x6, we can decompose the 4x6 into blocks of 2x2 or 4x4?
    # Actually, a simpler structural approach:
    # The rank of matrix multiplication $M_{n,m,p}$ is bounded by $M_{n,m,p} \le M_{n,m,p/2} + M_{n,2m,p/2}$? No.
    
    # Let's try a "Randomized Subspace" approach.
    # We look for a decomposition of rank R.
    # We fix U and V to be random (or structured) and solve for W? No, it's trilinear.
    
    # Better Strategy: **Iterative Rank Reduction via Linear Algebra**
    # We represent the decomposition as a set of vectors.
    # We start with the schoolbook basis.
    # We attempt to remove a term by checking if it is in the span of the others *in a compatible way*.
    # This is equivalent to checking if the tensor is in the span of the other rank-1 tensors.
    
    # Given the time constraint and the difficulty of exact rank minimization for 2x4x6,
    # I will implement a search for a **structured decomposition** based on the **Kronecker sum**.
    # Specifically, if we can find a decomposition for smaller sizes, we can combine them.
    # However, 2, 4, 6 are not powers of 2.
    
    # Let's try a simpler, robust improvement:
    # **Randomized Local Search for Cancellations**
    # 1. Initialize with Schoolbook.
    # 2. Randomly select a pair of terms (r1, r2).
    # 3. Check if we can form a new term r_new = r1 + r2 (or some linear combo) that is "sparser" or allows removal of one of the original terms?
    # This is complex.
    
    # Let's look at the "Inspiration 3" which claimed rank 42.
    # How to get 42?
    # 48 - 6 = 42.
    # Maybe there are 6 redundant terms?
    
    # I will implement a **Greedy Elimination** algorithm.
    # 1. Build the full tensor T as a 3D array (or flattened).
    # 2. Start with the schoolbook decomposition (list of triples of indices).
    # 3. For each pair of terms, check if their sum/difference creates a term that can replace them?
    #    Actually, we want to find a set of R rank-1 tensors that sum to T.
    #    This is the Tensor Rank problem, which is NP-hard.
    
    # Heuristic: **Randomized Gaussian Elimination on the Tensor Space**
    # We want to find the minimal R such that T is in the span of R rank-1 tensors.
    # We can try to "compress" the schoolbook decomposition.
    
    # Algorithm:
    # 1. Create list of schoolbook terms: `terms = [(i,j,k), ...]`
    # 2. Create matrices U, V, W corresponding to these terms.
    # 3. Iterate:
    #    - Pick a random term `t` to remove.
    #    - Try to express `t` as a linear combination of the remaining terms?
    #    - If `t` is in the span of the others (in the tensor space), remove it.
    #    - But "in the span" for rank-1 tensors is not just linear span of the vectorized tensors, because the sum must also be a valid decomposition?
    #    - No, if $\sum_{r \in S} T_r = T$, and we remove $T_{r0}$, we need $\sum_{r \in S \setminus \{r0\}} T_r = T - T_{r0}$.
    #    - This requires $T_{r0}$ to be representable by the others.
    
    # Let's use a different, more effective heuristic for small sizes:
    # **Brute Force Search over Small Ranks with Random Initialization**
    # We know rank is likely between 30 and 48.
    # We can try to find a decomposition of rank R by:
    # 1. Randomly initialize U, V, W with entries from {0, 1, -1}.
    # 2. Compute the resulting tensor $T_{approx}$.
    # 3. Calculate the error $E = T - T_{approx}$.
    # 4. If error is 0, we found it.
    # 5. Otherwise, use the error to adjust?
    
    # Given the 150s limit, a robust approach is to **hardcode a known better decomposition** if we can derive it, or use a **fast local search**.
    
    # Let's try to derive a decomposition using **Block Structure**.
    # A (2x4) can be seen as two 2x2 blocks? No, 2x4 is 1x2 blocks of 2x2?
    # B (4x6) can be seen as 2x3 blocks of 2x2?
    # C (2x6) can be seen as 1x3 blocks of 2x2?
    
    # Let's split the middle dimension (size 4) into two parts of size 2.
    # $A = [A_1, A_2]$ where $A_1, A_2$ are 2x2.
    # $B = \begin{pmatrix} B_1 \\ B_2 \end{pmatrix}$ where $B_1, B_2$ are 2x6.
    # $C = A_1 B_1 + A_2 B_2$.
    
    # This splits the problem into two multiplications:
    # 1. $C_1 = A_1 (2x2) \times B_1 (2x6)$
    # 2. $C_2 = A_2 (2x2) \times B_2 (2x6)$
    
    # We need the rank of $M_{2,2,6}$.
    # $M_{2,2,6}$ can be further split.
    # $B_1 (2x6) = [B_{1a}, B_{1b}, B_{1c}]$ where each is 2x2.
    # $C_1 = A_1 B_{1a} + A_1 B_{1b} + A_1 B_{1c}$.
    # This is 3 multiplications of $2x2$ by $2x2$.
    # Rank of $M_{2,2,2}$ is 7 (Strassen).
    # So Rank($M_{2,2,6}$) <= 3 * 7 = 21?
    # Wait, the "sum" of tensors is just concatenation of decompositions.
    # So if we use Strassen for each 2x2x2 block, we get rank 21 for the 2x2x6 part.
    # We have two such parts (for $A_1$ and $A_2$).
    # Total Rank = 21 + 21 = 42.
    
    # This matches the inspiration! Rank 42.
    
    # So the strategy is:
    # 1. Decompose the 2x4 x 4x6 multiplication into 6 sub-problems of 2x2 x 2x2.
    # 2. Use the Strassen algorithm (Rank 7) for each 2x2x2 sub-problem.
    # 3. Combine the decompositions.
    
    # Let's verify the structure.
    # A (2x4):
    #   A1 = A[:, 0:2] (2x2)
    #   A2 = A[:, 2:4] (2x2)
    # B (4x6):
    #   B is 4x6. We need to multiply A (2x4) by B (4x6).
    #   $C = A B$.
    #   $C = A1 B_{top} + A2 B_{bottom}$ where $B_{top}$ is top 2 rows of B, $B_{bottom}$ is bottom 2 rows.
    #   $B_{top}$ is 2x6. $B_{bottom}$ is 2x6.
    #   $C = A1 B_{top} + A2 B_{bottom}$.
    #   
    #   Now, $B_{top}$ (2x6) can be split into 3 blocks of 2x2: $B_{top,1}, B_{top,2}, B_{top,3}$.
    #   $B_{bottom}$ (2x6) can be split into 3 blocks of 2x2: $B_{bot,1}, B_{bot,2}, B_{bot,3}$.
    #   
    #   $C = A1 B_{top,1} + A1 B_{top,2} + A1 B_{top,3} + A2 B_{bot,1} + A2 B_{bot,2} + A2 B_{bot,3}$.
    #   
    #   Each term is a 2x2 matrix multiplication.
    #   There are 6 such terms.
    #   Each 2x2x2 multiplication can be done with rank 7 (Strassen).
    #   Total rank = 6 * 7 = 42.
    
    # We need to construct the U, V, W matrices for this decomposition.
    # The global indices:
    # U index: i * m + j. Here n=2, m=4.
    #   i in [0, 1], j in [0, 3].
    #   A1 uses j=0,1. A2 uses j=2,3.
    # V index: j' * p + k. Here m=4, p=6.
    #   j' in [0, 3], k in [0, 5].
    #   B_top uses j'=0,1. B_bot uses j'=2,3.
    #   B_top,1 uses k=0,1. B_top,2 uses k=2,3. B_top,3 uses k=4,5.
    #   B_bot,1 uses k=0,1. B_bot,2 uses k=2,3. B_bot,3 uses k=4,5.
    # W index: k * n + i. Here p=6, n=2.
    #   k in [0, 5], i in [0, 1].
    #   C is 2x6.
    #   C[:, 0:2] comes from B_top,1 and B_bot,1.
    #   C[:, 2:4] comes from B_top,2 and B_bot,2.
    #   C[:, 4:6] comes from B_top,3 and B_bot,3.
    
    # We need the Strassen decomposition for 2x2x2.
    # Standard Strassen:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
    # i = (a+d)(e+h) - ((d-c)(f-g) + (a-b)(e+g))
    # j = (b+d)(g+h)
    # k = (a+c)(e+f)
    # l = (a-d)(h-f) + (c+a)(e+f)
    
    # Let's define the 7 terms for a single 2x2x2 multiplication.
    # Inputs: A (2x2), B (2x2). Outputs: C (2x2).
    # Local indices for A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local indices for B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local indices for C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Strassen Terms (u, v, w) where u is coeff for A, v for B, w for C.
    # We need 7 rank-1 tensors.
    # T1: (a+d)(e+h) -> A:[1,1,0,1], B:[1,1,1,1], C:[1,0,0,1]
    # T2: (d-c)(f-g) -> A:[0,0,-1,1], B:[0,1,-1,0], C:[1,0,0,0] (with -1 sign)
    # T3: (a-b)(e+g) -> A:[1,-1,0,0], B:[1,0,1,0], C:[1,0,0,0] (with -1 sign)
    # T4: (b+d)(g+h) -> A:[0,1,0,1], B:[0,0,1,1], C:[0,1,0,0]
    # T5: (a+c)(e+f) -> A:[1,0,1,0], B:[1,1,0,0], C:[0,0,1,0]
    # T6: (a-d)(h-f) -> A:[1,0,0,-1], B:[0,1,0,-1], C:[0,0,0,1]
    # T7: (c+a)(e+f) -> A:[1,0,1,0], B:[1,1,0,0], C:[0,0,0,1]
    
    # Wait, let's verify T6 and T7.
    # l = (a-d)(h-f) + (c+a)(e+f)
    # T6: A:[1,0,0,-1], B:[0,1,0,-1] -> (a-d)(h-f). C contributes to l.
    # T7: A:[1,0,1,0], B:[1,1,0,0] -> (a+c)(e+f). C contributes to l.
    
    # Let's double check T2 and T3 signs.
    # i = T1 - T2 - T3
    # T1: (a+d)(e+h)
    # T2: (d-c)(f-g) = df - dg - cf + cg
    # T3: (a-b)(e+g) = ae + ag - be - bg
    # T1 - T2 - T3 = (ae + ah + de + dh) - (df - dg - cf + cg) - (ae + ag - be - bg)
    # = ae + ah + de + dh - df + dg + cf - cg - ae - ag + be + bg
    # = ah + de + dh - df + dg + cf - cg - ag + be + bg
    # We want i = ae + bg? No, i = ae + bg is wrong.
    # C[0,0] = ae + bg.
    # My expansion:
    # ah + de + dh - df + dg + cf - cg - ag + be + bg
    # This doesn't look like ae + bg.
    
    # Let's use the standard Strassen definitions carefully.
    # p1 = (a + d)(e + h)
    # p2 = (b + d)(e + f)
    # p3 = (a + c)(g + h)
    # p4 = (d - c)(e + f)
    # p5 = (a - b)(g + h)
    # p6 = (a + d)(f - h)
    # p7 = (b + c)(e - g)
    
    # C[0,0] = p1 - p4 - p5 + p7
    # C[0,1] = p2 + p6
    # C[1,0] = p3 + p5
    # C[1,1] = p1 - p2 + p3 - p6
    
    # Let's map these to vectors.
    # A indices: 0:a, 1:b, 2:c, 3:d
    # B indices: 0:e, 1:f, 2:g, 3:h
    # C indices: 0:i, 1:j, 2:k, 3:l
    
    # P1: A:[1,0,0,1], B:[1,1,0,0]? No, (e+h) is B[0]+B[3].
    # P1: A:[1,0,0,1], B:[1,0,0,1] -> (a+d)(e+h).
    # P2: (b+d)(e+f) -> A:[0,1,0,1], B:[1,1,0,0].
    # P3: (a+c)(g+h) -> A:[1,0,1,0], B:[0,0,1,1].
    # P4: (d-c)(e+f) -> A:[0,0,-1,1], B:[1,1,0,0].
    # P5: (a-b)(g+h) -> A:[1,-1,0,0], B:[0,0,1,1].
    # P6: (a+d)(f-h) -> A:[1,0,0,1], B:[0,1,0,-1].
    # P7: (b+c)(e-g) -> A:[0,1,1,0], B:[1,0,-1,0].
    
    # Coefficients for C:
    # C0 (i): P1 - P4 - P5 + P7
    # C1 (j): P2 + P6
    # C2 (k): P3 + P5
    # C3 (l): P1 - P2 + P3 - P6
    
    # We can represent each P as a rank-1 tensor.
    # The total decomposition for one 2x2x2 block consists of 7 terms.
    # Each term r has vectors u_r, v_r, w_r.
    # w_r is a vector of length 4 (for the 2x2 C block).
    # For P1, w_1 = [1, 0, 0, 1]
    # For P2, w_2 = [0, 1, 0, 0]
    # For P3, w_3 = [0, 0, 1, 0]
    # For P4, w_4 = [-1, 0, 0, 0]
    # For P5, w_5 = [-1, 0, 1, 0]
    # For P6, w_6 = [0, 1, 0, -1]
    # For P7, w_7 = [0, 0, 0, 1]
    
    # Now we map these to the global 2x4 x 4x6 problem.
    # We have 6 blocks:
    # Block 0: A1 (cols 0,1), B_top,1 (rows 0,1; cols 0,1) -> C[:, 0:2]
    # Block 1: A1 (cols 0,1), B_top,2 (rows 0,1; cols 2,3) -> C[:, 2:4]
    # Block 2: A1 (cols 0,1), B_top,3 (rows 0,1; cols 4,5) -> C[:, 4:6]
    # Block 3: A2 (cols 2,3), B_bot,1 (rows 2,3; cols 0,1) -> C[:, 0:2]
    # Block 4: A2 (cols 2,3), B_bot,2 (rows 2,3; cols 2,3) -> C[:, 2:4]
    # Block 5: A2 (cols 2,3), B_bot,3 (rows 2,3; cols 4,5) -> C[:, 4:6]
    
    # Global U (2*4 = 8 rows):
    # For Block b, we use A subset.
    # A1 corresponds to global A indices 0,1 (row 0) and 2,3 (row 1)?
    # Global A index = i * 4 + j.
    # A1 (2x2):
    #   Local (0,0) -> Global (0,0) -> idx 0
    #   Local (0,1) -> Global (0,1) -> idx 1
    #   Local (1,0) -> Global (1,0) -> idx 4
    #   Local (1,1) -> Global (1,1) -> idx 5
    # A2 (2x2):
    #   Local (0,0) -> Global (0,2) -> idx 2
    #   Local (0,1) -> Global (0,3) -> idx 3
    #   Local (1,0) -> Global (1,2) -> idx 6
    #   Local (1,1) -> Global (1,3) -> idx 7
    
    # Global V (4*6 = 24 rows):
    # Global B index = j * 6 + k.
    # B_top,1 (rows 0,1; cols 0,1):
    #   Local (0,0) -> Global (0,0) -> idx 0
    #   Local (0,1) -> Global (0,1) -> idx 1
    #   Local (1,0) -> Global (1,0) -> idx 6
    #   Local (1,1) -> Global (1,1) -> idx 7
    # B_top,2 (rows 0,1; cols 2,3):
    #   Local (0,0) -> Global (0,2) -> idx 2
    #   Local (0,1) -> Global (0,3) -> idx 3
    #   Local (1,0) -> Global (1,2) -> idx 8
    #   Local (1,1) -> Global (1,3) -> idx 9
    # B_top,3 (rows 0,1; cols 4,5):
    #   Local (0,0) -> Global (0,4) -> idx 4
    #   Local (0,1) -> Global (0,5) -> idx 5
    #   Local (1,0) -> Global (1,4) -> idx 10
    #   Local (1,1) -> Global (1,5) -> idx 11
    # B_bot,1 (rows 2,3; cols 0,1):
    #   Local (0,0) -> Global (2,0) -> idx 12
    #   Local (0,1) -> Global (2,1) -> idx 13
    #   Local (1,0) -> Global (3,0) -> idx 18
    #   Local (1,1) -> Global (3,1) -> idx 19
    # B_bot,2 (rows 2,3; cols 2,3):
    #   Local (0,0) -> Global (2,2) -> idx 14
    #   Local (0,1) -> Global (2,3) -> idx 15
    #   Local (1,0) -> Global (3,2) -> idx 20
    #   Local (1,1) -> Global (3,3) -> idx 21
    # B_bot,3 (rows 2,3; cols 4,5):
    #   Local (0,0) -> Global (2,4) -> idx 16
    #   Local (0,1) -> Global (2,5) -> idx 17
    #   Local (1,0) -> Global (3,4) -> idx 22
    #   Local (1,1) -> Global (3,5) -> idx 23
    
    # Global W (6*2 = 12 rows):
    # Global C index = k * 2 + i. (Task: W[k*n+i, r])
    # C[:, 0:2] corresponds to k=0,1.
    #   Local C (0,0) -> Global (i=0, k=0) -> idx 0*2+0 = 0
    #   Local C (0,1) -> Global (i=0, k=1) -> idx 1*2+0 = 2
    #   Local C (1,0) -> Global (i=1, k=0) -> idx 0*2+1 = 1
    #   Local C (1,1) -> Global (i=1, k=1) -> idx 1*2+1 = 3
    # C[:, 2:4] corresponds to k=2,3.
    #   Local C (0,0) -> Global (i=0, k=2) -> idx 2*2+0 = 4
    #   Local C (0,1) -> Global (i=0, k=3) -> idx 3*2+0 = 6
    #   Local C (1,0) -> Global (i=1, k=2) -> idx 2*2+1 = 5
    #   Local C (1,1) -> Global (i=1, k=3) -> idx 3*2+1 = 7
    # C[:, 4:6] corresponds to k=4,5.
    #   Local C (0,0) -> Global (i=0, k=4) -> idx 4*2+0 = 8
    #   Local C (0,1) -> Global (i=0, k=5) -> idx 5*2+0 = 10
    #   Local C (1,0) -> Global (i=1, k=4) -> idx 4*2+1 = 9
    #   Local C (1,1) -> Global (i=1, k=5) -> idx 5*2+1 = 11
    
    # We construct the 42 terms.
    # Rank = 42.
    
    U = np.zeros((8, 42))
    V = np.zeros((24, 42))
    W = np.zeros((12, 42))
    
    # Strassen Vectors (Local 2x2)
    # u, v, w are length 4.
    # P1
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1]
    # P2
    u2 = [0, 1, 0, 1]
    v2 = [1, 1, 0, 0]
    w2 = [0, 1, 0, 0]
    # P3
    u3 = [1, 0, 1, 0]
    v3 = [0, 0, 1, 1]
    w3 = [0, 0, 1, 0]
    # P4
    u4 = [0, 0, -1, 1]
    v4 = [1, 1, 0, 0]
    w4 = [-1, 0, 0, 0]
    # P5
    u5 = [1, -1, 0, 0]
    v5 = [0, 0, 1, 1]
    w5 = [-1, 0, 1, 0]
    # P6
    u6 = [1, 0, 0, 1]
    v6 = [0, 1, 0, -1]
    w6 = [0, 1, 0, -1]
    # P7
    u7 = [0, 1, 1, 0]
    v7 = [1, 0, -1, 0]
    w7 = [0, 0, 0, 1]
    
    strassen_terms = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    # Block Definitions
    # (A_offset, B_offset, C_offset)
    # A_offset: index in global A for local A[0,0]
    # B_offset: index in global B for local B[0,0]
    # C_offset: index in global W for local C[0,0]
    
    # Note: Global A index = i*4 + j.
    # A1 (rows 0,1; cols 0,1):
    #   Local (0,0) -> Global (0,0) -> 0
    #   Local (0,1) -> Global (0,1) -> 1
    #   Local (1,0) -> Global (1,0) -> 4
    #   Local (1,1) -> Global (1,1) -> 5
    # Mapping: Local idx 0->0, 1->1, 2->4, 3->5.
    
    # A2 (rows 0,1; cols 2,3):
    #   Local (0,0) -> Global (0,2) -> 2
    #   Local (0,1) -> Global (0,3) -> 3
    #   Local (1,0) -> Global (1,2) -> 6
    #   Local (1,1) -> Global (1,3) -> 7
    # Mapping: Local idx 0->2, 1->3, 2->6, 3->7.
    
    # B_top,1 (rows 0,1; cols 0,1):
    #   Local (0,0) -> Global (0,0) -> 0
    #   Local (0,1) -> Global (0,1) -> 1
    #   Local (1,0) -> Global (1,0) -> 6
    #   Local (1,1) -> Global (1,1) -> 7
    # Mapping: Local idx 0->0, 1->1, 2->6, 3->7.
    
    # B_top,2 (rows 0,1; cols 2,3):
    #   Local (0,0) -> Global (0,2) -> 2
    #   Local (0,1) -> Global (0,3) -> 3
    #   Local (1,0) -> Global (1,2) -> 8
    #   Local (1,1) -> Global (1,3) -> 9
    # Mapping: Local idx 0->2, 1->3, 2->8, 3->9.
    
    # B_top,3 (rows 0,1; cols 4,5):
    #   Local (0,0) -> Global (0,4) -> 4
    #   Local (0,1) -> Global (0,5) -> 5
    #   Local (1,0) -> Global (1,4) -> 10
    #   Local (1,1) -> Global (1,5) -> 11
    # Mapping: Local idx 0->4, 1->5, 2->10, 3->11.
    
    # B_bot,1 (rows 2,3; cols 0,1):
    #   Local (0,0) -> Global (2,0) -> 12
    #   Local (0,1) -> Global (2,1) -> 13
    #   Local (1,0) -> Global (3,0) -> 18
    #   Local (1,1) -> Global (3,1) -> 19
    # Mapping: Local idx 0->12, 1->13, 2->18, 3->19.
    
    # B_bot,2 (rows 2,3; cols 2,3):
    #   Local (0,0) -> Global (2,2) -> 14
    #   Local (0,1) -> Global (2,3) -> 15
    #   Local (1,0) -> Global (3,2) -> 20
    #   Local (1,1) -> Global (3,3) -> 21
    # Mapping: Local idx 0->14, 1->15, 2->20, 3->21.
    
    # B_bot,3 (rows 2,3; cols 4,5):
    #   Local (0,0) -> Global (2,4) -> 16
    #   Local (0,1) -> Global (2,5) -> 17
    #   Local (1,0) -> Global (3,4) -> 22
    #   Local (1,1) -> Global (3,5) -> 23
    # Mapping: Local idx 0->16, 1->17, 2->22, 3->23.
    
    # C[:, 0:2] (cols 0,1):
    #   Local (0,0) -> Global (i=0, k=0) -> 0
    #   Local (0,1) -> Global (i=0, k=1) -> 2
    #   Local (1,0) -> Global (i=1, k=0) -> 1
    #   Local (1,1) -> Global (i=1, k=1) -> 3
    # Mapping: Local idx 0->0, 1->2, 2->1, 3->3.
    
    # C[:, 2:4] (cols 2,3):
    #   Local (0,0) -> Global (i=0, k=2) -> 4
    #   Local (0,1) -> Global (i=0, k=3) -> 6
    #   Local (1,0) -> Global (i=1, k=2) -> 5
    #   Local (1,1) -> Global (i=1, k=3) -> 7
    # Mapping: Local idx 0->4, 1->6, 2->5, 3->7.
    
    # C[:, 4:6] (cols 4,5):
    #   Local (0,0) -> Global (i=0, k=4) -> 8
    #   Local (0,1) -> Global (i=0, k=5) -> 10
    #   Local (1,0) -> Global (i=1, k=4) -> 9
    #   Local (1,1) -> Global (i=1, k=5) -> 11
    # Mapping: Local idx 0->8, 1->10, 2->9, 3->11.
    
    blocks = [
        # (A_map, B_map, C_map)
        ([0, 1, 4, 5], [0, 1, 6, 7], [0, 2, 1, 3]),   # Block 0
        ([0, 1, 4, 5], [2, 3, 8, 9], [4, 6, 5, 7]),   # Block 1
        ([0, 1, 4, 5], [4, 5, 10, 11], [8, 10, 9, 11]), # Block 2
        ([2, 3, 6, 7], [12, 13, 18, 19], [0, 2, 1, 3]), # Block 3
        ([2, 3, 6, 7], [14, 15, 20, 21], [4, 6, 5, 7]), # Block 4
        ([2, 3, 6, 7], [16, 17, 22, 23], [8, 10, 9, 11]) # Block 5
    ]
    
    col = 0
    for b_idx, (a_map, b_map, c_map) in enumerate(blocks):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_terms):
            # Fill U
            for l in range(4):
                if u_loc[l] != 0:
                    U[a_map[l], col] += u_loc[l]
            # Fill V
            for l in range(4):
                if v_loc[l] != 0:
                    V[b_map[l], col] += v_loc[l]
            # Fill W
            for l in range(4):
                if w_loc[l] != 0:
                    W[c_map[l], col] += w_loc[l]
            col += 1
            
    if time.time() > deadline:
         # Fallback to schoolbook if something went wrong (should not happen)
         return _schoolbook()
         
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
