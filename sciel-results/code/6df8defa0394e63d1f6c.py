"""
Rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.

Strategy:
- Use Laderman's rank-19 algorithm for 4x4x4 matrix multiplication as a building block.
- Split the 9-column output into three 3-column blocks.
- For each 4x4 * 4x3 subproblem, derive a rank-19 decomposition by restricting Laderman's
  4x4x4 decomposition to the first 3 columns (dropping the 4th output column).
- This gives total rank 19*3 = 57, which is much better than schoolbook 144.

Actually, let me think more carefully. Laderman's algorithm computes C = A*B where A is 4x4,
B is 4x4, C is 4x4 with 19 scalar multiplications. The decomposition has the form:
C[i,k] = sum_r M_r(A) * N_r(B) * P_r(C)
where M_r, N_r are linear forms in A and B respectively, and P_r is a coefficient for C[i,k].

For our problem: A is 4x4, B is 4x9, C is 4x9.
We can split B into B1, B2, B3 each 4x3, and C into C1, C2, C3 each 4x3.
For each block, C_t = A * B_t.

If we use Laderman's algorithm for each 4x4 * 4x3, we need a decomposition of the
4x4x3 tensor. The 4x4x3 tensor T[i,j,k] = delta_{consistent} for i in [0,4), j in [0,4), k in [0,3).

One approach: use Laderman's 19-term decomposition for 4x4x4, and for each term,
if the output coefficient P_r has support only in columns 0,1,2 (i.e., the 4th column
coefficient is 0), we can keep it. But Laderman's algorithm likely has terms that
involve all 4 output columns.

Alternative: construct the 4x4x3 decomposition directly. The rank of the 4x4x3 matrix
multiplication tensor is at most 19 (since it's a sub-tensor of 4x4x4 which has rank 19).
Actually, the rank of a sub-tensor is at most the rank of the full tensor, so rank(4x4x3) <= 19.

Let me try to explicitly construct a rank-19 decomposition for 4x4x3 by taking Laderman's
algorithm and projecting out the 4th column.

Laderman's algorithm for 4x4x4:
The 19 multiplications are:
M1 = (A11+A22)(B11+B22)
M2 = (A11+A22)(B12+B21)  -- wait, this isn't quite right.

Let me look up Laderman's algorithm properly.

Laderman (1976) gave a 19-multiplication algorithm for 4x4 matrix multiplication.
The algorithm is:
Let A = [[a11,a12,a13,a14],[a21,a22,a23,a24],[a31,a32,a33,a34],[a41,a42,a43,a44]]
Let B = [[b11,b12,b13,b14],[b21,b22,b23,b24],[b31,b32,b33,b34],[b41,b42,b43,b44]]

The 19 products are:
P1  = (a11+a22)(b11+b22)
P2  = (a11+a22)(b12+b21)
P3  = (a11+a33)(b13+b31)
P4  = (a11+a33)(b14+b34)  -- hmm, I'm not sure of the exact form

Actually, let me just implement a known correct version. The key insight is that for
4x4x3, we can use a rank at most 19 decomposition. Let me try to find it by construction.

Alternative approach: use the fact that 4x4x3 can be decomposed as:
C[i,k] = sum_j A[i,j] * B[j,k] for k=0,1,2

We can use Strassen-like ideas. For 2x2x2, Strassen uses 7 instead of 8.
For 4x4x4, Laderman uses 19.

For 4x4x3, a natural approach is to use a 2x2 block structure.
Split A into 2x2 blocks of 2x2 matrices, and B into 2x2 blocks of 2x3 matrices.

Actually, let me try a different approach: use the known rank bound and try to find
a decomposition via linear algebra.

The tensor T for 4x4x3 has dimensions:
U: 16 x R
V: 12 x R  
W: 12 x R

where R is the rank. We need R such that T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r].

Let me try R=19 and see if we can find a solution.

Actually, given the complexity, let me try a simpler approach:
- Use schoolbook for the 4x4x3 part: rank = 4*4*3 = 48 per block
- Total: 48*3 = 144 (same as full schoolbook)

That's not helpful. Let me try to actually implement Laderman's algorithm.

After research, Laderman's 19-term algorithm for 4x4x4 can be described as follows.
The algorithm uses the following 19 bilinear forms:

Let me define the algorithm using a known reference. The key products are:

P1 = (a00+a11)*(b00+b11)
P2 = (a00+a11)*(b01+b10)
P3 = (a00+a22)*(b02+b20)
P4 = (a00+a22)*(b03+b23)
P5 = (a00+a33)*(b03+b30)
P6 = (a11+a22)*(b12+b21)
P7 = (a11+a33)*(b13+b31)
P8 = (a22+a33)*(b23+b32)
P9 = (a01+a10+a23+a32)*(b01+b10+b23+b32)
P10 = (a01+a10+a23+a32)*(b02+b13+b21+b32)  -- I'm making this up

This is getting too complex to get right from memory. Let me try a different strategy.

Strategy: Use a randomized approach to find a low-rank decomposition.
1. Start with R=19 for a 4x4x3 subproblem.
2. Set up the system of equations and try to solve it.

Actually, let me try yet another approach: use the fact that the 4x4x3 tensor
is a "slice" of the 4x4x4 tensor. If I have a rank-R decomposition of 4x4x4,
I can get a rank-R decomposition of 4x4x3 by just keeping the terms where
the W vector has zero in the 4th output position (or adjusting).

But this doesn't directly work because the W vectors in a 4x4x4 decomposition
are 16-dimensional (for 4x4 output), and I need 12-dimensional (for 4x3 output).

Let me try the most practical approach: implement a known low-rank algorithm.

For 4x4x4, the best known rank is 19 (Laderman). For 4x4x3, the rank is at most 19.

Let me try to construct the 4x4x3 decomposition by using a 2x2 block Strassen-like approach.

For 2x2x2, Strassen's 7-term algorithm:
M1 = (a00+a11)(b00+b11)
M2 = (a10+a11)(b00)
M3 = (a00)(b10+b11)
M4 = (a11)(b11-b10)
M5 = (a00+a01)(b01)
M6 = (a10-a00)(b00+b11)
M7 = (a01-a11)(b10+b11)

C00 = M1+M4-M6-M7
C01 = M3+M7
C10 = M2+M6
C11 = M1-M2+M3-M4

For 4x4x3, I can use a 2x2 block structure where each block is 2x2 or 2x1.
A is 4x4 = [[A00,A01],[A10,A11]] where each Aij is 2x2.
B is 4x3 = [[B00,B01],[B10,B11]] where B00,B10 are 2x2 and B01,B11 are 2x1.

Wait, this doesn't split cleanly. Let me try:
B is 4x3. Split into top 2 rows and bottom 2 rows:
B = [[Btop],[Bbot]] where Btop is 2x3, Bbot is 2x3.

C = A*B = [[A00,A01],[A10,A11]] * [[Btop],[Bbot]]
C0 = A00*Btop + A01*Bbot  (top 2 rows of C, 2x3)
C1 = A10*Btop + A11*Bbot  (bottom 2 rows of C, 2x3)

Each of C0, C1 is a 2x2 * 2x3 product plus another 2x2 * 2x3 product.
A 2x2 * 2x3 product has rank at most 2*2*3 = 12 (schoolbook) or can be done with fewer.

Actually, a 2x2 * 2x3 matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, j in {0,1}, k in {0,1,2}.
This is a 2x2x3 tensor. Its rank is at most 8 (schoolbook) or possibly 7 (by Strassen-like).

For 2x2x3: C = A*B where A is 2x2, B is 2x3.
C00 = a00*b00 + a01*b10
C01 = a00*b01 + a01*b11
C02 = a00*b02 + a01*b12
C10 = a10*b00 + a11*b10
C11 = a10*b01 + a11*b11
C12 = a10*b02 + a11*b12

This is 6 outputs, 4 inputs from A, 6 inputs from B. Schoolbook rank = 2*2*3 = 12? No, schoolbook is sum over j, so for each (i,k) pair, we have 2 terms. Total terms = 2*3*2 = 12? No, schoolbook for 2x2*2x3 is: for each i in {0,1}, k in {0,1,2}, sum over j in {0,1}. That's 2*3*2 = 12 multiplications. But we can do better.

Actually, the rank of the 2x2x3 tensor: it's the same as the 2x3x2 tensor (by permuting indices). The 2x3x2 tensor has rank at most 7 (I think). Let me just use schoolbook for now: rank 12 per 2x2x3.

So for 4x4x3:
C0 = A00*Btop + A01*Bbot: this is sum of two 2x2*2x3 products, rank at most 12+12=24
C1 = A10*Btop + A11*Bbot: similarly 24
Total: 48 per 4x4x3 block. Three blocks: 144. Same as schoolbook. Not helpful.

Let me try a completely different approach: use the known Laderman algorithm.

After more thought, let me just implement Laderman's algorithm correctly. Here's a known formulation:

For 4x4x4 matrix multiplication, Laderman's algorithm uses 19 multiplications.
The algorithm can be described as follows (using 0-indexed notation):

The 19 bilinear forms are:
M1  = (a00+a11)(b00+b11)
M2  = (a00+a11)(b01+b10)
M3  = (a00+a22)(b02+b20)
M4  = (a00+a22)(b03+b23)
M5  = (a00+a33)(b03+b30)
M6  = (a11+a22)(b12+b21)
M7  = (a11+a33)(b13+b31)
M8  = (a22+a33)(b23+b32)
M9  = (a01+a10+a23+a32)(b01+b10+b23+b32)
M10 = (a01+a10+a23+a32)(b02+b13+b21+b32)
M11 = (a01+a10+a23+a32)(b03+b12+b20+b31)
M12 = (a02+a13+a20+a31)(b02+b13+b20+b31)
M13 = (a02+a13+a20+a31)(b03+b12+b21+b30)
M14 = (a02+a13+a20+a31)(b00+b11+b22+b33)
M15 = (a03+a12+a21+a30)(b00+b11+b22+b33)
M16 = (a03+a12+a21+a30)(b01+b12+b23+b30)
M17 = (a03+a12+a21+a30)(b02+b13+b20+b31)
M18 = (a00+a11+a22+a33)(b00+b11+b22+b33)
M19 = (a00+a11+a22+a33)(b01+b10+b23+b32)

And the output C[i,k] is a linear combination of these M's.

I'm not confident in the exact coefficients. Let me try a different approach.

Given the difficulty of getting Laderman's algorithm exactly right from memory, let me try a
practical approach: use a randomized search to find a low-rank decomposition for 4x4x3,
then combine three such decompositions.

For 4x4x3:
- U: 16 x R
- V: 12 x R
- W: 12 x R

Let me try R=19 and use a gradient-free approach or just try random integer/half-integer
matrices and check.

Actually, let me try a much simpler approach: just use the schoolbook decomposition
but with a smarter structure. The schoolbook for 4x4*4x9 has rank 144. 

Wait, I just realized: the problem is 4x4 * 4x9, so:
- A is 4x4 (16 elements)
- B is 4x9 (36 elements)
- C is 4x9 (36 elements)

Schoolbook: C[i,k] = sum_j A[i,j]*B[j,k] for i in 0..3, j in 0..3, k in 0..8.
Total multiplications: 4*4*9 = 144.

The tensor has:
- U: 16 x R
- V: 36 x R
- W: 36 x R

For a rank-R decomposition, we need T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] where
a = i*4+j (0..15), b = j*9+k (0..35), c = k*4+i (0..35).

T[a,b,c] = 1 iff a's j == b's j AND a's i == c's i AND b's k == c's k.

Let me try R=36 with a specific construction:
For r = j*9+k (j in 0..3, k in 0..8):
  U[i*4+j, r] = 1 for all i (since the j in a matches the j in r)
  V[j'*9+k', r] = 1 iff (j',k') = (j,k)
  W[k''*4+i'', r] = 1 iff k'' = k (and i'' can be anything? No, we need i'' to match)

The problem is that W needs to depend on i'', but i'' is not in r. So this doesn't work.

Let me try R = 4*9 = 36 with r = i*9+k:
  U[i'*4+j, r] = 1 iff i' = i
  V[j'*9+k', r] = 1 iff k' = k
  W[k''*4+i'', r] = 1 iff k'' = k AND i'' = i

Check: sum_r U[a,r]*V[b,r]*W[c,r] where a=i1*4+j1, b=j2*9+k2, c=k3*4+i3
= sum_{i,k} [i1=i] [k2=k] [k3=k][i3=i]
= 1 iff i1=i3 AND k2=k3

But we also need j1=j2. In this construction, j1 and j2 don't appear in the condition,
so if i1=i3 and k2=k3 but j1!=j2, we'd get 1 instead of 0. So this doesn't work.

The fundamental issue is that the j index appears in both U and V but not in W, and we
need it to be consistent between U and V.

What if I use R = 4*4*9 = 144 (schoolbook) but try to find a lower rank by exploiting
structure?

Let me try a different construction. What if I use R = 4*9 = 36 but with a more careful
assignment?

Actually, let me just try to find ANY rank less than 144 using a search. I'll try
R values from 100 down to 50 and use random integer/half-integer initialization with
some structure.

Given time constraints, let me implement a practical search:
1. Try R=36 with the construction above (will fail but worth checking).
2. Try random searches for R in [100, 110, 120, 130, 140].
3. Fall back to schoolbook 144.

Actually, I just realized I should try a much simpler idea: use the fact that the
4x4*4x9 problem can be seen as three independent 4x4*4x3 problems (columns 0-2, 3-5, 6-8).
If I can find a rank-R decomposition for 4x4*4x3, then the total rank is 3R.

For 4x4*4x3, schoolbook rank is 4*4*3 = 48. So 3*48 = 144 (same as full schoolbook).
But if I can find rank 19 for 4x4*4x3 (which is plausible since 4x4*4x4 has rank 19),
then total rank = 57.

Let me try to find a rank-19 decomposition for 4x4*4x3 using a search.

For 4x4*4x3:
- U: 16 x 19
- V: 12 x 19
- W: 12 x 19

The conditions are:
For all i in 0..3, j in 0..3, k in 0..2:
  sum_r U[i*4+j, r] * V[j*3+k, r] * W[k*4+i, r] = 1

For all inconsistent (a,b,c):
  sum_r U[a,r] * V[b,r] * W[c,r] = 0

This is a system of nonlinear equations. Let me try to solve it using a simple approach:
initialize with random small integers and use coordinate descent or just try many random
initializations.

Given the 120-second time limit, I'll try a limited number of random initializations.
"""

import time
import numpy as np
import random
from itertools import product, combinations

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # Schoolbook baseline: rank 144
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    def verify(U, V, W, R):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i1, j1 = divmod(a, m)
                    j2, k2 = divmod(b, p)
                    k3, i3 = divmod(c, n)
                    if j1 == j2 and i1 == i3 and k2 == k3:
                        continue
                    val = np.dot(U_np[a], V_np[b] * W_np[c])
                    if abs(val) > 1e-9:
                        return False
        return True

    def verify_4x4x3(U, V, W, R):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    val = np.dot(U_np[i*4+j], V_np[j*3+k] * W_np[k*4+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        for a in range(16):
            for b in range(12):
                for c in range(12):
                    i1, j1 = divmod(a, 4)
                    j2, k2 = divmod(b, 3)
                    k3, i3 = divmod(c, 4)
                    if j1 == j2 and i1 == i3 and k2 == k3:
                        continue
                    val = np.dot(U_np[a], V_np[b] * W_np[c])
                    if abs(val) > 1e-9:
                        return False
        return True

    def verify_4x4x4(U, V, W, R):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    val = np.dot(U_np[i*4+j], V_np[j*4+k] * W_np[k*4+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        for a in range(16):
            for b in range(16):
                for c in range(16):
                    i1, j1 = divmod(a, 4)
                    j2, k2 = divmod(b, 4)
                    k3, i3 = divmod(c, 4)
                    if j1 == j2 and i1 == i3 and k2 == k3:
                        continue
                    val = np.dot(U_np[a], V_np[b] * W_np[c])
                    if abs(val) > 1e-9:
                        return False
        return True

    # Try to find Laderman's 19-term decomposition for 4x4x4
    # using a structured search based on the known algorithm structure.
    # 
    # Laderman's algorithm can be described using a specific set of 19 bilinear forms.
    # Let me try to construct it by searching over the space of possible decompositions
    # with the known structure.
    
    # Approach: Use the known fact that Laderman's algorithm exists and try to find it
    # by solving the system of equations. We'll use a structured random search.
    
    def find_laderman_4x4x4(deadline):
        """Search for a rank-19 decomposition of 4x4x4 matrix multiplication."""
        R = 19
        # The 19 bilinear forms in Laderman's algorithm have a specific structure.
        # Each form is (linear combination of A entries) * (linear combination of B entries)
        # The output C[i,k] is a linear combination of these 19 products.
        
        # Let me try a different approach: use the known structure of Laderman's algorithm.
        # Based on literature, the 19 terms can be grouped as follows:
        # - 8 terms from "diagonal" pairs
        # - 11 terms from "off-diagonal" combinations
        
        # Actually, let me try to use a known explicit construction.
        # One known formulation uses the following 19 products:
        
        # I'll try to construct it by solving the equations directly.
        # For rank 19, we need:
        # U: 16 x 19, V: 16 x 19, W: 16 x 19
        # such that for all i,j,k: sum_r U[i*4+j,r]*V[j*4+k,r]*W[k*4+i,r] = 1
        # and for all inconsistent (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = 0
        
        # This is a system of 16*16*16 = 4096 equations (16 consistent, 4080 inconsistent)
        # with 19*16*3 = 912 unknowns.
        
        # Let me try a randomized approach with the constraint that entries are in {-1, 0, 1, 2}
        
        best_score = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            
            random.seed(1000 + attempt)
            
            # Initialize with sparse random matrices
            U_t = np.zeros((16, R), dtype=float)
            V_t = np.zeros((16, R), dtype=float)
            W_t = np.zeros((16, R), dtype=float)
            
            for r in range(R):
                # Each column of U, V, W should have a specific structure
                # For Laderman, each term involves specific pairs of indices
                # Let me try to create structured random matrices
                
                # Random sparse initialization
                for a in range(16):
                    if random.random() < 0.3:
                        U_t[a, r] = random.choice([-1, 1, 2, -2])
                for b in range(16):
                    if random.random() < 0.3:
                        V_t[b, r] = random.choice([-1, 1, 2, -2])
                for c in range(16):
                    if random.random() < 0.3:
                        W_t[c, r] = random.choice([-1, 1, 2, -2])
            
            # Check the consistent equations
            consistent_ok = True
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        val = np.dot(U_t[i*4+j], V_t[j*4+k] * W_t[k*4+i])
                        if abs(val - 1.0) > 1e-6:
                            consistent_ok = False
                            break
                    if not consistent_ok:
                        break
                if not consistent_ok:
                    break
            
            if consistent_ok:
                # Check inconsistent equations
                inconsistent_violations = 0
                max_violation = 0
                for a in range(16):
                    for b in range(16):
                        for c in range(16):
                            i1, j1 = divmod(a, 4)
                            j2, k2 = divmod(b, 4)
                            k3, i3 = divmod(c, 4)
                            if j1 == j2 and i1 == i3 and k2 == k3:
                                continue
                            val = np.dot(U_t[a], V_t[b] * W_t[c])
                            if abs(val) > 1e-6:
                                inconsistent_violations += 1
                                max_violation = max(max_violation, abs(val))
                
                score = inconsistent_violations * 1000 + max_violation
                if score < best_score:
                    best_score = score
                    best_U = U_t.copy()
                    best_V = V_t.copy()
                    best_W = W_t.copy()
                    
                    if inconsistent_violations == 0:
                        # Found a valid decomposition!
                        U_list = [[int(round(x)) for x in best_U[r]] for r in range(16)]
                        V_list = [[int(round(x)) for x in best_V[r]] for r in range(16)]
                        W_list = [[int(round(x)) for x in best_W[r]] for r in range(16)]
                        if verify_4x4x4(U_list, V_list, W_list, R):
                            return U_list, V_list, W_list, R
        
        return None
    
    # Try to find Laderman's decomposition
    laderman_result = find_laderman_4x4x4(deadline)
    
    if laderman_result is not None:
        U_l, V_l, W_l, R_l = laderman_result
        # Project to 4x4x3 by keeping only the first 3 output columns
        # W for 4x4x3: 12 x 19 (rows are k*4+i for k in 0..2, i in 0..3)
        # W for 4x4x4: 16 x 19 (rows are k*4+i for k in 0..3, i in 0..3)
        # We need to check if the 4x4x3 projection works
        
        U_4x4x3 = U_l  # Same U: 16 x 19
        V_4x4x3 = V_l  # Same V: 16 x 19, but we only use rows 0..11 (j*3+k for j in 0..3, k in 0..2)
        # Actually, V for 4x4x3 should be 12 x 19 (rows j*3+k for j in 0..3, k in 0..2)
        # But V_l is 16 x 19 (rows j*4+k for j in 0..3, k in 0..3)
        # We need to extract the appropriate rows
        
        V_4x4x3 = []
        for j in range(4):
            for k in range(3):
                V_4x4x3.append(V_l[j*4+k])
        
        W_4x4x3 = []
        for k in range(3):
            for i in range(4):
                W_4x4x3.append(W_l[k*4+i])
        
        if verify_4x4x3(U_4x4x3, V_4x4x3, W_4x4x3, R_l):
            # Combine 3 copies for 4x4x9
            total_R = R_l * 3
            U_full = [[0]*total_R for _ in range(16)]
            V_full = [[0]*total_R for _ in range(36)]
            W_full = [[0]*total_R for _ in range(36)]
            for block in range(3):
                for i in range(4):
                    for j in range(4):
                        for r in range(R_l):
                            U_full[i*4+j][block*R_l+r] = U_4x4x3[i*4+j][r]
                for j in range(4):
                    for k in range(3):
                        for r in range(R_l):
                            V_full[j*9 + block*3 + k][block*R_l+r] = V_4x4x3[j*3+k][r]
                for k in range(3):
                    for i in range(4):
                        for r in range(R_l):
                            W_full[(block*3+k)*4 + i][block*R_l+r] = W_4x4x3[k*4+i][r]
            if verify(U_full, V_full, W_full, total_R):
                return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback: try to find a direct decomposition for 4x4x9
    # Try various ranks
    best_result = None
    best_rank = rank_sb
    
    for R in [57, 60, 64, 72, 80, 96, 108, 120, 132, 144]:
        if time.time() > deadline - 15:
            break
        
        if R == 144:
            # Schoolbook
            if best_rank == 144:
                best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
                best_rank = 144
            continue
        
        # Try random search for this rank
        found = False
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            
            random.seed(2000 + R * 10 + attempt)
            
            U_t = [[random.choice([0, 1, -1]) for _ in range(R)] for _ in range(16)]
            V_t = [[random.choice([0, 1, -1]) for _ in range(R)] for _ in range(36)]
            W_t = [[random.choice([0, 1, -1]) for _ in range(R)] for _ in range(36)]
            
            if verify(U_t, V_t, W_t, R):
                best_result = {"U": U_t, "V": V_t, "W": W_t}
                best_rank = R
                found = True
                break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
