"""Tensor decomposition for 2x3 x 3x8 matrix multiplication.

Strategy: partition the 3x8 input matrix into three 3x? blocks.
  - Split into two 3x4 blocks (k=0..3, k=4..7)
  - Within each 3x4 block, split the 3 columns into 2+1
  - The 2x2x2 subproblems use Strassen (rank 7)
  - The 2x1x2 subproblems use schoolbook (rank 2)
  - The 2x1x4 subproblems (j=2, all 4 k-values) use schoolbook (rank 8)

Total rank: 2 * (7 + 7 + 8) = 2 * 22 = 44

But we can do better by using a different partition:
  - Split 3 into 1+2 (instead of 2+1)
  - The 1x2x2 subproblem is just 2 multiplications (rank 2)
  - The 2x2x2 subproblem is Strassen (rank 7)
  - For 1x2x4: rank 4 (schoolbook)
  
  Per 3x4 block: 2 + 7 + 4 = 13
  Total: 2 * 13 = 26

Wait, let me reconsider. The 1x2x2 tensor (multiply 1x2 by 2x2) has rank 2.
The 2x2x2 tensor (multiply 2x2 by 2x2) has rank 7 (Strassen).
The 1x2x4 tensor (multiply 1x2 by 2x4) has rank 4 (schoolbook).

Per 3x4 block: rank(1x2x2) + rank(2x2x2) + rank(1x2x4) = 2 + 7 + 4 = 13
Total for two 3x4 blocks: 26

Actually, we can be even smarter. Let me think about the full 2x3x4 tensor.
The 2x3x4 tensor can be decomposed by partitioning the 3 columns:
  Columns 0,1: 2x2x2 subproblem -> rank 7 (Strassen)
  Column 2: 2x1x2 subproblem -> rank 2 (schoolbook)
Total per 2x3x2 block: 7 + 2 = 9

For 2x3x4:
  Columns 0,1: 2x2x2 -> rank 7
  Column 2,3: 2x1x2 -> rank 2... wait, that's not right.

Let me redo this. A 2x3x4 multiplication tensor T[i,j,k] = delta[i,k] * A[i,j] * B[j,k]
where i in {0,1}, j in {0,1,2}, k in {0,1,2,3}.

We can partition by j:
  j=0,1: 2x2x4 tensor
  j=2: 2x1x4 tensor

For 2x2x4: partition k into k=0,1 and k=2,3
  k=0,1: 2x2x2 -> rank 7
  k=2,3: 2x2x2 -> rank 7
  Total: 14

For 2x1x4: schoolbook -> rank 8

Total per 3x4 block: 14 + 8 = 22
Total for 3x8: 44

Alternatively, partition by k first:
  k=0,1: 2x3x2
  k=2,3: 2x3x2

For 2x3x2: partition j:
  j=0,1: 2x2x2 -> rank 7
  j=2: 2x1x2 -> rank 2
  Total: 9

So 2x3x4 = 9 + 9 = 18
2x3x8 = 18 + 18 = 36

That's better! Let me verify: 2x3x2 tensor.
T[i,j,k] where i in {0,1}, j in {0,1,2}, k in {0,1}.
Partition j into {0,1} and {2}:
  {0,1} x {0,1}: 2x2x2, rank 7
  {2} x {0,1}: 2x1x2, rank 2
Total: 9. Yes.

So rank 36 should work.
"""

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = 2, 3, 8
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 36
    V = [[0] * rank for _ in range(m * p)]  # 24 x 36
    W = [[0] * rank for _ in range(p * n)]  # 16 x 36
    
    col = 0
    
    # Strassen 2x2x2: multiplies (2x2) A * (2x2) B = (2x2) C
    # A indices: (i,j) for i in {0,1}, j in {0,1}
    # B indices: (j,k) for j in {0,1}, k in {0,1}
    # C indices: (i,k) for i in {0,1}, k in {0,1}
    # U row index: i*m + j  (within local 2x2 block of A)
    # V row index: j*p + k  (within local 2x2 block of B)
    # W row index: k*n + i  (within local 2x2 block of C)
    
    def add_strassen_222(i_base, j_base, k_base):
        nonlocal col
        # i_base, j_base, k_base are offsets in the full matrix indices
        # U rows: (i_base+li)*m + (j_base+lj) for li in {0,1}, lj in {0,1}
        # V rows: (j_base+lj)*p + (k_base+lk) for lj in {0,1}, lk in {0,1}
        # W rows: (k_base+lk)*n + (i_base+li) for lk in {0,1}, li in {0,1}
        
        def u(li, lj):
            return (i_base + li) * m + (j_base + lj)
        
        def v(lj, lk):
            return (j_base + lj) * p + (k_base + lk)
        
        def w(lk, li):
            return (k_base + lk) * n + (i_base + li)
        
        # m1: (a+d)(e+h)
        U[u(0,0)][col] += 1; U[u(1,1)][col] += 1
        V[v(0,0)][col] += 1; V[v(1,1)][col] += 1
        W[w(0,0)][col] += 1; W[w(1,1)][col] += 1
        col += 1
        
        # m2: (c+d)e
        U[u(1,0)][col] += 1; U[u(1,1)][col] += 1
        V[v(0,0)][col] += 1
        W[w(0,1)][col] += 1; W[w(1,1)][col] += -1
        col += 1
        
        # m3: a(f-h)
        U[u(0,0)][col] += 1
        V[v(0,1)][col] += 1; V[v(1,1)][col] += -1
        W[w(1,0)][col] += 1; W[w(1,1)][col] += 1
        col += 1
        
        # m4: d(g-e)
        U[u(1,1)][col] += 1
        V[v(1,0)][col] += 1; V[v(0,0)][col] += -1
        W[w(0,0)][col] += 1; W[w(0,1)][col] += 1
        col += 1
        
        # m5: (a+b)h
        U[u(0,0)][col] += 1; U[u(0,1)][col] += 1
        V[v(1,1)][col] += 1
        W[w(0,0)][col] += -1; W[w(1,0)][col] += 1
        col += 1
        
        # m6: (c-a)(e+f)
        U[u(1,0)][col] += 1; U[u(0,0)][col] += -1
        V[v(0,0)][col] += 1; V[v(0,1)][col] += 1
        W[w(1,1)][col] += 1
        col += 1
        
        # m7: (b-d)(g+h)
        U[u(0,1)][col] += 1; U[u(1,1)][col] += -1
        V[v(1,0)][col] += 1; V[v(1,1)][col] += 1
        W[w(0,0)][col] += 1
        col += 1
    
    # Schoolbook 2x1x2: multiplies (2x1) A * (1x2) B = (2x2) C
    # A indices: (i, j) for i in {0,1}, j fixed
    # B indices: (j, k) for j fixed, k in {0,1}
    # C indices: (i, k) for i in {0,1}, k in {0,1}
    
    def add_schoolbook_212(i_base, j_fixed, k_base):
        nonlocal col
        for i in range(2):
            for k in range(2):
                U[(i_base + i) * m + j_fixed][col] = 1
                V[j_fixed * p + (k_base + k)][col] = 1
                W[(k_base + k) * n + (i_base + i)][col] = 1
                col += 1
    
    # For each k-block of size 2 (k=0,1; k=2,3; k=4,5; k=6,7):
    #   j=0,1: Strassen 2x2x2 (rank 7)
    #   j=2: Schoolbook 2x1x2 (rank 2)
    # Total per k-block: 9
    # 4 k-blocks: 36
    
    for k_off in range(0, 8, 2):
        # Strassen for j=0,1; i=0,1; k=k_off, k_off+1
        add_strassen_222(0, 0, k_off)
        # Schoolbook for j=2; i=0,1; k=k_off, k_off+1
        add_schoolbook_212(0, 2, k_off)
    
    return {"U": U, "V": V, "W": W}
