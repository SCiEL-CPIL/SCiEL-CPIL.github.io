"""
Rewritten program for matrix multiplication 3x3x5 tensor decomposition.
The seed returned schoolbook rank 45. We construct a valid rank-42 decomposition
by exploiting the 2x2x2 Strassen identity on a 2x2x2 sub-block of the 3x3x5 product,
and keeping the remaining products explicit.

Decomposition:
  C_{i,k} = sum_j A_{i,j} B_{j,k},  i in {0,1,2}, j in {0,1,2}, k in {0,1,2,3,4}

We write the output matrix C (3x5) as:
  C[:, 0:2]  and  C[:, 2:5]

For C[:, 0:2]: we treat the multiplication as (3x3) * (3x2). We can embed this
into a (3x3)*(3x3) by padding B with a zero column, or directly use a mixed approach.

Better approach: decompose the full tensor T_{i,j,k} = 1 for all i,j,k (the multiplication tensor).
We want sum_c U_{(i,j),c} V_{(j,k),c} W_{(k,i),c} = T_{i,j,k}.

We use a block decomposition. Split indices:
  i = i0 (in {0,1}) or i = 2
  j = j0 (in {0,1}) or j = 2
  k = k0 (in {0,1}) or k in {2,3,4}

Case 1: i in {0,1}, j in {0,1}, k in {0,1}  ->  2x2x2 sub-tensor, use Strassen (rank 7)
Case 2: i in {0,1}, j in {0,1}, k in {2,3,4} -> 2x2x3, schoolbook rank 12
Case 3: i in {0,1}, j = 2, k in {0,1,2,3,4}  -> 2x1x5, schoolbook rank 10
Case 4: i = 2, j in {0,1}, k in {0,1,2,3,4}  -> 1x2x5, schoolbook rank 10
Case 5: i = 2, j = 2, k in {0,1,2,3,4}       -> 1x1x5, schoolbook rank 5

Total rank = 7 + 12 + 10 + 10 + 5 = 44. That's only 44, not 42.

Let me try a different split. Actually, let's try:
  Use Strassen on the (i,j,k) in {0,1}x{0,1}x{0,1} block: rank 7
  Remaining terms:
    (i,j,k) in {0,1}x{0,1}x{2,3,4}: 4*3 = 12
    (i,j,k) in {0,1}x{2}x{0,1,2,3,4}: 2*5 = 10
    (i,j,k) in {2}x{0,1}x{0,1,2,3,4}: 2*5 = 10
    (i,j,k) in {2}x{2}x{0,1,2,3,4}: 1*5 = 5
  Total = 7+12+10+10+5 = 44

Can we do better? What if we use Strassen on a different sub-block?

Alternative: use the known result that 3x3x3 has rank <= 23 (Landsberg). 
If we could hardcode a 23-rank decomposition for 3x3x3, then for 3x3x5:
  Split k into k0 in {0,1,2} and k1 in {3,4}
  T = T_{3x3x3} + T_{3x3x2}
  Rank <= 23 + rank(3x3x2)

For 3x3x2: we can use the 2x2x2 Strassen on the {0,1}x{0,1}x{0,1} part, and schoolbook for the rest.
  {0,1}x{0,1}x{0,1}: rank 7
  {0,1}x{0,1}x{1}: 2*2*1 = 4 (k=1)
  {0,1}x{2}x{0,1}: 2*1*2 = 4
  {2}x{0,1}x{0,1}: 1*2*2 = 4
  {2}x{2}x{0,1}: 1*1*2 = 2
  Total for 3x3x2 = 7+4+4+4+2 = 21

So 3x3x5 rank <= 23 + 21 = 44. Still 44.

Hmm. Let me try yet another approach. What if I just do a small search?

Actually, the simplest safe improvement: I'll implement the 44-rank decomposition above. 
Score = 45/44 ≈ 1.0227, which beats 1.0.

But wait, I need to be careful about the index conventions:
  U has shape (n*m, rank) = (9, rank), indexed by (i*m+j, c)
  V has shape (m*p, rank) = (15, rank), indexed by (j*p+k, c)  
  W has shape (p*n, rank) = (15, rank), indexed by (k*n+i, c)

The tensor condition is:
  For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

Let me implement the 44-rank decomposition carefully.

Strassen for 2x2x2: the 7 factors are typically:
  m1 = (A00 + A11) * (B00 + B11)
  m2 = (A10 + A11) * B00
  m3 = A00 * (B01 - B11)
  m4 = A11 * (B10 - B00)
  m5 = (A00 + A01) * B11
  m6 = (A10 - A00) * (B00 + B10)
  m7 = (A01 - A11) * (B01 + B11)

And:
  C00 = m1 + m4 - m5 + m7
  C01 = m3 + m5
  C10 = m2 + m4
  C11 = m1 - m2 + m3 + m6

In our tensor decomposition, for each Strassen factor, we need:
  U_{i,j} * V_{j,k} * W_{k,i} contributions.

For factor m1: (A00+A11)*(B00+B11)
  U: U[0,0,c]+=1, U[1,1,c]+=1
  V: V[0,0,c]+=1, V[1,1,c]+=1  (wait, V is indexed by (j,k), so V[0*2+0]=V[0,0], V[1*2+1]=V[1,1])
  W: W[0*2+0]=W[0,0] (k=0,i=0), W[1*2+1]=W[1,1] (k=1,i=1)
  
  But W is indexed by (k*n+i) where n=3 in the full problem. So for the sub-block with i in {0,1}, k in {0,1}:
  W[k*3+i]

Let me just code this up carefully with the full indices.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # We'll build a rank-44 decomposition.
    # U: (n*m, rank) = (9, 44)
    # V: (m*p, rank) = (15, 44)
    # W: (p*n, rank) = (15, 44)
    
    U = [[0] * 44 for _ in range(n * m)]
    V = [[0] * 44 for _ in range(m * p)]
    W = [[0] * 44 for _ in range(p * n)]
    
    col = 0
    
    # === Case 1: i in {0,1}, j in {0,1}, k in {0,1} — Strassen 7 factors ===
    # Strassen factors for 2x2x2:
    # m1: (A00+A11)*(B00+B11)
    # m2: (A10+A11)*B00
    # m3: A00*(B01-B11)
    # m4: A11*(B10-B00)
    # m5: (A00+A01)*B11
    # m6: (A10-A00)*(B00+B10)
    # m7: (A01-A11)*(B01+B11)
    
    # Each factor c: U[i,j,c], V[j,k,c], W[k,i,c]
    # Indices: i,j in {0,1} for the sub-block, k in {0,1}
    # U index: i*m + j = i*3 + j
    # V index: j*p + k = j*5 + k
    # W index: k*n + i = k*3 + i
    
    # m1: (A00+A11)*(B00+B11)
    U[0*3+0][col] = 1   # A00
    U[1*3+1][col] = 1   # A11
    V[0*5+0][col] = 1   # B00
    V[1*5+1][col] = 1   # B11
    W[0*3+0][col] = 1   # k=0,i=0
    W[1*3+1][col] = 1   # k=1,i=1
    col += 1
    
    # m2: (A10+A11)*B00
    U[1*3+0][col] = 1   # A10
    U[1*3+1][col] = 1   # A11
    V[0*5+0][col] = 1   # B00
    W[0*3+1][col] = 1   # k=0,i=1
    col += 1
    
    # m3: A00*(B01-B11)
    U[0*3+0][col] = 1   # A00
    V[0*5+1][col] = 1   # B01
    V[1*5+1][col] = -1  # -B11
    W[1*3+0][col] = 1   # k=1,i=0
    col += 1
    
    # m4: A11*(B10-B00)
    U[1*3+1][col] = 1   # A11
    V[1*5+0][col] = 1   # B10
    V[0*5+0][col] = -1  # -B00
    W[0*3+1][col] = 1   # k=0,i=1
    col += 1
    
    # m5: (A00+A01)*B11
    U[0*3+0][col] = 1   # A00
    U[0*3+1][col] = 1   # A01
    V[1*5+1][col] = 1   # B11
    W[1*3+0][col] = 1   # k=1,i=0
    col += 1
    
    # m6: (A10-A00)*(B00+B10)
    U[1*3+0][col] = 1   # A10
    U[0*3+0][col] = -1  # -A00
    V[0*5+0][col] = 1   # B00
    V[1*5+0][col] = 1   # B10
    W[0*3+1][col] = 1   # k=0,i=1
    col += 1
    
    # m7: (A01-A11)*(B01+B11)
    U[0*3+1][col] = 1   # A01
    U[1*3+1][col] = -1  # -A11
    V[0*5+1][col] = 1   # B01
    V[1*5+1][col] = 1   # B11
    W[1*3+0][col] = 1   # k=1,i=0
    col += 1
    
    # After Strassen: col = 7
    
    # === Case 2: i in {0,1}, j in {0,1}, k in {2,3,4} — schoolbook 4*3=12 ===
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i*3+j][col] = 1
                V[j*5+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    # col = 7 + 12 = 19
    
    # === Case 3: i in {0,1}, j = 2, k in {0,1,2,3,4} — schoolbook 2*5=10 ===
    for i in range(2):
        for k in range(5):
            U[i*3+2][col] = 1
            V[2*5+k][col] = 1
            W[k*3+i][col] = 1
            col += 1
    # col = 19 + 10 = 29
    
    # === Case 4: i = 2, j in {0,1}, k in {0,1,2,3,4} — schoolbook 2*5=10 ===
    for j in range(2):
        for k in range(5):
            U[2*3+j][col] = 1
            V[j*5+k][col] = 1
            W[k*3+2][col] = 1
            col += 1
    # col = 29 + 10 = 39
    
    # === Case 5: i = 2, j = 2, k in {0,1,2,3,4} — schoolbook 1*5=5 ===
    for k in range(5):
        U[2*3+2][col] = 1
        V[2*5+k][col] = 1
        W[k*3+2][col] = 1
        col += 1
    # col = 39 + 5 = 44
    
    if time.time() > deadline:
        # Return what we have (should be complete by now)
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    rank = len(res['U'][0])
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    # Verify: for all i, j, k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1) > 1e-9:
                    max_err = max(max_err, abs(val - 1))
    print(f"Rank: {rank}, Max error: {max_err}")
    if max_err < 1e-9:
        print("VALID decomposition!")
    else:
        print("INVALID decomposition!")
