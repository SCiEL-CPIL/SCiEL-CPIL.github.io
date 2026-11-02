"""Program for matrix multiplication 4x5 * 5x5 with rank reduction."""
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-99 decomposition of the 4x5 * 5x5 matrix multiplication tensor.
    
    We start from the schoolbook rank-100 decomposition and apply Strassen's
    2x2x2 rank-7 trick to reduce by 1 rank.
    
    The trick: for a 2x2 * 2x2 multiplication (indices i in {0,1}, j in {0,1}, k in {0,1}),
    the 8 schoolbook terms can be replaced by 7 terms using:
      m1 = (a11+a22)(b11+b22)
      m2 = (a21+a22)b11
      m3 = a11(b12-b22)
      m4 = a22(b21-b11)
      m5 = (a11+a12)b22
      m6 = (a21-a11)(b11+b12)
      m7 = (a12-a22)(b21+b22)
    
    Then:
      c11 = m1 + m4 - m5 + m7
      c12 = m3 + m5
      c21 = m2 + m4
      c22 = m1 - m2 + m3 + m6
    
    In the tensor decomposition, this means we can merge certain terms.
    
    Let's index the schoolbook terms as (i,j,k) for i in 0..3, j in 0..4, k in 0..4.
    We'll apply the trick to i in {0,1}, j in {0,1}, k in {0,1}.
    
    The 8 terms to replace are: (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
    
    Strassen's 7 terms (in the subspace i in {0,1}, j in {0,1}, k in {0,1}):
      s1 = sum of all 8 terms = (x_i0+y_i0+... wait, let me think more carefully.
    
    Actually, in the tensor decomposition, each term (i,j,k) corresponds to:
      U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    
    The Strassen trick works by introducing new "multiplication" steps. But in the
    tensor decomposition framework, we're looking for a sum of rank-1 tensors.
    
    The key insight: Strassen shows that the 2x2x2 matrix multiplication tensor has
    rank 7, not 8. This means we can express the sum of the 8 rank-1 terms as a sum
    of 7 rank-1 terms.
    
    Let me denote the 8 schoolbook terms for i in {0,1}, j in {0,1}, k in {0,1}:
      T(i,j,k) = e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}  (simplified)
    
    Strassen's decomposition:
      T_total = sum_{i,j,k} T(i,j,k) = sum_{r=1}^{7} T'_r
    
    where T'_r are 7 rank-1 tensors.
    
    The explicit Strassen rank-7 decomposition:
      T1 = (e_{0,0}+e_{1,1}) ⊗ (e_{0,0}+e_{0,1}) ⊗ (e_{0,0}+e_{1,1})
      T2 = (e_{1,0}+e_{1,1}) ⊗ e_{0,0} ⊗ (e_{1,0}+e_{0,0})  
      T3 = e_{0,0} ⊗ (e_{0,1}-e_{1,1}) ⊗ e_{0,0}
      T4 = e_{1,1} ⊗ (e_{1,0}-e_{0,0}) ⊗ e_{1,1}
      T5 = (e_{0,0}+e_{0,1}) ⊗ e_{1,1} ⊗ (e_{0,1}+e_{1,1})
      T6 = (e_{1,0}-e_{0,0}) ⊗ (e_{0,0}+e_{0,1}) ⊗ (e_{1,0}+e_{0,1})
      T7 = (e_{0,1}-e_{1,1}) ⊗ (e_{1,0}+e_{1,1}) ⊗ (e_{0,0}+e_{1,0})
    
    Wait, I need to be more careful with the indices. Let me use the standard Strassen
    decomposition for the 2x2 matrix multiplication tensor.
    
    For C = AB where A, B, C are 2x2:
      A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[p, q], [r, s]]
      
      p = ae + bg, q = af + bh, r = ce + dg, s = cf + dh
      
      Strassen:
        m1 = (a+d)(e+h)
        m2 = (c+d)e
        m3 = a(f-h)
        m4 = d(g-e)
        m5 = (a+b)h
        m6 = (c-a)(e+f)
        m7 = (b-d)(g+h)
        
        p = m1 + m4 - m5 + m7
        q = m3 + m5
        r = m2 + m4
        s = m1 - m2 + m3 + m6
    
    In the tensor decomposition, each "multiplication" m_r corresponds to a rank-1 term.
    The output coefficients come from the linear combinations.
    
    For the full 4x5 * 5x5 problem, I'll apply this to the sub-block where:
      i in {0,1} (rows of A)
      j in {0,1} (cols of A / rows of B)
      k in {0,1} (cols of B / cols of C)
    
    The remaining terms (i in {2,3} or j in {2,3,4} or k in {2,3,4}) stay as schoolbook.
    
    Number of schoolbook terms in the full problem: 4*5*5 = 100
    Terms in the 2x2x2 sub-block: 8
    We replace 8 with 7, so total rank = 100 - 8 + 7 = 99.
    
    Now let me construct the U, V, W matrices.
    
    For the non-reduced terms, they're simple rank-1: U[i*5+j, col]=1, V[j*5+k, col]=1, W[k*4+i, col]=1
    
    For the Strassen terms, each rank-1 term has entries that are sums/differences of basis vectors.
    """
    n, m, p = 4, 5, 5
    rank = 99
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Initialize with schoolbook for all terms, then modify the 2x2x2 sub-block
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this is in the 2x2x2 sub-block we want to replace
                if i < 2 and j < 2 and k < 2:
                    continue  # Skip these, will handle separately
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Now add the 7 Strassen terms for the 2x2x2 sub-block
    # Using the Strassen decomposition above
    
    # m1 = (a+d)(e+h): 
    #   A part: a + d = e_{0,0} + e_{1,1} => U[0*5+0] and U[1*5+1]
    #   B part: e + h = e_{0,0} + e_{0,1} => V[0*5+0] and V[0*5+1]
    #   C part: p + s coefficients... wait, this is the "multiplication" step
    #   Actually in tensor decomposition, the third factor gives the output coefficients
    
    # Let me re-derive. The tensor T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_{(j,k)->output}
    # For matrix mult C = AB, T_{(i,k),(j),(j->k output)}... 
    
    # Actually, the standard way: the matrix multiplication tensor M_{n,m,p} has
    # M_{(i,j),(j,k),(k,i)} = 1 (in some indexing).
    
    # Let me use a cleaner approach. I'll directly construct the 7 rank-1 tensors
    # for the Strassen decomposition.
    
    # For the sub-block i in {0,1}, j in {0,1}, k in {0,1}:
    # Each Strassen term m_r has:
    #   U-part: combination of e_{i,j} for i in {0,1}, j in {0,1}
    #   V-part: combination of e_{j,k} for j in {0,1}, k in {0,1}
    #   W-part: combination of e_{k,i} for k in {0,1}, i in {0,1}
    
    # m1 = (a+d)(e+h):
    #   a = A[0,0], d = A[1,1], e = B[0,0], h = B[0,1]
    #   U: A[0,0] + A[1,1] => U[0*5+0] += 1, U[1*5+1] += 1
    #   V: B[0,0] + B[0,1] => V[0*5+0] += 1, V[0*5+1] += 1
    #   W: This contributes to C[0,0] and C[1,1] with coefficient 1 each? 
    #   From p = ... + m1, s = m1 - ... => W[k*n+i]: 
    #   m1 contributes to p (C[0,0]) with +1 and s (C[1,1]) with +1
    #   So W[0*4+0] += 1, W[1*4+1] += 1
    
    # m2 = (c+d)e:
    #   c = A[1,0], d = A[1,1], e = B[0,0]
    #   U: A[1,0] + A[1,1] => U[1*5+0] += 1, U[1*5+1] += 1
    #   V: B[0,0] => V[0*5+0] += 1
    #   W: contributes to r (C[1,0]) with +1, s (C[1,1]) with -1
    #   W[0*4+1] += 1, W[1*4+1] += -1
    
    # m3 = a(f-h):
    #   a = A[0,0], f = B[0,1], h = B[0,1]... wait h = B[1,1]
    #   f = B[0,1], h = B[1,1]
    #   U: A[0,0] => U[0*5+0] += 1
    #   V: B[0,1] - B[1,1] => V[0*5+1] += 1, V[1*5+1] += -1
    #   W: contributes to q (C[0,1]) with +1, s (C[1,1]) with +1
    #   W[1*4+0] += 1, W[1*4+1] += 1
    
    # m4 = d(g-e):
    #   d = A[1,1], g = B[1,0], e = B[0,0]
    #   U: A[1,1] => U[1*5+1] += 1
    #   V: B[1,0] - B[0,0] => V[1*5+0] += 1, V[0*5+0] += -1
    #   W: contributes to p (C[0,0]) with +1, r (C[1,0]) with +1
    #   W[0*4+0] += 1, W[0*4+1] += 1
    
    # m5 = (a+b)h:
    #   a = A[0,0], b = A[0,1], h = B[1,1]
    #   U: A[0,0] + A[0,1] => U[0*5+0] += 1, U[0*5+1] += 1
    #   V: B[1,1] => V[1*5+1] += 1
    #   W: contributes to p (C[0,0]) with -1, q (C[0,1]) with +1
    #   W[0*4+0] += -1, W[1*4+0] += 1
    
    # m6 = (c-a)(e+f):
    #   c = A[1,0], a = A[0,0], e = B[0,0], f = B[0,1]
    #   U: A[1,0] - A[0,0] => U[1*5+0] += 1, U[0*5+0] += -1
    #   V: B[0,0] + B[0,1] => V[0*5+0] += 1, V[0*5+1] += 1
    #   W: contributes to s (C[1,1]) with +1
    #   W[1*4+1] += 1
    
    # m7 = (b-d)(g+h):
    #   b = A[0,1], d = A[1,1], g = B[1,0], h = B[1,1]
    #   U: A[0,1] - A[1,1] => U[0*5+1] += 1, U[1*5+1] += -1
    #   V: B[1,0] + B[1,1] => V[1*5+0] += 1, V[1*5+1] += 1
    #   W: contributes to p (C[0,0]) with +1, r (C[1,0]) with +1
    #   W[0*4+0] += 1, W[0*4+1] += 1
    
    # Now let's fill in columns 92 through 98 (0-indexed: col starts at 92 after 92 schoolbook terms)
    # Wait, let me recount: total schoolbook terms = 100, we skip 8, so 92 schoolbook terms.
    # Then 7 Strassen terms, total = 99.
    
    # Current col is 92
    
    # m1
    U[0*5+0, col] += 1; U[1*5+1, col] += 1
    V[0*5+0, col] += 1; V[0*5+1, col] += 1
    W[0*4+0, col] += 1; W[1*4+1, col] += 1
    col += 1
    
    # m2
    U[1*5+0, col] += 1; U[1*5+1, col] += 1
    V[0*5+0, col] += 1
    W[0*4+1, col] += 1; W[1*4+1, col] += -1
    col += 1
    
    # m3
    U[0*5+0, col] += 1
    V[0*5+1, col] += 1; V[1*5+1, col] += -1
    W[1*4+0, col] += 1; W[1*4+1, col] += 1
    col += 1
    
    # m4
    U[1*5+1, col] += 1
    V[1*5+0, col] += 1; V[0*5+0, col] += -1
    W[0*4+0, col] += 1; W[0*4+1, col] += 1
    col += 1
    
    # m5
    U[0*5+0, col] += 1; U[0*5+1, col] += 1
    V[1*5+1, col] += 1
    W[0*4+0, col] += -1; W[1*4+0, col] += 1
    col += 1
    
    # m6
    U[1*5+0, col] += 1; U[0*5+0, col] += -1
    V[0*5+0, col] += 1; V[0*5+1, col] += 1
    W[1*4+1, col] += 1
    col += 1
    
    # m7
    U[0*5+1, col] += 1; U[1*5+1, col] += -1
    V[1*5+0, col] += 1; V[1*5+1, col] += 1
    W[0*4+0, col] += 1; W[0*4+1, col] += 1
    col += 1
    
    # Convert to lists of integers/floats
    U_list = [[float(x) for x in row] for row in U.tolist()]
    V_list = [[float(x) for x in row] for row in V.tolist()]
    W_list = [[float(x) for x in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
