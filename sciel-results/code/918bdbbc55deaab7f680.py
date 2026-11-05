"""Strassen-style block decomposition for 3x5 * 5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# Deadline guard: must finish well before 150s wall-clock limit.
_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-12):
    """
    Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    The tensor is T[i,j,k] = delta_{j,k} (for A[i,j] * B[j,k] -> C[i,k]).
    We check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if j==k, else 0.
    Actually, the correct check: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal
    1 if j==k (since the tensor element for (i,j,k) in matmul is 1 when j==k, 0 otherwise).
    Wait, let me reconsider. The matrix multiplication tensor T has T[i,j,k] = 1 if j==k, else 0? 
    No. The tensor is defined by T_{ijk} = delta_{jk} for the bilinear form sum_{j,k} A_{ij} B_{jk} = C_{ik}.
    Actually the standard tensor: C_{ik} = sum_j A_{ij} B_{jk}, so T_{i,j,k} = delta_{j,k} is not right.
    The correct tensor: T_{i,j,k} = 1 if j==k... no.
    
    The matrix multiplication tensor M has entries M_{i,j,k} = 1 if j==k, 0 otherwise? 
    Let me think again. The bilinear map is (A,B) -> C where C_{ik} = sum_j A_{ij} B_{jk}.
    So the tensor is T_{i,j,k} = delta_{j,k}? No, that's not standard.
    
    Actually, the tensor is indexed as T[i,j,k] and the bilinear form is:
    C_{ik} = sum_{j,k} T[i,j,k] A[i,j] B[j,k]
    For standard matmul: C_{ik} = sum_j A[i,j] B[j,k], so T[i,j,k] = 1 if j==k, else 0.
    
    Wait, that's not right either. Let me re-read the task.
    "The output factor follows the official trace convention W[k*n+i,r]."
    
    So the decomposition is: C_{ik} = sum_r U[i*m+j, r] ... no.
    
    Standard tensor decomposition of matmul: 
    C_{ik} = sum_r U_{i,m,j,r} V_{j,p,k,r} W_{k,n,i,r}
    But the shapes are U: (n*m, rank), V: (m*p, rank), W: (p*n, rank).
    
    So U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r], summed over r.
    And this should equal 1 if j==k (since C_{ik} = sum_j A_{ij} B_{jk}), else 0.
    
    Wait no: C_{ik} = sum_j A_{ij} B_{jk}. So the coefficient of A_{ij} B_{jk} in C_{ik} is 1 if j==k, 0 otherwise? 
    No, the coefficient is 1 for all j (each term A_{ij}B_{jk} contributes to C_{ik} with coefficient 1).
    So T[i,j,k] = 1 if the indices are consistent... 
    
    Let me think of it as: the tensor T has T[i,j,k] = 1 for all i,j,k where the product A[i,j]*B[j,k] contributes to C[i,k]. 
    In standard matmul, every (i,j,k) triple contributes: C[i,k] = sum_j A[i,j]B[j,k]. 
    So for fixed i,k, we sum over j. The term A[i,j]*B[j,k] appears with coefficient 1.
    So T[i,j,k] = 1 for all valid i,j,k? That's not right because j is the summation index.
    
    I think the correct interpretation: the tensor T_{ijk} = delta_{jk} is WRONG.
    The correct tensor for matmul (n,m,p) has T_{ijk} = 1 if j==k... 
    
    Let me look at the schoolbook code: for each (i,j,k), U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    So the product U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for the column corresponding to (i,j,k), and 0 otherwise.
    Summed over r (columns), for a fixed (i,j,k), the sum is 1 (from the column where all three are 1).
    So T[i,j,k] = 1 for ALL i,j,k. That means the tensor is the all-ones tensor?
    
    That can't be right for matmul. Let me reconsider.
    
    Oh I see - the decomposition represents the bilinear map. The tensor T[i,j,k] should be 1 if the term A[i,j]*B[j,k] appears in C[i,k]. 
    In standard matmul, C[i,k] = sum_j A[i,j]*B[j,k]. So for each i,k,j, the term appears. 
    But the tensor is 3D with indices (i,j,k), and T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to... what?
    
    I think the convention is: T[i,j,k] = 1 if j==k is NOT the case. 
    Actually, I believe T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p) is WRONG.
    
    Let me re-examine: the schoolbook decomposition has rank n*m*p and for each (i,j,k) triple, it sets one entry in each of U,V,W.
    The verification should be: for all i,j,k, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = T[i,j,k].
    
    What is T[i,j,k] for matrix multiplication? 
    C[i,k] = sum_j A[i,j] B[j,k].
    This is a trilinear form: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] ... no, that gives a scalar.
    
    The correct way: the bilinear map has output indexed by (i,k), and for each output index (i,k), the coefficient of A[i',j']*B[j',k'] is 1 if i'=i and k'=k, 0 otherwise.
    So the tensor T has 4 indices? No, it's a 3-way tensor in the context of tensor rank for bilinear maps.
    
    For bilinear maps, the tensor rank is the rank of the 3-tensor T where T[a,b,c] represents the coefficient of input1[a]*input2[b] in output[c].
    But here input1 is A (n x m), input2 is B (m x p), output is C (n x p).
    So we flatten: input1 has n*m entries indexed by (i,j), input2 has m*p entries indexed by (j,k), output has n*p entries indexed by (i,k).
    The tensor T is of shape (n*m, m*p, n*p) where T[i*m+j, j'*p+k, i'*n+k'] = 1 if (i,j) == (i',j') and (j,k) == (j',k')... 
    
    No wait. T[a, b, c] where a indexes A, b indexes B, c indexes C.
    T[a,b,c] = 1 if the term A[a]*B[b] contributes to C[c].
    A[a] = A[i,j] where a = i*m+j.
    B[b] = B[j,k] where b = j*p+k.
    C[c] = C[i,k] where c = i*n+k.
    
    A[i,j]*B[j,k] contributes to C[i,k]. So T[i*m+j, j*p+k, i*n+k] = 1.
    Note: the j in A[i,j] and the j in B[j,k] must be the same for the product to appear in matmul.
    So T[a,b,c] = 1 if a=i*m+j, b=j'*p+k, c=i'*n+k' AND i=i', k=k', j=j'.
    Otherwise 0.
    
    So T[i*m+j, j*p+k, i*n+k] = 1, and all other entries are 0.
    
    The schoolbook decomposition: for each (i,j,k), set U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1.
    Note W is indexed as k*n+i, not i*n+k. The task says "W[k*n+i,r]" which is a specific ordering.
    
    So for the schoolbook, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k'*n+i',r] where we evaluate at specific (i,j,k,i',j',k').
    Actually, the verification is: for all (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c].
    
    For schoolbook: U[a,r] is 1 only for the column r corresponding to (i,j) where a=i*m+j, and only if r is that specific column.
    So sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 if there exists a column r where all three are 1.
    In schoolbook, column (i,j,k) has U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    So the sum is 1 when (a=i*m+j, b=j*p+k, c=k*n+i) for some i,j,k, i.e., when a=i*m+j, b=j*p+k, c=k*n+i with consistent i,j,k.
    This means T[i*m+j, j*p+k, k*n+i] = 1.
    
    But we said T[i*m+j, j*p+k, i*n+k] should be 1 for matmul. 
    There's a mismatch in the W indexing! The task uses W[k*n+i] but the natural matmul tensor would use W[i*n+k].
    
    Hmm, let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    So W is indexed as W[k*n+i, r] where k is the column of B (and row of... no).
    Let me just accept the convention and verify that schoolbook works with it.
    
    Actually, I think the key insight is: the decomposition must satisfy:
    For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    And for all other combinations, it should be 0.
    
    But what are "other combinations"? The full condition is:
    For all (a,b,c) where a in [0,n*m), b in [0,m*p), c in [0,p*n):
    sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    And T[a,b,c] = 1 if a=i*m+j, b=j'*p+k, c=k'*n+i' with i=i', j=j', k=k', else 0.
    Wait, that requires j=j' (the inner dimension matches) AND i=i' AND k=k'.
    So T[i*m+j, j*p+k, k*n+i] = 1, and T[a,b,c] = 0 otherwise.
    
    Yes! So the tensor has T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k, and 0 elsewhere.
    
    Let me verify schoolbook satisfies this:
    sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r]
    In schoolbook, for the column corresponding to (i,j,k): U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    So the sum includes this term = 1.
    For other columns r', U[i*m+j,r']=0 or V[j*p+k,r'=0 or W[k*n+i,r']=0.
    So the sum is exactly 1.
    
    Now check T[a,b,c] for a different combination, say a=i*m+j, b=j'*p+k, c=k*n+i where j != j'.
    sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k*n+i,r]
    In schoolbook, U[i*m+j,r] is nonzero only for columns corresponding to (i,j,k') for various k'.
    For such a column r=(i,j,k'), V[j'*p+k,r] = V[j'*p+k, (i,j,k')] which is 1 only if j'=j and k'=k.
    Since j != j', V is 0. So the sum is 0. Good.
    
    So the verification condition is:
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    And for all other (a,b,c) not of the form (i*m+j, j*p+k, k*n+i) with consistent indices, the sum should be 0.
    
    But checking all (a,b,c) is expensive. A sufficient check is to verify the tensor equation.
    We can do this by: for each r, form the matrix M_r = outer(U[:,r], V[:,r]) ... no, that's not right.
    
    Actually, we can verify by: the 3-tensor T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product) should equal the target tensor.
    We can check this by: for a random or systematic set of (i,j,k) where j is the inner index, check the value.
    
    Let me just check all valid (i,j,k) triples and also some invalid ones.
    """
    rank = U.shape[1]
    # Check all valid (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1) > tol:
                    return False
    # Check some invalid triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for j2 in range(m):
                    if j2 == j:
                        continue
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j2*p+k, r] * W[k*n+i, r]
                    if abs(s) > tol:
                        return False
    return True

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication (rank 7).
    U: (2*2, 7), V: (2*2, 7), W: (2*2, 7) with W indexed as k*n+i.
    n=m=p=2.
    """
    # Standard Strassen 7 products for 2x2 matmul
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Here A is 2x2, B is 2x2, C is 2x2.
    # U[i*m+j, r] where i,j in {0,1}, m=2
    # V[j*p+k, r] where j,k in {0,1}, p=2
    # W[k*n+i, r] where k,i in {0,1}, n=2
    #
    # The 7 Strassen products correspond to 7 terms.
    # For each Strassen product t, we need to express it as sum of products of linear combinations.
    
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indices: A[i,j], B[j,k], C[i,k]
    # M1 = (A00+A11)(B00+B11)
    # Expand: A00*B00 + A00*B11 + A11*B00 + A11*B11
    # For W[k*n+i]: 
    #   A00*B00 contributes to C00: W[0*2+0,0]=W[0,0]
    #   A00*B11 contributes to C01: W[1*2+0,0]=W[2,0]
    #   A11*B00 contributes to C10: W[0*2+1,0]=W[1,0]
    #   A11*B11 contributes to C11: W[1*2+1,0]=W[3,0]
    # But Strassen uses the SUM as one product, so we need to distribute.
    
    # Actually, in the tensor decomposition, each of the 7 Strassen products is ONE term.
    # The product M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # This single product contributes to C as:
    # C00 gets +1 * M1
    # C01 gets +1 * M1
    # C10 gets +1 * M1
    # C11 gets +1 * M1
    # Wait no, M1 is a scalar, and it contributes to multiple C entries.
    
    # In the tensor decomposition: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # For a given r (Strassen product), U[i*m+j,r] is the coefficient of A[i,j] in that product,
    # V[j*p+k,r] is the coefficient of B[j,k] in that product,
    # W[k*n+i,r] is the coefficient that goes into C[i,k].
    
    # For M1 = (A00+A11)(B00+B11):
    # U[0*2+0, 0] = 1 (A00)
    # U[1*2+1, 0] = 1 (A11)
    # V[0*2+0, 0] = 1 (B00)
    # V[1*2+1, 0] = 1 (B11)
    # Now W: M1 contributes to C00, C01, C10, C11 each with coefficient 1.
    # W[0*2+0, 0] = 1 (C00)
    # W[1*2+0, 0] = 1 (C01)
    # W[0*2+1, 0] = 1 (C10)
    # W[1*2+1, 0] = 1 (C11)
    
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = 1; W[3, 0] = 1  # C00, C01, C10, C11
    
    # M2 = (A10+A11)B00
    # U[1*2+0, 1] = 1 (A10)
    # U[1*2+1, 1] = 1 (A11)
    # V[0*2+0, 1] = 1 (B00)
    # M2 contributes to C10 with +1 and C11 with -1
    # W[0*2+1, 1] = 1 (C10)
    # W[1*2+1, 1] = -1 (C11)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B10-B11)
    # U[0*2+0, 2] = 1 (A00)
    # V[1*2+0, 2] = 1 (B10)
    # V[1*2+1, 2] = -1 (B11)
    # M3 contributes to C01 with +1 and C11 with +1
    # W[1*2+0, 2] = 1 (C01)
    # W[1*2+1, 2] = 1 (C11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B01-B00)
    # U[1*2+1, 3] = 1 (A11)
    # V[0*2+1, 3] = 1 (B01)
    # V[0*2+0, 3] = -1 (B00)
    # M4 contributes to C00 with +1 and C10 with +1
    # W[0*2+0, 3] = 1 (C00)
    # W[0*2+1, 3] = 1 (C10)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11
    # U[0*2+0, 4] = 1 (A00)
    # U[0*2+1, 4] = 1 (A01)
    # V[1*2+1, 4] = 1 (B11)
    # M5 contributes to C00 with -1 and C01 with +1
    # W[0*2+0, 4] = -1 (C00)
    # W[1*2+0, 4] = 1 (C01)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    # U[1*2+0, 5] = 1 (A10)
    # U[0*2+0, 5] = -1 (A00)
    # V[0*2+0, 5] = 1 (B00)
    # V[0*2+1, 5] = 1 (B01)
    # M6 contributes to C11 with +1
    # W[1*2+1, 5] = 1 (C11)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # U[0*2+1, 6] = 1 (A01)
    # U[1*2+1, 6] = -1 (A11)
    # V[1*2+0, 6] = 1 (B10)
    # V[1*2+1, 6] = 1 (B11)
    # M7 contributes to C00 with +1
    # W[0*2+0, 6] = 1 (C00)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _block_decomposition(n, m, p):
    """
    Try to build a block decomposition using Strassen where possible.
    For 3x5 * 5x6:
    We can try to decompose into blocks. 
    n=3, m=5, p=6.
    
    One approach: partition A into 2x2 blocks where possible, use Strassen for those blocks.
    But 3, 5, 6 don't divide evenly into 2s.
    
    Alternative: use a mixed decomposition. 
    Let's try: n=3 -> 1+2, m=5 -> 1+4 or 2+3, p=6 -> 2+4 or 3+3.
    
    Actually, let me try a simpler approach: use Strassen on a 2x2x2 subproblem embedded in the full problem.
    If we can identify a 2x2x2 substructure, we can replace the 8 schoolbook terms with 7 Strassen terms, saving 1.
    
    For 3x5 * 5x6, let's pick rows 0,1 of A, columns 0,1 of B, and produce C[0,1;x,0,1] (a 2x2 block of C).
    This subproblem is 2x2 * 2x2 -> 2x2, which can be done with Strassen rank 7 instead of 8.
    The remaining terms use schoolbook.
    
    Total schoolbook rank: 3*5*6 = 90.
    We replace 8 terms (for the 2x2x2 subblock) with 7, saving 1. Rank = 89.
    
    But we might be able to do better with more careful partitioning.
    
    Let's first implement this simple improvement.
    """
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    rank = U.shape[1]
    
    # Find a 2x2x2 subblock to replace with Strassen
    # Pick i1,i2 from {0,1,2}, j1,j2 from {0..4}, k1,k2 from {0..5}
    # Let's use i1=0,i2=1, j1=0,j2=1, k1=0,k2=1
    i1, i2 = 0, 1
    j1, j2 = 0, 1
    k1, k2 = 0, 1
    
    # The 8 schoolbook columns for this subblock are:
    # (i1,j1,k1), (i1,j1,k2), (i1,j2,k1), (i1,j2,k2),
    # (i2,j1,k1), (i2,j1,k2), (i2,j2,k1), (i2,j2,k2)
    # Find their column indices
    cols_to_replace = []
    for ii in [i1, i2]:
        for jj in [j1, j2]:
            for kk in [k1, k2]:
                # Find column where U[ii*m+jj] = 1 and V[jj*p+kk] = 1 and W[kk*n+ii] = 1
                for col in range(rank):
                    if (abs(U[ii*m+jj, col] - 1) < 1e-12 and 
                        abs(V[jj*p+kk, col] - 1) < 1e-12 and 
                        abs(W[kk*n+ii, col] - 1) < 1e-12):
                        cols_to_replace.append(col)
                        break
    
    if len(cols_to_replace) != 8:
        # Fallback to schoolbook
        return U, V, W
    
    # Remove these 8 columns and replace with 7 Strassen columns
    # New rank = 90 - 8 + 7 = 89
    new_rank = rank - 1
    
    U_new = np.zeros((n*m, new_rank))
    V_new = np.zeros((m*p, new_rank))
    W_new = np.zeros((p*n, new_rank))
    
    # Copy all columns except the 8 we're replacing
    keep_cols = [c for c in range(rank) if c not in cols_to_replace]
    for old_col, new_col in enumerate(keep_cols):
        U_new[:, new_col] = U[:, old_col]
        V_new[:, new_col] = V[:, old_col]
        W_new[:, new_col] = W[:, old_col]
    
    # Add 7 Strassen columns
    S_U, S_V, S_W = _strassen_2x2x2()
    for r in range(7):
        new_col = len(keep_cols) + r
        # Map Strassen local indices to global
        # Local i=0,1 -> global i1,i2
        # Local j=0,1 -> global j1,j2
        # Local k=0,1 -> global k1,k2
        for li in range(2):
            for lj in range(2):
                U_new[[i1, i2][li]*m + [j1, j2][lj], new_col] = S_U[li*2+lj, r]
        for lj in range(2):
            for lk in range(2):
                V_new[[j1, j2][lj]*p + [k1, k2][lk], new_col] = S_V[lj*2+lk, r]
        for lk in range(2):
            for li in range(2):
                W_new[[k1, k2][lk]*n + [i1, i2][li], new_col] = S_W[lk*2+li, r]
    
    return U_new, V_new, W_new

def construct():
    n, m, p = DIMENSIONS
    # Try block decomposition
    U, V, W = _block_decomposition(n, m, p)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        # Convert to lists with integer/half-integer entries
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
