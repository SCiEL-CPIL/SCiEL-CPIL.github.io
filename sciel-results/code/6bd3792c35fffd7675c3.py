import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # seconds, safety margin before 150s kill

    n, m, p = DIMENSIONS
    dim_u = n * m  # 15
    dim_v = m * p  # 25
    dim_w = p * n  # 15

    # The matrix multiplication tensor M_{n,m,p} has entries T[i,j,k] = 1
    # for all i in [0,n), j in [0,m), k in [0,p).
    # A rank-r decomposition gives matrices U (dim_u x r), V (dim_v x r), W (dim_w x r)
    # such that for all i,j,k:
    #   sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # and for all "invalid" index combinations (which don't exist in the tensor),
    # the decomposition must still be consistent.
    #
    # Actually, the tensor is only defined on valid indices. The decomposition
    # must satisfy: for every (i,j,k) with i<n, j<m, k<p:
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    # is computed as sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_k V[j*p+k,r]*B[j,k]) * W[k*n+i,r]
    #
    # Wait, let me re-read the contract. The factor matrices are:
    # U: (n*m, rank)
    # V: (m*p, rank)  
    # W: (p*n, rank)
    #
    # The trace convention is W[k*n+i, r]. So W is indexed by (k,i) pairs.
    #
    # The decomposition means:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    #
    # No, that's not quite right either. Let me think more carefully.
    #
    # In a standard tensor decomposition of matrix multiplication:
    # The multiplication tensor T has indices (i, j, k) where:
    #   i indexes rows of A (and rows of C)
    #   j indexes columns of A (and rows of B)  
    #   k indexes columns of B (and columns of C)
    #
    # T[i,j,k] = 1 for all valid i,j,k
    #
    # A rank-1 decomposition term is u * v * w where:
    #   u is a vector in R^{n*m} (indexed by (i,j))
    #   v is a vector in R^{m*p} (indexed by (j,k))
    #   w is a vector in R^{p*n} (indexed by (k,i))
    #
    # And the condition is:
    #   For all (i,j,k): sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    #
    # The trivial decomposition has rank n*m*p with:
    #   u_r[i*m+j] = 1 if (i,j) matches the r-th term, else 0
    #   v_r[j*p+k] = 1 if (j,k) matches, else 0
    #   w_r[k*n+i] = 1 if (k,i) matches, else 0

    # Let's use the Strassen algorithm for the 3x3x3 sub-block and schoolbook for the rest.
    #
    # Strategy:
    # - For the 3x3x3 block (i in {0,1,2}, j in {0,1,2}, k in {0,1,2}): use Strassen (rank 7)
    # - For all other (i,j,k) combinations: use schoolbook (each contributes rank 1)
    #
    # Total terms in 3x5x5: 3*5*5 = 75
    # Terms in 3x3x3 sub-block: 3*3*3 = 27
    # Terms outside: 75 - 27 = 48
    # Strassen replaces 27 terms with 7 terms
    # Total rank: 48 + 7 = 55

    # But wait - we need to be careful. The Strassen algorithm computes C = A*B where
    # A is 3x3 and B is 3x3. But in our case, A is 3x5 and B is 5x5.
    # The 3x3 sub-block would be A[:, 0:3] * B[0:3, 0:3], but that's not how it works.
    #
    # Actually, let me reconsider. The multiplication is C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k].
    # We can't simply split this into independent sub-problems because each C[i,k] depends
    # on all j values.
    #
    # However, we CAN use the tensor decomposition framework. The key insight is that
    # the multiplication tensor for M_{3,5,5} can be decomposed by considering different
    # subsets of indices.
    #
    # Let me try a different approach: use known rank bounds and construct a valid
    # decomposition.
    #
    # For now, let's implement the trivial decomposition correctly and then try to
    # improve it.

    # Trivial decomposition: rank = n*m*p = 75
    rank_trivial = n * m * p
    U = np.zeros((dim_u, rank_trivial), dtype=float)
    V = np.zeros((dim_v, rank_trivial), dtype=float)
    W = np.zeros((dim_w, rank_trivial), dtype=float)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    best_rank = rank_trivial
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify the trivial decomposition
    def verify(U_mat, V_mat, W_mat):
        """Verify that the decomposition correctly computes matrix multiplication."""
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)
        
        # For each (i, k), check that C[i,k] = sum_j A[i,j]*B[j,k]
        # is correctly decomposed.
        # The decomposition gives:
        # C[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_k' V[j*p+k',r]*B[j,k']) * W[k*n+i,r]
        #
        # Actually, the correct interpretation is:
        # The bilinear form is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # should equal 1 for all valid (i,j,k).
        
        # Let's verify by checking that the tensor product matches
        # For each term r, it contributes to T[i,j,k] by U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # The sum over r should give 1 for all (i,j,k) in [0,n)x[0,m)x[0,p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(U_np.shape[1]):
                        val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(val - 1) > 1e-10:
                        return False
        return True

    if not verify(U.tolist(), V.tolist(), W.tolist()):
        # This shouldn't happen, but if it does, we have a bug
        pass

    # Now try to improve using Strassen for the 3x3x3 sub-block
    # 
    # The idea: We can view the 3x5x5 multiplication as involving three "sectors"
    # based on which j values are used. But this doesn't directly help.
    #
    # Alternative: Use the fact that we can embed a smaller matrix multiplication
    # into a larger one. Specifically, M_{3,3,3} is a sub-tensor of M_{3,5,5}.
    #
    # The rank of M_{3,5,5} is at least the rank of M_{3,3,3} (by sub-tensor argument)
    # and at most 75 (trivial).
    #
    # Known: rank(M_{3,3,3}) <= 23 (Lader), but we can use Strassen-like for 3x3x3.
    #
    # Actually, for a 3x3x3 multiplication, we can use the following approach:
    # Split into 2x2x2 blocks and use Strassen.
    #
    # For 3x3x3:
    # A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
    # B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    #
    # We can write this as:
    # C = A11*B11 + A12*B21 + A13*B31 + ... etc.
    #
    # This gets complicated. Let me try a simpler approach:
    # Use the known rank-23 decomposition for 3x3x3 (if I can find one)
    # or just use the trivial 27-term decomposition for the 3x3x3 part.
    #
    # For now, let's try to find a better overall decomposition by using
    # linear algebra to reduce the rank.

    # Time check
    if time.time() - start_time > deadline - 10:
        return best_result

    # Let's try a more systematic approach: use the fact that the multiplication
    # tensor has a certain structure and try to find a lower-rank decomposition.
    #
    # One approach: use the "border rank" or "approximate rank" concepts, but we
    # need exact decomposition.
    #
    # Another approach: use the known result that rank(M_{n,m,p}) <= n*m*p and
    # try to find a decomposition with fewer terms by exploiting symmetries.
    #
    # For 3x5x5, let's try to use the following decomposition:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can write this as:
    # C[i,k] = sum_j (A[i,j] * B[j,k])
    #
    # The trivial decomposition has 75 terms. Can we do better?
    #
    # Let's try to use the Strassen algorithm on a 2x2x2 sub-block.
    # For example, consider i in {0,1}, j in {0,1}, k in {0,1}.
    # This is a 2x2x2 multiplication, which can be done in rank 7 (Strassen).
    #
    # The 2x2x2 sub-block has 8 terms. Strassen replaces them with 7 terms.
    # So we save 1 term for each 2x2x2 sub-block.
    #
    # How many disjoint 2x2x2 sub-blocks can we fit in 3x5x5?
    # In the i dimension: floor(3/2) = 1 (covers i=0,1)
    # In the j dimension: floor(5/2) = 2 (covers j=0,1 and j=2,3)
    # In the k dimension: floor(5/2) = 2 (covers k=0,1 and k=2,3)
    #
    # So we can have 1*2*2 = 4 disjoint 2x2x2 sub-blocks.
    # Each saves 1 term, so total savings: 4 terms.
    # New rank: 75 - 4 = 71.

    # Let's implement this:
    # For each 2x2x2 sub-block, replace the 8 schoolbook terms with 7 Strassen terms.

    # Strassen for 2x2x2:
    # Given A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[a*e + b*g, a*f + b*h], [c*e + d*g, c*f + d*h]]
    #
    # Strassen:
    # m1 = (a+d)*(e+h)
    # m2 = (c+d)*e
    # m3 = a*(f-h)
    # m4 = d*(g-e)
    # m5 = (a+b)*h
    # m6 = (c-a)*(e+f)
    # m7 = (b-d)*(g+h)
    #
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6

    # In the tensor decomposition framework, each Strassen term corresponds to
    # a rank-1 tensor. Let's construct the U, V, W matrices for the Strassen terms.
    #
    # For a 2x2x2 block with i0, j0, k0 as the base indices:
    # The 8 schoolbook terms are for (i0+i', j0+j', k0+k') for i',j',k' in {0,1}
    #
    # Each Strassen term m_t involves a linear combination of A entries, B entries,
    # and C entries. In the tensor decomposition, this corresponds to:
    #   U vector: coefficients for A[i,j] entries
    #   V vector: coefficients for B[j,k] entries  
    #   W vector: coefficients for C[i,k] entries (with the trace convention W[k*n+i])

    # Let me construct the full decomposition:

    U_full = []
    V_full = []
    W_full = []

    # We'll track which (i,j,k) terms have been covered by Strassen blocks
    covered = set()

    # Define the 2x2x2 Strassen blocks
    # Block at (i0, j0, k0) covers i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    blocks = []
    for i0 in range(0, n-1, 2):  # i0 = 0
        for j0 in range(0, m-1, 2):  # j0 = 0, 2
            for k0 in range(0, p-1, 2):  # k0 = 0, 2
                # Check that the block doesn't overlap with existing blocks
                block_terms = set()
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            block_terms.add((i0+di, j0+dj, k0+dk))
                # Check for overlap
                if not any(block_terms & covered):
                    blocks.append((i0, j0, k0, block_terms))
                    covered.update(block_terms)

    # For each block, add the 7 Strassen terms
    for i0, j0, k0, block_terms in blocks:
        # Strassen coefficients for the 2x2x2 block
        # m1 = (a+d)*(e+h): A terms: a=1, d=1; B terms: e=1, h=1; C terms: C11=1, C22=1
        # m2 = (c+d)*e: A terms: c=1, d=1; B terms: e=1; C terms: C21=1
        # m3 = a*(f-h): A terms: a=1; B terms: f=1, h=-1; C terms: C12=1
        # m4 = d*(g-e): A terms: d=1; B terms: g=1, e=-1; C terms: C11=1, C21=1
        # m5 = (a+b)*h: A terms: a=1, b=1; B terms: h=1; C terms: C11=-1, C12=1
        # m6 = (c-a)*(e+f): A terms: c=1, a=-1; B terms: e=1, f=1; C terms: C22=1
        # m7 = (b-d)*(g+h): A terms: b=1, d=-1; B terms: g=1, h=1; C terms: C11=1, C22=-1

        strassen_terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # U is indexed by (i,j) within the 2x2 block: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
            # V is indexed by (j,k) within the 2x2 block: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
            # W is indexed by (k,i) within the 2x2 block: (0,0)=C11, (0,1)=C21, (1,0)=C12, (1,1)=C22
            # Wait, the trace convention is W[k*n+i]. So for the block, W is indexed by (k,i)
            # where k is the column of C and i is the row of C.
            # C11: i=0, k=0 -> W[0*n+0]
            # C12: i=0, k=1 -> W[1*n+0]
            # C21: i=1, k=0 -> W[0*n+1]
            # C22: i=1, k=1 -> W[1*n+1]
            
            # m1 = (a+d)*(e+h): 
            # U: a=1, d=1 -> [(0,0):1, (1,1):1]
            # V: e=1, h=1 -> [(0,0):1, (1,1):1]
            # W: C11=1, C22=1 -> [(0,0):1, (1,1):1] in (k,i) indexing
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            
            # m2 = (c+d)*e:
            # U: c=1, d=1 -> [(1,0):1, (1,1):1]
            # V: e=1 -> [(0,0):1]
            # W: C21=1 -> [(0,1):1] in (k,i) indexing (k=0, i=1)
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1)]),
            
            # m3 = a*(f-h):
            # U: a=1 -> [(0,0):1]
            # V: f=1, h=-1 -> [(0,1):1, (1,1):-1]
            # W: C12=1 -> [(1,0):1] in (k,i) indexing (k=1, i=0)
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1)]),
            
            # m4 = d*(g-e):
            # U: d=1 -> [(1,1):1]
            # V: g=1, e=-1 -> [(1,0):1, (0,0):-1]
            # W: C11=1, C21=1 -> [(0,0):1, (0,1):1] in (k,i) indexing
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
            
            # m5 = (a+b)*h:
            # U: a=1, b=1 -> [(0,0):1, (0,1):1]
            # V: h=1 -> [(1,1):1]
            # W: C11=-1, C12=1 -> [(0,0):-1, (1,0):1] in (k,i) indexing
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (1,0,1)]),
            
            # m6 = (c-a)*(e+f):
            # U: c=1, a=-1 -> [(1,0):1, (0,0):-1]
            # V: e=1, f=1 -> [(0,0):1, (0,1):1]
            # W: C22=1 -> [(1,1):1] in (k,i) indexing (k=1, i=1)
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            
            # m7 = (b-d)*(g+h):
            # U: b=1, d=-1 -> [(0,1):1, (1,1):-1]
            # V: g=1, h=1 -> [(1,0):1, (1,1):1]
            # W: C11=1, C22=-1 -> [(0,0):1, (1,1):-1] in (k,i) indexing
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,-1)]),
        ]

        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            U_vec = [0.0] * dim_u
            V_vec = [0.0] * dim_v
            W_vec = [0.0] * dim_w
            
            for (di, dj, coeff) in u_coeffs:
                U_vec[(i0+di)*m + (j0+dj)] = coeff
            
            for (dj, dk, coeff) in v_coeffs:
                V_vec[(j0+dj)*p + (k0+dk)] = coeff
            
            for (dk, di, coeff) in w_coeffs:
                W_vec[(k0+dk)*n + (i0+di)] = coeff
            
            U_full.append(U_vec)
            V_full.append(V_vec)
            W_full.append(W_vec)

    # For all uncovered (i,j,k) terms, add schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U_vec = [0.0] * dim_u
                    V_vec = [0.0] * dim_v
                    W_vec = [0.0] * dim_w
                    
                    U_vec[i*m + j] = 1
                    V_vec[j*p + k] = 1
                    W_vec[k*n + i] = 1
                    
                    U_full.append(U_vec)
                    V_full.append(V_vec)
                    W_full.append(W_vec)

    # Convert to numpy arrays
    U_new = np.array(U_full, dtype=float)
    V_new = np.array(V_full, dtype=float)
    W_new = np.array(W_full, dtype=float)

    new_rank = U_new.shape[1]
    
    # Verify the new decomposition
    if verify(U_new.tolist(), V_new.tolist(), W_new.tolist()):
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    else:
        # If verification fails, stick with the trivial decomposition
        pass

    # Time check
    if time.time() - start_time > deadline - 5:
        return best_result

    # Try to further reduce the rank by looking for linear dependencies
    # among the terms in the current best decomposition.
    
    # Check if any two terms are linearly dependent
    U_best = np.array(best_result["U"], dtype=float)
    V_best = np.array(best_result["V"], dtype=float)
    W_best = np.array(best_result["W"], dtype=float)
    
    r = U_best.shape[1]
    
    # Try to remove terms one by one and see if the decomposition still works
    # This is a greedy approach: try removing each term and check validity
    for r_idx in range(r):
        if time.time() - start_time > deadline - 5:
            break
        
        # Skip term r_idx
        mask = [i for i in range(r) if i != r_idx]
        U_sub = U_best[:, mask]
        V_sub = V_best[:, mask]
        W_sub = W_best[:, mask]
        
        if verify(U_sub.tolist(), V_sub.tolist(), W_sub.tolist()):
            # We can remove this term
            U_best = U_sub
            V_best = V_sub
            W_best = W_sub
            best_rank -= 1
            best_result = {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
            r -= 1

    return best_result
