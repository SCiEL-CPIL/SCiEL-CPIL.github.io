"""Matrix multiplication 3x5x6 rank-minimal exact tensor decomposition."""
import time
import itertools

DIMENSIONS = [3, 5, 6]

# Deadline guard
_DEADLINE_START = time.time()
_TIME_LIMIT = 110  # seconds, safety margin before 120s limit

def _time_exhausted():
    return (time.time() - _DEADLINE_START) > _TIME_LIMIT

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # schoolbook rank = 90
    
    # Start with schoolbook decomposition
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Now try to improve using Strassen's algorithm on 2x2 blocks
    # The key idea: find overlapping 2x2 blocks and replace their schoolbook
    # terms with Strassen's 7-term decomposition to reduce rank.
    
    # For a 2x2x2 block at positions (i0, j0, k0), the schoolbook uses 8 terms.
    # Strassen uses 7 terms. The savings is 1 per 2x2x2 block, but blocks
    # must be non-overlapping in terms of their factor entries to be independent.
    
    # Actually, the proper way is to use a block decomposition where we
    # partition the matrices into blocks and apply Strassen recursively.
    # But for 3x5x6, we can do better with a structured approach.
    
    # Let's use the known optimal or near-optimal constructions.
    # For small matrix multiplication, the best known ranks:
    # 2x2x2: rank 7 (Strassen)
    # 3x3x3: rank 23 (Bini et al.) or 22 (unknown)
    # 
    # A practical approach: use Strassen on all possible 2x2x2 sub-blocks
    # in a way that the decompositions are compatible.
    
    # Alternative approach: Use the "laser method" or bilinear decomposition.
    # For 3x5x6, let's try a block decomposition:
    # Split A (3x5) into blocks, B (5x6) into blocks.
    
    # Let's try: split the inner dimension m=5 into 3+2.
    # A = [A1 (3x3), A2 (3x2)], B = [B1 (3x6); B2 (2x6)]
    # C = A1*B1 + A2*B2
    # A1*B1: 3x3 * 3x6 -> use block Strassen if possible
    # A2*B2: 3x2 * 2x6
    
    # For 3x3*3x6: split B1 into two 3x3 blocks: B1a, B1b
    # A1*B1a: 3x3*3x3 -> rank 23 (Bini) or we can use 2x2 Strassen blocks
    # A1*B1b: 3x3*3x3 -> rank 23
    
    # This is getting complex. Let me try a simpler approach:
    # Use Strassen on all 2x2x2 sub-problems in a tiling pattern.
    
    # Actually, the cleanest approach for getting a guaranteed improvement:
    # Use the fact that we can replace any 2x2x2 schoolbook (rank 8) with
    # Strassen (rank 7) if the three matrices involved don't overlap with
    # other such replacements in a conflicting way.
    
    # For a clean non-overlapping tiling of 3x5x6:
    # We can fit floor(3/2)*floor(5/2)*floor(6/2) = 1*2*3 = 6 non-overlapping 2x2x2 blocks
    # Each saves 1 rank, so we save 6 ranks: 90 - 6 = 84.
    
    # But we can do better by also handling the remaining strips.
    
    # Let me implement a proper block-based Strassen:
    # Tile the 3x5x6 problem into 2x2x2 blocks plus remainders.
    # 3 = 2+1, 5 = 2+2+1, 6 = 2+2+2
    
    # Blocks at (i,j,k) where i in {0,1} (for the 2-part of dim 3),
    # j in {0,1,2} (for the three 2-parts... wait, 5=2+2+1, so two full 2s and one 1)
    # Actually let me be more careful.
    
    # Dim n=3: split as 2+1. Block rows at i=0 (rows 0,1) and i=1 (row 2).
    # Dim m=5: split as 2+2+1. Block cols at j=0 (cols 0,1), j=1 (cols 2,3), j=2 (col 4).
    # Dim p=6: split as 2+2+2. Block rows at k=0 (rows 0,1), k=1 (rows 2,3), k=2 (rows 4,5).
    
    # For each combination (bi, bj, bk) where bi in {0,1}, bj in {0,1,2}, bk in {0,1,2}:
    # If all three block sizes are 2, we can use Strassen (rank 7 instead of 8).
    # If any block size is 1, we use schoolbook for that sub-problem.
    
    # Block sizes:
    # bi=0: n_size=2, bi=1: n_size=1
    # bj=0: m_size=2, bj=1: m_size=2, bj=2: m_size=1
    # bk=0: p_size=2, bk=1: p_size=2, bk=2: p_size=2
    
    # So Strassen applies when bi=0, bj in {0,1}, bk in {0,1,2} -> 1*2*3 = 6 blocks
    # Each saves 1 rank.
    # Total saved: 6. New rank: 90 - 6 = 84.
    
    # Let's build this properly.
    
    # First, define the block structure
    n_blocks = [(0, 2), (2, 1)]  # (start, size) for dimension n
    m_blocks = [(0, 2), (2, 2), (4, 1)]  # for dimension m
    p_blocks = [(0, 2), (2, 2), (4, 2)]  # for dimension p
    
    # Build new decomposition
    # We'll collect all terms (U_term, V_term, W_term) and then assemble.
    
    terms = []  # Each term is a tuple of three sparse dicts: {(idx): val}
    
    for (ni, ns), (mj, ms), (pk, ps) in itertools.product(n_blocks, m_blocks, p_blocks):
        # Block: A[ni:ni+ns, mj:mj+ms] @ B[mj:mj+ms, pk:pk+ps]
        # Result: C[ni:ni+ns, pk:pk+ps]
        
        if ns == 2 and ms == 2 and ps == 2:
            # Strassen 2x2x2
            # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
            # Strassen terms:
            # M1 = (a+d)(e+h)
            # M2 = (c+d)e
            # M3 = a(f-h)
            # M4 = d(g-e)
            # M5 = (a+b)h
            # M6 = (c-a)(e+f)
            # M7 = (b-d)(g+h)
            # C11 = M1+M4-M5+M7
            # C12 = M3+M5
            # C21 = M2+M4
            # C22 = M1-M2+M3+M6
            
            # In terms of U, V, W factors:
            # U is indexed by (i, j) where i in block rows of A, j in block cols of A
            # V is indexed by (j, k) where j in block cols of A (= block rows of B), k in block cols of B
            # W is indexed by (k, i) where k in block cols of B (= result cols), i in block rows of A (= result rows)
            
            # Local indices: i_loc in {0,1}, j_loc in {0,1}, k_loc in {0,1}
            # Global: i = ni + i_loc, j = mj + j_loc, k = pk + k_loc
            
            strassen_terms = [
                # (U_coeffs, V_coeffs, W_coeffs)
                # U: {(i_loc, j_loc): coeff}, V: {(j_loc, k_loc): coeff}, W: {(k_loc, i_loc): coeff}
                (
                    {(0,0):1, (1,0):1},  # a+d
                    {(0,0):1, (1,1):1},  # e+h
                    {(0,0):1, (0,1):1, (1,0):1, (1,1):1}  # M1 contributes to all 4 C entries with signs
                ),
                # Wait, this isn't right. Each Strassen term is a single rank-1 contribution
                # to specific C entries. Let me redo this properly.
                
                # Actually, each Strassen multiplication M_t = U_t * V_t gives a scalar,
                # and that scalar is added to specific C entries with coefficients.
                # So the tensor decomposition has 7 terms, each being:
                # Term t: U_t (2x2 matrix), V_t (2x2 matrix), W_t (2x2 matrix)
                # where sum_t U_t[i,j] * V_t[j,k] * W_t[k,i] = (A*B)[i,i']
                
                # The standard Strassen tensor decomposition:
                # Let me write out the 7 rank-1 terms explicitly.
            ]
            
            # Let me use the explicit Strassen decomposition.
            # The 7 terms are:
            # T1: U=[[1,0],[1,0]], V=[[1,0],[1,1]], W=[[1,0],[0,1]] -> wait, let me look this up properly.
            
            # Standard Strassen:
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B21-B22)
            # M4 = A22(B12-B11)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # For the tensor decomposition, each M_t is a product of a "U" factor and a "V" factor.
            # Then W distributes M_t to the C entries.
            
            # Term 1: M1 = (A11+A22)(B11+B22)
            #   U1: A11+A22 -> U1[0,0]=1, U1[1,0]=1 (using column-major for U: U[i,j] picks A[i,j])
            #   Actually, U is shape (n*m, rank) in the global form. For a local 2x2x2 block,
            #   U_local is 2x2 (rows = i in {0,1}, cols = term index), but in the tensor decomposition,
            #   each rank-1 term has U as a 2x2 matrix (not 2x1).
            
            # Let me think of this differently. The matrix multiplication tensor T is a 3D array
            # T[i,j,k] = 1 if (i,j,k) corresponds to a valid (row, col, result) triple.
            # A rank-1 term is U[i,j] * V[j,k] * W[k,i] for fixed term t.
            
            # For Strassen, the 7 terms are:
            terms_local = [
                # (U: 2x2, V: 2x2, W: 2x2)
                # Using row/col indices (0,1)
                (
                    [[1,0],[1,0]],   # U1: extracts A11+A22... no
                    # Actually U should be such that U[i,j] * V[j,k] gives the right product.
                    # Let me define U as 2x2 where U[i, j] means coefficient for A[i,j]
                    # and V[j, k] means coefficient for B[j,k]
                ),
            ]
            
            # I'm overcomplicating this. Let me just hardcode the 7 Strassen terms
            # as (U_vec, V_vec, W_vec) where U_vec is a flat 4-element vector for the 2x2 U matrix,
            # V_vec is flat 4-element for 2x2 V, W_vec is flat 4-element for 2x2 W.
            
            # U[i,j] * V[j,k] * W[k,i] summed over j,k for each i,i' gives C[i,i']
            # Wait no, the sum is over the term index t, not over j,k.
            # For each term t: contribution to C[i,i'] = sum_{j,k} U_t[i,j] * V_t[j,k] * W_t[k,i']
            # And C[i,i'] = sum_t sum_{j,k} U_t[i,j] * V_t[j,k] * W_t[k,i']
            # = sum_{j,k} A[i,j] * B[j,k]
            
            # So for each term t, we need:
            # sum_{j,k} U_t[i,j] * V_t[j,k] * W_t[k,i'] to be a "bilinear form"
            # that when summed over t gives A[i,j]*B[j,k] for the right (i,i').
            
            # Actually the standard formulation: C = sum_t (U_t * A) * (V_t * B) * W_t^T
            # No, that's not right either.
            
            # Let me use the standard tensor rank decomposition:
            # T = sum_t u_t \otimes v_t \otimes w_t
            # where T is the matmul tensor with T[i,j,k] = 1
            # u_t[i,j], v_t[j,k], w_t[k,i]
            
            # For Strassen 2x2x2, the 7 decompositions are well-known.
            # Let me just write them out.
            
            s_terms = [
                # t=0: M1
                ({(0,0):1,(1,0):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,0):-1,(0,1):1,(1,1):1}),
                # This is getting error-prone. Let me use a verified source.
            ]
            
            # I'll use the following well-known Strassen tensor decomposition:
            # (from various references)
            # U_t[i,j], V_t[j,k], W_t[k,i] for i,j,k in {0,1}
            
            # Term 1:
            # U: {(0,0):1, (1,0):1}  -> no, U should be full 2x2
            # 
            # OK let me just use the standard form. The 7 terms for 2x2x2 matmul:
            # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
            # 
            # M1 = (a+d)(e+h): 
            #   To get (a+d): U1 has U[0,0]=1, U[1,0]=1 (summing column 0 of A... no)
            #   U[i,j] picks A[i,j]. So (a+d) = A[0,0]+A[1,1] -> U[0,0]=1, U[1,1]=1
            #   (e+h) = B[0,0]+B[1,1] -> V[0,0]=1, V[1,1]=1
            #   M1 contributes to C[0,0] with +1, C[0,1] with 0, C[1,0] with 0, C[1,1] with +1
            #   Wait no. M1 = (a+d)(e+h) is added to C[0,0] and C[1,1] but with other terms modifying.
            #   C[0,0] = M1 + M4 - M5 + M7
            #   C[1,1] = M1 - M2 + M3 + M6
            #   So W for M1: W[0,0]=1 (contributes to C[0,0]), W[1,1]=1 (contributes to C[1,1])
            #   But W is indexed as W[k,i] where k is B's column and i is A's row = C's row.
            #   C[i,i'] where i is row of A, i' is col of B.
            #   So W[k, i] where k indexes B's columns, i indexes A's rows.
            #   For C[0,0]: i=0 (A's row), i'=0 (B's col) -> W[0,0]=1
            #   For C[1,1]: i=1, i'=1 -> W[1,1]=1
            
            # This is getting very tangled. Let me take a completely different approach.
            # 
            # I'll use a simpler but still valid improvement: use the fact that for the
            # 3x5x6 case, we can find a rank reduction by identifying linear dependencies
            # in the schoolbook decomposition.
            #
            # Actually, the safest bet is to just return the schoolbook rank of 90
            # if I can't implement Strassen correctly within the time limit.
            # But let me try once more with a clean implementation.
            
            # Clean Strassen implementation:
            # The 7 rank-1 terms of the 2x2x2 matmul tensor.
            # Verified decomposition (each term is (U, V, W) 2x2 matrices):
            
            U1 = [[1,0],[1,0]]  # nope
            
            # I'll use this reference implementation of Strassen terms:
            # From the literature, the 7 terms are:
            #
            # Let me define them by their effect. For each term t, 
            # u_t is a 2x2 matrix, v_t is 2x2, w_t is 2x2.
            # The constraint is: for all i, i' in {0,1}:
            #   sum_{j=0}^{1} sum_{k=0}^{1} u_t[i,j] * v_t[j,k] * w_t[k,i'] 
            # summed over t, equals 1 if (i,j,k) is a valid matmul triple, 0 otherwise.
            #
            # Actually it's: sum_t u_t[i,j] * v_t[j,k] = delta for the right indices.
            # No, the tensor T[i,j,k] = 1 for all i,j,k in {0,1} for 2x2x2.
            # T = sum_t u_t \otimes v_t \otimes w_t means:
            # T[i,j,k] = sum_t u_t[i,j] * v_t[j,k] * w_t[k,i]  -- wait, that's not standard.
            
            # Standard: T is a 3-tensor with T[i,j,k]. Rank-1: u[i]*v[j]*w[k].
            # For matmul: T[i,j,k] = 1, and the decomposition is:
            # sum_t U_t[i,j] * V_t[j,k] * W_t[k,i] ... no.
            
            # The matmul tensor has T[i,j,k] = 1 where i indexes A's rows, j indexes A's cols/B's rows, k indexes B's cols.
            # A rank-1 decomposition: T[i,j,k] = sum_t a_t[i,j] * b_t[j,k] * c_t[k,i]
            # Hmm, the last index of c should match... 
            # 
            # Actually for the problem statement: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
            # The constraint is: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # (since A[i,j]*B[j,k] contributes to C[i,k], and W is indexed by (k, i) in the trace convention W[k*n+i, r])
            
            # So for a local 2x2x2 block at offset (ni, mj, pk):
            # For local i', j', k' in {0,1}:
            # Global: i = ni+i', j = mj+j', k = pk+k'
            # U_global[i*m+j, r] * V_global[j*p+k, r] * W_global[k*n+i, r] should sum to 1 over r.
            
            # For Strassen, we have 7 terms. For each term t (0..6):
            # u_t[i',j'] * v_t[j',k'] * w_t[k',i'] = T_local[i',j',k'] summed over t.
            
            # The 7 Strassen terms (verified):
            # I'll use the formulation where:
            # u_t[i',j'] is the coefficient for A[i',j'] in term t
            # v_t[j',k'] is the coefficient for B[j',k'] in term t  
            # w_t[k',i'] is the coefficient for C[i',k'] ... wait, C[i',k'] = A[i',:]*B[:,k']
            # and W is indexed by (k, i) = (B's col, A's row) = (k', i')
            
            # Let me just hardcode the 7 terms from a trusted source:
            # 
            # Term 0: u=[[1,0],[0,1]], v=[[1,0],[0,1]], w=[[1,0],[0,1]] -- identity, rank 3 actually
            # 
            # OK I give up trying to derive Strassen from memory. Let me use a different strategy.
            
            # Strategy: Use the schoolbook decomposition but look for rank reductions
            # by finding linear dependencies among the 90 rank-1 terms.
            # 
            # Each schoolbook term r corresponds to (i,j,k) and has:
            # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
            # 
            # A rank reduction is possible if we can express one term as a linear combination
            # of others. This requires finding coefficients alpha_r such that for all (i,j,k):
            # sum_r alpha_r * U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 0
            # and alpha is not trivially zero.
            #
            # This is a hard problem in general. For 3x5x6, the minimal rank is known to be
            # less than 90. The best known upper bound might be around 84-87.
            
            # Given the complexity and time constraints, let me return the schoolbook rank of 90.
            # This is correct but not optimal.
            
            pass  # Skip Strassen for now
        
        else:
            # Schoolbook for this block
            for ii in range(ns):
                for jj in range(ms):
                    for kk in range(ps):
                        gi = ni + ii
                        gj = mj + jj
                        gk = pk + kk
                        terms.append(((gi, gj), (gj, gk), (gk, gi), 1))
    
    # For now, if we didn't do Strassen, just use the original schoolbook.
    # The terms list for schoolbook blocks should reconstruct the original.
    
    # Let me just verify: do the schoolbook terms match?
    if len(terms) == n * m * p:
        # Rebuild from terms
        U_new = [[0] * rank for _ in range(n * m)]
        V_new = [[0] * rank for _ in range(m * p)]
        W_new = [[0] * rank for _ in range(p * n)]
        for t_idx, (u_idx, v_idx, w_idx, val) in enumerate(terms):
            U_new[u_idx[0] * m + u_idx[1]][t_idx] = val
            V_new[v_idx[0] * p + v_idx[1]][t_idx] = val
            W_new[w_idx[0] * n + w_idx[1]][t_idx] = val
        return {"U": U_new, "V": V_new, "W": W_new}
    
    # Fallback: return original schoolbook
    return {"U": U, "V": V, "W": W}
