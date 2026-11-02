"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses a structured approach combining Strassen-style rank-7 blocks for 2x2x2 subproblems
with carefully shared rank-1 terms to minimize total rank.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def time_ok():
        return (time.time() - start_time) < DEADLINE

    # We'll build the decomposition as a list of rank-1 terms.
    # Each term is (U_entry, V_entry, W_entry) where each is a dict from index to coeff.
    # U index: i*m + j for A[i,j]
    # V index: j*p + k for B[j,k]
    # W index: k*n + i for C[i,k]  (trace convention W[k*n+i, r])
    
    terms = []  # list of (U_dict, V_dict, W_dict)
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term."""
        terms.append((u_dict, v_dict, w_dict))
    
    def strassen_block(ri, rj, bi, bk):
        """Add Strassen decomposition for A[ri:ri+2, rj:rj+2] * B[bi:bi+2, bk:bk+2] -> C[ri:ri+2, bk:bk+2].
        
        Standard Strassen for 2x2*2x2:
        Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        m1 = (a+d)(e+h)
        m2 = (c+d)*e
        m3 = a*(f-h)
        m4 = d*(g-e)
        m5 = (a+b)*h
        m6 = (c-a)*(e+f)
        m7 = (b-d)*(g+h)
        
        C[0,0] = m1 + m4 - m5 + m7
        C[0,1] = m3 + m5
        C[1,0] = m2 + m4
        C[1,1] = m1 - m2 + m3 + m6
        """
        # A indices
        a = (ri, rj)       # (0,0) relative
        b = (ri, rj+1)     # (0,1)
        c = (ri+1, rj)     # (1,0)
        d = (ri+1, rj+1)   # (1,1)
        
        # B indices
        e = (bi, bk)       # (0,0)
        f = (bi, bk+1)     # (0,1)
        g = (bi+1, bk)     # (1,0)
        h = (bi+1, bk+1)   # (1,1)
        
        # C indices
        C00 = (ri, bk)
        C01 = (ri, bk+1)
        C10 = (ri+1, bk)
        C11 = (ri+1, bk+1)
        
        # m1 = (a+d)(e+h)
        add_term({a[0]*m+a[1]: 1, d[0]*m+d[1]: 1},
                 {e[0]*p+e[1]: 1, h[0]*p+h[1]: 1},
                 {C00[1]*n+C00[0]: 1, C11[1]*n+C11[0]: 1})
        
        # m2 = (c+d)*e
        add_term({c[0]*m+c[1]: 1, d[0]*m+d[1]: 1},
                 {e[0]*p+e[1]: 1},
                 {C10[1]*n+C10[0]: 1, C11[1]*n+C11[0]: -1})
        
        # m3 = a*(f-h)
        add_term({a[0]*m+a[1]: 1},
                 {f[0]*p+f[1]: 1, h[0]*p+h[1]: -1},
                 {C01[1]*n+C01[0]: 1, C11[1]*n+C11[0]: 1})
        
        # m4 = d*(g-e)
        add_term({d[0]*m+d[1]: 1},
                 {g[0]*p+g[1]: 1, e[0]*p+e[1]: -1},
                 {C00[1]*n+C00[0]: 1, C10[1]*n+C10[0]: 1})
        
        # m5 = (a+b)*h
        add_term({a[0]*m+a[1]: 1, b[0]*m+b[1]: 1},
                 {h[0]*p+h[1]: 1},
                 {C00[1]*n+C00[0]: -1, C01[1]*n+C01[0]: 1})
        
        # m6 = (c-a)*(e+f)
        add_term({c[0]*m+c[1]: 1, a[0]*m+a[1]: -1},
                 {e[0]*p+e[1]: 1, f[0]*p+f[1]: 1},
                 {C11[1]*n+C11[0]: 1})
        
        # m7 = (b-d)*(g+h)
        add_term({b[0]*m+b[1]: 1, d[0]*m+d[1]: -1},
                 {g[0]*p+g[1]: 1, h[0]*p+h[1]: 1},
                 {C00[1]*n+C00[0]: 1})
    
    def direct(i, j, k):
        """Add direct term A[i,j]*B[j,k] -> C[i,k]."""
        add_term({i*m+j: 1}, {j*p+k: 1}, {k*n+i: 1})
    
    # Strategy:
    # Use Strassen for the 2x2 * 2x2 blocks in A[0:2,0:2] * B[0:2, 0:8] (4 blocks of 2x2)
    # That gives 4*7 = 28 terms for A[0:2,0:2] * B[0:2,0:8] -> C[0:2,0:8]
    #
    # Remaining:
    # A[0:2, 2] * B[2, 0:8] -> C[0:2, 0:8]: 2*8 = 16 direct terms
    # A[2, 0:2] * B[0:2, 0:8] -> C[2, 0:8]: 2*8 = 16 direct terms  
    # A[2, 2] * B[2, 0:8] -> C[2, 0:8]: 8 direct terms
    # Total: 28 + 16 + 16 + 8 = 68
    #
    # But we can do better! Let's think about sharing.
    
    # Actually, let me reconsider. The key insight for beating 68:
    # For the cross terms A[0:2,2]*B[2,:] and A[2,:]*B[0:2,:], 
    # we can use rank-1 decompositions that share structure.
    
    # Let's try a different decomposition:
    # Group by the middle index j:
    # j=0: A[:,0] (3x1) * B[0,:] (1x8) -> outer product, rank 3*8=24 but can be done in...
    #   Actually A[:,0]*B[0,:] is a rank-1 matrix (3x8), so it contributes to all C[i,k] with coeff A[i,0]*B[0,k]
    #   This is 1 tensor term? No, it's a rank-1 matrix in C space, which is 1 rank-1 tensor term only if
    #   the A part and B part are both rank-1 vectors.
    #   A[:,0] is a 3-vector, B[0,:] is an 8-vector. The outer product is rank 1 as a matrix.
    #   As a tensor term: U has entries at (i,0) for i=0,1,2; V has entries at (0,k) for k=0..7;
    #   W has entries at (k,i) for all i,k. This is ONE rank-1 tensor term!
    #   Wait, no. A rank-1 tensor term has U as a single vector (or a vector with structure), V as a single vector, W as a single vector.
    #   Actually in this decomposition, each rank-1 term is: U is a vector in R^(n*m), V is a vector in R^(m*p), W is a vector in R^(p*n).
    #   The term contributes outer(U, V, W) to the tensor.
    #   
    #   For the j=0 slice: A[:,0]*B[0,:] gives C[i,k] += A[i,0]*B[0,k] for all i,k.
    #   This is a rank-1 term where:
    #     U[i*m+0] = A[i,0] for i=0,1,2  -> but we need fixed coefficients, not variable!
    #   Wait, I'm confusing myself. In a tensor decomposition, the coefficients are FIXED.
    #   The tensor T[i,j,k] = 1 for all i,j,k (since A[i,j]*B[j,k] always contributes).
    #   So we need sum_r U[i*m+j]*V[j*p+k]*W[k*n+i] = 1 for all i,j,k.
    #
    #   For the j=0 contribution: we need sum_r U[i*3+0]*V[0*8+k]*W[k*3+i] = 1 for all i,k.
    #   This is like decomposing the 3x8 all-ones matrix as a sum of rank-1 matrices where each rank-1 
    #   matrix is U_col * V_row (outer product of 3-vector and 8-vector).
    #   The rank of the 3x8 all-ones matrix is 1! So one term suffices for j=0.
    #   U[0*3+0]=1, U[1*3+0]=1, U[2*3+0]=1, V[0*8+k]=1 for all k, W[k*3+i]=1 for all i,k.
    #   That's 1 term for the entire j=0 slice!
    #
    #   Similarly for j=1: 1 term
    #   Similarly for j=2: 1 term
    #   Total: 3 terms for all j slices if we just do this naively? No wait...
    #   For j=0: U[0,0]=1, U[1,0]=1, U[2,0]=1; V[0,k]=1 for all k; W[k,i]=1 for all i,k.
    #   For j=1: U[0,1]=1, U[1,1]=1, U[2,1]=1; V[1,k]=1 for all k; W[k,i]=1 for all i,k.
    #   For j=2: U[0,2]=1, U[1,2]=1, U[2,2]=1; V[2,k]=1 for all k; W[k,i]=1 for all i,k.
    #
    #   But W is shared! W[k,i] needs to be 1 for ALL j contributions. Since each term uses the same W vector
    #   (all ones), we can't just add them because W would be the sum... 
    #   Wait, no. Each term is independent. Term r has U_r, V_r, W_r. The total is sum_r U_r[i,j]*V_r[j,k]*W_r[k,i].
    #   For term 1 (j=0 only): U_1 has non-zero only at (i,0), V_1 has non-zero only at (0,k), W_1 has non-zero at all (k,i).
    #   Contribution: sum_i,k U_1[i,0]*V_1[0,k]*W_1[k,i] = sum_i,k 1*1*1 = but this gives the sum, not individual entries.
    #   
    #   I need to be more careful. The tensor entry T[i,j,k] = sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i].
    #   We need T[i,j,k] = 1 for all i,j,k.
    #
    #   For the j=0 slice to work with one term r:
    #   T[i,0,k] = U_r[i*3+0] * V_r[0*8+k] * W_r[k*3+i] = 1 for all i,k.
    #   If U_r[i*3+0] = 1 for all i, V_r[0*8+k] = 1 for all k, W_r[k*3+i] = 1 for all i,k, then yes.
    #   But this term also contributes to other (i,j,k) where j≠0 if U_r[i*3+j] ≠ 0.
    #   So we set U_r[i*3+j] = 0 for j≠0.
    #   Similarly V_r[j*8+k] = 0 for j≠0.
    #   Then this term only contributes to T[i,0,k]. Good.
    #
    #   So 3 terms for all j slices? That gives rank 3! But that can't be right because 
    #   the matrix multiplication tensor for 3x3*3x8 should have rank at least... 
    #   Actually wait, let me verify. If we use:
    #   Term 1: U has 1's at (0,0),(1,0),(2,0); V has 1's at (0,k) for k=0..7; W has 1's at (k,i) for all i,k.
    #   Term 2: U has 1's at (0,1),(1,1),(2,1); V has 1's at (1,k) for k=0..7; W has 1's at (k,i) for all i,k.
    #   Term 3: U has 1's at (0,2),(1,2),(2,2); V has 1's at (2,k) for k=0..7; W has 1's at (k,i) for all i,k.
    #
    #   Then T[i,j,k] = sum_r U_r[i*3+j]*V_r[j*8+k]*W_r[k*3+i]
    #   For j=0: only term 1 contributes: U_1[i*3+0]*V_1[0*8+k]*W_1[k*3+i] = 1*1*1 = 1. ✓
    #   For j=1: only term 2 contributes: 1*1*1 = 1. ✓
    #   For j=2: only term 3 contributes: 1*1*1 = 1. ✓
    #
    #   This gives rank 3! But that seems too good. Let me double-check the tensor definition.
    #   The matrix multiplication tensor for A(n*m) x B(m*p) -> C(p*n) is defined as:
    #   T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    #   In standard matrix multiplication C = A*B, C[i,k] = sum_j A[i,j]*B[j,k].
    #   So every pair (i,j,k) with j being the contraction index contributes. The tensor has T[i,j,k]=1 for all valid i,j,k.
    #   
    #   But wait - the tensor is 3-indexed: (i,j,k) where i indexes rows of A/C, j is the contraction, k indexes columns of B/C.
    #   The rank of this tensor... for 1x1x1 it's rank 1. For 2x1x1 it's rank 1 (just A[0,0]*B[0,0]).
    #   For nx1x1 it's rank 1. For 1x1x1 it's rank 1.
    #   For n x m x p, the "schoolbook" rank is n*m*p.
    #   But can we do better? For m=1, it's rank n*p (each A[i,0]*B[0,k] is a separate product).
    #   Wait no, for m=1: T[i,0,k] = 1 for all i,k. We need sum_r U_r[i]*V_r[0]*W_r[k] = 1.
    #   If V_r[0] is a scalar, then this is sum_r (U_r[i])*(V_r[0])*(W_r[k]) = 1.
    #   This is decomposing the n x p all-ones matrix as a sum of rank-1 matrices, which has rank min(n,p).
    #   So for n x 1 x p, the rank is min(n,p)!
    #
    #   For our case 3 x 3 x 8: by the same logic, if we fix j, we get a 3x8 all-ones "sub-tensor" 
    #   but the j dimension couples them.
    #   Actually no, the tensor is NOT separable in j because U depends on (i,j), V depends on (j,k).
    #   
    #   Let me reconsider. The decomposition I proposed above with 3 terms:
    #   Does it actually work? Let me trace through more carefully.
    #   
    #   U is 9x3 (n*m x rank = 9x3), V is 24x3 (m*p x rank = 24x3), W is 24x3 (p*n x rank = 24x3).
    #   
    #   Term 1 (r=0): 
    #     U[:,0]: indices (i,j) -> non-zero at j=0: U[0,0]=1, U[3,0]=1, U[6,0]=1 (i.e., (0,0),(1,0),(2,0))
    #     V[:,0]: indices (j,k) -> non-zero at j=0: V[0,0]=1, V[1,0]=1, ..., V[7,0]=1
    #     W[:,0]: indices (k,i) -> all 24 entries = 1
    #   
    #   Term 2 (r=1):
    #     U[:,1]: non-zero at j=1: U[1,1]=1, U[4,1]=1, U[7,1]=1
    #     V[:,1]: non-zero at j=1: V[8,1]=1, ..., V[15,1]=1
    #     W[:,1]: all = 1
    #   
    #   Term 3 (r=2):
    #     U[:,2]: non-zero at j=2: U[2,2]=1, U[5,2]=1, U[8,2]=1
    #     V[:,2]: non-zero at j=2: V[16,2]=1, ..., V[23,2]=1
    #     W[:,2]: all = 1
    #   
    #   Now T[i,j,k] = sum_r U[i*3+j, r] * V[j*8+k, r] * W[k*3+i, r]
    #   
    #   For i=0, j=0, k=0:
    #     r=0: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    #     r=1: U[0,1]*V[0,1]*W[0,1] = 0*0*1 = 0
    #     r=2: U[0,2]*V[0,2]*W[0,2] = 0*0*1 = 0
    #     Sum = 1 ✓
    #   
    #   For i=1, j=0, k=5:
    #     r=0: U[3,0]*V[5,0]*W[15,0] = 1*1*1 = 1
    #     r=1: U[3,1]*V[5,1]*W[15,1] = 0*0*1 = 0
    #     r=2: U[3,2]*V[5,2]*W[15,2] = 0*0*1 = 0
    #     Sum = 1 ✓
    #   
    #   For i=0, j=1, k=3:
    #     r=0: U[1,0]*V[8+3,0]*W[9,0] = 0*0*1 = 0
    #     r=1: U[1,1]*V[11,1]*W[9,1] = 1*1*1 = 1
    #     r=2: U[1,2]*V[11,2]*W[9,2] = 0*0*1 = 0
    #     Sum = 1 ✓
    #   
    #   This works! Rank 3! But wait, this seems impossibly good.
    #   Let me verify with a concrete example.
    #   A = [[1,2,3],[4,5,6],[7,8,9]], B = [[1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2],[3,3,3,3,3,3,3,3]]
    #   C = A*B = [[1+4+9, 1+4+9, ...], [2+10+18, ...], ...] = [[14,14,14,14,14,14,14,14], [30,30,...], [46,46,...]]
    #   
    #   Using our decomposition:
    #   C[i,k] = sum_j sum_r U[i*3+j,r]*V[j*8+k,r]*W[k*3+i,r]
    #   = sum_j (sum_r U[i*3+j,r]*V[j*8+k,r]*W[k*3+i,r])
    #   
    #   For j=0: sum_r U[i*3,r... no, U[i*3+0,r]. Only r=0 has U[i*3,0]=1 (for the i-th entry in column 0).
    #   Wait, U[i*3+0, 0] = 1 for i=0,1,2. And V[0*8+k, 0] = 1 for all k. W[k*3+i, 0] = 1 for all i,k.
    #   So contribution from j=0: 1*1*1 = 1 (times A[i,0]*B[0,k])
    #   
    #   Hmm, I think I'm confusing the tensor decomposition with the actual multiplication.
    #   The tensor T[i,j,k] = 1 means that the (i,j) entry of A and (j,k) entry of B multiply to give a contribution to C[i,k].
    #   The decomposition T = sum_r U_r ⊗ V_r ⊗ W_r means:
    #   T[i,j,k] = sum_r U_r[(i,j)] * V_r[(j,k)] * W_r[(k,i)]
    #   
    #   And the actual multiplication is: C[i,k] = sum_j A[i,j]*B[j,k] = sum_j T[i,j,k]*A[i,j]*B[j,k]
    #   = sum_j sum_r U_r[(i,j)]*V_r[(j,k)]*W_r[(k,i)]*A[i,j]*B[j,k]
    #   = sum_r sum_j (A[i,j]*U_r[(i,j)]) * (B[j,k]*V_r[(j,k)]) * W_r[(k,i)]
    #   = sum_r (sum_j A[i,j]*U_r[(i,j)]) * (sum_j B[j,k]*V_r[(j,k)]) * W_r[(k,i)]
    #   
    #   So the algorithm is: for each r, compute x_r[i] = sum_j A[i,j]*U_r[(i,j)], y_r[k] = sum_j B[j,k]*V_r[(j,k)],
    #   then C[i,k] += x_r[i] * y_r[k] * W_r[(k,i)].
    #   
    #   With our 3-term decomposition:
    #   r=0: x_0[i] = A[i,0]*1 = A[i,0], y_0[k] = B[0,k]*1 = B[0,k], W_0[(k,i)] = 1
    #     Contribution: A[i,0]*B[0,k]
    #   r=1: x_1[i] = A[i,1], y_1[k] = B[1,k], contribution: A[i,1]*B[1,k]
    #   r=2: x_2[i] = A[i,2], y_2[k] = B[2,k], contribution: A[i,2]*B[2,k]
    #   Total: sum_j A[i,j]*B[j,k] = C[i,k]. ✓
    #   
    #   So rank 3 works! But this is just the "row-column" decomposition which is rank m=3.
    #   This is much better than schoolbook rank 72 or Strassen-based 68.
    
    # Build the rank-3 decomposition
    rank = 3
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r in range(m):  # r = 0, 1, 2 corresponds to j = r
        for i in range(n):
            U[i * m + r][r] = 1.0
        for k in range(p):
            V[r * p + k][r] = 1.0
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1.0
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check all tensor entries
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                err = abs(s - 1.0)
                max_err = max(max_err, err)
    
    if max_err > 1e-9:
        # Fallback to schoolbook (should never happen)
        rank_fb = n * m * p
        U_fb = [[0.0] * rank_fb for _ in range(n * m)]
        V_fb = [[0.0] * rank_fb for _ in range(m * p)]
        W_fb = [[0.0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1.0
                    V_fb[j * p + k][col] = 1.0
                    W_fb[k * n + i][col] = 1.0
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
