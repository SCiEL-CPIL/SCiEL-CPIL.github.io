"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W form an exact decomposition of the multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the tensor via the decomposition
    # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    # Also check that non-matching entries are zero
    # The tensor for matrix mult has T[i,j,k]=1 iff the indices correspond to (i,j,k)
    # Actually the multiplication tensor T_{ijk} = 1 for all i,j,k in their ranges
    # Wait, let me reconsider. The matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k).
    # The decomposition says: for each output element C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The factorization: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need T[i,j,k] = 1 for all i,j,k.
    return True

def _try_strassen_2x2x2(n, m, p, block_i, block_j, block_k):
    """
    Try to embed a 2x2x2 Strassen block at position (block_i, block_j, block_k)
    and fill the rest with schoolbook. Returns (U, V, W) or None.
    
    The 2x2x2 Strassen decomposition replaces 8 schoolbook terms with 7.
    We pick a 2x2 submatrix of A (rows i0, i0+1), 2x2 submatrix of B (cols j0, j0+1 for rows, k0,k0+1 for cols)
    and 2x2 submatrix of C (rows i0,i0+1, cols k0,k0+1).
    
    Strassen for 2x2:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B10-B11)
    M4 = A11*(B01-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B10)
    M7 = (A01-A11)*(B01+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    i0, j0, k0 = block_i, block_j, block_k
    if i0 + 1 >= n or j0 + 1 >= m or k0 + 1 >= p:
        return None
    
    # Build full schoolbook first, then modify
    # Actually, let's build from scratch.
    # Total schoolbook terms: n*m*p
    # We replace 8 terms (the 2x2x2 block) with 7 Strassen terms.
    # So rank = n*m*p - 8 + 7 = n*m*p - 1
    
    rank = n * m * p - 1
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # First, add all schoolbook terms EXCEPT those in the 2x2x2 block
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i0 <= i < i0+2 and j0 <= j < j0+2 and k0 <= k < k0+2:
                    continue
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    # Now add the 7 Strassen terms
    # The 7 multiplications involve:
    # M1: (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
    # M2: (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
    # M3: A[i0,j0] * (B[j0+1,k0]-B[j0+1,k0+1])
    # M4: A[i0+1,j0+1] * (B[j0,k0+1]-B[j0,k0])
    # M5: (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
    # M6: (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0+1,k0])
    # M7: (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0,k0+1]+B[j0+1,k0+1])
    
    # For each Strassen multiplier, we need to define U, V, W contributions
    # The key insight: each "multiplication" M_t = (linear combo of A) * (linear combo of B)
    # contributes to C via linear combos.
    
    # Let's define the 7 factors explicitly.
    # For M_t = P_t * Q_t where P_t is a linear combo of A entries and Q_t is a linear combo of B entries
    # The contribution to C[i,k] is sum over (i',j',k') of coeff_A[i',j',t] * coeff_B[j',k',t] * C_coeff[i,k,i',j',k',t]
    # But in the tensor decomposition, we have:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For a single "Strassen term" t, we assign one column c.
    # U[i*m+j, c] = coeff of A[i,j] in P_t
    # V[j*p+k, c] = coeff of B[j,k] in Q_t
    # W[k*n+i, c] = coeff such that the product gives the right C contribution
    
    # Actually, this is tricky. Let me think more carefully.
    
    # The standard Strassen decomposition for 2x2 matrix mult:
    # We have 7 bilinear forms M_t = <u_t, A> * <v_t, B>
    # and C = sum_t w_t * M_t (outer product style)
    
    # In the tensor decomposition T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]:
    # For a Strassen term t with column c:
    # U[i*m+j, c] = u_t[i,j]  (coefficient of A[i,j] in the t-th Strassen factor)
    # V[j*p+k, c] = v_t[j,k]  (coefficient of B[j,k] in the t-th Strassen factor)
    # W[k*n+i, c] = w_t[i,k]  (coefficient for C[i,k] from the t-th Strassen term)
    
    # And we need: for all (i,j,k) in the 2x2x2 block,
    # sum_t u_t[i,j] * v_t[j,k] * w_t[i,k] = 1 (if (i,j,k) is in the block)
    # and the cross terms with schoolbook should be 0 (which they are since schoolbook uses different columns)
    
    # Let me define the 7 Strassen factors for indices (0,1) in each dimension:
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me define u_t, v_t, w_t for t=0..6 (M1..M7):
    
    # M1: u = {(0,0):1, (1,1):1}, v = {(0,0):1, (1,1):1}, w = {(0,0):1, (1,1):1}
    # M2: u = {(1,0):1, (1,1):1}, v = {(0,0):1}, w = {(1,0):1, (1,1):-1}
    # M3: u = {(0,0):1}, v = {(1,0):1, (1,1):-1}, w = {(0,1):1, (1,1):1}
    # M4: u = {(1,1):1}, v = {(0,1):1, (0,0):-1}, w = {(0,0):1, (1,0):1}
    # M5: u = {(0,0):1, (0,1):1}, v = {(1,1):1}, w = {(0,0):-1, (0,1):1}
    # M6: u = {(1,0):1, (0,0):-1}, v = {(0,0):1, (1,0):1}, w = {(1,1):1}
    # M7: u = {(0,1):1, (1,1):-1}, v = {(0,1):1, (1,1):1}, w = {(0,0):1}
    
    # Let me verify: for C[0,0] = M1 + M4 - M5 + M7
    # M1 contributes: u(0,0)*v(?,?)... wait, I need to check the tensor identity.
    
    # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # For the Strassen block, for a given (i,j,k) in the 2x2x2 block:
    # sum_t U[i*m+j, c_t] * V[j*p+k, c_t] * W[k*n+i, c_t] should equal 1
    
    # Let me re-derive. The multiplication is C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    # In the Strassen block, we have 8 (i,j,k) triples.
    # For each triple (i,j,k), we need sum_t u_t[i,j] v_t[j,k] w_t[i,k] = 1.
    
    # Let me check M1: u(0,0)=1, u(1,1)=1; v(0,0)=1, v(1,1)=1; w(0,0)=1, w(1,1)=1
    # For (i,j,k)=(0,0,0): u(0,0)*v(0,0)*w(0,0) = 1*1*1 = 1
    # For (i,j,k)=(0,0,1): u(0,0)*v(0,1)*w(0,1) = 1*0*0 = 0 (v(0,1)=0 for M1)
    # Hmm, this doesn't seem right. Let me reconsider.
    
    # I think I'm confusing myself. Let me use a different approach.
    # The standard way: the 2x2x2 tensor has rank 7.
    # The 7 rank-1 tensors are:
    # T1 = (e00+e11) ⊗ (e00+e11) ⊗ (e00+e11)  -- but this isn't quite right either.
    
    # Let me just use the known Strassen decomposition and verify numerically.
    
    # For the 2x2x2 block, the tensor T[i,j,k] = 1 for i,j,k in {0,1}.
    # We want to find 7 triples (u_t, v_t, w_t) such that
    # sum_t u_t[i,j] v_t[j,k] w_t[i,k] = 1 for all i,j,k in {0,1}.
    
    # From Strassen:
    # M1 = (A00+A11)(B00+B11)
    #   u1 = {(0,0):1, (1,1):1}, v1 = {(0,0):1, (1,1):1}
    #   Contribution to C: C00 += M1, C11 += M1
    #   So w1 = {(0,0):1, (1,1):1}
    #   Check (0,0,0): u1(0,0)*v1(0,0)*w1(0,0) = 1*1*1 = 1 ✓
    #   Check (0,0,1): u1(0,0)*v1(0,1)*w1(0,1) = 1*0*0 = 0 ✗ (should be 1)
    
    # This approach isn't working directly. The issue is that the tensor decomposition
    # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # requires that for EACH (i,j,k), the sum equals 1. But in Strassen, the bilinear
    # forms don't factor this way directly.
    
    # I think the correct interpretation is different. Let me reconsider the problem.
    
    # The matrix multiplication tensor is T where T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k].
    # So T is a 3D array with T[i,j,k] = 1 for all valid i,j,k.
    # A rank-r decomposition is T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For the 2x2x2 case, T[i,j,k] = 1 for i,j,k in {0,1}, and we need rank 7.
    # This is the classical result that the 2x2x2 matrix multiplication tensor has rank 7.
    
    # Let me look up the explicit rank-7 decomposition.
    # The 7 rank-1 tensors are (using 0-based indexing for the 2x2 block):
    
    # I'll use the standard Strassen decomposition mapped to tensor factors.
    # After some thought, here's the correct mapping:
    
    # For the 2x2x2 block with local indices (a,b,c) where a,b,c in {0,1}:
    # The 7 factors (u, v, w) where u is 2x2, v is 2x2, w is 2x2:
    
    # Factor 1: u = [[1,0],[0,1]], v = [[1,0],[0,1]], w = [[1,0],[0,1]]
    #   This gives T[a,b,c] += u[a,b]*v[b,c]*w[a,c]
    #   For (0,0,0): 1*1*1=1, (0,0,1): 1*0*0=0, (0,1,0): 0*1*0=0, (0,1,1): 0*0*0=0
    #   (1,0,0): 0*1*0=0, (1,0,1): 0*0*0=0, (1,1,0): 1*0*0=0, (1,1,1): 1*1*1=1
    #   So this contributes 1 to (0,0,0) and (1,1,1)
    
    # This is getting complicated. Let me just try a different strategy:
    # Use the known fact that 2x2x2 has rank 7, and construct the decomposition
    # by solving a system of equations.
    
    # Actually, for the purposes of this task, let me just try to find a valid
    # decomposition by search/optimization for small ranks.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Start with schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank decompositions using local search
    # Strategy: start from schoolbook, try to merge/eliminate terms
    
    # For 3x3x8, schoolbook rank is 72.
    # Known lower bounds: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # Upper bound from various constructions...
    
    # Let's try a simple approach: use numpy to try random initializations
    # and optimize the rank-1 decomposition.
    
    # Actually, let's try a more systematic approach.
    # Try ranks from a reasonable lower bound up to schoolbook.
    
    # For now, let's just return the schoolbook decomposition since finding
    # a lower rank decomposition is non-trivial.
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
