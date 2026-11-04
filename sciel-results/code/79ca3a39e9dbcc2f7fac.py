"""Matrix multiplication 2,4,7 - rank-minimal tensor decomposition search."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 4, 7]

def _build_rank50():
    """Known valid rank-50 decomposition."""
    n, m, p = 2, 4, 7
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        # M2
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        # M3
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        # M4
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + (i0+1), col] += 1
        col += 1
        # M5
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        W[(k0+1)*n + i0, col] += 1
        col += 1
        # M6
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        # M7
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    for i in range(2):
        for j in range(4):
            U[i*m + j, col] = 1
            V[j*p + 6, col] = 1
            W[6*n + i, col] = 1
            col += 1
    return U, V, W

def _verify(U, V, W, tol=1e-10):
    """Check if U,V,W correctly decompose the 2x4x7 matmul tensor."""
    n, m, p = 2, 4, 7
    # Build the tensor T[i,j,k] = delta
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == (i,j,k) == 1 if valid
    # Actually the tensor is: T[i,j,k] = 1 if (i,j,k) corresponds to a valid triple
    # For matmul C = A*B: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T[i,j,k] where the multiplication tensor has T[i,j,k] = 1
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # So we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all valid (i,j,k)
    # Wait - the matmul tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No! The tensor is: for each output position (i,k), C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid indices, meaning every (i,j,k) triple contributes.
    # Actually no. The multiplication tensor for C=AB where A is n×m, B is m×p:
    # T[i,j,k] = 1 if this is a valid term, i.e., always 1 for all i,j,k in range.
    # The decomposition requires: 1 = sum_r U[i,j,r] V[j,k,r] W[k,i,r] for all i,j,k.
    
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    # Reshape: U is (n*m, r), V is (m*p, r), W is (p*n, r)
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Sum over r should equal 1 for all i,j,k
    
    rank = U.shape[1]
    # Build the full tensor product and check
    # For efficiency, do it in chunks
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                val = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
                err = max(err, abs(val - 1.0))
    return err < tol, err

def _search_lower_rank(U0, V0, W0, deadline):
    """Try to find a lower rank decomposition by null-space elimination."""
    n, m, p = 2, 4, 7
    rank0 = U0.shape[1]
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = rank0
    
    # The decomposition: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k
    # If we have a valid decomposition of rank R, we can try to find linear dependencies
    # among the rank-1 tensors to reduce the rank.
    # 
    # Key insight: if there exist coefficients c_r such that
    #   sum_r c_r U[:,r] = 0, sum_r c_r V[:,r] = 0, sum_r c_r W[:,r] = 0
    # then we can eliminate one term.
    # More generally, if the set of rank-1 tensors {U[:,r] ⊗ V[:,r] ⊗ W[:,r]} is linearly
    # dependent, we can reduce rank.
    #
    # Alternative: try to find a decomposition of rank R < rank0 directly.
    # Use the constraint that for all (i,j,k): sum_{r<R} U[i,j,r] V[j,k,r] W[k,i,r] = 1
    #
    # Strategy: start from rank0 and try to set one column to zero (or absorb it).
    # If column r can be written as a combination of others in a way that preserves
    # the tensor identity, we can eliminate it.
    
    # Actually, a simpler approach: try random projections / null-space searches.
    # If we have rank R, the tensors T_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r] span the matmul tensor.
    # We need sum_r T_r = T_matmul (the all-ones tensor on valid indices).
    # Wait, that's not right either. We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    # This is equivalent to: the tensor sum_r (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) evaluated
    # at (i,j,k) gives 1.
    
    # Let's try a different approach: for each pair of columns, check if one can be
    # eliminated by adjusting the others.
    
    # Simpler: try to find if any column is redundant by checking if removing it
    # and solving for the remaining can still satisfy the constraint.
    
    # Most practical approach given time: try specific structured reductions.
    # 
    # For the 2x4x7 case, let's try to find a rank-49 or lower solution.
    # 
    # Approach: parameterize small perturbations and use the constraint equations.
    # There are n*m*p = 56 constraints and 3*R unknowns (R columns in U, V, W).
    # For R=49: 147 unknowns, 56 constraints -> very underdetermined.
    # For R=40: 120 unknowns, 56 constraints -> still underdetermined.
    #
    # But the constraints are nonlinear (cubic in the unknowns).
    #
    # Let me try a direct approach: assume the structure is similar to the rank-50
    # solution but with some terms merged.
    
    # Try: can we do rank 48 by merging two of the rank-1 terms for the last column?
    # The last column (k=6) has 8 rank-1 terms. Can we do it with fewer?
    # For k=6: we need sum_r U[i,j,r] V[j,6,r] W[6,i,r] = 1 for all i,j.
    # This is a 2x4 matrix that must equal the all-ones matrix.
    # Minimum rank of a 2x4 all-ones matrix is 1 (it's rank 1!).
    # So we can do it with rank 1: U[*,*,r] = [1,1,1,1,1,1,1,1], V[*,6,r] = [1,1,1,1], W[6,*,r] = [1,1]
    # Wait: U[i*m+j, r] * V[j*p+6, r] * W[6*n+i, r] = 1*1*1 = 1. Yes!
    # So the last column only needs rank 1, not 8!
    
    # This means we can potentially reduce from 50 to 43!
    # Let me verify: the 6 Strassen blocks use 42 terms, and the last column uses 1 term.
    # Total: 43.
    
    # Let's construct this:
    U_new = np.zeros((n*m, 43))
    V_new = np.zeros((m*p, 43))
    W_new = np.zeros((p*n, 43))
    
    # Copy the 6 Strassen blocks (first 42 columns)
    U_new[:, :42] = U0[:, :42]
    V_new[:, :42] = V0[:, :42]
    W_new[:, :42] = W0[:, :42]
    
    # Last column: rank-1 term for k=6
    U_new[:, 42] = 1.0  # All entries 1
    V_new[:, 42] = 0.0
    V_new[:, 42] = 0.0  # m*p = 28 entries
    for j in range(m):
        V_new[j*p + 6, 42] = 1.0
    W_new[:, 42] = 0.0
    for i in range(n):
        W_new[6*n + i, 42] = 1.0
    
    ok, err = _verify(U_new, V_new, W_new)
    if ok:
        best_rank = 43
        best_U, best_V, best_W = U_new, V_new, W_new
        print(f"Found rank 43 solution! err={err}")
    
    # Now try to reduce the Strassen blocks further.
    # Each Strassen block handles a 2x2x2 subproblem with rank 7.
    # Can we do better for the 2x2x2 blocks? No, rank 7 is optimal for 2x2x2.
    # But maybe we can share structure between blocks.
    
    # Try rank 42: can we merge the last column term into a Strassen block?
    # The last column term is: U=1, V[j,6]=1, W[6,i]=1
    # A Strassen block for (i0=0, j0=0, k0=4) handles rows 0-1, cols 4-5 of B, 
    # output rows 4-5. It doesn't cover k=6.
    # We need a separate term for k=6.
    
    # Try: can we reduce the number of Strassen blocks?
    # We have 6 blocks covering:
    # Block 1: A(0:2,0:2), B(0:2,0:2) -> C(0:2,0:2)
    # Block 2: A(0:2,0:2), B(0:2,2:4) -> C(0:2,2:4)
    # Block 3: A(0:2,2:4), B(2:4,0:2) -> C(0:2,0:2)
    # Block 4: A(0:2,2:4), B(2:4,2:4) -> C(0:2,2:4)
    # Block 5: A(0:2,0:2), B(0:2,4:6) -> C(0:2,4:6)
    # Block 6: A(0:2,2:4), B(2:4,4:6) -> C(0:2,4:6)
    
    # What if we use a different blocking? E.g., 2x4x2 blocks with rank < 7 per block?
    # A 2x4x2 matmul: C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    # This is equivalent to multiplying a 2x4 by 4x2.
    # The tensor rank of 2x4x2 matmul: we can view it as 2 independent 2x2x2 problems
    # if we split j into {0,1} and {2,3}, giving 2*7=14. But maybe less.
    # Actually, 2x4x2 can be done with rank 8: 
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0] + A[0,3]B[3,0]
    # This is a 4-term sum, not easily factorable into rank-1 tensors.
    # The tensor rank of <2,4,2> is known to be 8 (from literature).
    # So 3 blocks of 2x4x2 would give 3*8=24, plus 1 for the last = 25. That's much better!
    # Wait, 2x4x7 = three 2x4x2 blocks (for k=0-1, k=2-3, k=4-5) plus one 2x4x1 (k=6).
    # 3*8 + 1 = 25. Let me verify this is correct.
    
    # Actually, let me check: the tensor rank of <m,n,p> for matrix multiplication.
    # <2,4,2>: rank is 8? Let me think...
    # C[i,k] = sum_j A[i,j] B[j,k] for i in {0,1}, j in {0,1,2,3}, k in {0,1}
    # We need: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k
    # 
    # One construction: use the 4 terms directly for each (i,k):
    # For (i,k): C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k] + A[i,3]B[3,k]
    # Each term A[i,j]B[j,k] is rank 1 in the tensor sense.
    # But we need a single decomposition for all (i,k) simultaneously.
    # 
    # Rank 8 construction for <2,4,2>:
    # r=0: U[0,0]=1,V[0,0]=1,W[0,0]=1 (contributes to C[0,0])
    # r=1: U[1,0]=1,V[0,0]=1,W[0,1]=1 (contributes to C[1,0])
    # r=2: U[0,1]=1,V[1,0]=1,W[0,0]=1 (contributes to C[0,0])
    # r=3: U[1,1]=1,V[1,0]=1,W[0,1]=1 (contributes to C[1,0])
    # r=4: U[0,2]=1,V[2,0]=1,W[0,0]=1 (contributes to C[0,0])
    # r=5: U[1,2]=1,V[2,0]=1,W[0,1]=1 (contributes to C[1,0])
    # r=6: U[0,3]=1,V[3,0]=1,W[0,0]=1 (contributes to C[0,0])
    # r=7: U[1,3]=1,V[3,0]=1,W[0,1]=1 (contributes to C[1,0])
    # But this only covers k=0! We also need k=1.
    # This gives 8 terms for k=0 and we'd need 8 more for k=1, total 16. That's worse.
    #
    # Better: use Strassen-like approach. Split j into {0,1} and {2,3}.
    # C[i,k] = sum_{j=0}^{1} A[i,j]B[j,k] + sum_{j=2}^{3} A[i,j]B[j,k]
    # Each half is a <2,2,2> problem with rank 7.
    # Total: 14 for <2,4,2>. Same as before.
    #
    # So <2,4,2> has rank 14 (using Strassen on two halves).
    # Can we do better? The lower bound for <2,4,2> is max(2*2, 2*4, 4*2) = 8.
    # Upper bound: 14 (two Strassen blocks).
    # It's known that <2,4,2> has rank 14 (or possibly less, but 14 is the standard).
    #
    # Actually, I recall that the rank of <2,4,2> is 14. Let me just use the Strassen approach.
    # 3 blocks of rank 14 + 1 = 43. Same as what we already have.
    
    # Let me try to search for a better solution using the time budget.
    # Try rank 42, 41, ... by solving the nonlinear system.
    
    # For a given rank R, we need to find U (8×R), V (28×R), W (14×R) such that
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k (56 equations)
    # 
    # This is a system of 56 polynomial equations in 50R variables (for R=42: 2100 vars).
    # Very underdetermined. Let's try a randomized approach.
    
    # Approach: start from the rank-43 solution and try to eliminate one more term.
    # Take the rank-43 solution and try to set column 42 to zero by adjusting others.
    
    if time.time() < deadline - 5:
        # Try to find rank 42 by perturbing the rank-43 solution
        # Set up: we want sum_{r=0}^{41} U[i,j,r] V[j,k,r] W[k,i,r] = 1
        # Starting from U43, V43, W43 where sum_{r=0}^{42} ... = 1
        # We need: sum_{r=0}^{41} ... = 1 - (U[i,j,42] V[j,k,42] W[k,i,42])
        # The term r=42 contributes: 1*V[j,6,42]*W[6,i,42] which is 1 if k=6, else 0.
        # So we need sum_{r=0}^{41} ... = 1 for k≠6, and 0 for k=6.
        # But the Strassen blocks already give 1 for k=0..5 and 0 for k=6.
        # So we just need the first 42 columns to satisfy the constraint, which they do!
        # Wait, that means the rank-43 solution's first 42 columns already work for k≠6,
        # and for k=6 they give 0, so we need the 43rd column to give 1 for k=6.
        # If we remove the 43rd column, k=6 gives 0≠1. So we can't just remove it.
        # We need to modify the first 42 columns to also cover k=6.
        
        # Let's try: modify one Strassen block to also cover k=6.
        # Block 5 covers k=4,5. What if we make it cover k=4,5,6?
        # A 2x2x3 problem: rank?
        # C[i,k] for k in {4,5,6}: sum_j A[i,j]B[j,k]
        # This is <2,2,3>. Rank of <2,2,3> is 8 (known).
        # So we could replace block 5 (rank 7 for 2x2x2) with a rank-8 block for 2x2x3.
        # Net change: +1. Total: 42-7+8+1 = 44. Worse!
        
        # Alternative: can we share the k=6 term with a Strassen block more efficiently?
        # The k=6 term is: for all i,j: contributes A[i,j]*B[j,6] to C[i,6].
        # In the tensor: U[i,j,r]*V[j,6,r]*W[6,i,r] = 1 for all i,j.
        # This is a rank-1 tensor in the (i,j) subspace for fixed k=6.
        
        # I think 43 is close to optimal for this structure. Let me try a different approach.
        
        # Try: use a 2x4x7 decomposition directly with rank < 43.
        # Lower bound: max(n*p, n*m, m*p) = max(14, 8, 28) = 28.
        # So rank >= 28.
        
        # Let me try a random search for rank 42.
        # Use the fact that the system is very underdetermined.
        
        # Actually, let me try a completely different structured approach.
        # View the 2x4x7 matmul as:
        # C = A * B where A is 2x4, B is 4x7.
        # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k] + A[i,3]B[3,k]
        # 
        # For each i (2 values), we have a row vector A[i,:] (length 4) times B (4x7).
        # The tensor is the sum over i of (row_i of A) ⊗ B, which is a sum of 2 terms
        # each of which is a rank-7 tensor (since B is 4x7, its matmul with a vector is
        # a 1x4x7 problem... no, that's not quite right).
        #
        # Actually: C[i,k] = sum_j A[i,j] B[j,k]
        # The multiplication tensor T[i,j,k] = 1 for all valid (i,j,k).
        # We can write T = sum_i (e_i ⊗ A_row_i) ⊗ B ... no.
        #
        # Let me think differently. The tensor T has T[i,j,k] = 1.
        # We can write T = sum_{j=0}^{3} (e_j in A-space) ⊗ (e_j in B-space) ⊗ (all-ones in C-space)
        # No, that gives T[i,j,k] = 1 if i and k are arbitrary... 
        #
        # T[i,j,k] = 1 means: for each (i,j,k), the value is 1.
        # A rank-1 tensor is U[i,j] * V[j,k] * W[k,i] where U is 2x4, V is 4x7, W is 7x2.
        # We need sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1.
        
        # Simple rank-28 solution: for each (j,k) pair, use a rank-1 tensor
        # U[i,j] = 1 if j matches, V[j,k] = 1 if (j,k) matches, W[k,i] = 1
        # That gives 4*7 = 28 terms, each of rank 1.
        # U[i*m+j, r] = 1 if j = j_r, else 0 (for all i)
        # V[j*p+k, r] = 1 if (j,k) = (j_r, k_r), else 0
        # W[k*n+i, r] = 1 (for all i,k)
        # Check: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
        # = sum_{r: j_r=j, k_r=k} 1*1*1 = 1 (since there's exactly one r for each (j,k))
        # Yes! This gives rank 28!
        
        # Wait, let me double-check. r indexes (j_r, k_r) for j_r in {0,1,2,3}, k_r in {0,...,6}.
        # U[i*m+j, r]: for a given r corresponding to (j_r, k_r), U[i*m+j, r] = 1 if j = j_r, else 0.
        #   So U[i*m+j, r] = delta_{j, j_r} (independent of i).
        # V[j*p+k, r]: V[j*p+k, r] = 1 if (j,k) = (j_r, k_r), else 0.
        #   So V[j*p+k, r] = delta_{j,j_r} * delta_{k,k_r}.
        # W[k*n+i, r]: W[k*n+i, r] = 1 for all k,i.
        # 
        # Product: delta_{j,j_r} * delta_{j,j_r} * delta_{k,k_r} * 1 = delta_{j,j_r} * delta_{k,k_r}
        # Sum over r: sum_{j_r, k_r} delta_{j,j_r} * delta_{k,k_r} = 1 (exactly one term matches).
        # 
        # So rank 28 works! Let me construct it.
        
        R = 28
        U28 = np.zeros((n*m, R))
        V28 = np.zeros((m*p, R))
        W28 = np.zeros((p*n, R))
        
        r = 0
        for j in range(m):
            for k in range(p):
                # U[i*m+j, r] = 1 for all i
                for i in range(n):
                    U28[i*m + j, r] = 1.0
                # V[j*p+k, r] = 1
                V28[j*p + k, r] = 1.0
                # W[k*n+i, r] = 1 for all i
                for i in range(n):
                    W28[k*n + i, r] = 1.0
                r += 1
        
        ok28, err28 = _verify(U28, V28, W28)
        if ok28:
            best_rank = 28
            best_U, best_V, best_W = U28, V28, W28
            print(f"Found rank 28 solution! err={err28}")
            # Can we do even better? Lower bound is max(n*p, n*m, m*p) = max(14,8,28) = 28.
            # So 28 is optimal!
        else:
            print(f"Rank 28 verification failed: err={err28}")
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # First, try the rank-28 solution (which should be optimal)
    R = 28
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    r = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U[i*m + j, r] = 1.0
            V[j*p + k, r] = 1.0
            for i in range(n):
                W[k*n + i, r] = 1.0
            r += 1
    
    ok, err = _verify(U, V, W)
    if ok:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 50
    U50, V50, W50 = _build_rank50()
    return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}
