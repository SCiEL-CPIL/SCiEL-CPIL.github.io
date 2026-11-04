"""Rewritten matrix-multiplication tensor decomposition search for (3,3,7)."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin below 150s wall-clock
    rng = np.random.default_rng(12345)

    # Build the matmul tensor T[i, j, k] = 1 iff i < n, j < m, k < p and
    # this corresponds to the (i,j) entry of product, i.e. standard matmul tensor.
    # Indices: i in [0,n), j in [0,m), k in [0,p)
    # The tensor has shape (n*m, m*p, p*n) with:
    #   row index = i*m + j
    #   col index = j*p + k
    #   third index = k*n + i
    #   T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    def verify(U, V, W):
        """Exact verification: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]"""
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.max(np.abs(recon - T)) < 1e-9

    def verify_halfint(U, V, W):
        """Verify after converting to Fraction (half-integers)."""
        Uf = np.zeros_like(U, dtype=object)
        Vf = np.zeros_like(V, dtype=object)
        Wf = np.zeros_like(W, dtype=object)
        for idx in np.ndindex(U.shape):
            val = U[idx] * 2
            if abs(val - round(val)) < 1e-9:
                Uf[idx] = Fraction(int(round(val)), 2)
            else:
                return False
        for idx in np.ndindex(V.shape):
            val = V[idx] * 2
            if abs(val - round(val)) < 1e-9:
                Vf[idx] = Fraction(int(round(val)), 2)
            else:
                return False
        for idx in np.ndindex(W.shape):
            val = W[idx] * 2
            if abs(val - round(val)) < 1e-9:
                Wf[idx] = Fraction(int(round(val)), 2)
            else:
                return False
        # Verify with fractions
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    s = Fraction(0)
                    for r in range(U.shape[1]):
                        s += Uf[i, r] * Vf[j, r] * Wf[k, r]
                    target = 1.0 if (i % m == j // p and j % p * n + i // m == k) else 0.0
                    # Actually simpler check:
                    # T[i,j,k]=1 iff i=m*j+... let's just use the float check
                    pass
        # Fallback: use float check on the fractional representation
        recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    recon[i, j, k] = float(sum(float(Uf[i, r]) * float(Vf[j, r]) * float(Wf[k, r]) for r in range(U.shape[1])))
        return np.max(np.abs(recon - T)) < 1e-9

    # Strategy: Use structured low-rank approaches
    # 1. Schoolbook = 63
    # 2. Try Kronecker product structure: decompose 3x3 x 3x7 as sum of rank-1 terms
    #    A known approach: for 3x3, Strassen-like ideas. For 3x7, combine.
    #    Rank of 3x3 matmul is known to be at most 7 (Strassen for 2x2 gives 7 for 2x2,
    #    but for 3x3 the best known is 23 by Lader. Actually let's not rely on that.)
    #    
    #    For 3x3 x 3x7, we can view it as: C[i,k] = sum_j A[i,j]*B[j,k]
    #    Split B into 7 columns: B[:, 0:7]. 
    #    If we have a rank-r decomposition of 3x3 matmul (A x B where B is 3x3),
    #    then for 3x7 we need rank-r * 7 / 3 * ... hmm, not straightforward.
    #    
    #    Alternative: Use the fact that 3x7 matmul can be decomposed using
    #    the 3x3 matmul tensor (rank r3) applied to each "column block" plus corrections.
    #
    #    Let me try a different structural approach:
    #    - Randomized low-rank with rounding
    #    - Greedy rank reduction
    #    - Structured: fix some entries, solve for others

    # Start with schoolbook
    sb_rank = n * m * p
    U = np.zeros((n*m, sb_rank), dtype=np.float64)
    V = np.zeros((m*p, sb_rank), dtype=np.float64)
    W = np.zeros((p*n, sb_rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = sb_rank

    # Try to reduce rank via linear algebra:
    # The tensor equation is: for each (i,j,k), sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
    # This is a 3-linear equation. We can fix W (or U, or V) and solve for others.
    
    # Approach 1: Fix W to have Kronecker structure
    # Let W[k*n+i, r] depend on k and i. Try W as a matrix product.
    
    # Approach 2: Use the "sliding" structure
    # For 3x3 x 3x7: C[i,k] = sum_j A[i,j] B[j,k]
    # Split k into groups. If we use a rank-r scheme for 3x3 matmul,
    # then for 3x7 we need to handle 7 columns of B.
    # If rank of 3x3 is r3, then 3x3 x 3x(3q) has rank at most r3 * q (by applying to each block).
    # 3x3 x 3x6 has rank <= r3 * 2. Then 3x3 x 3x1 adds at most 9.
    # So 3x3 x 3x7 has rank <= r3 * 2 + 9.
    # If r3 = 23 (Lader), that gives 55. If r3 = 22, gives 53.
    # But we don't have explicit Lader decomposition.
    
    # Let's try: rank = 48, 45, 42, ... using structured initialization
    
    # Key insight: Use integer/half-integer entries. The tensor T has 0/1 entries.
    # If U, V, W have integer entries, the product sum will be integer.
    # So we need: for "1" entries, the sum = 1; for "0" entries, sum = 0.
    
    # Strategy: Try random integer matrices of decreasing rank, verify, and use
    # gradient-like corrections.
    
    # Better strategy: Use the algebraic structure.
    # For 3x3 x 3x7, write B as a 3x7 matrix.
    # C = A * B where A is 3x3, B is 3x7, C is 3x7.
    # C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k]
    
    # Rank-1 tensor: u_i * v_j * w_k
    # For matmul, we need u[i*m+j], v[j*p+k], w[k*n+i]
    
    # Let me try a structured low-rank based on the column structure.
    # For each output column k (0..6), C[:,k] = A * B[:,k]
    # This is a 3x3 matmul of A with a 3x1 vector B[:,k].
    # The tensor for 3x3 x 3x1 has rank at most 9 (schoolbook) but can be less.
    
    # Actually, the most practical approach within time budget:
    # 1. Start from schoolbook (63)
    # 2. Try to merge columns using linear dependencies
    # 3. Use random restarts with smaller rank
    
    # Try structured approach: exploit that 3x3 x 3x7 = 3x3 x (3x6 + 3x1)
    # Use rank-r3 scheme for 3x3, replicated 2 times for 3x6, plus 9 for 3x1.
    
    # But we need explicit rank-r3 matrices. Let's use schoolbook for 3x3 (rank 9)
    # giving total rank 9*2 + 9 = 27? No, that's not right either.
    # Actually 3x3 x 3x6: for each of the 6 columns, we do 3x3 x 3x1.
    # If 3x3 x 3x1 has rank r1, then 3x3 x 3x6 has rank <= 6*r1? No.
    # The tensor is not simply a sum of independent 3x3 x 3x1 tensors.
    
    # Let me just try random search with structured initialization.
    
    def try_rank(target_rank, n_tries=100):
        """Try to find a valid decomposition at target_rank."""
        if time.time() > deadline - 5:
            return None
        
        # Initialize with structured pattern
        # Use the fact that entries should be small integers or half-integers
        # Try patterns based on Kronecker-like structures
        
        for attempt in range(n_tries):
            if time.time() > deadline - 5:
                return None
            
            # Generate random structured matrices
            # U has shape (9, r), V has shape (21, r), W has shape (21, r)
            
            # Try: U entries in {-1, 0, 1, 2, 3}, V and W similar
            # Or use half-integers: {-1, -0.5, 0, 0.5, 1, 1.5, 2}
            
            if attempt % 3 == 0:
                # Integer entries
                U = rng.integers(-2, 3, size=(n*m, target_rank)).astype(np.float64)
                V = rng.integers(-2, 3, size=(m*p, target_rank)).astype(np.float64)
                W = rng.integers(-2, 3, size=(p*n, target_rank)).astype(np.float64)
            elif attempt % 3 == 1:
                # Half-integer entries
                U = rng.integers(-4, 5, size=(n*m, target_rank)).astype(np.float64) / 2.0
                V = rng.integers(-4, 5, size=(m*p, target_rank)).astype(np.float64) / 2.0
                W = rng.integers(-4, 5, size=(p*n, target_rank)).astype(np.float64) / 2.0
            else:
                # Small random with some structure: reuse patterns
                U = rng.integers(-1, 2, size=(n*m, target_rank)).astype(np.float64)
                V = rng.integers(-1, 2, size=(m*p, target_rank)).astype(np.float64)
                W = rng.integers(-1, 2, size=(p*n, target_rank)).astype(np.float64)
            
            # Check if this random assignment happens to work (very unlikely for large rank)
            if verify(U, V, W):
                # Check half-integer condition
                if verify_halfint(U, V, W):
                    return U, V, W
        
        return None

    # The random approach above is hopeless for rank > ~30.
    # Let me use a smarter approach: gradient descent on the non-convex problem,
    # then snap to nearest half-integers.
    
    def gradient_search(target_rank, max_iter=2000, lr=0.01):
        """Find low-rank decomposition via gradient descent, then snap."""
        if time.time() > deadline - 5:
            return None
        
        best_err = float('inf')
        best_result = None
        
        for restart in range(5):
            if time.time() > deadline - 10:
                return best_result
            
            # Initialize from schoolbook columns with random perturbation
            cols = rng.choice(sb_rank, size=target_rank, replace=False)
            U = best_U[:, cols].copy()
            V = best_V[:, cols].copy()
            W = best_W[:, cols].copy()
            
            # Add random noise
            noise = 0.5
            U += rng.normal(0, noise, size=U.shape)
            V += rng.normal(0, noise, size=V.shape)
            W += rng.normal(0, noise, size=W.shape)
            
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # Compute current reconstruction
                recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.mean((recon - T)**2)
                
                if err < 1e-10:
                    break
                
                # Gradients
                dU = lr * np.einsum('ijk,jr,kr->ir', T - recon, V, W)
                dV = lr * np.einsum('ijk,ir,kr->jr', T - recon, U, W)
                dW = lr * np.einsum('ijk,ir,jr->kr', T - recon, U, V)
                
                U += dU
                V += dV
                W += dW
                
                # Snap check periodically
                if it % 50 == 49:
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    if verify(U_snap, V_snap, W_snap):
                        if verify_halfint(U_snap, V_snap, W_snap):
                            return U_snap, V_snap, W_snap
                    
                    # Track best
                    snap_err = np.mean((np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap) - T)**2)
                    if snap_err < best_err:
                        best_err = snap_err
                        best_result = (U_snap.copy(), V_snap.copy(), W_snap.copy())
            
            # Final check
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            if verify(U_snap, V_snap, W_snap):
                if verify_halfint(U_snap, V_snap, W_snap):
                    return U_snap, V_snap, W_snap
            else:
                # Also check without snapping
                if verify(U, V, W):
                    if verify_halfint(U, V, W):
                        return U, V, W
        
        return best_result

    # Try decreasing ranks
    for target_rank in range(sb_rank - 1, 20, -1):
        if time.time() > deadline - 15:
            break
        
        result = gradient_search(target_rank)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break

    # Also try: structured low-rank using the block structure
    # 3x3 x 3x7: write B as [B1 | B2 | b3] where B1, B2 are 3x3, b3 is 3x1
    # C = A*B = [A*B1 | A*B2 | A*b3]
    # If we have a rank-r scheme for 3x3 matmul (U3, V3, W3),
    # then A*B1 uses rank r, A*B2 uses rank r, A*b3 uses rank r.
    # Total rank: r * 2 + r = 3r? No, the tensors overlap in the A indices.
    # Actually the full tensor for 3x3 x 3x7 has indices (i,j) for A, (j,k) for B, (k,i) for C.
    # It's not a simple block sum.
    
    # Let me try yet another approach: use the known fact that for 2x2 matmul,
    # Strassen gives rank 7. For 3x3, the best known is 23 (Lader).
    # For 3x3 x 3x7, a trivial bound is 23*7/3... no that doesn't work.
    
    # Practical approach within time:
    # Try to construct a rank-48, 45, 42, 39, 36 decomposition using
    # the "repeated" structure with random integer entries.
    
    # Final strategy: use scipy to solve the system for smaller target ranks
    # The equations are: for each (i,j,k), sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
    # This is 9*21*21 = 3969 equations.
    # If we fix two of U,V,W, the remaining is linear (or at least structured).
    
    # Fix W to be identity-like and solve for U,V? Not straightforward.
    
    # Let's try: parameterize as U[i,r] * V[j,r] * W[k,r]
    # Fix U to have a specific structure (e.g., only depends on i, not j)
    # Then V[j,r]*W[k,r] = T[i,j,k]/U[i,r] for each i... but this must hold for all i,
    # which is over-constrained.
    
    # Alternative: Fix U[i*m+j, r] to depend only on i (not j):
    # U[i*m+j, r] = a[i, r] for all j
    # Then: sum_r a[i,r] * V[j,r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
    # = 1 if (i,j,k) is valid, 0 otherwise
    # For fixed i: sum_r a[i,r] * V[j,r] * W[k*n+i, r] = delta(i,j,k)
    # where delta is 1 for valid (j,k) pairs (all j in 0..2, k in 0..6) and 0 for invalid.
    # But ALL (i,j,k) with i<3, j<3, k<7 are valid! So T=1 for all 63 entries, 0 otherwise.
    # The "otherwise" entries are those where the index pattern doesn't match.
    
    # Let me re-examine: T[a, b, c] = 1 iff there exist i,j,k such that
    # a = i*m+j, b = j*p+k, c = k*n+i, i.e., a = i*3+j, b = j*7+k, c = k*3+i.
    # So T is 1 only at 63 specific positions out of 9*21*21 = 3969.
    
    # OK so my "fix U" approach:
    # For each i (0,1,2), the sub-tensor T_i[j,k] = T[i*3+j, j*7+k, k*3+i]
    # must equal sum_r a[i,r] * V[j,r] * W[k*3+i, r]
    
    # Let me fix a specific simple structure and see if it works.
    # Try: a[i,r] = 1 for all i,r (i.e., U depends only on which "row block" i we're in)
    # Then we need: sum_r V[j,r] * W[k*3+i, r] = T[i*3+j, j*7+k, k*3+i] for all i,j,k
    
    # For i=0: sum_r V[j,r] * W[k*3, r] = 1 for all j,k (since T[0*3+j, j*7+k, k*3+0]=1 for all j,k)
    # Wait no: T[i*3+j, j*7+k, k*3+i] = 1 always (for valid i,j,k).
    # And T is 0 elsewhere. But in the sub-problem, we only see entries that ARE 1.
    # The entries that are 0 correspond to different index patterns.
    
    # This is getting complex. Let me just go with the gradient approach and hope.
    
    # Actually, let me try a completely different structural approach:
    # Use the fact that we only need half-integer entries, and try to construct
    # explicit solutions for small ranks.
    
    # For rank 1: need U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = T[...] for all
    # This is impossible since T has both 0 and 1 entries with the same (i,j) patterns.
    
    # Minimum rank is at least 9 (since 3x3 x 3x1 requires at least 3 terms,
    # and we have 7 output columns, but they share the A matrix).
    
    # Let me try rank = 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48
    # with the gradient approach, starting from good initializations.
    
    # Actually, the gradient search above should work. Let me make it more robust.
    
    # Final attempt: if we haven't improved below 63, return the schoolbook.
    # But first let me try a few specific structured decompositions.
    
    # Structured try: rank = 27 (3*3*3, treating the 7 as 3+3+1 with overlap)
    # This is a shot in the dark.
    
    # Let me just run the gradient search for a few key ranks and hope.
    for target_rank in [48, 45, 42, 39, 36, 33, 30, 27, 24, 21, 18]:
        if time.time() > deadline - 10:
            break
        
        result = gradient_search(target_rank, max_iter=1000, lr=0.005)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break

    # Prepare output
    # Ensure entries are integer or half-integer
    U_out = [[float(round(x*2))/2.0 for x in row] for row in best_U]
    V_out = [[float(round(x*2))/2.0 for x in row] for row in best_V]
    W_out = [[float(round(x*2))/2.0 for x in row] for row in best_W]
    
    # Final verification
    Uf = np.array(U_out)
    Vf = np.array(V_out)
    Wf = np.array(W_out)
    if not verify(Uf, Vf, Wf):
        # Fall back to schoolbook
        Uf = U.copy()
        Vf = V.copy()
        Wf = W.copy()
        U_out = [[float(x) for x in row] for row in Uf]
        V_out = [[float(x) for x in row] for row in Vf]
        W_out = [[float(x) for x in row] for row in Wf]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
