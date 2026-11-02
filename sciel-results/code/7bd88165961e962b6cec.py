"""
Matrix multiplication 2x2 * 2x11: rank-minimal exact tensor decomposition.

Strategy:
  1. Build the schoolbook rank-44 decomposition.
  2. Use a randomized linear-algebra approach to find a lower-rank decomposition:
     - Represent the multiplication tensor as a set of linear constraints.
     - Use iterative rank-reduction: start from schoolbook, try to merge columns
       by solving for dependencies.
  3. Apply SVD-based numerical rank detection to identify linearly dependent
       columns in the constraint system.
  4. Greedy column removal: iteratively try removing each column and check
       whether the remaining columns still span the required image.

  The key insight for M(2,2,11): we can exploit the structure that A is 2x2.
  Each output entry C[i,k] = sum_j A[i,j]*B[j,k].
  The tensor has 2*2*11 = 44 elementary multiplications.

  A rank-r decomposition requires r bilinear forms. We search for the minimal r.

  Approach:
  - Flatten the problem: for each of the 4 entries of A (indices (i,j)),
    we have 11 output entries (k=0..10) that depend on B[j,k].
  - The multiplication tensor T_{(i,j),(j',k),(i',k')} = delta_{i,i'} * delta_{j,j'}
    (with the W convention: W[k*n+i, r]).

  Actually, let me be more careful with the tensor structure.

  The multiplication: C = A * B where A is 2x2, B is 2x11, C is 2x11.
  C[i,k] = sum_j A[i,j] * B[j,k]

  The tensor T has indices (i,j) for A, (j,k) for B, (i,k) for C.
  T[(i,j),(j',k),(i',k')] = 1 if i=i' and j=j' and k=k', else 0.

  A rank-r decomposition: T = sum_{c=0}^{r-1} u_c * v_c * w_c
  where u_c is a vector in R^{n*m} = R^4, v_c in R^{m*p} = R^22, w_c in R^{p*n} = R^22.

  Constraint: for all (i,j), (j',k), (i',k'):
    sum_c U[(i,j),c] * V[(j',k),c] * W[(i',k'),c] = T[(i,j),(j',k),(i',k')]

  This is a system of bilinear equations in the entries of U, V, W.

  For small problems, we can use the following approach:
  - Fix the structure: since n=2, m=2, p=11, the problem is small enough
    that we can try to find a low-rank decomposition using numerical methods.

  Practical approach:
  1. Start with schoolbook (rank 44).
  2. Try to reduce rank by identifying that some of the 44 terms can be combined.
     Specifically, for each pair (i,j) of A-indices, we have 11 terms.
     The B-indices are (j,k) for k=0..10.
     The C-indices are (k,i) for k=0..10.

  3. Use the fact that the problem decomposes: for a fixed i, the 11 outputs C[i,k]
     for k=0..10 each depend on A[i,0]*B[0,k] + A[i,1]*B[1,k].
     So for fixed i, we need to compute 11 linear combinations of the 4 A-entries
     (A[i,0], A[i,1]) with the 22 B-entries.

  4. The key observation: for each i, the map from (A[i,0], A[i,1]) to
     (C[i,0], ..., C[i,10]) is linear with coefficient matrix B (2x11).
     This is a 2-to-11 linear map. The rank of this map (as a bilinear form)
     is at most 2*11 = 22, but we can do better.

  5. Actually, the bilinear complexity of computing y = B*x where x is 2-dimensional
     and B is 2x11 is at most 11 (just compute each y_k = b_{0,k}*x_0 + b_{1,k}*x_1
     using 2 multiplications each, total 22, but we can share: x_0 and x_1 are
     reused across all 11 outputs).

  Wait, the bilinear complexity of a 2x11 matrix-vector product:
  We need to compute 11 outputs, each is a linear combination of 2 inputs.
  The bilinear rank of a 2x11 matrix (as a bilinear form) is at most 2+11-1 = 12
  (by the standard bound for matrix multiplication M(2,11) <= 2*11 but can be lower).

  Actually, for a single matrix-vector product y = Bx where B is 2x11,
  the bilinear complexity is at most 2*11 = 22 (schoolbook), but it can be
  reduced. The minimum bilinear complexity of a 2x11 matrix is known to be
  at least 11 (since we need 11 outputs) and at most 22.

  But in our case, we have TWO such matrix-vector products (one for each i in 0,1),
  and they share the same B but different A-rows.

  Let me think differently. The total problem is:
  Given A (2x2) and B (2x11), compute C = AB (2x11).

  This is equivalent to: for each column k of B, compute A * b_k where b_k is
  the k-th column of B (a 2-vector). So we need to compute 11 matrix-vector
  products of a 2x2 matrix with a 2-vector.

  The bilinear complexity of M(2,2,1) is 4 (schoolbook) or 7 (Strassen gives 7
  for 2x2*2x2, but for 2x2*2x1 it's just 4).

  Actually M(2,2,1): A is 2x2, B is 2x1, C is 2x1.
  C[0] = A[0,0]*B[0] + A[0,1]*B[1]
  C[1] = A[1,0]*B[0] + A[1,1]*B[1]
  This requires 4 multiplications in schoolbook. Can we do better?
  The bilinear rank of this is 4 (it's a 2x2 matrix times a vector, which is
  just 4 scalar multiplications, and I believe the rank is exactly 4).

  So for 11 columns, naively 11*4 = 44. But we can share computations across
  columns since B[0] and B[1] appear in all 11 products.

  Hmm, but in the bilinear model, each "multiplication" is a product of a linear
  form in A and a linear form in B. So we can form linear combinations of the
  B-entries.

  Let me use a more systematic approach: I'll implement a numerical search for
  low-rank decompositions using the following method:

  Method: "Bilinear complexity via linear algebra"
  1. The multiplication tensor T can be viewed as a linear map from
     (A ⊗ B) to C. Specifically, T: R^4 ⊗ R^22 -> R^22.
     T(A ⊗ B) = AB (flattened).

  2. A rank-r decomposition corresponds to writing this map as a sum of r
     rank-1 tensors, i.e., T = sum_c u_c ⊗ v_c ⊗ w_c.

  3. Equivalently, there exist matrices U (4xr), V (22xr), W (22xr) such that
     for all A, B: C = U^T (A ⊙ (V^T B)) W ... no, this isn't quite right.

  Let me just implement the direct constraint satisfaction approach:

  The constraints are:
  For each (i, j, j', k, i', k'):
    sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}

  Since the RHS is 1 only when i=i', j=j', k=k', and 0 otherwise.

  For the "1" cases (i=i', j=j', k=k'):
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

  For the "0" cases, the sum must be 0.

  This is a system of trilinear equations. Hard to solve directly.

  Alternative: use the "slice" approach. Fix one index and view the rest as a
  matrix. For fixed i (say i=0), the constraints involving A[0,j] for j=0,1
  and B[j,k] for j=0,1, k=0..10, producing C[0,k] for k=0..10.

  This is getting complex. Let me just implement a practical search:

  1. Start with schoolbook rank 44.
  2. Use SVD on the "flattened" tensor to find a low-rank approximation.
  3. Round to exact rational values.
  4. Verify.

  Actually, for this specific problem, let me try a direct construction:

  Observation: The problem M(2,2,11) can be seen as computing 11 independent
  M(2,2,1) problems, but sharing the A-matrix. However, in the bilinear model,
  sharing is automatic since we can form linear combinations of B-entries.

  Key insight: For each k, C[:,k] = A * B[:,k]. The bilinear complexity of
  computing A * v for a fixed v (2-vector) is 4. But if v varies, we can
  precompute linear combinations of v.

  Actually, the bilinear complexity of M(2,2,p) is known. For p=1, it's 4.
  For general p, it's at most 4p (schoolbook) but can be lower.

  A standard result: M(n,m,p) <= M(n,m,1) * p is not tight. The bilinear
  complexity of M(n,m,p) is the same as the bilinear rank of the n x m x p
  tensor.

  For M(2,2,11), I'll try to find a decomposition of rank < 44 using
  numerical methods.

  Let me implement:
  1. Build the 44x(22*22) constraint matrix (or use a more efficient representation).
  2. Use iterative refinement to reduce rank.

  Given the time constraints, let me implement a practical approach:
  - Use the schoolbook as baseline.
  - Try to find dependencies using SVD on appropriately constructed matrices.
  - If a lower rank is found, verify and return it.

  Here's my concrete plan:
  1. Construct the multiplication tensor T as a 3D array of shape (4, 22, 22).
  2. For each possible rank r from 44 down to some lower bound, try to find
     a decomposition.
  3. Use a randomized approach: initialize U, V, W randomly, then use
     alternating least squares or gradient descent to minimize the error.
  4. Check if the error is below tolerance, and if so, round to exact values.

  Given the 120s time limit, I'll limit the search.

  Let me code this up.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 * 2x11 matrix multiplication.
    Returns {'U': (4, rank), 'V': (22, rank), 'W': (22, rank)} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill

    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if i==i' and j==j' and k==k' else 0
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition (rank = n*m*p = 44)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def verify(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose T."""
        r = U.shape[1]
        # Compute the reconstructed tensor
        # T_recon[i*m+j, j'*p+k, k'*n+i'] = sum_c U[i*m+j,c] * V[j'*p+k,c] * W[k'*n+i',c]
        # Efficient computation: for each c, outer product of U[:,c] and (V[:,c] * W[:,c] broadcast)
        # Actually: T_recon = sum_c np.einsum('a,c,b,c,d,c->abd', U, V, W)
        # More efficiently: for each c, add U[:,c][:,None,None] * V[:,c][None,:,None] * W[:,c][None,None,:]
        T_recon = np.zeros_like(T)
        for c in range(r):
            T_recon += U[:, c][:, None, None] * V[:, c][None, :, None] * W[:, c][None, None, :]
        return np.max(np.abs(T_recon - T)) < tol

    # Try to find lower-rank decompositions using numerical optimization
    # Strategy: use alternating least squares with random initialization

    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb

    # Try various ranks from best_rank-1 down to a lower bound
    # Lower bound: at least max(n*m, m*p, p*n) = max(4, 22, 22) = 22
    # But actually, the rank must be at least the "slice rank" which is at least max of matricization ranks
    # T as (4, 22*22) matrix: rank <= 4, so slice rank >= 4
    # T as (22, 4*22) matrix: rank <= 22, so slice rank >= 22
    # T as (22, 4*22) matrix: rank <= 22, so slice rank >= 22
    # So lower bound is 22.

    np.random.seed(42)

    for target_rank in range(best_rank - 1, 21, -1):
        if time.time() > deadline - 30:
            break

        # Try multiple random initializations
        found = False
        for trial in range(50):
            if time.time() > deadline - 30:
                break

            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)

            # Scale to have reasonable magnitudes
            U /= np.linalg.norm(U) * np.sqrt(target_rank)
            V /= np.linalg.norm(V) * np.sqrt(target_rank)
            W /= np.linalg.norm(W) * np.sqrt(target_rank)

            # Alternating least squares: fix two, solve for the third
            # The objective is min ||T - sum_c U_c V_c W_c||_F^2
            # This is not convex, but ALS can find local minima.

            # For fixed V, W: min_U ||T - sum_c U_c (V_c * W_c)^T||_F^2
            # This is a linear least squares problem for U.
            # Similarly for V and W.

            # Flatten T to (n*m, m*p * p*n)
            T_flat = T.reshape(n * m, m * p * p * n)

            for iteration in range(200):
                if time.time() > deadline - 30:
                    break

                # Update U: for each row a of U, solve min ||T[a] - sum_c U[a,c] * (V[:,c] * W[:,c]).flatten()||^2
                # This is: T_flat[a] = sum_c U[a,c] * (V[:,c] * W[:,c]).flatten()
                # So: T_flat = U * M where M[c, b] = V[b,c] * W[b,c] ... wait, dimensions don't match.

                # Let me reconsider. T[a, b, d] = sum_c U[a,c] * V[b,c] * W[d,c]
                # For fixed V, W: T[a, :, :] = sum_c U[a,c] * (V[:,c] * W[:,c])[:,None] * ... no.
                # T[a, b, d] = sum_c U[a,c] * V[b,c] * W[d,c]
                # For fixed V,W: the coefficient of U[a,c] in T[a,b,d] is V[b,c]*W[d,c].
                # So: T[a] (as a (m*p, p*n) matrix) = sum_c U[a,c] * (V[:,c] * W[:,c])[:,None] * ones? No.
                # T[a,b,d] = sum_c U[a,c] * V[b,c] * W[d,c]
                # Let M_c be the outer product V[:,c] * W[:,c]^T, which is (m*p) x (p*n).
                # Then T[a] = sum_c U[a,c] * M_c.
                # So: T_flat[a] = sum_c U[a,c] * M_c.flatten()
                # This is a linear system: T_flat = U * M_flat where M_flat[c, (b,d)] = V[b,c]*W[d,c]

                # Build M_flat: shape (target_rank, m*p * p*n)
                M_flat = (V[:, :, None] * W[:, None, :]).reshape(target_rank, m * p * p * n)

                # Solve: T_flat = U * M_flat  =>  U = T_flat * pinv(M_flat)
                try:
                    U_new, residuals, rank_m, s = np.linalg.lstsq(M_flat.T, T_flat.T, rcond=None)
                    U = U_new.T
                except:
                    break

                # Update V: T[a,b,d] = sum_c U[a,c] * V[b,c] * W[d,c]
                # For fixed U, W: T[:,b,:] = sum_c V[b,c] * (U[:,c] * W[:,c])[:,None] * ...
                # T[a,b,d] = sum_c V[b,c] * U[a,c] * W[d,c]
                # Let N_c be outer product U[:,c] * W[:,c]^T, shape (n*m) x (p*n).
                # T[:,b,:] = sum_c V[b,c] * N_c
                # T_flat2 = V * N_flat where N_flat[c, (a,d)] = U[a,c]*W[d,c]

                N_flat = (U[:, :, None] * W[:, None, :]).reshape(target_rank, n * m * p * n)
                T_flat2 = T.reshape(m * p, n * m * p * n)
                try:
                    V_new, residuals, rank_n, s = np.linalg.lstsq(N_flat.T, T_flat2.T, rcond=None)
                    V = V_new.T
                except:
                    break

                # Update W: T[a,b,d] = sum_c W[d,c] * U[a,c] * V[b,c]
                # For fixed U, V: T[:,:,d] = sum_c W[d,c] * (U[:,c] * V[:,c])[:,None] * ...
                # T[a,b,d] = sum_c W[d,c] * U[a,c] * V[b,c]
                # Let P_c be outer product U[:,c] * V[:,c]^T, shape (n*m) x (m*p).
                # T[:,:,d] = sum_c W[d,c] * P_c
                # T_flat3 = W * P_flat where P_flat[c, (a,b)] = U[a,c]*V[b,c]

                P_flat = (U[:, :, None] * V[:, None, :]).reshape(target_rank, n * m * m * p)
                T_flat3 = T.reshape(p * n, n * m * m * p)
                try:
                    W_new, residuals, rank_p, s = np.linalg.lstsq(P_flat.T, T_flat3.T, rcond=None)
                    W = W_new.T
                except:
                    break

            # Check if we found a valid decomposition
            if verify(U, V, W, tol=1e-8):
                # Try to round to exact values
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2

                if verify(U_rounded, V_rounded, W_rounded, tol=1e-10):
                    best_result = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    best_rank = target_rank
                    found = True
                    break
                else:
                    # Try rounding to integers
                    U_rounded = np.round(U)
                    V_rounded = np.round(V)
                    W_rounded = np.round(W)
                    if verify(U_rounded, V_rounded, W_rounded, tol=1e-10):
                        best_result = {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                        best_rank = target_rank
                        found = True
                        break

        if found:
            # Check if we can go lower
            continue
        else:
            # If we didn't find a decomposition at this rank, stop
            break

    # If no improvement found, return schoolbook
    if best_rank == rank_sb:
        return best_result

    return best_result

# EVOLVE-BLOCK-END
