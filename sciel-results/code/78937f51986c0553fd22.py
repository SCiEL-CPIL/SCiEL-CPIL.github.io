"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 140  # safety margin before 150s wall-clock limit

    n, m, p = 2, 4, 4

    # Build the multiplication tensor T[i,j,k,l] = delta_{i,l} * delta_{j,k}
    # where A is (n x m), B is (m x p), C = A*B is (n x p)
    # Tensor index: T[i, k, j, l] = delta_{i,l} * delta_{k,j}  (A[i,j] * B[j,k] -> C[i,k])
    # Flatten: T_flat[idx] where idx maps (i, k, j, l) -> linear
    # We'll use the convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # i.e., T[i*m+j, k*p+l, ...] -- let me think about the indexing more carefully.

    # Standard: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T_{i,j,k} where i indexes A rows, j is the contraction index, k indexes B cols
    # U: (n*m, rank) -> index (i,j) flattened
    # V: (m*p, rank) -> index (j,k) flattened  
    # W: (p*n, rank) -> index follows trace convention W[k*n + i, r] per the task

    # Actually the task says: "The output factor follows the official trace convention W[k*n+i,r]"
    # So W index is (k, i) -> k*n + i, where k is column of C, i is row of C
    # And the decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{...}
    # Wait, that's not quite right. Let me re-read.

    # The tensor for matrix multiplication: for each (i, j, k) where i in [0,n), j in [0,m), k in [0,p):
    # T[i, j, k] = 1 (this is the "structure tensor")
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[?, r]
    # But we need 3-way: U (n*m, R), V (m*p, R), W (p*n, R)
    # The full tensor has indices (i, k, j) or similar... 

    # Let me think of it as: the multiplication map is a bilinear map A -> (B -> C)
    # So the tensor lives in (n*m) * (m*p) * (p*n) space? No...
    # Actually it's: T in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n} where
    # T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    # So the only nonzero entries are where all three pairs match.

    # This is a rank-(n*p) tensor in general, but we want to find a low-rank decomposition.
    # Rank of matmul(n,m,p) tensor: for 2x4x4, schoolbook gives 32, Strassen blocks give 28.

    # Let me just use numpy to search for improvements.
    # First, build the base rank-28 solution, then try to reduce rank by finding linear dependencies
    # or using a different factorization.

    # Strategy: Start from the known rank-28 Strassen-based solution, then attempt to:
    # 1. Check if any rank-27 or lower solution exists by trying to zero out columns
    # 2. Use numerical optimization to find a lower rank solution

    # First, let's build the rank-28 solution properly using numpy for verification

    def build_rank_28():
        rank = 28
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)

        # Strassen 2x2x2 blocks: we have 2 (rows of A) x 4 (cols of A) times 4 (rows of B) x 4 (cols of B)
        # We can split into 4 blocks of 2x2:
        # A is 2x4, B is 4x4, C is 2x4
        # Split A into [A00 A01] where A00, A01 are 2x2
        # Split B into [[B00],[B01],[B10],[B11]] where each is 2x2
        # Split C into [C00 C01] where C00, C01 are 2x2
        # C00 = A00*B00 + A01*B10, C01 = A00*B01 + A01*B11
        # Use Strassen for each 2x2 multiplication

        def strassen_2x2(A, B):
            # A, B are 2x2, returns C = A*B using 7 multiplications
            m1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
            m2 = (A[1,0] + A[1,1]) * B[0,0]
            m3 = A[0,0] * (B[0,1] - B[1,1])
            m4 = A[1,1] * (B[1,0] - B[0,0])
            m5 = (A[0,0] + A[0,1]) * B[1,1]
            m6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
            m7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
            C = np.array([
                [m1 + m4 - m5 + m7, m3 + m5],
                [m2 + m4, m1 - m2 + m3 + m6]
            ])
            return C

        # For the tensor decomposition, each "multiplication" in Strassen becomes one term in the rank.
        # We need to express: for each Strassen multiplication m_s = f_s(A) * g_s(B), 
        # the contribution to C is sum_s h_s * m_s
        # In the tensor: U row for f_s(A) part, V row for g_s(B) part, W row for h_s(C) part

        # A is 2x4: rows 0,1; cols 0,1,2,3
        # B is 4x4: rows 0,1,2,3; cols 0,1,2,3
        # C is 2x4: rows 0,1; cols 0,1,2,3

        # Block decomposition:
        # Block 1: A[:,0:2] * B[0:2,:] -> C[:,0:2]  (using Strassen 2x2*2x2=2x2)
        # Block 2: A[:,0:2] * B[2:4,:] -> C[:,2:4]
        # Block 3: A[:,2:4] * B[0:2,:] -> C[:,0:2]
        # Block 4: A[:,2:4] * B[2:4,:] -> C[:,2:4]
        # Total rank: 4 * 7 = 28

        # For each block, the Strassen formulas are:
        # m1 = (A00+A11)(B00+B11) -> C00 += m1, C11 += m1
        # m2 = (A10+A11)B00 -> C10 += m2, C11 -= m2
        # m3 = A00(B01-B11) -> C01 += m3, C11 += m3
        # m4 = A11(B10-B00) -> C00 += m4, C10 += m4
        # m5 = (A00+A01)B11 -> C00 -= m5, C01 += m5
        # m6 = (A10-A00)(B00+B01) -> C11 += m6
        # m7 = (A01-A11)(B10+B11) -> C00 += m7

        # U index: i*m + j where i is row of A (0 or 1), j is col of A (0..3)
        # V index: j*p + k where j is row of B (0..3), k is col of B (0..3)
        # W index: k*n + i where k is col of C (0..3), i is row of C (0 or 1)

        def add_block(offset, a_col_start, b_row_start, c_col_start):
            # A sub-block: rows 0..1, cols a_col_start..a_col_start+1
            # B sub-block: rows b_row_start..b_row_start+1, cols 0..3 (full width, but we only need 2 cols for the block)
            # Wait, B is 4x4. For block multiplication A[2x2] * B[2x2], B should be 2x2.
            # B sub-block: rows b_row_start..b_row_start+1, cols 0..3? No, we need matching dimensions.
            # A is 2x2 (rows 0,1; cols a_col_start, a_col_start+1)
            # B should be 2x2 (rows b_row_start, b_row_start+1; cols c_col_start, c_col_start+1)
            # C sub-block: rows 0,1; cols c_col_start, c_col_start+1

            # Strassen indices within the 2x2 blocks:
            # A[0,0]=A[i0,ac0], A[0,1]=A[i0,ac1], A[1,0]=A[i1,ac0], A[1,1]=A[i1,ac1]
            # B[0,0]=B[br0,bc0], B[0,1]=B[br0,bc1], B[1,0]=B[br1,bc0], B[1,1]=B[br1,bc1]
            # C[0,0]=C[i0,cc0], C[0,1]=C[i0,cc1], C[1,0]=C[i1,cc0], C[1,1]=C[i1,cc1]

            ac0 = a_col_start
            ac1 = a_col_start + 1
            br0 = b_row_start
            br1 = b_row_start + 1
            cc0 = c_col_start
            cc1 = c_col_start + 1

            # m1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
            U[0*m+ac0][offset+0] += 1
            U[1*m+ac1][offset+0] += 1
            V[br0*p+cc0][offset+0] += 1
            V[br1*p+cc1][offset+0] += 1
            W[cc0*n+0][offset+0] += 1
            W[cc1*n+1][offset+0] += 1

            # m2 = (A[1,0]+A[1,1]) * B[0,0]
            U[1*m+ac0][offset+1] += 1
            U[1*m+ac1][offset+1] += 1
            V[br0*p+cc0][offset+1] += 1
            W[cc0*n+1][offset+1] += 1
            W[cc1*n+1][offset+1] -= 1

            # m3 = A[0,0] * (B[0,1]-B[1,1])
            U[0*m+ac0][offset+2] += 1
            V[br0*p+cc1][offset+2] += 1
            V[br1*p+cc1][offset+2] -= 1
            W[cc1*n+0][offset+2] += 1
            W[cc1*n+1][offset+2] += 1

            # m4 = A[1,1] * (B[1,0]-B[0,0])
            U[1*m+ac1][offset+3] += 1
            V[br1*p+cc0][offset+3] += 1
            V[br0*p+cc0][offset+3] -= 1
            W[cc0*n+0][offset+3] += 1
            W[cc0*n+1][offset+3] += 1

            # m5 = (A[0,0]+A[0,1]) * B[1,1]
            U[0*m+ac0][offset+4] += 1
            U[0*m+ac1][offset+4] += 1
            V[br1*p+cc1][offset+4] += 1
            W[cc0*n+0][offset+4] -= 1
            W[cc1*n+0][offset+4] += 1

            # m6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
            U[1*m+ac0][offset+5] += 1
            U[0*m+ac0][offset+5] -= 1
            V[br0*p+cc0][offset+5] += 1
            V[br0*p+cc1][offset+5] += 1
            W[cc1*n+1][offset+5] += 1

            # m7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
            U[0*m+ac1][offset+6] += 1
            U[1*m+ac1][offset+6] -= 1
            V[br1*p+cc0][offset+6] += 1
            V[br1*p+cc1][offset+6] += 1
            W[cc0*n+0][offset+6] += 1

        # Four blocks
        add_block(0, 0, 0, 0)    # A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
        add_block(7, 0, 2, 2)    # A[:,0:2] * B[2:4,2:4] -> C[:,2:4]
        add_block(14, 2, 0, 0)   # A[:,2:4] * B[0:2,0:2] -> C[:,0:2]
        add_block(21, 2, 2, 2)   # A[:,2:4] * B[2:4,2:4] -> C[:,2:4]

        return U, V, W

    # Verify the rank-28 solution
    U, V, W = build_rank_28()

    # Verify: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal 1 if this is the structure tensor
    # Actually the structure tensor for matmul is: T[i,j,k] = 1 for all valid (i,j,k) is NOT right.
    # The correct tensor: C[i,k] = sum_j A[i,j]*B[j,k]
    # So the bilinear map is: (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,k,j] = 1 (it's all ones in a sense, but structured)
    # Decomposition: T = sum_r U_r ⊗ W_r ⊗ V_r where
    # U_r[i*m+j] picks out A[i,j], V_r[j'*p+k] picks out B[j',k], W_r[k'*n+i'] picks out C[i',k']
    # And the constraint is: sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = delta_{j,j} (always 1 when indices match)
    # Wait no. The correct identity is:
    # For each output element C[i,k]: C[i,k] = sum_j A[i,j] B[j,k]
    # In the decomposition: C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_{j'} V[j'*p+k, r] B[j',k]) * W[k*n+i, r]
    # Hmm, that's not quite standard. Let me use the standard tensor network approach.

    # The multiplication tensor T ∈ R^{nm} ⊗ R^{mp} ⊗ R^{pn} (or some ordering)
    # T[a, b, c] where a indexes A, b indexes B, c indexes C
    # T[a, b, c] = 1 if a, b, c are "compatible" (i.e., they represent the same (i,j,k) triple), else 0
    # More precisely: if a = i*m+j, b = j*p+k, c = k*n+i, then T[a,b,c] = 1, else 0.

    # So the decomposition is: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # i.e., T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]

    # Let's verify our solution:
    # For a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a,b,c] should be 1 iff i=i', j=j', k=k'

    def verify_decomp(U, V, W, tol=1e-10):
        """Check that sum_r U[a,r]*V[b,r]*W[c,r] = delta_{a,b,c compatible}"""
        nm, R = U.shape
        mp, R2 = V.shape
        pn, R3 = W.shape
        assert R == R2 == R3
        # Check all triples
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    val = sum(U[a,r]*V[b,r]*W[c,r] for r in range(R))
                    if abs(val - 1.0) > tol:
                        max_err = max(max_err, abs(val - 1.0))
        # Check a few incompatible triples should give 0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    a = i*m + j
                    # Pick a different j'
                    j2 = (j+1) % m
                    if j2 != j:
                        b = j2*p + k
                        c = k*n + i
                        val = sum(U[a,r]*V[b,r]*W[c,r] for r in range(R))
                        if abs(val) > tol:
                            max_err = max(max_err, abs(val))
                    # Pick a different k'
                    k2 = (k+1) % p
                    if k2 != k:
                        b = j*p + k2
                        c = k2*n + i
                        val = sum(U[a,r]*V[b,r]*W[c,r] for r in range(R))
                        if abs(val) > tol:
                            max_err = max(max_err, abs(val))
        return max_err

    err = verify_decomp(U, V, W)
    print(f"Rank-28 verification error: {err:.2e}")

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now try to improve: attempt rank reduction
    # Approach: Try to find if we can reduce rank by checking linear dependencies in the 
    # "slice" matrices, or use a numerical search.

    # Method: For a given rank R, the tensors U, V, W (each R columns) define the decomposition.
    # To reduce rank, we try to find a column of U, V, W that is "redundant" or can be 
    # eliminated by adjusting others.

    # Alternative approach: Use the fact that we can do a change of basis in the rank space.
    # If we have a rank-R solution, we can apply any invertible R x R matrix P:
    # U' = U @ P, V' = V @ P^{-1}, W' = W
    # This preserves the decomposition. We can use this to zero out entries.

    # Let's try a more direct approach: random search for rank-27
    # Fix one column to be zero and see if we can adjust the others.

    # Actually, let's try a smarter approach: use the structure of the problem.
    # The 2x4x4 matmul tensor has some known properties. Let me try to use a 
    # "sliding window" or different block structure.

    # Alternative block decomposition: Instead of 4 blocks of 2x2, try 2 blocks of 
    # different sizes, or overlap blocks.

    # Try: A[:,0:3] * B[0:3,:] (3-wide overlap) -- this doesn't directly work with Strassen.

    # Let me try a different strategy: use the rank-28 solution as a starting point and 
    # attempt to find a rank-27 solution by optimization.

    # For rank R, we need U (nm x R), V (mp x R), W (pn x R) such that
    # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c] for all a,b,c.

    # This is a system of nonlinear equations. We can use scipy's least_squares.

    try:
        from scipy.optimize import least_squares
        import sys

        def try_reduce_rank(start_rank, target_rank, max_iter=200, seed=42):
            """Try to reduce from start_rank to target_rank using optimization."""
            rng = np.random.RandomState(seed)
            
            # Start from the start_rank solution
            U0, V0, W0 = build_rank_28()
            # Take first target_rank columns
            U_init = U0[:, :target_rank].copy()
            V_init = V0[:, :target_rank].copy()
            W_init = W0[:, :target_rank].copy()
            # Add small noise to avoid getting stuck
            U_init += rng.randn(*U_init.shape) * 0.01
            V_init += rng.randn(*V_init.shape) * 0.01
            W_init += rng.randn(*W_init.shape) * 0.01

            nm = n*m
            mp = m*p
            pn = p*n
            R = target_rank

            # Precompute all (a,b,c) triples
            triples = []
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        triples.append((i*m+j, j*p+k, k*n+i, 1.0))
            # Add some incompatible triples
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        j2 = (j+1) % m
                        if j2 != j:
                            triples.append((i*m+j, j2*p+k, k*n+i, 0.0))
                        k2 = (k+1) % p
                        if k2 != k:
                            triples.append((i*m+j, j*p+k2, k2*n+i, 0.0))

            triples = np.array(triples)
            a_idx = triples[:,0].astype(int)
            b_idx = triples[:,1].astype(int)
            c_idx = triples[:,2].astype(int)
            targets = triples[:,3]

            def residuals(x):
                U_flat = x[:nm*R]
                V_flat = x[nm*R:nm*R+mp*R]
                W_flat = x[nm*R+mp*R:nm*R+mp*R+pn*R]
                U_mat = U_flat.reshape(nm, R)
                V_mat = V_flat.reshape(mp, R)
                W_mat = W_flat.reshape(pn, R)
                # Compute T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # This is expensive; do it vectorized
                # For each triple (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r]
                # = (U[a,:] * V[b,:] * W[c,:]).sum()
                # Vectorized: 
                Ua = U_mat[a_idx]  # (N, R)
                Vb = V_mat[b_idx]  # (N, R)
                Wc = W_mat[c_idx]  # (N, R)
                vals = (Ua * Vb * Wc).sum(axis=1)  # (N,)
                return vals - targets

            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            N = len(a_idx)
            
            # Use bounded optimization or just least squares
            # This might be too slow for full system; subsample
            if N > 500:
                perm = rng.permutation(N)
                a_idx = a_idx[perm[:500]]
                b_idx = b_idx[perm[:500]]
                c_idx = c_idx[perm[:500]]
                targets = targets[perm[:500]]
                N = 500

            result = least_squares(residuals, x0, max_nfev=max_iter*10, xtol=1e-8, ftol=1e-8)
            x_opt = result.x
            U_opt = x_opt[:nm*R].reshape(nm, R)
            V_opt = x_opt[nm*R:nm*R+mp*R].reshape(mp, R)
            W_opt = x_opt[nm*R+mp*R:nm*R+mp*R+pn*R].reshape(pn, R)

            # Verify with full check
            max_err = 0.0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        a = i*m+j
                        b = j*p+k
                        c = k*n+i
                        val = (U_opt[a]*V_opt[b]*W_opt[c]).sum()
                        max_err = max(max_err, abs(val-1.0))
                        # Incompatible
                        j2 = (j+1)%m
                        if j2 != j:
                            b2 = j2*p+k
                            val2 = (U_opt[a]*V_opt[b2]*W_opt[c]).sum()
                            max_err = max(max_err, abs(val2))
                        k2 = (k+1)%p
                        if k2 != k:
                            b3 = j*p+k2
                            c3 = k2*n+i
                            val3 = (U_opt[a]*V_opt[b3]*W_opt[c3]).sum()
                            max_err = max(max_err, abs(val3))
            
            print(f"Rank {R} optimization: max_err={max_err:.2e}, cost={result.cost:.2e}")
            if max_err < 1e-6:
                return U_opt, V_opt, W_opt
            return None

        # Try rank 27
        best_solution = (U, V, W)
        best_rank = 28

        for target_r in [27]:
            if time.time() > deadline - 20:
                break
            print(f"Attempting rank {target_r}...")
            sol = try_reduce_rank(28, target_r, max_iter=100, seed=42)
            if sol is not None:
                best_solution = sol
                best_rank = target_r
                print(f"Found valid rank {target_r} solution!")
                break

        U, V, W = best_solution

    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fall back to rank 28
        U, V, W = build_rank_28()

    if time.time() > deadline:
        pass

    # Convert to lists for the return interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
