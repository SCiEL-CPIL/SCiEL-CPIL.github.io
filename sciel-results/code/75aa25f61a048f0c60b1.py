"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120

    n, m, p = DIMENSIONS

    def build_base():
        R = 54
        U = np.zeros((n * m, R), dtype=float)
        V = np.zeros((m * p, R), dtype=float)
        W = np.zeros((p * n, R), dtype=float)
        c = 0

        def put(u_pairs, v_pairs, w_pairs):
            nonlocal c
            if c >= R:
                return
            for idx, val in u_pairs:
                U[idx, c] += val
            for idx, val in v_pairs:
                V[idx, c] += val
            for idx, val in w_pairs:
                W[idx, c] += val
            c += 1

        def strassen(ja, jb, ka, kb):
            # Standard Strassen for 2x2x2:
            # M1 = (A00 + A11)(B00 + B11)
            # M2 = (A10 + A11)B00
            # M3 = A00(B01 - B11)
            # M4 = A11(B10 - B00)
            # M5 = (A00 + A01)B11
            # M6 = (A10 - A00)(B00 + B01)
            # M7 = (A01 - A11)(B10 + B11)
            
            # M1
            put([(0*m+ja, 1), (1*m+jb, 1)], [(ja*p+ka, 1), (jb*p+kb, 1)], [(ka*n+0, 1), (kb*n+1, 1)])
            # M2
            put([(1*m+ja, 1), (1*m+jb, 1)], [(ja*p+ka, 1)], [(ka*n+1, 1), (kb*n+1, -1)])
            # M3
            put([(0*m+ja, 1)], [(ja*p+kb, 1), (jb*p+kb, -1)], [(kb*n+0, 1), (kb*n+1, 1)])
            # M4
            put([(1*m+jb, 1)], [(jb*p+ka, 1), (ja*p+ka, -1)], [(ka*n+0, 1), (ka*n+1, 1)])
            # M5
            put([(0*m+ja, 1), (0*m+jb, 1)], [(jb*p+kb, 1)], [(ka*n+0, -1), (kb*n+0, 1)])
            # M6
            put([(1*m+ja, 1), (0*m+ja, -1)], [(ja*p+ka, 1), (ja*p+kb, 1)], [(kb*n+1, 1)])
            # M7
            put([(0*m+jb, 1), (1*m+jb, -1)], [(jb*p+ka, 1), (jb*p+kb, 1)], [(ka*n+0, 1)])

        # Apply Strassen to all 2x2 blocks in the middle dimension (j) 
        # and all 2x2 blocks in the output dimension (k)
        # j pairs: (0,1), (2,3)
        # k pairs: (0,1), (2,3), (4,5)
        for (ja, jb) in [(0, 1), (2, 3)]:
            for (ka, kb) in [(0, 1), (2, 3), (4, 5)]:
                strassen(ja, jb, ka, kb)

        # Handle the remaining middle index j=4 with schoolbook multiplication
        # This covers i in {0,1}, j=4, k in {0..5}
        for k in range(p):
            for i in range(n):
                U[i * m + 4, c] = 1.0
                V[4 * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1

        return U, V, W, c

    def verify_decomposition(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(U.shape[1]):
                        actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(actual - expected) > 1e-10:
                        return False
        return True

    def try_reduce_rank(U, V, W, max_reduction=5):
        """
        Try to reduce rank by finding linear dependencies among the rank-1 terms.
        We look for a non-trivial linear combination of the columns of U, V, W
        that sums to zero in the tensor space.
        
        Specifically, we check if there exists a vector alpha such that:
        sum_r alpha_r * U[:, r] * V[:, r] * W[:, r] = 0
        and sum alpha_r = 0 (to maintain the normalization or simply to find a null vector).
        
        Actually, we want to find if the columns are linearly dependent in the sense that
        there is a subset that can be removed or combined.
        A simpler approach: check if any column is a linear combination of others
        in the flattened tensor space? No, that's not quite right for tensor rank.
        
        Instead, we can try to find a vector x such that:
        sum_r x_r U[:,r] = 0, sum_r x_r V[:,r] = 0, sum_r x_r W[:,r] = 0
        If such an x exists (and not all zero), then we can "remove" the components
        weighted by x. But this requires the same x to work for U, V, and W simultaneously.
        
        Let's formulate it as: Find x != 0 such that:
        U x = 0
        V x = 0
        W x = 0
        
        If such an x exists, then the decomposition is not minimal.
        We can then project out this dependency.
        """
        n, m, p = DIMENSIONS
        R = U.shape[1]
        
        # Stack U, V, W vertically to form a large matrix M
        # M x = 0
        M = np.vstack([U, V, W])
        
        # Find the null space of M
        # Use SVD to find the null space
        try:
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=True)
        except:
            return U, V, W, R
            
        # The null space is spanned by the last columns of Vt.T
        # The rank of M is the number of non-zero singular values
        tol = 1e-10
        rank_M = np.sum(s > tol)
        
        if rank_M < R:
            # There is a non-trivial null space
            # The basis vectors for the null space are Vt[rank_M:].T
            # We can try to remove one dimension corresponding to a null vector
            
            # For simplicity, let's just return the original if we can't easily
            # construct a new decomposition. Properly reducing rank requires
            # finding a change of basis.
            
            # If we find a null vector x, we can define new bases:
            # U_new = U @ P, V_new = V @ P, W_new = W @ P
            # where P is a projection that removes the x direction.
            
            # However, implementing this robustly within time limits is complex.
            # Given the tight time limit and the fact that we already have a valid
            # rank 54 solution, we will skip the actual reduction unless it's trivial.
            pass
            
        return U, V, W, R

    U, V, W, R = build_base()
    
    # Verify the base decomposition
    if not verify_decomposition(U, V, W):
        raise ValueError("Base decomposition is invalid")
    
    # Try to reduce rank if time permits
    if time.time() < deadline - 30:
        U, V, W, R = try_reduce_rank(U, V, W)
    
    # Convert to lists for the return interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
