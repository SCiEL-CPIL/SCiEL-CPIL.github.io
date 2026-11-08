"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    deadline = time.time() + 140  # Safety margin
    
    def validate(U, V, W, rank):
        """Check if U, V, W correctly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def strassen_2x2x2(U, V, W, col, i0, i1, j0, j1, k0, k1):
        """Add Strassen decomposition for 2x2x2 block. Returns new col."""
        # P1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        U[i0*m+j0, col] += 1; U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k0*n+i0, col] += 1; W[k1*n+i1, col] += 1
        col += 1
        # P2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        U[i1*m+j0, col] += 1; U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1
        W[k0*n+i1, col] += 1; W[k1*n+i1, col] -= 1
        col += 1
        # P3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
        U[i0*m+j0, col] += 1
        V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
        W[k1*n+i0, col] += 1; W[k1*n+i1, col] += 1
        col += 1
        # P4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
        U[i1*m+j1, col] += 1
        V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
        W[k0*n+i0, col] += 1; W[k0*n+i1, col] += 1
        col += 1
        # P5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        U[i0*m+j0, col] += 1; U[i0*m+j1, col] += 1
        V[j1*p+k1, col] += 1
        W[k0*n+i0, col] -= 1; W[k1*n+i0, col] += 1
        col += 1
        # P6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        U[i1*m+j0, col] += 1; U[i0*m+j0, col] -= 1
        V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
        W[k1*n+i1, col] += 1
        col += 1
        # P7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        U[i0*m+j1, col] += 1; U[i1*m+j1, col] -= 1
        V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k0*n+i0, col] += 1
        col += 1
        return col
    
    def schoolbook_block(U, V, W, col, i_range, j_range, k_range):
        """Add schoolbook terms for a block. Returns new col."""
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i*m+j, col] += 1
                    V[j*p+k, col] += 1
                    W[k*n+i, col] += 1
                    col += 1
        return col
    
    def try_reduce_rank(U, V, W, rank):
        """Try to reduce rank by finding linearly dependent rank-1 terms.
        
        Strategy: For each pair of columns, check if their contribution can be
        expressed as a linear combination of others. If so, merge them.
        
        A more practical approach: build the tensor and use SVD-based rank reduction
        on the mode-3 unfolding to identify redundant directions.
        """
        # Build the full tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # and compare with the target tensor
        # 
        # For rank reduction, we look for linear dependencies in the space of
        # rank-1 tensors. Two rank-1 terms u1⊗v1⊗w1 and u2⊗v2⊗w2 are linearly
        # dependent if one is a scalar multiple of the other.
        #
        # A practical approach: try removing each column and see if the remaining
        # columns can be adjusted to still decompose the tensor.
        #
        # Simpler approach: check if any column is zero or duplicate.
        
        # Check for zero columns
        for r in range(rank):
            if np.allclose(U[:, r], 0) and np.allclose(V[:, r], 0) and np.allclose(W[:, r], 0):
                # This column contributes nothing, remove it
                return try_reduce_rank(U[:, [r] + list(range(rank-1))], 
                                       V[:, [r] + list(range(rank-1))], 
                                       W[:, [r] + list(range(rank-1))], 
                                       rank - 1)
        
        # Check for duplicate columns (up to scaling)
        for r in range(rank):
            for s in range(r+1, rank):
                # Check if column r and s are proportional
                if not np.allclose(U[:, r], 0) and not np.allclose(V[:, r], 0) and not np.allclose(W[:, r], 0):
                    if not np.allclose(U[:, s], 0) and not np.allclose(V[:, s], 0) and not np.allclose(W[:, s], 0):
                        # Check proportionality
                        try:
                            alpha_u = U[0, s] / U[0, r] if U[0, r] != 0 else None
                            if alpha_u is not None:
                                if (np.allclose(U[:, s], alpha_u * U[:, r], atol=1e-10) and
                                    np.allclose(V[:, s], alpha_u * V[:, r], atol=1e-10) and
                                    np.allclose(W[:, s], alpha_u * W[:, r], atol=1e-10)):
                                    # Columns are proportional, keep one
                                    keep = [c for c in range(rank) if c != s]
                                    return try_reduce_rank(U[:, keep], V[:, keep], W[:, keep], rank - 1)
                        except:
                            pass
        
        return U, V, W, rank
    
    def try_pairwise_merge(U, V, W, rank):
        """Try to merge pairs of columns that span a 2-dimensional subspace."""
        # For each pair (r, s), check if the 2D subspace they span can be
        # represented by a single rank-1 term plus adjustments to other terms.
        # This is complex; instead, try a simpler heuristic:
        # 
        # If two columns have the same support pattern (same non-zero positions),
        # they might be mergeable.
        
        # For now, just return as-is
        return U, V, W, rank
    
    # Build the rank-54 decomposition
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for j0, j1 in [(0,1), (2,3)]:
        for k0, k1 in [(0,1), (2,3), (4,5)]:
            col = strassen_2x2x2(U, V, W, col, 0, 1, j0, j1, k0, k1)
    col = schoolbook_block(U, V, W, col, [0,1], [4], range(p))
    
    # Verify
    if not validate(U, V, W, rank):
        # Fallback to pure schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = schoolbook_block(U, V, W, 0, range(n), range(m), range(p))
    
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
    
    # Try to reduce rank using linear algebra
    if time.time() < deadline - 60:
        U_red, V_red, W_red, rank_red = try_reduce_rank(U.copy(), V.copy(), W.copy(), rank)
        if rank_red < best_rank and validate(U_red, V_red, W_red, rank_red):
            best_U, best_V, best_W, best_rank = U_red, V_red, W_red, rank_red
    
    # Try more aggressive reduction: use tensor unfolding and SVD
    if time.time() < deadline - 30:
        # Unfold the tensor along mode 3: T_3[a*b, c] where c indexes the rank-1 terms
        # Actually, let's build the tensor and check its numerical rank
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Check if our decomposition matches
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T, T_decomp, atol=1e-9):
            # Now try to find a lower-rank decomposition
            # Use the fact that the tensor has a specific structure
            # For matrix multiplication, the tensor is "diagonal" in the middle index
            # 
            # Try: use the known lower bounds and known results
            # R(2,5,6) is known to be at most 54 (from our construction)
            # Lower bound: R(2,5,6) >= max(R(2,5,1), R(2,1,6), R(1,5,6)) = max(10, 12, 30) = 30
            # But better bounds exist.
            #
            # For now, try to see if we can find rank 53 by checking
            # if the 54-dimensional space has a 1-dimensional nullspace
            # in the appropriate tensor product space.
            pass
    
    # Final: try to see if we can do better with a different block structure
    if time.time() < deadline - 20:
        # Try: split m as {0,1,2,3} and {4}, p as {0,1,2} and {3,4,5}
        # 2x4x3 + 2x1x3 + 2x4x3 + 2x1x3
        # R(2,4,3) <= R(2,2,3) + R(2,2,3) = 11+11 = 22
        # R(2,1,3) = 6
        # Total: 22+6+22+6 = 56 (worse than 54)
        pass
        
        # Try: split m as {0,1,2} and {3,4}, p as {0,1,2,3} and {4,5}
        # 2x3x4 + 2x2x4 + 2x3x2 + 2x2x2
        # R(2,3,4) <= R(2,3,2) + R(2,3,2) = ? 
        # R(2,2,4) <= R(2,2,2) + R(2,2,2) = 7+7 = 14
        # R(2,3,2) <= ? At most 2*3*2=12, possibly less
        # R(2,2,2) = 7
        # This is getting complex; the 54 solution is likely near-optimal
        pass
    
    # Return the best solution found
    U_list = [[int(U[i, c]) if abs(U[i, c] - round(U[i, c])) < 1e-9 else float(U[i, c]) for c in range(best_rank)] for i in range(n*m)]
    V_list = [[int(V[i, c]) if abs(V[i, c] - round(V[i, c])) < 1e-9 else float(V[i, c]) for c in range(best_rank)] for i in range(m*p)]
    W_list = [[int(W[i, c]) if abs(W[i, c] - round(W[i, c])) < 1e-9 else float(W[i, c]) for c in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
