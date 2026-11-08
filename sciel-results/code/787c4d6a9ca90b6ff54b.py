import numpy as np
import random
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    dim_u = n * m  # 8
    dim_v = m * p  # 20
    dim_w = p * n  # 10
    
    # Try to find a rank-minimal decomposition via random search + verification
    # The trivial rank is n*m*p = 40
    # Let's try to find something better
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
        # Check that sum_r U[.,r] * V[.,r] * W[.,r] = T_{matmul}
        # T[i,j,k] = 1 if (i,m+k) is in A and (j,p+k) is in B... 
        # Actually, the tensor T has entries T[i,j,k] where:
        # U has entries for (i,j) pairs of A (n*m entries)
        # V has entries for (j,k) pairs of B (m*p entries)
        # W has entries for (k,i) pairs of C (p*n entries)
        # The tensor decomposition says: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
        # where T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (k,i) entry of C
        # In standard matmul: C[k,i] = sum_j A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
        
        # Build the full tensor
        T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Compute the decomposition
        T_decomp = np.zeros_like(T)
        for r in range(rank):
            T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        return np.allclose(T, T_decomp, atol=1e-10)
    
    # Start with the trivial decomposition (rank 40)
    rank_trivial = n * m * p  # 40
    U_trivial = np.zeros((dim_u, rank_trivial))
    V_trivial = np.zeros((dim_v, rank_trivial))
    W_trivial = np.zeros((dim_w, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m+j, col] = 1.0
                V_trivial[j*p+k, col] = 1.0
                W_trivial[k*n+i, col] = 1.0
                col += 1
    
    # Verify trivial
    assert verify_decomposition(U_trivial, V_trivial, W_trivial, rank_trivial)
    
    best_rank = rank_trivial
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    
    # Try rank 39, 38, ... using random search
    # Strategy: use integer entries and try random sign patterns with structured patterns
    
    # For each rank r < 40, try to find a valid decomposition
    # Use a more structured approach: try to merge some terms
    
    # Key insight: we can try to find a decomposition by looking for linear dependencies
    # in the tensor and exploiting them.
    
    # Let's try a different approach: use the known result that for 2x4 by 4x5,
    # the rank might be reducible. Let's try systematic search for smaller ranks.
    
    # Actually, let's try to use a more clever construction.
    # For 2xm by mxp, the rank is at most 2mp (trivial).
    # Can we do better?
    
    # Let's try rank 38: merge pairs that share structure
    # Try to combine terms where possible
    
    # Random search for better rank
    random.seed(42)
    
    for target_rank in range(35, 40):
        if time.time() > deadline:
            break
            
        # Try multiple random attempts for this rank
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Generate a random decomposition with integer entries
            # Use a structured random approach
            U_candidate = np.zeros((dim_u, target_rank))
            V_candidate = np.zeros((dim_v, target_rank))
            W_candidate = np.zeros((dim_w, target_rank))
            
            # Start from trivial and try to merge
            # Create a random subset of the trivial terms and see if we can absorb others
            all_terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
            random.shuffle(all_terms)
            
            # Assign each term to a random rank slot
            # But this won't work directly; we need the decomposition to be exact
            
            # Different approach: use the fact that we can write the tensor as a sum
            # of rank-1 tensors, and look for linear combinations
            pass
        
    # Since random search is too slow, let's stick with the known rank 36 decomposition
    # from the current program and verify it
    
    # Build the rank-36 Strassen-based decomposition
    rank = 36
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1: (y00 + y11) * (z00 + z11)
        U[u_idx(y00[0], y00[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M2: (y10 + y11) * z00
        U[u_idx(y10[0], y10[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
        col += 1
        
        # M3: y00 * (z01 - z11)
        U[u_idx(y00[0], y00[1])][col] += 1
        V[v_idx(z01[0], z01[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M4: y11 * (z10 - z00)
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z10[0], z10[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] -= 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        col += 1
        
        # M5: (y00 + y01) * z11
        U[u_idx(y00[0], y00[1])][col] += 1
        U[u_idx(y01[0], y01[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        col += 1
        
        # M6: (y10 - y00) * (z00 + z01)
        U[u_idx(y10[0], y10[1])][col] += 1
        U[u_idx(y00[0], y00[1])][col] -= 1
        V[v_idx(z00[0], z00[1])][col] += 1
        V[v_idx(z01[0], z01[1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M7: (y01 - y11) * (z10 + z11)
        U[u_idx(y01[0], y01[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] -= 1
        V[v_idx(z10[0], z10[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        col += 1
    
    # Block 1: A_L B_11 (2x2 * 2x2)
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y1, z1, w1)
    
    # Block 2: A_R B_21 (2x2 * 2x2)
    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y2, z2, w2)
    
    # Block 3: A_L B_12 (2x2 * 2x2)
    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    w3 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y3, z3, w3)
    
    # Block 4: A_R B_22 (2x2 * 2x2)
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y4, z4, w4)
    
    # Block 5: A_L B_15 (2x2 * 2x1) - rank 4
    U[u_idx(0,0)][col] += 1
    V[v_idx(0,4)][col] += 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(0,1)][col] += 1
    V[v_idx(1,4)][col] += 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(1,0)][col] += 1
    V[v_idx(0,4)][col] += 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    U[u_idx(1,1)][col] += 1
    V[v_idx(1,4)][col] += 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # Block 6: A_R B_25 (2x2 * 2x1) - rank 4
    U[u_idx(0,2)][col] += 1
    V[v_idx(2,4)][col] += 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(0,3)][col] += 1
    V[v_idx(3,4)][col] += 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(1,2)][col] += 1
    V[v_idx(2,4)][col] += 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    U[u_idx(1,3)][col] += 1
    V[v_idx(3,4)][col] += 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify
    if not verify_decomposition(U, V, W, rank):
        # Fall back to trivial
        return {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
