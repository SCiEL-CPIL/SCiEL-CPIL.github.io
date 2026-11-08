import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def build_rank33():
        rank = 33
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        col = 0
        
        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + (j_off + 0)][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 0) * p + (k_off + 0)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[(k_off + 0) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P2 = (A10 + A11) B00
            U[1 * m + (j_off + 0)][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 0) * p + (k_off + 0)][c] += 1
            W[(k_off + 0) * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            
            # P3 = A00 (B01 - B11)
            U[0 * m + (j_off + 0)][c] += 1
            V[(j_off + 0) * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 0)][c] += 1
            V[(j_off + 0) * p + (k_off + 0)][c] -= 1
            W[(k_off + 0) * n + 0][c] += 1
            W[(k_off + 0) * n + 1][c] += 1
            c += 1
            
            # P5 = (A00 + A01) B11
            U[0 * m + (j_off + 0)][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[(k_off + 0) * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            
            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + (j_off + 0)][c] += 1
            U[0 * m + (j_off + 0)][c] -= 1
            V[(j_off + 0) * p + (k_off + 0)][c] += 1
            V[(j_off + 0) * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + (k_off + 0)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[(k_off + 0) * n + 0][c] += 1
            c += 1
            
            return c

        # 4 Strassen blocks
        col = add_strassen(0, 0, col)  # J1 x K1
        col = add_strassen(0, 2, col)  # J1 x K2
        col = add_strassen(2, 0, col)  # J2 x K1
        col = add_strassen(2, 2, col)  # J2 x K2
        
        # Boundary terms (Rank 1 each)
        
        # J1 x K3 (j in 0,1; k=4)
        for i in range(n):
            for j in [0, 1]:
                U[i * m + j][col] = 1
        for j in [0, 1]:
            V[j * p + 4][col] = 1
        for i in range(n):
            W[4 * n + i][col] = 1
        col += 1
        
        # J2 x K3 (j in 2,3; k=4)
        for i in range(n):
            for j in [2, 3]:
                U[i * m + j][col] = 1
        for j in [2, 3]:
            V[j * p + 4][col] = 1
        for i in range(n):
            W[4 * n + i][col] = 1
        col += 1
        
        # J3 x K1 (j=4; k in 0,1)
        for i in range(n):
            U[i * m + 4][col] = 1
        for k in [0, 1]:
            V[4 * p + k][col] = 1
        for i in range(n):
            for k in [0, 1]:
                W[k * n + i][col] = 1
        col += 1
        
        # J3 x K2 (j=4; k in 2,3)
        for i in range(n):
            U[i * m + 4][col] = 1
        for k in [2, 3]:
            V[4 * p + k][col] = 1
        for i in range(n):
            for k in [2, 3]:
                W[k * n + i][col] = 1
        col += 1
        
        # J3 x K3 (j=4; k=4)
        for i in range(n):
            U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        for i in range(n):
            W[4 * n + i][col] = 1
        col += 1
        
        return U, V, W

    # MILP-based search: fix V to sparse basis, solve for U and W
    def milp_search(rank, time_limit=30):
        """
        Fix V to a sparse canonical basis (one 1 per row across rank columns).
        Then solve linear system for U and W with entries in {-1, -0.5, 0, 0.5, 1}.
        """
        if time.time() - start_time > DEADLINE - 5:
            return None
        
        # Canonical V: each row of V has exactly one 1 in some column
        # We need to assign each of V_dim rows to a column in [0, rank)
        # Such that the resulting system is solvable
        
        # For each column r, let S_r = set of V-indices assigned to column r
        # Then for each (u,v,w) with T[u,v,w] != 0:
        #   sum_r U[u,r] * V[v,r] * W[w,r] = T[u,v,w]
        # Since V[v,r] = 1 if v is assigned to column r, else 0:
        #   U[u, col(v)] * W[w, col(v)] = T[u,v,w]
        
        # This means for each v (assigned to column c(v)):
        #   For all u,w: U[u,c(v)] * W[w,c(v)] = T[u,v,w]
        
        # The product U[u,c]*W[w,c] must equal T[u,v,w] for all u,w.
        # This is a rank-1 constraint on the submatrix T[:,v,:]
        
        # So first, check which v-indices have T[:,v,:] of rank 1
        # (or zero)
        
        valid_v = []
        for v in range(V_dim):
            sub = T[:, v, :]
            if np.allclose(sub, 0, atol=1e-9):
                valid_v.append(('zero', v))
            else:
                # Check if sub is rank 1
                rank_sub = np.linalg.matrix_rank(sub)
                if rank_sub <= 1:
                    valid_v.append(('rank1', v))
                else:
                    valid_v.append(('invalid', v))
        
        # Count how many v-indices are valid
        n_valid = sum(1 for t, v in valid_v if t != 'invalid')
        n_invalid = sum(1 for t, v in valid_v if t == 'invalid')
        
        # For invalid v-indices, we need to use multiple columns
        # This gets complex. Let's try a different approach:
        # Allow V entries to be 0, 0.5, or 1 (sparse)
        
        # Simpler approach: try random sparse V assignments and solve LP
        
        from scipy.optimize import linprog
        
        # Variables: U[u,r] and W[w,r] for all u,w,r
        # But this is too many variables. Let's use the structure.
        
        # Alternative: for a given V assignment, the constraints are:
        # For each (u,v,w): sum_r U[u,r]*V[v,r]*W[w,r] = T[u,v,w]
        # This is bilinear in U and W. Not linear.
        
        # Let's use a different strategy: 
        # Try to find a decomposition by solving a sequence of LPs.
        
        # Actually, let's try a heuristic: start from rank 33 and try to remove columns.
        
        return None

    # Heuristic: start from rank 33 and try to merge/remove columns
    def try_reduce_rank(U0, V0, W0, max_rank_reduction=5):
        """Try to reduce rank by removing or merging columns."""
        U, V, W = U0.copy(), V0.copy(), W0.copy()
        current_rank = U.shape[1]
        
        # Try removing each column and checking if we can redistribute
        for col_to_remove in range(current_rank):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            U_test = np.delete(U, col_to_remove, axis=1)
            V_test = np.delete(V, col_to_remove, axis=1)
            W_test = np.delete(W, col_to_remove, axis=1)
            
            # Check if the remaining decomposition still works
            # (It won't in general, but let's see if we can fix it)
            
            # Try to add back the missing contributions by modifying other columns
            # This is complex. Skip for now.
        
        return None

    # Build the rank 33 baseline
    U33, V33, W33 = build_rank33()
    
    best_result = {
        "U": [[float(x) for x in row] for row in U33],
        "V": [[float(x) for x in row] for row in V33],
        "W": [[float(x) for x in row] for row in W33]
    }
    best_rank = 33
    
    if not validate(U33, V33, W33):
        # Fallback to rank 50
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
        best_rank = 50
    
    # Try to improve using MILP-like approach
    # Key insight: for 2x5x5, we can try to find a better block structure
    
    # Try rank 30: use 3x3 Strassen-like blocks more efficiently
    def build_rank30():
        rank = 30
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        col = 0
        
        # Use a more efficient decomposition for 2x5x5
        # Split into: 2x2x2 (Strassen, 7 mults) + boundary terms
        
        # Core 2x2x2 block using Strassen (7 multiplications)
        def add_strassen_core(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + (j_off + 0)][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 0) * p + (k_off + 0)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[(k_off + 0) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P2 = (A10 + A11) B00
            U[1 * m + (j_off + 0)][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 0) * p + (k_off + 0)][c] += 1
            W[(k_off + 0) * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            
            # P3 = A00 (B01 - B11)
            U[0 * m + (j_off + 0)][c] += 1
            V[(j_off + 0) * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 0)][c] += 1
            V[(j_off + 0) * p + (k_off + 0)][c] -= 1
            W[(k_off + 0) * n + 0][c] += 1
            W[(k_off + 0) * n + 1][c] += 1
            c += 1
            
            # P5 = (A00 + A01) B11
            U[0 * m + (j_off + 0)][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[(k_off + 0) * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            
            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + (j_off + 0)][c] += 1
            U[0 * m + (j_off + 0)][c] -= 1
            V[(j_off + 0) * p + (k_off + 0)][c] += 1
            V[(j_off + 0) * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + (k_off + 0)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[(k_off + 0) * n + 0][c] += 1
            c += 1
            
            return c

        # 2x2x2 core blocks
        col = add_strassen_core(0, 0, col)  # j in {0,1}, k in {0,1}
        col = add_strassen_core(0, 2, col)  # j in {0,1}, k in {2,3}
        col = add_strassen_core(2, 0, col)  # j in {2,3}, k in {0,1}
        col = add_strassen_core(2, 2, col)  # j in {2,3}, k in {2,3}
        # That's 28 multiplications
        
        # Boundary: j=4 or k=4
        # j=4, k=0: rank 1
        for i in range(n):
            U[i * m + 4][col] = 1
        V[4 * p + 0][col] = 1
        for i in range(n):
            W[0 * n + i][col] = 1
        col += 1
        
        # j=4, k=1: rank 1
        for i in range(n):
            U[i * m + 4][col] = 1
        V[4 * p + 1][col] = 1
        for i in range(n):
            W[1 * n + i][col] = 1
        col += 1
        
        # j=4, k=2: rank 1
        for i in range(n):
            U[i * m + 4][col] = 1
        V[4 * p + 2][col] = 1
        for i in range(n):
            W[2 * n + i][col] = 1
        col += 1
        
        # j=4, k=3: rank 1
        for i in range(n):
            U[i * m + 4][col] = 1
        V[4 * p + 3][col] = 1
        for i in range(n):
            W[3 * n + i][col] = 1
        col += 1
        
        # j=4, k=4: rank 1
        for i in range(n):
            U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        for i in range(n):
            W[4 * n + i][col] = 1
        col += 1
        
        # j in {0,1}, k=4: can we do better than rank 2?
        # T[:, j, 4, :] for j in {0,1}
        # These are 2x2 submatrices. If they're rank 1, we can share.
        
        # j=0, k=4 and j=1, k=4
        # Try rank 1: U[:, r] * W[:, r] should match T[:, 0, 4, :] and T[:, 1, 4, :]
        # T[:, 0, 4, :] has 1s at (i, i) for i in {0,1}
        # T[:, 1, 4, :] has 1s at (i, i) for i in {0,1}
        # They're the same! So rank 1 suffices.
        
        for i in range(n):
            U[i * m + 0][col] = 1
            U[i * m + 1][col] = 1
        V[0 * p + 4][col] = 1
        V[1 * p + 4][col] = 1
        for i in range(n):
            W[4 * n + i][col] = 1
        col += 1
        
        # j in {2,3}, k=4
        for i in range(n):
            U[i * m + 2][col] = 1
            U[i * m + 3][col] = 1
        V[2 * p + 4][col] = 1
        V[3 * p + 4][col] = 1
        for i in range(n):
            W[4 * n + i][col] = 1
        col += 1
        
        # Total: 28 + 5 + 2 = 35. That's worse than 33!
        
        return None  # This approach doesn't help

    # Try a different decomposition: use the fact that n=2
    # The output is 5x2. Maybe we can exploit the small n.
    
    def build_rank28():
        """Try rank 28 using a smarter decomposition."""
        rank = 28
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        col = 0
        
        # For each pair (j, k), the contribution to C is:
        # C[i, k'] += A[i, j] * B[j, k] * delta(k, k')
        # So the tensor T[u, v, w] = 1 when u=(i,j), v=(j,k), w=(k,i)
        
        # Key observation: for fixed j, the sub-tensor T[:, (j,:), :] has a special structure.
        # T[i*m+j, j*p+k, k*n+i] = 1
        
        # Let's try to decompose by j:
        # For each j, we need to handle the 2x5x5 subproblem (i in {0,1}, k in {0..4})
        
        # For fixed j, the sub-tensor is:
        # T_sub[i, k, k', i'] = delta(i, i') * delta(k, k') ... no that's not right.
        # T[i*m+j, j*p+k, k'*n+i'] = 1 iff i=i' and k=k'
        
        # So for fixed j, it's a 2x5x2x5 tensor that's 1 when (i,k) = (i',k')
        # This is like an identity map from R^{2x5} to R^{5x2}
        
        # The rank of this sub-tensor is... let's think.
        # It's the tensor for the map (i,k) -> (k,i), which is a permutation.
        # The tensor rank of a permutation tensor... 
        
        # Actually, for fixed j, we have:
        # sum over r: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i', r] = delta(i,i')*delta(k,k')
        
        # This is a 2x5x5x2 tensor. Let's see if we can decompose it efficiently.
        
        # Try: for each k, use one multiplication:
        # U[i*m+j, r_k] = 1 for all i
        # V[j*p+k, r_k] = 1
        # W[k*n+i, r_k] = 1 for all i
        # That's 5 multiplications per j, so 25 total. Plus we need to handle the i-dependence.
        
        # Wait, that doesn't work because W[k*n+i, r] = 1 for all i means
        # the output doesn't depend on i correctly.
        
        # Let me reconsider. For fixed j:
        # We need: U[i, r] * V[k, r] * W[k', i'] = delta(i,i') * delta(k,k')
        # where U is 2xR, V is 5xR, W is 5x2xR... no, W is 10xR.
        
        # W index is k'*n + i', so W[k'*2 + i', r]
        
        # So: U[i, r] * V[k, r] * W[k'*2 + i', r] = delta(i,i') * delta(k,k')
        
        # One approach: for each (k, i), use one term:
        # U[i, r_{k,i}] = 1
        # V[k, r_{k,i}] = 1  
        # W[k*2 + i, r_{k,i}] = 1
        # That's 10 terms per j, 50 total. Too many.
        
        # Better: for each k, use 2 terms (one for each i):
        # Same as above, 10 per j.
        
        # Or: use Strassen-like tricks for the 2x5 part.
        
        # For fixed j, the problem is: multiply 2x5 matrix by 5x5 matrix? No.
        # For fixed j, A[:, j] is a 2x1 vector, B[j, :] is a 1x5 vector.
        # Their outer product is 2x5. The contribution to C is:
        # C[i, k'] += A[i, j] * B[j, k] * delta(k, k')
        # So C[i, k] += A[i, j] * B[j, k]
        
        # This is just a rank-1 update: C += A_col_j * B_row_j
        # The tensor for this is: T[i, k, k', i'] = delta(i,i') * delta(k,k') * A[i,j] * B[j,k]
        
        # In the decomposition, for fixed j, we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k'*2+i', r] = delta(i,i')*delta(k,k')
        
        # This is the tensor for the "identity" map from (i,k) to (k,i) in the output.
        # The rank of this 2x5x5x2 tensor... 
        
        # Actually, I think the minimal rank for this sub-problem is 5 (one per k).
        # For each k: U[:, r_k] = [1,1], V[k, r_k] = 1, W[k*2: k*2+2, r_k] = [1,1]
        # Check: U[i, r_k]*V[k', r_k]*W[k''*2+i', r_k] = 1*delta(k',k)*delta(k'',k)*1 = delta(k',k)*delta(k'',k)
        # But we need delta(i,i')*delta(k,k'). The i,i' dependence is missing!
        
        # So that doesn't work. We need to capture the i,i' dependence.
        
        # For each (k, i): U[i, r_{k,i}] = 1, V[k, r_{k,i}] = 1, W[k*2+i, r_{k,i}] = 1
        # Check: sum_r U[i,r]*V[k',r]*W[k''*2+i',r] = sum_{k,i} delta(i,i_r)*delta(k',k_r)*delta(k'',k_r)*delta(i',i_r)
        # = sum_{k,i} delta(i,i)*delta(k',k)*delta(k'',k)*delta(i',i)
        # = sum_k delta(k',k)*delta(k'',k) = delta(k',k'')
        # But we need delta(i,i')*delta(k,k'). This gives delta(k,k') but not the i,i' part correctly.
        
        # I'm confusing myself. Let me just return the rank 33 solution.
        return None

    # Try to find a rank 32 solution by modifying the rank 33
    def try_rank32():
        """Try to find a rank 32 decomposition."""
        if time.time() - start_time > DEADLINE - 20:
            return None
        
        # Start from rank 33 and try to merge two columns
        U, V, W = build_rank33()
        
        # Try merging column pairs
        for r1 in range(33):
            for r2 in range(r1 + 1, 33):
                if time.time() - start_time > DEADLINE - 10:
                    return None
                
                # Try to see if columns r1 and r2 can be merged
                # by finding a linear combination that preserves the tensor
                pass
        
        return None

    # Just return the best we have
    if time.time() - start_time > DEADLINE - 5:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
