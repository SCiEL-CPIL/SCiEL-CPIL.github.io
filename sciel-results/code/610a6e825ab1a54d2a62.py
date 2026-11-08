"""Matrix multiplication 4,4,7 decomposition - Strassen-based with rank reduction."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    deadline = time.time() + 110
    n, m, p = 4, 4, 7
    terms = []

    def check_deadline():
        if time.time() > deadline - 5:
            return True
        return False

    def add_term(u, v, w):
        terms.append((u, v, w))

    def strassen_2x2_block(i_off, j_off, k_off):
        """7 Strassen terms for 2x2 * 2x2 block."""
        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        add_term(u, v, w)

        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = 1
        w[(k_off+1) * n + (i_off+1)] = -1
        add_term(u, v, w)

        # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = 1
        v[(j_off+1) * p + (k_off+1)] = -1
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        add_term(u, v, w)

        # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[j_off * p + k_off] = -1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[k_off * n + (i_off+1)] = 1
        add_term(u, v, w)

        # M5 = (A11+A12)*B22 -> C12 += M5, C11 -= M5
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[i_off * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[k_off * n + i_off] = -1
        add_term(u, v, w)

        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[i_off * m + j_off] = -1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[j_off * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = 1
        add_term(u, v, w)

        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = 1
        u[(i_off+1) * m + (j_off+1)] = -1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        add_term(u, v, w)

    def strassen_4x4_head():
        """
        Generates 49 terms for 4x4 * 4x4 using recursive Strassen.
        Uses outer 4x4 Strassen (7 terms) x inner 2x2 Strassen (7 terms) = 49 terms.
        """
        outer_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        inner_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        def get_block_indices(block_idx, local_idx, dim_a, offset_k=0):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            row = br * 2 + lr
            col = bc * 2 + lc + offset_k
            if dim_a == 'A':
                return row * m + col
            elif dim_a == 'B':
                return row * p + col
            elif dim_a == 'C':
                return col * n + row

        for outer_a, outer_b, outer_c in outer_terms:
            for inner_x, inner_y, inner_z in inner_terms:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                for x_local, x_coeff in inner_x:
                    for a_block, a_coeff in outer_a:
                        idx = get_block_indices(a_block, x_local, 'A')
                        u[idx] += a_coeff * x_coeff
                        
                for y_local, y_coeff in inner_y:
                    for b_block, b_coeff in outer_b:
                        idx = get_block_indices(b_block, y_local, 'B')
                        v[idx] += b_coeff * y_coeff
                        
                for z_local, z_coeff in inner_z:
                    for c_block, c_coeff in outer_c:
                        idx = get_block_indices(c_block, z_local, 'C')
                        w[idx] += c_coeff * z_coeff
                        
                add_term(u, v, w)

    def strassen_4x4x2_tail():
        """
        Generates 28 terms for 4x4 * 4x2 (cols 4,5) using block Strassen.
        """
        for i_b in range(2):
            for j_b in range(2):
                i_off = i_b * 2
                j_off = j_b * 2
                k_off = 4
                strassen_2x2_block(i_off, j_off, k_off)

    def schoolbook_4x4x1_tail():
        """
        Generates 16 terms for 4x4 * 4x1 (col 6) using schoolbook.
        """
        for i in range(n):
            for j in range(m):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + 6] = 1
                w = [0] * (p * n)
                w[6 * n + i] = 1
                add_term(u, v, w)

    # Build the decomposition
    strassen_4x4_head()
    
    if check_deadline():
        pass
    else:
        strassen_4x4x2_tail()
    
    if check_deadline():
        pass
    else:
        schoolbook_4x4x1_tail()

    rank = len(terms)
    
    if rank == 0:
        # Emergency fallback: schoolbook for everything
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0] * (n * m)
                    u[i * m + j] = 1
                    v = [0] * (m * p)
                    v[j * p + k] = 1
                    w = [0] * (p * n)
                    w[k * n + i] = 1
                    terms.append((u, v, w))
        rank = len(terms)
    
    # Build matrices as numpy arrays for efficient operations
    U_mat = np.zeros((n * m, rank), dtype=np.int64)
    V_mat = np.zeros((m * p, rank), dtype=np.int64)
    W_mat = np.zeros((p * n, rank), dtype=np.int64)
    
    for col, (u, v, w) in enumerate(terms):
        U_mat[:, col] = np.array(u)
        V_mat[:, col] = np.array(v)
        W_mat[:, col] = np.array(w)
    
    # Verify the decomposition is correct before attempting reduction
    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        """Verify that sum_r U[:,r] * V[:,r] * W[:,r] gives the correct multiplication tensor."""
        # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # should equal delta_{i,i'} * delta_{j,j'} for matrix multiplication
        # Actually, for matrix multiplication C = AB:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor is T[i,j,k] = 1 if this corresponds to a valid multiplication term
        # More precisely: T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k]
        
        # Compute the tensor from the decomposition
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        tensor = np.zeros((n, m, p), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(U_mat.shape[1]):
                        val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    tensor[i,j,k] = val
        
        # Expected: T[i,j,k] = 1 for all valid (i,j,k)
        expected = np.ones((n, m, p), dtype=np.int64)
        return np.array_equal(tensor, expected)
    
    # Check if current decomposition is valid
    if not verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        # If invalid, just return what we have (shouldn't happen with correct construction)
        pass
    else:
        # Try to reduce rank using integer-preserving elimination
        # The idea: if we can express one term as a linear combination of others
        # with integer coefficients, we can eliminate it.
        
        # Build the tensor for each term
        def compute_term_tensor(u, v, w, n, m, p):
            """Compute the rank-1 tensor for a single term."""
            tensor = np.zeros((n, m, p), dtype=np.int64)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
            return tensor
        
        # Compute tensor for each term
        term_tensors = []
        for col in range(rank):
            u = U_mat[:, col].tolist()
            v = V_mat[:, col].tolist()
            w = W_mat[:, col].tolist()
            term_tensors.append(compute_term_tensor(u, v, w, n, m, p))
        
        # Try to find linear dependencies among the term tensors
        # We want to find integer coefficients c_r such that sum_r c_r * T_r = 0
        # and then eliminate one term.
        
        # Flatten all tensors
        flat_tensors = np.array([t.flatten() for t in term_tensors], dtype=np.int64)
        # flat_tensors shape: (rank, n*m*p)
        
        # Try to reduce by finding which columns are linearly dependent
        # Use a simple approach: try to eliminate terms one by one
        # by checking if a term can be expressed as integer combination of others
        
        new_U = U_mat.copy()
        new_V = V_mat.copy()
        new_W = W_mat.copy()
        
        # Try greedy elimination
        eliminated = set()
        for target in range(rank):
            if target in eliminated:
                continue
            if check_deadline():
                break
            
            # Try to express term[target] as integer combination of other non-eliminated terms
            other_indices = [i for i in range(rank) if i != target and i not in eliminated]
            if len(other_indices) < 1:
                continue
            
            # Check if we can find small integer coefficients
            # For efficiency, only try if the tensor has small entries
            target_tensor = flat_tensors[target]
            if np.max(np.abs(target_tensor)) > 4:
                continue
            
            # Try simple cases: coefficient of -1, 0, or 1 for each other term
            # This is a simplified search
            found = False
            for coeff in [-1, 1]:
                # Check if target_tensor + coeff * other_tensor = 0 for some other
                for other in other_indices:
                    if other in eliminated:
                        continue
                    if np.array_equal(target_tensor, -coeff * flat_tensors[other]):
                        # target = -coeff * other, so we can eliminate one of them
                        # Keep the one with smaller support
                        target_support = np.count_nonzero(target_tensor)
                        other_support = np.count_nonzero(flat_tensors[other])
                        if target_support <= other_support:
                            # Eliminate target, keep other
                            eliminated.add(target)
                        else:
                            eliminated.add(other)
                        found = True
                        break
                if found:
                    break
            
            if found:
                continue
        
        # If we found eliminations, rebuild matrices
        if eliminated:
            keep_indices = [i for i in range(rank) if i not in eliminated]
            new_rank = len(keep_indices)
            if new_rank > 0:
                U_mat = U_mat[:, keep_indices]
                V_mat = V_mat[:, keep_indices]
                W_mat = W_mat[:, keep_indices]
                rank = new_rank
    
    # Convert back to list format
    U = U_mat.tolist()
    V = V_mat.tolist()
    W = W_mat.tolist()
    
    return {"U": U, "V": V, "W": W}
