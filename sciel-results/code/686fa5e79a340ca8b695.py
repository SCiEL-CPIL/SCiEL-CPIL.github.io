import time
import numpy as np
from itertools import combinations

DEADLINE = 145.0

def construct():
    start_time = time.time()
    n, m, p = 2, 5, 6
    total_entries = n * m * p  # 60

    def check_time():
        return time.time() - start_time < DEADLINE

    def verify_solution(U, V, W, rank):
        """Verify that the decomposition correctly computes matrix multiplication."""
        A_test = np.random.randint(-10, 10, (n, m))
        B_test = np.random.randint(-10, 10, (m, p))
        C_test = A_test @ B_test
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_calc[i, k] = val
        return np.allclose(C_test, C_calc, atol=1e-8)

    def try_reduce_via_nullspace(U, V, W, rank):
        """Try to reduce rank by finding linear dependencies among the rank-1 terms."""
        if rank < 2:
            return None
        
        # Build the tensor matrix: each column is a rank-1 tensor
        tensor_mat = np.zeros((total_entries, rank))
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor_mat[i*m*p + j*p + k, r] = u[i*m+j] * v[j*p+k] * w[k*n+i]
        
        # Find null space of tensor_mat
        try:
            U_sv, s_vals, Vt = np.linalg.svd(tensor_mat, full_matrices=False)
            tol = 1e-10 * (s_vals[0] if len(s_vals) > 0 else 1)
            null_dim = rank - len(s_vals[s_vals >= tol])
            if null_dim <= 0:
                return None
            
            # Get null space basis
            null_basis = Vt[len(s_vals[s_vals >= tol]):].T
            
            # Try eliminating one term at a time
            best_result = None
            for elim in range(rank):
                # Find a null vector that has significant component at position elim
                best_vec = None
                best_val = 0
                for vec in null_basis:
                    if abs(vec[elim]) > best_val:
                        best_val = abs(vec[elim])
                        best_vec = vec
                
                if best_vec is None or best_val < 1e-10:
                    continue
                
                # Use this null vector to eliminate term 'elim'
                # new_coeffs[t] = old_coeff[t] - (coeff[elim] * vec[t] / vec[elim]) for t != elim
                # We need to work with the actual coefficient structure
                # The tensor_mat columns are the rank-1 tensors, so we can directly
                # form a new set of rank-1 tensors by linear combination
                
                new_U = np.zeros((n*m, rank-1))
                new_V = np.zeros((m*p, rank-1))
                new_W = np.zeros((p*n, rank-1))
                
                col = 0
                for t in range(rank):
                    if t == elim:
                        continue
                    # Scale to make the eliminated term zero
                    # We want: sum_r c_r * T_r = sum_t d_t * T'_t where T'_t are new rank-1 tensors
                    # This is more complex - let's just check if removing 'elim' works
                    pass
                
                # Simpler approach: check if the rank-1 tensor at position 'elim' is in the span
                # of the others by solving a linear system
                others = [t for t in range(rank) if t != elim]
                if len(others) >= 1:
                    # Check if tensor_mat[:, elim] is in span of tensor_mat[:, others]
                    A_sub = tensor_mat[:, others]
                    b = tensor_mat[:, elim]
                    try:
                        coeffs, residuals, rank_sub, _ = np.linalg.lstsq(A_sub, b, rcond=1e-10)
                        if len(residuals) > 0 and residuals[0] < 1e-8:
                            # We can eliminate term 'elim'
                            new_U = np.column_stack([U[:, t] for t in others])
                            new_V = np.column_stack([V[:, t] for t in others])
                            new_W = np.column_stack([W[:, t] for t in others])
                            
                            # Verify
                            if verify_solution(new_U, new_V, new_W, rank-1):
                                if best_result is None or (rank-1) < best_result[0]:
                                    best_result = (rank-1, new_U, new_V, new_W)
                    except:
                        pass
            
            return best_result
        except:
            return None

    # Build candidate pool of rank-1 tensors
    candidates = []

    def add_candidate(u_sparse, v_sparse, w_sparse):
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        for (i, j), val in u_sparse.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_sparse.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_sparse.items():
            w_vec[k * n + i] = val
        contrib = np.zeros(total_entries)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contrib[i * m * p + j * p + k] = u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
        candidates.append((u_vec, v_vec, w_vec, contrib))

    # Type 1: Standard unit tensors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_candidate({(i,j): 1}, {(j,k): 1}, {(k,i): 1})

    # Type 2: Strassen 2x2x2 blocks
    def strassen_terms(io, jo, ko):
        terms = []
        terms.append(({(io,jo): 1, (io+1,jo+1): 1}, 
                      {(jo,ko): 1, (jo+1,ko+1): 1}, 
                      {(ko,io): 1, (ko+1,io+1): 1}))
        terms.append(({(io+1,jo): 1, (io+1,jo+1): 1}, 
                      {(jo,ko): 1}, 
                      {(ko,io+1): 1, (ko+1,io+1): -1}))
        terms.append(({(io,jo): 1}, 
                      {(jo,ko+1): 1, (jo+1,ko+1): -1}, 
                      {(ko+1,io): 1, (ko+1,io+1): 1}))
        terms.append(({(io+1,jo+1): 1}, 
                      {(jo+1,ko): 1, (jo,ko): -1}, 
                      {(ko,io): 1, (ko,io+1): 1}))
        terms.append(({(io,jo): 1, (io,jo+1): 1}, 
                      {(jo+1,ko+1): 1}, 
                      {(ko,io): -1, (ko+1,io): 1}))
        terms.append(({(io+1,jo): 1, (io,jo): -1}, 
                      {(jo,ko): 1, (jo,ko+1): 1}, 
                      {(ko+1,io+1): 1}))
        terms.append(({(io,jo+1): 1, (io+1,jo+1): -1}, 
                      {(jo+1,ko): 1, (jo+1,ko+1): 1}, 
                      {(ko,io): 1}))
        return terms

    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 1):
                for u_c, v_c, w_c in strassen_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)

    # Type 3: 2x3x2 blocks
    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 1):
                for u_c, v_c, w_c in strassen_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)
                for u_c, v_c, w_c in strassen_terms(io, jo+1, ko):
                    add_candidate(u_c, v_c, w_c)
                add_candidate({(io,jo): 1}, {(jo,ko): 1}, {(ko,io): 1})
                add_candidate({(io+1,jo): 1}, {(jo,ko): 1}, {(ko,io+1): 1})
                add_candidate({(io,jo+2): 1}, {(jo+2,ko): 1}, {(ko,io): 1})
                add_candidate({(io+1,jo+2): 1}, {(jo+2,ko): 1}, {(ko,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+2): 1}, {(jo,ko+1): 1, (jo+2,ko): 1}, {(ko,io): 1, (ko+1,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+2): 1}, {(jo,ko+1): 1, (jo+2,ko): -1}, {(ko+1,io): 1, (ko,io+1): 1})

    # Type 4: 2x2x3 blocks
    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 2):
                for u_c, v_c, w_c in strassen_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)
                for u_c, v_c, w_c in strassen_terms(io, jo, ko+1):
                    add_candidate(u_c, v_c, w_c)
                add_candidate({(io,jo): 1}, {(jo,ko): 1}, {(ko,io): 1})
                add_candidate({(io+1,jo): 1}, {(jo,ko): 1}, {(ko,io+1): 1})
                add_candidate({(io,jo): 1}, {(jo,ko+2): 1}, {(ko+2,io): 1})
                add_candidate({(io+1,jo): 1}, {(jo,ko+2): 1}, {(ko+2,io+1): 1})
                add_candidate({(io,jo+1): 1}, {(jo+1,ko+2): 1, (jo,ko+2): 1}, {(ko+2,io): 1, (ko,io): 1})
                add_candidate({(io+1,jo+1): 1}, {(jo+1,ko+2): 1, (jo,ko+2): 1}, {(ko+2,io+1): 1, (ko,io+1): 1})

    # Type 5: 2x3x3 blocks
    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 2):
                for j_sub in range(2):
                    for k_sub in range(2):
                        for u_c, v_c, w_c in strassen_terms(io, jo+j_sub, ko+k_sub):
                            add_candidate(u_c, v_c, w_c)
                for j_off in range(3):
                    for k_off in range(3):
                        add_candidate({(io,jo+j_off): 1}, {(jo+j_off,ko+k_off): 1}, {(ko+k_off,io): 1})
                        add_candidate({(io+1,jo+j_off): 1}, {(jo+j_off,ko+k_off): 1}, {(ko+k_off,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+2): 1}, {(jo,ko+2): 1, (jo+2,ko): 1}, {(ko,io): 1, (ko+2,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+2): 1}, {(jo,ko+2): 1, (jo+2,ko): -1}, {(ko+2,io): 1, (ko,io+1): 1})
                add_candidate({(io,jo+2): 1, (io+1,jo): 1}, {(jo+2,ko): 1, (jo,ko+2): 1}, {(ko,io): 1, (ko+2,io+1): 1})
                add_candidate({(io,jo+2): 1, (io+1,jo): 1}, {(jo+2,ko): 1, (jo,ko+2): -1}, {(ko+2,io): 1, (ko,io+1): 1})

    # Type 6: 2x4x2 blocks
    for io in range(n - 1):
        for jo in range(m - 3):
            for ko in range(p - 1):
                for j_sub in range(3):
                    for u_c, v_c, w_c in strassen_terms(io, jo+j_sub, ko):
                        add_candidate(u_c, v_c, w_c)
                for j_off in range(4):
                    add_candidate({(io,jo+j_off): 1}, {(jo+j_off,ko): 1}, {(ko,io): 1})
                    add_candidate({(io+1,jo+j_off): 1}, {(jo+j_off,ko): 1}, {(ko,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+3): 1}, {(jo,ko+1): 1, (jo+3,ko): 1}, {(ko,io): 1, (ko+1,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+3): 1}, {(jo,ko+1): 1, (jo+3,ko): -1}, {(ko+1,io): 1, (ko,io+1): 1})

    # Type 7: 2x2x4 blocks
    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 3):
                for k_sub in range(3):
                    for u_c, v_c, w_c in strassen_terms(io, jo, ko+k_sub):
                        add_candidate(u_c, v_c, w_c)
                for k_off in range(4):
                    add_candidate({(io,jo): 1}, {(jo,ko+k_off): 1}, {(ko+k_off,io): 1})
                    add_candidate({(io+1,jo): 1}, {(jo,ko+k_off): 1}, {(ko+k_off,io+1): 1})
                add_candidate({(io,jo+1): 1, (io+1,jo+1): 1}, {(jo+1,ko): 1, (jo+1,ko+3): 1}, {(ko,io): 1, (ko+3,io+1): 1})
                add_candidate({(io,jo+1): 1, (io+1,jo+1): 1}, {(jo+1,ko): 1, (jo+1,ko+3): -1}, {(ko+3,io): 1, (ko,io+1): 1})

    # Type 8: 2x4x3 blocks
    for io in range(n - 1):
        for jo in range(m - 3):
            for ko in range(p - 2):
                for j_sub in range(3):
                    for k_sub in range(2):
                        for u_c, v_c, w_c in strassen_terms(io, jo+j_sub, ko+k_sub):
                            add_candidate(u_c, v_c, w_c)
                for j_off in range(4):
                    for k_off in range(3):
                        add_candidate({(io,jo+j_off): 1}, {(jo+j_off,ko+k_off): 1}, {(ko+k_off,io): 1})
                        add_candidate({(io+1,jo+j_off): 1}, {(jo+j_off,ko+k_off): 1}, {(ko+k_off,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+3): 1}, {(jo,ko+2): 1, (jo+3,ko): 1}, {(ko,io): 1, (ko+2,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+3): 1}, {(jo,ko+2): 1, (jo+3,ko): -1}, {(ko+2,io): 1, (ko,io+1): 1})

    # Type 9: 2x3x4 blocks
    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 3):
                for j_sub in range(2):
                    for k_sub in range(3):
                        for u_c, v_c, w_c in strassen_terms(io, jo+j_sub, ko+k_sub):
                            add_candidate(u_c, v_c, w_c)
                for j_off in range(3):
                    for k_off in range(4):
                        add_candidate({(io,jo+j_off): 1}, {(jo+j_off,ko+k_off): 1}, {(ko+k_off,io): 1})
                        add_candidate({(io+1,jo+j_off): 1}, {(jo+j_off,ko+k_off): 1}, {(ko+k_off,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+2): 1}, {(jo,ko+3): 1, (jo+2,ko): 1}, {(ko,io): 1, (ko+3,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+2): 1}, {(jo,ko+3): 1, (jo+2,ko): -1}, {(ko+3,io): 1, (ko,io+1): 1})

    # Type 10: 2x4x4 blocks
    for io in range(n - 1):
        for jo in range(m - 3):
            for ko in range(p - 3):
                for j_sub in range(3):
                    for k_sub in range(3):
                        for u_c, v_c, w_c in strassen_terms(io, jo+j_sub, ko+k_sub):
                            add_candidate(u_c, v_c, w_c)
                for j_off in range(4):
                    for k_off in range(4):
                        add_candidate({(io,jo+j_off): 1}, {(jo+j_off,ko+k_off): 1}, {(ko+k_off,io): 1})
                        add_candidate({(io+1,jo+j_off): 1}, {(jo+j_off,ko+k_off): 1}, {(ko+k_off,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+3): 1}, {(jo,ko+3): 1, (jo+3,ko): 1}, {(ko,io): 1, (ko+3,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+3): 1}, {(jo,ko+3): 1, (jo+3,ko): -1}, {(ko+3,io): 1, (ko,io+1): 1})
                add_candidate({(io,jo+1): 1, (io+1,jo+2): 1}, {(jo+1,ko+3): 1, (jo+2,ko): 1}, {(ko,io): 1, (ko+3,io+1): 1})
                add_candidate({(io,jo+1): 1, (io+1,jo+2): 1}, {(jo+1,ko+3): 1, (jo+2,ko): -1}, {(ko+3,io): 1, (ko,io+1): 1})

    num_candidates = len(candidates)
    
    # Precompute contribution matrix
    contrib_matrix = np.zeros((num_candidates, total_entries))
    for idx in range(num_candidates):
        contrib_matrix[idx] = candidates[idx][3]
    
    norms = np.sum(contrib_matrix ** 2, axis=1)
    
    T_target = np.ones(total_entries)
    residual = T_target.copy()
    
    selected_indices = []
    selected_coeffs = []
    
    # Greedy selection with multiple restarts
    best_solution = None
    best_rank = float('inf')
    
    for restart in range(3):
        if not check_time():
            break
        
        residual = T_target.copy()
        selected_indices = []
        selected_coeffs = []
        
        max_iter = 150
        for _ in range(max_iter):
            if not check_time():
                break
            
            dots = contrib_matrix @ residual
            valid = norms > 1e-12
            safe_norms = np.where(valid, norms, 1)
            reductions = np.where(valid, dots**2 / safe_norms, 0)
            
            # Add some randomness to avoid getting stuck
            if restart > 0 and np.random.random() < 0.1:
                top_k = min(10, num_candidates)
                top_indices = np.argsort(reductions)[-top_k:]
                best_idx = top_indices[np.random.randint(len(top_indices))]
            else:
                best_idx = int(np.argmax(reductions))
            
            best_reduction = reductions[best_idx]
            
            if best_reduction < 1e-9:
                break
            
            best_coeff = dots[best_idx] / norms[best_idx]
            selected_indices.append(best_idx)
            selected_coeffs.append(best_coeff)
            residual -= best_coeff * contrib_matrix[best_idx]
            
            if np.linalg.norm(residual) < 1e-9:
                break
        
        # Check if we got an exact solution
        if np.linalg.norm(residual) < 1e-9:
            rank_found = len(selected_indices)
            
            if rank_found < best_rank:
                best_rank = rank_found
                U = np.zeros((n * m, rank_found))
                V = np.zeros((m * p, rank_found))
                W = np.zeros((p * n, rank_found))
                
                for r, idx in enumerate(selected_indices):
                    coeff = selected_coeffs[r]
                    u_vec, v_vec, w_vec, _ = candidates[idx]
                    U[:, r] = u_vec * coeff
                    V[:, r] = v_vec
                    W[:, r] = w_vec
                
                # Try rank reduction
                reduced = try_reduce_via_nullspace(U, V, W, rank_found)
                if reduced is not None:
                    new_rank, U, V, W = reduced
                    rank_found = new_rank
                    best_rank = rank_found
                
                # Verify
                if verify_solution(U, V, W, rank_found):
                    best_solution = (rank_found, U, V, W)
    
    if best_solution is not None:
        rank_found, U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Build the known 54-rank solution, then try to reduce it
    n, m, p = 2, 5, 6
    U_list = []
    V_list = []
    W_list = []
    
    def add_strassen_block(i_off, j_off, k_off):
        U_vec = np.zeros(n*m)
        V_vec = np.zeros(m*p)
        W_vec = np.zeros(p*n)
        U_vec[i_off*m + j_off] = 1
        U_vec[(i_off+1)*m + (j_off+1)] = 1
        V_vec[j_off*p + k_off] = 1
        V_vec[(j_off+1)*p + (k_off+1)] = 1
        W_vec[k_off*n + i_off] = 1
        W_vec[(k_off+1)*n + (i_off+1)] = 1
        U_list.append(U_vec)
        V_list.append(V_vec)
        W_list.append(W_vec)
        
        U_vec = np.zeros(n*m)
        V_vec = np.zeros(m*p)
        W_vec = np.zeros(p*n)
        U_vec[(i_off+1)*m + j_off] = 1
        U_vec[(i_off+1)*m + (j_off+1)] = 1
        V_vec[j_off*p + k_off] = 1
        W_vec[k_off*n + (i_off+1)] = 1
        W_vec[(k_off+1)*n + (i_off+1)] = -1
        U_list.append(U_vec)
        V_list.append(V_vec)
        W_list.append(W_vec)
        
        U_vec = np.zeros(n*m)
        V_vec = np.zeros(m*p)
        W_vec = np.zeros(p*n)
        U_vec[i_off*m + j_off] = 1
        V_vec[j_off*p + (k_off+1)] = 1
        V_vec[(j_off+1)*p + (k_off+1)] = -1
        W_vec[(k_off+1)*n + i_off] = 1
        W_vec[(k_off+1)*n + (i_off+1)] = 1
        U_list.append(U_vec)
        V_list.append(V_vec)
        W_list.append(W_vec)
        
        U_vec = np.zeros(n*m)
        V_vec = np.zeros(m*p)
        W_vec = np.zeros(p*n)
        U_vec[(i_off+1)*m + (j_off+1)] = 1
        V_vec[(j_off+1)*p + k_off] = 1
        V_vec[j_off*p + k_off] = -1
        W_vec[k_off*n + i_off] = 1
        W_vec[k_off*n + (i_off+1)] = 1
        U_list.append(U_vec)
        V_list.append(V_vec)
        W_list.append(W_vec)
        
        U_vec = np.zeros(n*m)
        V_vec = np.zeros(m*p)
        W_vec = np.zeros(p*n)
        U_vec[i_off*m + j_off] = 1
        U_vec[i_off*m + (j_off+1)] = 1
        V_vec[(j_off+1)*p + (k_off+1)] = 1
        W_vec[k_off*n + i_off] = -1
        W_vec[(k_off+1)*n + i_off] = 1
        U_list.append(U_vec)
        V_list.append(V_vec)
        W_list.append(W_vec)
        
        U_vec = np.zeros(n*m)
        V_vec = np.zeros(m*p)
        W_vec = np.zeros(p*n)
        U_vec[(i_off+1)*m + j_off] = 1
        U_vec[i_off*m + j_off] = -1
        V_vec[j_off*p + k_off] = 1
        V_vec[j_off*p + (k_off+1)] = 1
        W_vec[(k_off+1)*n + (i_off+1)] = 1
        U_list.append(U_vec)
        V_list.append(V_vec)
        W_list.append(W_vec)
        
        U_vec = np.zeros(n*m)
        V_vec = np.zeros(m*p)
        W_vec = np.zeros(p*n)
        U_vec[i_off*m + (j_off+1)] = 1
        U_vec[(i_off+1)*m + (j_off+1)] = -1
        V_vec[(j_off+1)*p + k_off] = 1
        V_vec[(j_off+1)*p + (k_off+1)] = 1
        W_vec[k_off*n + i_off] = 1
        U_list.append(U_vec)
        V_list.append(V_vec)
        W_list.append(W_vec)

    def add_standard_block(i_off, j_off, k_off):
        U_vec = np.zeros(n*m)
        V_vec = np.zeros(m*p)
        W_vec = np.zeros(p*n)
        U_vec[i_off*m + j_off] = 1
        V_vec[j_off*p + k_off] = 1
        W_vec[k_off*n + i_off] = 1
        U_list.append(U_vec)
        V_list.append(V_vec)
        W_list.append(W_vec)
        
        U_vec = np.zeros(n*m)
        V_vec = np.zeros(m*p)
        W_vec = np.zeros(p*n)
        U_vec[(i_off+1)*m + j_off] = 1
        V_vec[j_off*p + k_off] = 1
        W_vec[k_off*n + (i_off+1)] = 1
        U_list.append(U_vec)
        V_list.append(V_vec)
        W_list.append(W_vec)

    # Part 1: A[:, 0:2] * B[0:2, :]
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # Part 2a: A[:, 2:3] * B[2:3, :]
    for k in range(6):
        add_standard_block(0, 2, k)
        
    # Part 2b: A[:, 3:5] * B[3:5, :]
    add_strassen_block(0, 3, 0)
    add_strassen_block(0, 3, 2)
    add_strassen_block(0, 3, 4)
    
    U_new = np.column_stack(U_list)
    V_new = np.column_stack(V_list)
    W_new = np.column_stack(W_list)
    rank_new = U_new.shape[1]
    
    # Try to reduce the rank
    reduced = try_reduce_via_nullspace(U_new, V_new, W_new, rank_new)
    if reduced is not None:
        rank_new, U_new, V_new, W_new = reduced
    
    # Final verification
    if verify_solution(U_new, V_new, W_new, rank_new):
        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    # If verification fails, return the original
    U_orig = np.column_stack(U_list)
    V_orig = np.column_stack(V_list)
    W_orig = np.column_stack(W_list)
    return {"U": U_orig.tolist(), "V": V_orig.tolist(), "W": W_orig.tolist()}
