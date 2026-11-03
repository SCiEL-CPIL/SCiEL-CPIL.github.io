"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Build the rank-36 decomposition first
    rank_init = 36
    U = [[0] * rank_init for _ in range(n * m)]
    V = [[0] * rank_init for _ in range(m * p)]
    W = [[0] * rank_init for _ in range(p * n)]
    
    col = 0

    def add_rank1_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank_init:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1

    def add_strassen_block(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        def map_a(local):
            return (a_base_i + local // 2) * m + (a_base_j + local % 2)
        def map_b(local):
            return (b_base_i + local // 2) * p + (b_base_j + local % 2)
        def map_c(local):
            i = c_base_i + local // 2
            k = c_base_j + local % 2
            return k * n + i

        u = {map_a(0): 1, map_a(3): 1}
        v = {map_b(0): 1, map_b(3): 1}
        w = {map_c(0): 1, map_c(3): 1}
        add_rank1_term(u, v, w)

        u = {map_a(2): 1, map_a(3): 1}
        v = {map_b(0): 1}
        w = {map_c(2): 1, map_c(3): -1}
        add_rank1_term(u, v, w)

        u = {map_a(0): 1}
        v = {map_b(1): 1, map_b(3): -1}
        w = {map_c(1): 1, map_c(3): 1}
        add_rank1_term(u, v, w)

        u = {map_a(3): 1}
        v = {map_b(2): 1, map_b(0): -1}
        w = {map_c(0): 1, map_c(2): 1}
        add_rank1_term(u, v, w)

        u = {map_a(0): 1, map_a(1): 1}
        v = {map_b(3): 1}
        w = {map_c(0): -1, map_c(1): 1}
        add_rank1_term(u, v, w)

        u = {map_a(2): 1, map_a(0): -1}
        v = {map_b(0): 1, map_b(1): 1}
        w = {map_c(3): 1}
        add_rank1_term(u, v, w)

        u = {map_a(1): 1, map_a(3): -1}
        v = {map_b(2): 1, map_b(3): 1}
        w = {map_c(0): 1}
        add_rank1_term(u, v, w)

    def add_schoolbook_2x2_1(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        def map_a(local):
            return (a_base_i + local // 2) * m + (a_base_j + local % 2)
        def map_b(local):
            return (b_base_i + local) * p + b_base_j
        def map_c(local):
            i = c_base_i + local
            k = c_base_j
            return k * n + i

        for l in range(4):
            a_local = l
            b_local = l % 2
            c_local = l // 2
            u = {map_a(a_local): 1}
            v = {map_b(b_local): 1}
            w = {map_c(c_local): 1}
            add_rank1_term(u, v, w)

    add_strassen_block(0, 0, 0, 0, 0, 0)
    add_strassen_block(0, 0, 0, 2, 0, 2)
    add_schoolbook_2x2_1(0, 0, 0, 4, 0, 4)
    add_strassen_block(0, 2, 2, 0, 0, 0)
    add_strassen_block(0, 2, 2, 2, 0, 2)
    add_schoolbook_2x2_1(0, 2, 2, 4, 0, 4)
    
    # Convert to numpy arrays
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check if we can reduce rank by finding linear dependencies
    # The tensor is T = sum_k u_k ⊗ v_k ⊗ w_k
    # We can check if any of the 36 rank-1 vectors are in the span of the others
    # by computing the rank of the matrix formed by flattening each term
    
    dim_u = n * m  # 8
    dim_v = m * p  # 20
    dim_w = p * n  # 10
    
    # Form matrix where each column is a flattened rank-1 term
    # Shape: (dim_u * dim_v * dim_w, rank_init)
    terms = np.zeros((dim_u * dim_v * dim_w, rank_init))
    for k in range(rank_init):
        terms[:, k] = np.outer(U_np[:, k], np.outer(V_np[:, k], W_np[:, k])).flatten()
    
    # Compute rank of the term matrix
    try:
        rank_of_terms = np.linalg.matrix_rank(terms, tol=1e-10)
    except np.linalg.LinAlgError:
        rank_of_terms = rank_init
    
    if rank_of_terms < rank_init:
        # There are linear dependencies. Try to find a minimal subset.
        # Use greedy elimination: remove terms that are in the span of the rest
        keep = list(range(rank_init))
        for _ in range(rank_init - rank_of_terms):
            # Try removing each term and check if the remaining span still covers the tensor
            removed = False
            for i in range(len(keep)):
                candidate = keep[:i] + keep[i+1:]
                if len(candidate) < 1:
                    continue
                # Check if the tensor is still in the span of the remaining terms
                # The tensor is the sum of all terms, so we need:
                # sum_{k in keep} terms[:,k] = sum_{k in candidate} terms[:,k] + terms[:, removed_k]
                # This is equivalent to: terms[:, removed_k] is in the span of {terms[:,k] : k in candidate}
                col_to_check = terms[:, keep[i]]
                remaining_cols = terms[:, candidate]
                # Check if col_to_check is in the column space of remaining_cols
                try:
                    # Least squares: find x such that remaining_cols @ x ≈ col_to_check
                    x, residuals, rank_rem, sv = np.linalg.lstsq(remaining_cols, col_to_check, rcond=1e-10)
                    if residuals.size > 0 and residuals[0] < 1e-8 * (np.linalg.norm(col_to_check) + 1e-10):
                        # This term is in the span of the others, remove it
                        keep = candidate
                        removed = True
                        break
                except np.linalg.LinAlgError:
                    continue
            if not removed:
                break
        
        if len(keep) < rank_init:
            U_out = U_np[:, keep]
            V_out = V_np[:, keep]
            W_out = W_np[:, keep]
            rank_final = len(keep)
        else:
            U_out = U_np
            V_out = V_np
            W_out = W_np
            rank_final = rank_init
    else:
        U_out = U_np
        V_out = V_np
        W_out = W_np
        rank_final = rank_init
    
    # Verify the decomposition is correct
    # Check that sum_k U_out[:,k] * V_out[:,k] * W_out[:,k] matches the multiplication tensor
    # For a few random test cases
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        # Compute C from the decomposition
        # C[i,k] = sum_j A[i,j] B[j,k]
        # In the decomposition: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # But we need to be careful: U has shape (n*m, rank), so U[i*m+j, r]
        # V has shape (m*p, rank), so V[j*p+k, r]
        # W has shape (p*n, rank), so W[k*n+i, r]
        
        C_computed = np.zeros((n, p))
        for r in range(rank_final):
            u_vec = U_out[:, r]
            v_vec = V_out[:, r]
            w_vec = W_out[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_computed[i, k] += u_vec[i * m + j] * v_vec[j * p + k] * w_vec[k * n + i]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback to the original 36-rank decomposition
            U_out = U_np
            V_out = V_np
            W_out = W_np
            rank_final = rank_init
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
