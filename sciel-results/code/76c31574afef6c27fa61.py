"""Matrix multiplication tensor decomposition 4x4 x 4x9."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p  # 4 * 4 * 9 = 144
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def check_validity(U, V, W, tol=1e-8):
        """Check if the decomposition is exact."""
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0.0
                        for c in range(U.shape[1]):
                            s += W[k * n + i, c] * U[i * m + j1, c] * V[j2 * p + k, c]
                        expected = 1.0 if j1 == j2 else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True

    def solve_subtensor(n_sub, m_sub, p_sub, target_rank, seed=42):
        """
        Solve for rank-target_rank decomposition of n_sub x m_sub x p_sub multiplication.
        Returns U, V, W for the sub-tensor.
        """
        rng = np.random.RandomState(seed)
        
        # Pre-compute b vectors for the sub-tensor
        # w index corresponds to (k_sub, i_sub)
        b_vectors = []
        for k in range(p_sub):
            for i in range(n_sub):
                b = np.zeros(n_sub * m_sub * m_sub * p_sub)
                # u index is i * m_sub + j1
                # v index is j2 * p_sub + k
                # Condition: sum_c W[k,i,c] U[i,j1,c] V[j2,k,c] = 1 if j1==j2 else 0
                # This is equivalent to:
                # For each (j1, j2) pair, we have an equation.
                # Actually, the standard formulation is:
                # Tensor T[i, j, k] = delta_{j, j'} where j is from A, j' from B?
                # No, A is n x m, B is m x p.
                # A_{i, j}, B_{j, k} -> C_{i, k}
                # T_{i, j, k} = 1 if we are looking at the term?
                # The tensor is defined by T[i, j, k] = 1 if the indices match the multiplication structure.
                # Specifically, we want C_{i, k} = sum_j A_{i, j} B_{j, k}.
                # So the tensor T has T[i, j, k] = 1 if... wait.
                # The decomposition is T = sum_c U_c \otimes V_c \otimes W_c.
                # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c] ?
                # Let's stick to the seed's convention:
                # U shape (n*m, r), V shape (m*p, r), W shape (p*n, r).
                # Indexing: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c].
                # Check: sum_c W[k*n+i, c] U[i*m+j, c] V[j*p+k, c] == 1 if j==j' ?
                # In the seed's check_validity:
                # s += W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c]
                # expected = 1.0 if j1 == j2 else 0.0
                # This corresponds to (A B)_{i, k} = sum_j A_{i, j} B_{j, k}.
                # Here j1 is the column of A, j2 is the row of B.
                # So yes, the tensor is the multiplication tensor.
                
                for j1 in range(m_sub):
                    for j2 in range(m_sub):
                        if j1 == j2:
                            # The term in the sum is U[i, j1] * V[j1, k] * W[k, i]
                            # In the flattened vector for the linear system on W:
                            # We are solving for W[k, i, c].
                            # The equation is sum_c W[k, i, c] * (U[i, j1, c] * V[j2, k, c]) = delta_{j1, j2}
                            # Wait, the linear system is usually set up differently.
                            # Let's use the seed's build_A and build_b logic adapted for sub-tensor.
                            pass
                b_vectors.append(b)
        
        # Re-implement build_A and build_b for sub-tensor to be safe
        def build_A_sub(w_idx, U_sub, V_sub):
            r = U_sub.shape[1]
            # Number of equations is n_sub * m_sub * m_sub * p_sub?
            # No, the check iterates i, k, j1, j2.
            # Total equations: n_sub * p_sub * m_sub * m_sub.
            A = np.zeros((n_sub * p_sub * m_sub * m_sub, r))
            # w_idx is (k, i).
            k_w = w_idx // n_sub
            i_w = w_idx % n_sub
            
            for c in range(r):
                col_u = U_sub[:, c]
                col_v = V_sub[:, c]
                # Flatten outer product of U and V?
                # U is (n_sub * m_sub, r). V is (m_sub * p_sub, r).
                # The term in the sum is U[i_w * m_sub + j1, c] * V[j2 * p_sub + k_w, c].
                # We need to place this into the correct slot in A[:, c].
                # Slot index for (j1, j2) is (j1 * m_sub + j2) + (i_w, k_w) offset?
                # Let's index the equations by (i, k, j1, j2).
                # Total size: n_sub * p_sub * m_sub * m_sub.
                # Index: ((i * p_sub + k) * m_sub + j1) * m_sub + j2
                
                for j1 in range(m_sub):
                    u_val = col_u[i_w * m_sub + j1]
                    for j2 in range(m_sub):
                        v_val = col_v[j2 * p_sub + k_w]
                        eq_idx = ((i_w * p_sub + k_w) * m_sub + j1) * m_sub + j2
                        A[eq_idx, c] = u_val * v_val
            return A

        def build_b_sub(w_idx):
            k_w = w_idx // n_sub
            i_w = w_idx % n_sub
            b = np.zeros(n_sub * p_sub * m_sub * m_sub)
            for j1 in range(m_sub):
                for j2 in range(m_sub):
                    if j1 == j2:
                        eq_idx = ((i_w * p_sub + k_w) * m_sub + j1) * m_sub + j2
                        b[eq_idx] = 1.0
            return b

        b_vectors = [build_b_sub(w) for w in range(p_sub * n_sub)]

        for restart in range(50):
            if time.time() > deadline - 5:
                return None
            
            U_sub = rng.randn(n_sub * m_sub, target_rank)
            V_sub = rng.randn(m_sub * p_sub, target_rank)
            
            W_sub = np.zeros((p_sub * n_sub, target_rank))
            valid = True
            
            for w in range(p_sub * n_sub):
                A = build_A_sub(w, U_sub, V_sub)
                b = b_vectors[w]
                
                try:
                    W_sub[w, :], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                if len(residuals) > 0 and residuals[0] > 1e-6:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Verify sub-tensor
            for i in range(n_sub):
                for k in range(p_sub):
                    for j1 in range(m_sub):
                        for j2 in range(m_sub):
                            s = 0.0
                            for c in range(target_rank):
                                s += W_sub[k * n_sub + i, c] * U_sub[i * m_sub + j1, c] * V_sub[j2 * p_sub + k, c]
                            expected = 1.0 if j1 == j2 else 0.0
                            if abs(s - expected) > 1e-8:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if not valid:
                continue
                
            return U_sub, V_sub, W_sub

    # First, get the schoolbook solution as a fallback
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb

    # Try to find rank-12 decomposition for 4x4x3 sub-tensor
    # There are 3 such blocks in the 4x4x9 problem
    # B is 4x9, so columns 0-2, 3-5, 6-8
    # C is 4x9, so columns 0-2, 3-5, 6-8
    # A is 4x4
    
    sub_U, sub_V, sub_W = None, None, None
    for seed in range(100):
        if time.time() > deadline - 10:
            break
        res = solve_subtensor(4, 4, 3, 12, seed=seed)
        if res is not None:
            sub_U, sub_V, sub_W = res
            break

    if sub_U is not None:
        # Construct full decomposition
        # Rank will be 3 * 12 = 36
        full_rank = 36
        U_full = np.zeros((n * m, full_rank))
        V_full = np.zeros((m * p, full_rank))
        W_full = np.zeros((p * n, full_rank))
        
        # sub_U is 16x12, sub_V is 12x12, sub_W is 12x12
        # Block 0: cols 0-2 of B, cols 0-2 of C
        # U is independent of k, so it's the same for all blocks?
        # Wait, U depends on i, j. It does NOT depend on k.
        # So U_full will have the same U_sub in all 3 blocks?
        # No, the decomposition is a sum of rank-1 tensors.
        # T = sum_{c=1}^{36} U_c \otimes V_c \otimes W_c.
        # The tensor is block diagonal in the k-index (columns of B/C).
        # So we can just repeat the sub-decomposition 3 times, shifted in the V and W indices.
        
        for block in range(3):
            offset = block * 12
            # U: indices i, j. Same for all blocks.
            U_full[:, offset:offset+12] = sub_U
            
            # V: indices j, k.
            # k ranges from block*3 to block*3+2.
            # sub_V is indexed by j, k_sub.
            # V_full index is j*p + k.
            for j in range(4):
                for k_sub in range(3):
                    k_full = block * 3 + k_sub
                    V_full[j * p + k_full, offset:offset+12] = sub_V[j * 3 + k_sub, :]
            
            # W: indices k, i.
            # W_full index is k*n + i.
            for k_sub in range(3):
                k_full = block * 3 + k_sub
                for i in range(4):
                    W_full[k_full * n + i, offset:offset+12] = sub_W[k_sub * 4 + i, :]
        
        # Verify full decomposition
        if check_validity(U_full, V_full, W_full):
            best_U, best_V, best_W, best_rank = U_full, V_full, W_full, full_rank

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
