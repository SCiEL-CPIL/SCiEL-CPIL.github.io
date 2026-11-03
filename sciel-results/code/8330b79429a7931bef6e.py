import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = 3, 4, 6

    def verify(U, V, W):
        T_target = np.ones((n, m, p))
        T_computed = np.einsum('it,jt,kt->ijk', U, V, W)
        return np.allclose(T_computed, T_target, atol=1e-9)

    def build_rank66():
        """Build the rank-66 decomposition from the current best."""
        rank = 66
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)

            U[A00, col] += 1; U[A11, col] += 1
            V[B00, col] += 1; V[B11, col] += 1
            W[C00, col] += 1; W[C11, col] += 1
            col += 1
            U[A10, col] += 1; U[A11, col] += 1
            V[B00, col] += 1
            W[C10, col] += 1; W[C11, col] -= 1
            col += 1
            U[A00, col] += 1
            V[B01, col] += 1; V[B11, col] -= 1
            W[C01, col] += 1; W[C11, col] += 1
            col += 1
            U[A11, col] += 1
            V[B10, col] += 1; V[B00, col] -= 1
            W[C00, col] += 1; W[C10, col] += 1
            col += 1
            U[A00, col] += 1; U[A01, col] += 1
            V[B11, col] += 1
            W[C00, col] -= 1; W[C01, col] += 1
            col += 1
            U[A10, col] += 1; U[A00, col] -= 1
            V[B00, col] += 1; V[B01, col] += 1
            W[C11, col] += 1
            col += 1
            U[A01, col] += 1; U[A11, col] -= 1
            V[B10, col] += 1; V[B11, col] += 1
            W[C00, col] += 1
            col += 1

        def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r

            U[A0, col] += 1; V[B00, col] += 1; W[C0, col] += 1
            col += 1
            U[A1, col] += 1; V[B10, col] += 1; W[C0, col] += 1
            col += 1
            U[A0, col] += 1; V[B01, col] += 1; W[C1, col] += 1
            col += 1
            U[A1, col] += 1; V[B11, col] += 1; W[C1, col] += 1
            col += 1

        add_1x2_2x2(0, 0, 0, 0, 0, 0)
        add_1x2_2x2(0, 2, 2, 0, 0, 0)
        add_1x2_2x2(0, 0, 0, 2, 0, 2)
        add_1x2_2x2(0, 2, 2, 2, 0, 2)
        add_1x2_2x2(0, 0, 0, 4, 0, 4)
        add_1x2_2x2(0, 2, 2, 4, 0, 4)
        add_strassen(1, 0, 0, 0, 1, 0)
        add_strassen(1, 2, 2, 0, 1, 0)
        add_strassen(1, 0, 0, 2, 1, 2)
        add_strassen(1, 2, 2, 2, 1, 2)
        add_strassen(1, 0, 0, 4, 1, 4)
        add_strassen(1, 2, 2, 4, 1, 4)

        return U, V, W

    def try_eliminate(U, V, W, col_idx):
        """
        Check if term col_idx is in the linear span of other terms in tensor space.
        """
        r = U.shape[1]
        if r <= 1:
            return None
        
        mask = np.ones(r, dtype=bool)
        mask[col_idx] = False
        other_cols = np.where(mask)[0]
        
        target = np.einsum('i,j,k->ijk', U[:, col_idx], V[:, col_idx], W[:, col_idx]).reshape(-1)
        
        M = np.zeros((n * m * p, len(other_cols)))
        for idx, t in enumerate(other_cols):
            M[:, idx] = np.einsum('i,j,k->ijk', U[:, t], V[:, t], W[:, t]).reshape(-1)
        
        try:
            alpha, res, rank_M, _ = np.linalg.lstsq(M, target, rcond=1e-10)
        except:
            return None
        
        residual = np.linalg.norm(M @ alpha - target)
        if residual < 1e-7:
            new_U = np.column_stack([U[:, col] for col in other_cols])
            new_V = np.column_stack([V[:, col] for col in other_cols])
            new_W = np.column_stack([W[:, col] for col in other_cols])
            return new_U, new_V, new_W
        
        return None

    def local_search(U, V, W, max_time):
        """Iteratively eliminate redundant columns."""
        U, V, W = U.copy(), V.copy(), W.copy()
        
        while time.time() < max_time:
            r = U.shape[1]
            if r <= 1:
                break
            
            eliminated = False
            for col_idx in range(r):
                if time.time() > max_time:
                    break
                result = try_eliminate(U, V, W, col_idx)
                if result is not None:
                    U, V, W = result
                    eliminated = True
                    break
            
            if not eliminated:
                break
        
        return U, V, W

    # Build initial decomposition
    U, V, W = build_rank66()
    assert verify(U, V, W), "Initial decomposition failed"
    
    # Perform elimination
    U, V, W = local_search(U, V, W, deadline - 5)
    
    final_rank = U.shape[1]
    
    # Verify
    if not verify(U, V, W):
        U, V, W = build_rank66()
        final_rank = U.shape[1]
    
    # Convert to half-integer lists
    U_list = [[int(round(U[i, j] * 2)) / 2 for j in range(final_rank)] for i in range(n * m)]
    V_list = [[int(round(V[i, j] * 2)) / 2 for j in range(final_rank)] for i in range(m * p)]
    W_list = [[int(round(W[i, j] * 2)) / 2 for j in range(final_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
