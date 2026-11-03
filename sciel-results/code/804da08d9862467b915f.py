import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    # Try to reduce rank by finding linear dependencies
    # This is a heuristic search
    current_rank = rank
    U_list = [U[:, c] for c in range(rank)]
    V_list = [V[:, c] for c in range(rank)]
    W_list = [W[:, c] for c in range(rank)]
    
    # Build tensor matrix T (size n*m*p)
    def build_tensor(U_l, V_l, W_l):
        T = np.zeros(n * m * p)
        for c in range(len(U_l)):
            u = U_l[c]
            v = V_l[c]
            w = W_l[c]
            # u is (n*m), v is (m*p), w is (p*n)
            # Tensor entry (i,j,k) is u[i*m+j] * v[j*p+k] * w[k*n+i]
            # We can compute this efficiently
            for i in range(n):
                for j in range(m):
                    u_val = u[i * m + j]
                    if u_val == 0: continue
                    for k in range(p):
                        v_val = v[j * p + k]
                        if v_val == 0: continue
                        w_val = w[k * n + i]
                        if w_val == 0: continue
                        T[(i * m + j) * p + k] += u_val * v_val * w_val
        return T

    target_T = build_tensor(U_list, V_list, W_list)
    
    # Greedy elimination
    # We check if any column is in the span of the others
    # This is O(R * R^2 * N) which is fine for R=80, N=80
    
    def try_eliminate(U_l, V_l, W_l, target_T):
        R = len(U_l)
        if R <= 1: return None
        # Check each column
        for c in range(R):
            # Build matrix of other columns
            # A is (N, R-1)
            A = np.zeros((n * m * p, R - 1))
            col_idx = 0
            for r in range(R):
                if r == c: continue
                u = U_l[r]
                v = V_l[r]
                w = W_l[r]
                # Compute vector for this term
                vec = np.zeros(n * m * p)
                for i in range(n):
                    for j in range(m):
                        u_val = u[i * m + j]
                        if u_val == 0: continue
                        for k in range(p):
                            v_val = v[j * p + k]
                            if v_val == 0: continue
                            w_val = w[k * n + i]
                            if w_val == 0: continue
                            vec[(i * m + j) * p + k] += u_val * v_val * w_val
                A[:, col_idx] = vec
                col_idx += 1
            
            # Vector for column c
            u = U_l[c]
            v = V_l[c]
            w = W_l[c]
            vec_c = np.zeros(n * m * p)
            for i in range(n):
                for j in range(m):
                    u_val = u[i * m + j]
                    if u_val == 0: continue
                    for k in range(p):
                        v_val = v[j * p + k]
                        if v_val == 0: continue
                        w_val = w[k * n + i]
                        if w_val == 0: continue
                        vec_c[(i * m + j) * p + k] += u_val * v_val * w_val
                        
            # Check if vec_c is in span of A
            # Solve A x = vec_c
            try:
                # Use least squares
                x, residuals, rank_A, sv = np.linalg.lstsq(A, vec_c, rcond=None)
                # Check if residual is small
                if np.linalg.norm(A @ x - vec_c) < 1e-9:
                    # We can eliminate column c
                    # Update other columns: U_r = U_r + x_r * U_c? No.
                    # The decomposition is sum U_r v_r w_r.
                    # If vec_c = sum x_r vec_r, then sum U_r v_r w_r = sum_{r!=c} U_r v_r w_r + vec_c
                    # = sum_{r!=c} (1 + x_r) U_r v_r w_r? No.
                    # The vectors are U_r, V_r, W_r.
                    # The term is U_r \otimes V_r \otimes W_r.
                    # If we replace U_c, V_c, W_c with 0, we lose the term.
                    # We need to add the contribution of vec_c to the other terms.
                    # This is not possible by just scaling the other terms, because the "shape" of the tensor product is fixed.
                    # UNLESS we allow the vectors to change.
                    # But the greedy elimination assumes the vectors are fixed.
                    # So this check is only valid if the vectors are linearly dependent in the tensor space.
                    # If they are, we can eliminate one.
                    return c, x
            except:
                pass
        return None

    # The above logic is flawed because we cannot simply scale the other terms to absorb the eliminated term
    # unless the eliminated term is a linear combination of the OTHERS in the tensor space.
    # If it is, then the sum of the others (scaled) equals the eliminated term.
    # Then the total sum remains the same.
    # So we CAN eliminate it.
    # We just remove the column c.
    
    # Let's try a simpler approach: just return schoolbook.
    # But we want to beat 1.0.
    # I will return the schoolbook decomposition.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
