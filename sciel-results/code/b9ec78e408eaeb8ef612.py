import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    n, m, p = 3, 3, 4
    
    # Build the target tensor
    T_expected = np.zeros((9, 12, 12), dtype=int)
    for i in range(3):
        for j in range(3):
            for k in range(4):
                T_expected[i*3+j, j*4+k, k*3+i] = 1
                
    def verify(U, V, W):
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
        return np.array_equal(T_recon, T_expected)

    # Start with the known rank-34 construction
    rank = 34
    U = np.zeros((9, rank), dtype=int)
    V = np.zeros((12, rank), dtype=int)
    W = np.zeros((12, rank), dtype=int)
    
    col = 0
    # Strassen Left (A11 * B1L)
    U[0, col] = 1; U[4, col] = 1; V[0, col] = 1; V[5, col] = 1; W[0, col] = 1; W[4, col] = 1; col += 1
    U[3, col] = 1; U[4, col] = 1; V[0, col] = 1; W[1, col] = 1; W[4, col] = -1; col += 1
    U[0, col] = 1; V[1, col] = 1; V[5, col] = -1; W[3, col] = 1; W[4, col] = 1; col += 1
    U[4, col] = 1; V[4, col] = 1; V[0, col] = -1; W[0, col] = 1; W[1, col] = 1; col += 1
    U[0, col] = 1; U[1, col] = 1; V[5, col] = 1; W[0, col] = -1; W[3, col] = 1; col += 1
    U[3, col] = 1; U[0, col] = -1; V[0, col] = 1; V[1, col] = 1; W[4, col] = 1; col += 1
    U[1, col] = 1; U[4, col] = -1; V[4, col] = 1; V[5, col] = 1; W[0, col] = 1; col += 1
    U[2, col] = 1; V[8, col] = 1; W[0, col] = 1; col += 1
    U[2, col] = 1; V[9, col] = 1; W[3, col] = 1; col += 1
    U[5, col] = 1; V[8, col] = 1; W[1, col] = 1; col += 1
    U[5, col] = 1; V[9, col] = 1; W[4, col] = 1; col += 1
    U[6, col] = 1; V[0, col] = 1; W[2, col] = 1; col += 1
    U[6, col] = 1; V[1, col] = 1; W[5, col] = 1; col += 1
    U[7, col] = 1; V[4, col] = 1; W[2, col] = 1; col += 1
    U[7, col] = 1; V[5, col] = 1; W[5, col] = 1; col += 1
    U[8, col] = 1; V[8, col] = 1; W[2, col] = 1; col += 1
    U[8, col] = 1; V[9, col] = 1; W[5, col] = 1; col += 1
    # Strassen Right (A11 * B1R)
    U[0, col] = 1; U[4, col] = 1; V[2, col] = 1; V[7, col] = 1; W[6, col] = 1; W[10, col] = 1; col += 1
    U[3, col] = 1; U[4, col] = 1; V[2, col] = 1; W[7, col] = 1; W[10, col] = -1; col += 1
    U[0, col] = 1; V[3, col] = 1; V[7, col] = -1; W[9, col] = 1; W[10, col] = 1; col += 1
    U[4, col] = 1; V[6, col] = 1; V[2, col] = -1; W[6, col] = 1; W[7, col] = 1; col += 1
    U[0, col] = 1; U[1, col] = 1; V[7, col] = 1; W[6, col] = -1; W[9, col] = 1; col += 1
    U[3, col] = 1; U[0, col] = -1; V[2, col] = 1; V[3, col] = 1; W[10, col] = 1; col += 1
    U[1, col] = 1; U[4, col] = -1; V[6, col] = 1; V[7, col] = 1; W[6, col] = 1; col += 1
    U[2, col] = 1; V[10, col] = 1; W[6, col] = 1; col += 1
    U[2, col] = 1; V[11, col] = 1; W[9, col] = 1; col += 1
    U[5, col] = 1; V[10, col] = 1; W[7, col] = 1; col += 1
    U[5, col] = 1; V[11, col] = 1; W[10, col] = 1; col += 1
    U[6, col] = 1; V[2, col] = 1; W[8, col] = 1; col += 1
    U[6, col] = 1; V[3, col] = 1; W[11, col] = 1; col += 1
    U[7, col] = 1; V[6, col] = 1; W[8, col] = 1; col += 1
    U[7, col] = 1; V[7, col] = 1; W[11, col] = 1; col += 1
    U[8, col] = 1; V[10, col] = 1; W[8, col] = 1; col += 1
    U[8, col] = 1; V[11, col] = 1; W[11, col] = 1; col += 1

    if not verify(U, V, W):
        # Fallback to trivial rank-36
        rank = 36
        U = np.zeros((9, rank), dtype=int)
        V = np.zeros((12, rank), dtype=int)
        W = np.zeros((12, rank), dtype=int)
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    U[i*3+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 34
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Try to find rank-33 decomposition
    # Strategy: Check for linear dependencies among columns that allow merging
    # Two columns r1, r2 can be merged if U[:, r1] = alpha U[:, r2], V[:, r1] = alpha V[:, r2], W[:, r1] = alpha W[:, r2]
    # Or more generally, if the contribution of r1 can be absorbed into r2 by scaling.
    
    # A more powerful check: 
    # Remove column r, and see if the remaining tensor T' can be corrected by modifying one existing column c.
    # T_expected = T' + U_c V_c W_c^T (outer product)
    # We want T_expected = T' + (U_c + dU) (V_c + dV) (W_c + dW)^T
    # This is complex.
    
    # Simpler: Try all pairs (r1, r2) and see if they are proportional.
    for r1 in range(34):
        for r2 in range(r1 + 1, 34):
            if time.time() > deadline - 10:
                break
            # Check if U[:, r1] is proportional to U[:, r2]
            # And V, W similarly
            # Use cross product or simple ratio check
            # If U[:, r1] = alpha U[:, r2]
            # Find alpha from non-zero elements
            idx = np.where(U[:, r2] != 0)[0]
            if len(idx) == 0:
                continue
            alpha = U[idx[0], r1] / U[idx[0], r2]
            if np.allclose(U[:, r1], alpha * U[:, r2]) and \
               np.allclose(V[:, r1], alpha * V[:, r2]) and \
               np.allclose(W[:, r1], alpha * W[:, r2]):
                # We can merge them!
                # Actually, if they are proportional, we can just sum them?
                # No, if T = sum U_r V_r W_r^T, and U_1 = a U_2, etc.
                # Then U_1 V_1 W_1^T = a U_2 (a V_2) (a W_2)^T = a^3 U_2 V_2 W_2^T.
                # This doesn't reduce rank unless a=0 or something.
                # Wait, if U_1 = U_2, V_1 = V_2, W_1 = W_2, then we have 2 identical terms.
                # We can combine them into 1 term with coefficient 2.
                # But the rank is the number of terms.
                # If we have identical terms, we can merge them into one term with scaled values?
                # No, the decomposition is a sum of rank-1 tensors.
                # If two rank-1 tensors are identical, their sum is 2 * that tensor.
                # 2 * (u v w^T) = (2u) v w^T = u (2v) w^T = u v (2w)^T.
                # So yes, we can merge them into one term.
                # Create new U, V, W without r1, and add r1 to r2?
                # No, just remove r1 and scale r2?
                # Actually, if U1=U2, V1=V2, W1=W2, then T = ... + U1 V1 W1^T + U2 V2 W2^T + ...
                # = ... + 2 U1 V1 W1^T + ...
                # We can replace the two terms with one term: U' = 2 U1, V' = V1, W' = W1.
                # Or U' = U1, V' = 2 V1, W' = W1.
                # So rank reduces by 1.
                
                # Let's verify this logic.
                # If we merge, we remove r1 and update r2.
                U_new = U.copy()
                V_new = V.copy()
                W_new = W.copy()
                
                # Remove r1
                mask = [i for i in range(34) if i != r1]
                U_new = U_new[:, mask]
                V_new = V_new[:, mask]
                W_new = W_new[:, mask]
                
                # Find index of r2 in new array
                r2_new = mask.index(r2)
                
                # Scale r2_new by 2 (since we merged two identical terms)
                # Wait, the values are integers. If we scale by 2, they remain integers.
                U_new[:, r2_new] = 2 * U_new[:, r2_new]
                # V_new, W_new unchanged?
                # T_term = U V W^T.
                # 2 * (U V W^T) = (2U) V W^T.
                # So yes.
                
                if verify(U_new, V_new, W_new):
                    best_rank = 33
                    best_U = U_new
                    best_V = V_new
                    best_W = W_new
                    break
        if best_rank == 33:
            break

    # If no direct proportional merge found, try a local search with random perturbations
    # to see if we can find a rank-33 solution that is not a simple merge of the initial 34.
    # This is computationally expensive, so we limit it.
    if best_rank > 33 and time.time() < deadline - 20:
        # Try to find if any column is redundant in a different way
        # e.g. U_r = 0? No, all columns are non-zero.
        
        # Try to eliminate one column and solve for the others?
        # This is hard.
        
        # Instead, we will just return the best found.
        pass

    if best_rank < 34:
        U, V, W = best_U, best_V, best_W
        rank = best_rank
    else:
        rank = 34

    if not verify(U, V, W):
        # Fallback
        rank = 36
        U = np.zeros((9, rank), dtype=int)
        V = np.zeros((12, rank), dtype=int)
        W = np.zeros((12, rank), dtype=int)
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    U[i*3+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
