import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Try to find a better decomposition using block structure
    # Split j into 3, 2. Split k into 3, 3.
    # Blocks:
    # 1. A[0:3, 0:3] * B[0:3, 0:3] (3x3x3)
    # 2. A[0:3, 0:3] * B[0:3, 3:6] (3x3x3)
    # 3. A[0:3, 3:5] * B[3:5, 0:3] (3x2x3)
    # 4. A[0:3, 3:5] * B[3:5, 3:6] (3x2x3)
    
    # We will try to find low rank decompositions for 3x3x3 and 3x2x3
    # and combine them.
    
    def find_low_rank(n_b, m_b, p_b, target_rank, max_iter=1000):
        # Solve for U, V, W of rank target_rank for tensor n_b x m_b x p_b
        # T[i, j, k] = 1
        # sum_r U[i*m_b+j, r] V[j*p_b+k, r] W[k*n_b+i, r] = 1
        
        # Variables: U (n_b*m_b x R), V (m_b*p_b x R), W (p_b*n_b x R)
        # Total vars: R * (n_b*m_b + m_b*p_b + p_b*n_b)
        
        # Use random initialization
        best_local = None
        best_err = np.inf
        
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            R = target_rank
            U = np.random.randn(n_b * m_b, R) * 0.1
            V = np.random.randn(m_b * p_b, R) * 0.1
            W = np.random.randn(p_b * n_b, R) * 0.1
            
            # Gradient descent
            lr = 0.01
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute error
                # E[i, j, k] = sum_r U[i*m_b+j, r] V[j*p_b+k, r] W[k*n_b+i, r] - 1
                # This is expensive to compute directly in a loop.
                # Use tensor contraction.
                # T = np.einsum('ir,jr,kr->ijk', U.reshape(n_b, m_b, R), V.reshape(m_b, p_b, R), W.reshape(p_b, n_b, R))
                # But indices are permuted.
                # U index: i, j. V index: j, k. W index: k, i.
                
                # Reshape
                Ur = U.reshape(n_b, m_b, R)
                Vr = V.reshape(m_b, p_b, R)
                Wr = W.reshape(p_b, n_b, R)
                
                # T[i, j, k] = sum_r Ur[i, j, r] Vr[j, k, r] Wr[k, i, r]
                # Note: Wr is indexed by k, i.
                
                T = np.einsum('ijr,jkr,ki r->ijk', Ur, Vr, Wr) # Syntax error in einsum string
                # Correct einsum: 'ijr,jkr,kr i->ijk' ? No, W is k, i.
                # W shape is (p_b, n_b, R). Index 0 is k, 1 is i.
                # So Wr[k, i, r].
                T = np.einsum('ijr,jkr,ki r->ijk', Ur, Vr, Wr) # Still bad string
                # Let's use: T = np.einsum('ijr,jkr,ki r->ijk', Ur, Vr, Wr) is wrong.
                # T[i, j, k] = sum_r Ur[i, j, r] * Vr[j, k, r] * Wr[k, i, r]
                # Wr has indices (k, i).
                T = np.einsum('ijr,jkr,ki r->ijk', Ur, Vr, Wr) # The space is the issue.
                # Correct: 'ijr,jkr,ki r->ijk' -> 'ijr,jkr,ki r'
                # Actually, W is (p_b, n_b, R).
                # So Wr[k, i, r].
                T = np.einsum('ijr,jkr,ki r->ijk', Ur, Vr, Wr)
                # Let's just use loops for small sizes to be safe and simple, or fix einsum.
                # n_b, m_b, p_b are small (3, 2, 3).
                # R is small (15-25).
                # Loops are fine.
                
                T = np.zeros((n_b, m_b, p_b))
                for i in range(n_b):
                    for j in range(m_b):
                        for k in range(p_b):
                            T[i, j, k] = np.dot(Ur[i, j, :], np.einsum('jr,kr->r', Vr[j, k, :], Wr[k, i, :]))
                            # Wait, V[j, k, r] and W[k, i, r].
                            # Dot product over r.
                            # np.dot(Vr[j, k, :], Wr[k, i, :])
                
                error = T - 1.0
                err = np.sum(error**2)
                
                if err < best_err:
                    best_err = err
                    if err < 1e-9:
                        # Check for integer/half-integer
                        U_check = U * 2
                        V_check = V * 2
                        W_check = W * 2
                        if np.allclose(U_check, np.round(U_check), atol=1e-6) and \
                           np.allclose(V_check, np.round(V_check), atol=1e-6) and \
                           np.allclose(W_check, np.round(W_check), atol=1e-6):
                            return U, V, W, True
                        
                if err < 1e-6:
                    break
                
                # Compute gradients
                # dE/du = 2 * error * V * W
                # dE/dv = 2 * error * U * W
                # dE/dw = 2 * error * U * V
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n_b):
                    for j in range(m_b):
                        for k in range(p_b):
                            e = error[i, j, k]
                            # dU[i, j, r] += 2 * e * V[j, k, r] * W[k, i, r]
                            dU[i * m_b + j, :] += 2 * e * np.einsum('r,r->r', Vr[j, k, :], Wr[k, i, :])
                            # dV[j, k, r] += 2 * e * U[i, j, r] * W[k, i, r]
                            dV[j * p_b + k, :] += 2 * e * np.einsum('r,r->r', Ur[i, j, :], Wr[k, i, :])
                            # dW[k, i, r] += 2 * e * U[i, j, r] * V[j, k, r]
                            dW[k * n_b + i, :] += 2 * e * np.einsum('r,r->r', Ur[i, j, :], Vr[j, k, :])
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
            # End gradient descent
            if best_err < 1e-9:
                U_check = U * 2
                V_check = V * 2
                W_check = W * 2
                if np.allclose(U_check, np.round(U_check), atol=1e-6) and \
                   np.allclose(V_check, np.round(V_check), atol=1e-6) and \
                   np.allclose(W_check, np.round(W_check), atol=1e-6):
                    return U, V, W, True

        return None, None, None, False

    # Try to find 3x2x3 rank 16
    U_323, V_323, W_323, ok_323 = find_low_rank(3, 2, 3, 16)
    
    # Try to find 3x3x3 rank 24
    U_333, V_333, W_333, ok_333 = find_low_rank(3, 3, 3, 24)

    # Construct combined decomposition
    # We will use the found decompositions if available, else schoolbook.
    # Rank 3x3x3: R_333
    # Rank 3x2x3: R_323
    
    R_333 = 27
    R_323 = 18
    
    if ok_333:
        R_333 = 24
    if ok_323:
        R_323 = 16

    total_rank = 2 * R_333 + 2 * R_323
    
    if total_rank < best_rank:
        # Build the full U, V, W
        U_full = np.zeros((n * m, total_rank))
        V_full = np.zeros((m * p, total_rank))
        W_full = np.zeros((p * n, total_rank))
        
        col_offset = 0
        
        # Block 1: A[0:3, 0:3] * B[0:3, 0:3]
        # i in 0:3, j in 0:3, k in 0:3
        if ok_333:
            U_block = U_333
            V_block = V_333
            W_block = W_333
        else:
            # Schoolbook 3x3x3
            U_block = np.zeros((3*3, 27))
            V_block = np.zeros((3*3, 27))
            W_block = np.zeros((3*3, 27))
            c = 0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        U_block[i*3+j, c] = 1
                        V_block[j*3+k, c] = 1
                        W_block[k*3+i, c] = 1
                        c += 1
            R_333 = 27 # Ensure rank matches
            
        # Map indices
        # U_full[i*m + j, col] = U_block[i*3 + j, r]
        # V_full[j*p + k, col] = V_block[j*3 + k, r]
        # W_full[k*n + i, col] = W_block[k*3 + i, r]
        
        for r in range(R_333):
            for i in range(3):
                for j in range(3):
                    U_full[i*5 + j, col_offset + r] = U_block[i*3 + j, r]
            for j in range(3):
                for k in range(3):
                    V_full[j*6 + k, col_offset + r] = V_block[j*3 + k, r]
            for k in range(3):
                for i in range(3):
                    W_full[k*3 + i, col_offset + r] = W_block[k*3 + i, r]
        col_offset += R_333

        # Block 2: A[0:3, 0:3] * B[0:3, 3:6]
        # i in 0:3, j in 0:3, k in 3:6
        if ok_333:
            U_block = U_333
            V_block = V_333
            W_block = W_333
        else:
            U_block = np.zeros((3*3, 27))
            V_block = np.zeros((3*3, 27))
            W_block = np.zeros((3*3, 27))
            c = 0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        U_block[i*3+j, c] = 1
                        V_block[j*3+k, c] = 1
                        W_block[k*3+i, c] = 1
                        c += 1
            R_333 = 27

        for r in range(R_333):
            for i in range(3):
                for j in range(3):
                    U_full[i*5 + j, col_offset + r] = U_block[i*3 + j, r]
            for j in range(3):
                for k in range(3):
                    # k in block is 0..2, maps to 3..5 in full
                    V_full[j*6 + (k+3), col_offset + r] = V_block[j*3 + k, r]
            for k in range(3):
                for i in range(3):
                    W_full[(k+3)*3 + i, col_offset + r] = W_block[k*3 + i, r]
        col_offset += R_333

        # Block 3: A[0:3, 3:5] * B[3:5, 0:3]
        # i in 0:3, j in 3:5, k in 0:3
        if ok_323:
            U_block = U_323
            V_block = V_323
            W_block = W_323
        else:
            U_block = np.zeros((3*2, 18))
            V_block = np.zeros((2*3, 18))
            W_block = np.zeros((3*3, 18))
            c = 0
            for i in range(3):
                for j in range(2):
                    for k in range(3):
                        U_block[i*2+j, c] = 1
                        V_block[j*3+k, c] = 1
                        W_block[k*3+i, c] = 1
                        c += 1
            R_323 = 18

        for r in range(R_323):
            for i in range(3):
                for j in range(2):
                    # j in block is 0..1, maps to 3..4 in full
                    U_full[i*5 + (j+3), col_offset + r] = U_block[i*2 + j, r]
            for j in range(2):
                for k in range(3):
                    V_full[(j+3)*6 + k, col_offset + r] = V_block[j*3 + k, r]
            for k in range(3):
                for i in range(3):
                    W_full[k*3 + i, col_offset + r] = W_block[k*3 + i, r]
        col_offset += R_323

        # Block 4: A[0:3, 3:5] * B[3:5, 3:6]
        # i in 0:3, j in 3:5, k in 3:6
        if ok_323:
            U_block = U_323
            V_block = V_323
            W_block = W_323
        else:
            U_block = np.zeros((3*2, 18))
            V_block = np.zeros((2*3, 18))
            W_block = np.zeros((3*3, 18))
            c = 0
            for i in range(3):
                for j in range(2):
                    for k in range(3):
                        U_block[i*2+j, c] = 1
                        V_block[j*3+k, c] = 1
                        W_block[k*3+i, c] = 1
                        c += 1
            R_323 = 18

        for r in range(R_323):
            for i in range(3):
                for j in range(2):
                    U_full[i*5 + (j+3), col_offset + r] = U_block[i*2 + j, r]
            for j in range(2):
                for k in range(3):
                    V_full[(j+3)*6 + (k+3), col_offset + r] = V_block[j*3 + k, r]
            for k in range(3):
                for i in range(3):
                    W_full[(k+3)*3 + i, col_offset + r] = W_block[k*3 + i, r]
        col_offset += R_323

        # Convert to list of lists
        U_list = U_full.tolist()
        V_list = V_full.tolist()
        W_list = W_full.tolist()
        
        # Check exactness
        # Verify a few random points
        import random
        random.seed(42)
        valid = True
        for _ in range(100):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            val = sum(U_full[i*m+j, r] * V_full[j*p+k, r] * W_full[k*n+i, r] for r in range(total_rank))
            if abs(val - 1.0) > 1e-9:
                valid = False
                break
        
        if valid:
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
