"""Rewritten program for matrix multiplication 2x4 * 4x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = 140  # 10s safety margin

    # Schoolbook decomposition (rank 40) as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    def verify_decomp(U, V, W, tol=1e-8):
        rank = U.shape[1]
        T = np.einsum('ijc,jkc,kic->ijk', U.reshape(n, m, rank), V.reshape(m, p, rank), W.reshape(p, n, rank))
        return np.allclose(T, np.ones((n, m, p)), atol=tol)

    def polish_rational(U, V, W):
        """Round to half-integers or integers if still valid."""
        for scale in [2, 1]:
            U_new = np.round(U * scale) / scale
            V_new = np.round(V * scale) / scale
            W_new = np.round(W * scale) / scale
            if verify_decomp(U_new, V_new, W_new, tol=1e-10):
                return U_new, V_new, W_new, True
        return U, V, W, False

    # Try structured Strassen-based decomposition for rank 20
    # A (2x4) = [A1 A2], B (4x5) = [B1; B2]
    # C = A1@B1 + A2@B2
    # Each 2x2 @ 2x5 can be done with rank 7 (Strassen-like)
    # Total rank: 14 for the two products, but we can share some terms
    # Actually, let's try a direct rank-20 construction

    # For 2x4 @ 4x5, use the following structure:
    # Split middle dimension 4 into 2+2
    # A = [A11 A12; A21 A22] no, A is 2x4 so A = [A1 A2] with A1,A2 2x2
    # B = [B1; B2] with B1,B2 2x5
    # C = A1@B1 + A2@B2
    
    # Use Strassen for each 2x2 @ 2x2 subproduct, extend to 2x5
    # 2x2 @ 2x5: treat as two 2x2 @ 2x2 plus 2x2 @ 2x1
    # Or use a rank-7 scheme for 2x2 @ 2x2 and handle the 5th column separately
    
    # Let's try a systematic approach: rank r from 20 to 39
    # Use gradient-based optimization with structured init
    
    def try_rank(target_rank, max_time_per_attempt=2.0):
        """Try to find decomposition of given rank using iterative refinement."""
        attempt_start = time.time()
        rng = np.random.RandomState(target_rank * 1000 + 42)
        
        # Multiple initialization strategies
        inits = []
        
        # Random init
        U0 = rng.randn(n * m, target_rank) * 0.3
        V0 = rng.randn(m * p, target_rank) * 0.3
        W0 = rng.randn(p * n, target_rank) * 0.3
        inits.append((U0, V0, W0))
        
        # Structured init based on block structure
        if target_rank >= 14:
            U_s = np.zeros((n * m, target_rank))
            V_s = np.zeros((m * p, target_rank))
            W_s = np.zeros((p * n, target_rank))
            # Strassen-like for 2x2 blocks
            # For each of the 2 row-blocks of A and 2 col-blocks of B
            # Use 7 Strassen terms per 2x2@2x2, scaled for 2x5
            stride = 7
            for rb in range(2):  # row block of A (0 or 1)
                for cb in range(2):  # col block of B (0 or 1)
                    base = rb * stride + cb * stride
                    for s in range(stride):
                        c = base + s
                        if c >= target_rank:
                            break
                        # Strassen terms for 2x2 @ 2x2
                        # M1 = (A21-A11)(B12-B22)
                        # M2 = (A11+A21)B22
                        # M3 = A11(B11-B22)
                        # M4 = (A22-A11)(B11+B22)
                        # M5 = (A11+A22)B11
                        # M6 = (A21-A22)(B22+B11)
                        # M7 = (A12-A22)(B21+B12)
                        # For 2x2 @ 2x5, extend by repeating for columns
                        U_s[rb * 2 + 0, c] = [1, -1, 1, -1, 1, 0, -1][s]
                        U_s[rb * 2 + 1, c] = [0, 1, 0, 1, 1, 1, 1][s]
                        V_s[cb * 2 + 0, c] = [0, 1, 1, 0, 1, 1, 1][s]
                        V_s[cb * 2 + 1, c] = [1, 0, -1, 1, 0, 0, 1][s]
                        # W for output: C[i,k] = sum over terms
                        # This is getting complex, just use random for W
                        W_s[c, :] = rng.randn(p * n) * 0.1
            inits.append((U_s, V_s, W_s))
        
        best_loss = np.inf
        best_UVW = None
        
        for U_init, V_init, W_init in inits:
            if time.time() - attempt_start > max_time_per_attempt:
                break
                
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()
            
            # Alternating least squares with momentum
            for it in range(200):
                if time.time() - attempt_start > max_time_per_attempt:
                    break
                    
                # Update U: for each (i,j), solve M_U @ u = ones(p)
                for i in range(n):
                    for j in range(m):
                        M = np.einsum('kc,kic->ck', V_curr[j*p:(j+1)*p, :], W_curr[:, i])
                        if M.shape[0] < target_rank:
                            continue
                        try:
                            U_curr[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        M = np.einsum('ic,kic->ci', U_curr[:, j], W_curr[k*n:(k+1)*n, :])
                        if M.shape[0] < target_rank:
                            continue
                        try:
                            V_curr[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        M = np.einsum('jc,jkc->cj', U_curr[i*m:(i+1)*m, :], V_curr[:, k])
                        if M.shape[0] < target_rank:
                            continue
                        try:
                            W_curr[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check
                if verify_decomp(U_curr, V_curr, W_curr, tol=1e-7):
                    U_p, V_p, W_p, ok = polish_rational(U_curr, V_curr, W_curr)
                    if ok:
                        return U_p, V_p, W_p, True
                    return U_curr, V_curr, W_curr, True
                
                # Compute loss for tracking
                T = np.einsum('ijc,jkc,kic->ijk', U_curr.reshape(n,m,target_rank), 
                             V_curr.reshape(m,p,target_rank), W_curr.reshape(p,n,target_rank))
                loss = np.sum((T - 1.0)**2)
                if loss < best_loss:
                    best_loss = loss
                    best_UVW = (U_curr.copy(), V_curr.copy(), W_curr.copy())
        
        # Check best found
        if best_UVW is not None:
            U_b, V_b, W_b = best_UVW
            if verify_decomp(U_b, V_b, W_b, tol=1e-6):
                U_p, V_p, W_p, ok = polish_rational(U_b, V_b, W_b)
                if ok:
                    return U_p, V_p, W_p, True
                return U_b, V_b, W_b, True
        
        return None, None, None, False

    # Try ranks from 20 upward
    for target_rank in range(20, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple seeds
        for seed_offset in range(5):
            if time.time() > deadline:
                break
            
            # Modify the seed by adjusting RNG
            np.random.seed(target_rank * 100 + seed_offset)
            
            U_r, V_r, W_r, valid = try_rank(target_rank, max_time_per_attempt=3.0)
            if valid:
                best_rank = target_rank
                best_U = U_r.copy()
                best_V = V_r.copy()
                best_W = W_r.copy()
                break  # Found this rank, move to next lower rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
