"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    # Build target tensor for verification
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W):
        T_rec = np.zeros((n * m, m * p, p * n))
        for col in range(U.shape[1]):
            T_rec += np.einsum('i,j,k->ijk', U[:, col], V[:, col], W[:, col])
        return np.linalg.norm(T_rec - T)

    # Fallback: Rank-42 Strassen decomposition (6 blocks of 2x2x2)
    def build_strassen_rank42():
        rank = 42
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        # Standard Strassen rank-7 terms for 2x2x2
        # Indices in U, V, W are (i, j) for U, (j, k) for V, (k, i) for W
        # Here we use local indices 0..3 for U, 0..3 for V, 0..3 for W
        # U[i*2+j], V[j*2+k], W[k*2+i]
        strassen_products = [
            # 1: (A00+A11)(B00+B11)
            {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
            # 2: (A10+A11)B00
            {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1)]},
            # 3: A00(B10-B11)
            {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1)]},
            # 4: A11(B01-B00)
            {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
            # 5: (A00+A01)B11
            {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
            # 6: (A10-A00)(B00+B01)
            {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
            # 7: (A01-A11)(B10+B11)
            {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, -1), (3, 1)]},
        ]

        # Map 2x2x2 blocks to 2x2x12
        # p=12, so 6 blocks of size 2.
        # Block t corresponds to k in [2t, 2t+1]
        for t in range(6):
            for s in range(7):
                col = 7 * t + s
                for idx, coeff in strassen_products[s]['U']:
                    U[idx, col] = coeff
                for idx, coeff in strassen_products[s]['V']:
                    j = idx // 2
                    k_local = idx % 2
                    k = 2 * t + k_local
                    V[j * p + k, col] = coeff
                for idx, coeff in strassen_products[s]['W']:
                    k_local = idx // 2
                    i = idx % 2
                    k = 2 * t + k_local
                    W[k * n + i, col] = coeff

        return U, V, W

    # Primary Strategy: Block-Circulant Decomposition (Rank 28)
    # Partition p=12 into 4 groups of size 3.
    # Each group is treated as a multiplication in R = Q[x]/(x^3-1).
    # R is isomorphic to Q * Q(omega) where omega = exp(2*pi*i/3).
    # Multiplication by a fixed element in R can be represented by a matrix.
    # We want to compute C = A * B where A is 2x2, B is 2x12.
    # Let B be viewed as 4 blocks of 2x3.
    # Actually, the tensor structure T[a,b,c] depends on i,j,k.
    # a = i*2+j, b = j*12+k, c = k*2+i.
    
    # The block-circulant approach exploits the symmetry in k.
    # If we assume a circulant structure in the k-dimension (size 3 per block),
    # we can diagonalize using DFT.
    # However, the problem is not fully circulant in the standard sense because 
    # the output index c depends on k and i in a specific way.
    
    # Let's try a direct analytic construction for Rank 28.
    # We partition the 12 k-values into 4 sets of 3: K_t = {3t, 3t+1, 3t+2}.
    # For each t, we solve for a rank-7 decomposition of the sub-tensor 
    # restricted to k in K_t.
    # The sub-tensor for block t has non-zeros at:
    # T[i*2+j, j*12+k, k*2+i] = 1 for k in {3t, 3t+1, 3t+2}.
    
    # Let's check if the sub-tensor for a single block of 3 k-values 
    # (i.e., 2x2x3 multiplication) has rank 7.
    # Actually, rank(2x2x3) is known to be 5? Or 6?
    # Literature: rank(2,2,3) = 5.
    # If rank(2,2,3) = 5, then 4 blocks * 5 = 20.
    # If rank(2,2,3) = 6, then 4 blocks * 6 = 24.
    # If we can find a rank-5 decomposition for 2x2x3, total rank is 20.
    
    # Let's attempt to find a rank-5 decomposition for 2x2x3.
    # Tensor T_s for 2x2x3:
    # a in 0..3, b in 0..5, c in 0..5.
    # T_s[i*2+j, j*3+k, k*2+i] = 1.
    
    def build_2x2x3_rank5():
        """
        Returns U, V, W for 2x2x3 multiplication with rank 5.
        U: 4x5, V: 6x5, W: 6x5.
        """
        # We use a known rank-5 decomposition for 2x2x3.
        # One such decomposition uses the following structure:
        # It is related to the fact that 2x2x3 is a sub-tensor of 2x2x4 (rank 8) 
        # or can be derived from Strassen-like tricks.
        
        # Let's try to construct it analytically or use a known one.
        # A common rank-5 decomposition for 2x2x3:
        # The tensor has 12 non-zero entries.
        # T[0,0,0]=1, T[0,1,2]=1, T[0,2,4]=1
        # T[1,3,0]=1, T[1,4,2]=1, T[1,5,4]=1
        # T[2,0,1]=1, T[2,1,3]=1, T[2,2,5]=1
        # T[3,3,1]=1, T[3,4,3]=1, T[3,5,5]=1
        
        # Let's try to fit 5 rank-1 terms.
        # This is a system of equations.
        # Since we need a robust solution, let's try to solve it numerically 
        # with a fixed seed and verify. If it fails, we fallback to rank 6 or 7.
        
        n_s, m_s, p_s = 2, 2, 3
        T_s = np.zeros((n_s * m_s, m_s * p_s, p_s * n_s))
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    a = i * m_s + j
                    b = j * p_s + k
                    c = k * n_s + i
                    T_s[a, b, c] = 1.0

        rank_sub = 5
        U_s = np.zeros((4, rank_sub))
        V_s = np.zeros((6, rank_sub))
        W_s = np.zeros((6, rank_sub))
        
        # We will try to find a solution using least squares with random restarts.
        # To ensure determinism and speed, we use a small number of restarts.
        
        np.random.seed(42)
        best_err = np.inf
        best_params = None
        
        for attempt in range(100):
            # Initialize with sparse random vectors
            U0 = np.zeros((4, rank_sub))
            V0 = np.zeros((6, rank_sub))
            W0 = np.zeros((6, rank_sub))
            
            for r in range(rank_sub):
                # Random support
                u_nz = np.random.choice(4, size=np.random.randint(2, 4), replace=False)
                v_nz = np.random.choice(6, size=np.random.randint(2, 4), replace=False)
                w_nz = np.random.choice(6, size=np.random.randint(2, 4), replace=False)
                
                U0[u_nz, r] = np.random.choice([-1, 1], size=len(u_nz))
                V0[v_nz, r] = np.random.choice([-1, 1], size=len(v_nz))
                W0[w_nz, r] = np.random.choice([-1, 1], size=len(w_nz))
            
            # Flatten
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def objective(x):
                U_curr = x[:4*rank_sub].reshape(4, rank_sub)
                V_curr = x[4*rank_sub:10*rank_sub].reshape(6, rank_sub)
                W_curr = x[10*rank_sub:].reshape(6, rank_sub)
                T_rec = np.zeros_like(T_s)
                for r in range(rank_sub):
                    T_rec += np.einsum('i,j,k->ijk', U_curr[:, r], V_curr[:, r], W_curr[:, r])
                return (T_rec - T_s).flatten()
            
            try:
                import scipy.optimize
                result = scipy.optimize.least_squares(objective, x0, method='trf', max_nfev=2000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                err = np.linalg.norm(result.fun)
                if err < best_err:
                    best_err = err
                    best_params = result.x.copy()
                    if err < 1e-10:
                        break
            except Exception:
                pass
        
        if best_params is not None and best_err < 1e-8:
            U_s = best_params[:4*rank_sub].reshape(4, rank_sub)
            V_s = best_params[4*rank_sub:10*rank_sub].reshape(6, rank_sub)
            W_s = best_params[10*rank_sub:].reshape(6, rank_sub)
            
            # Snap to half-integers if close
            U_snap = np.round(U_s * 2) / 2
            V_snap = np.round(V_s * 2) / 2
            W_snap = np.round(W_s * 2) / 2
            
            # Verify snapped version
            T_rec = np.zeros_like(T_s)
            for r in range(rank_sub):
                T_rec += np.einsum('i,j,k->ijk', U_snap[:, r], V_snap[:, r], W_snap[:, r])
            if np.linalg.norm(T_rec - T_s) < 1e-9:
                return U_snap, V_snap, W_snap
            
            # If snap failed, use original if error is very small
            if best_err < 1e-6:
                return U_s, V_s, W_s
        
        return None

    # Try to build the rank-20 solution (4 blocks of rank-5)
    best_U, best_V, best_W = None, None, None
    
    if time.time() < deadline - 30:
        res_2x2x3 = build_2x2x3_rank5()
        if res_2x2x3 is not None:
            U_s, V_s, W_s = res_2x2x3
            rank_total = 4 * 5
            U_total = np.zeros((n * m, rank_total))
            V_total = np.zeros((m * p, rank_total))
            W_total = np.zeros((p * n, rank_total))
            
            for g in range(4):
                for r in range(5):
                    col = 5 * g + r
                    U_total[:, col] = U_s[:, r]
                    for b_idx in range(6):
                        j = b_idx // 3
                        k_local = b_idx % 3
                        k = 3 * g + k_local
                        V_total[j * p + k, col] = V_s[b_idx, r]
                    for c_idx in range(6):
                        k_local = c_idx // 2
                        i = c_idx % 2
                        k = 3 * g + k_local
                        W_total[k * n + i, col] = W_s[c_idx, r]
            
            err = verify(U_total, V_total, W_total)
            if err < 1e-8:
                best_U, best_V, best_W = U_total, V_total, W_total

    # Fallback to Rank 42 if Rank 20 failed or time is up
    if best_U is None:
        U42, V42, W42 = build_strassen_rank42()
        err42 = verify(U42, V42, W42)
        if err42 < 1e-8:
            best_U, best_V, best_W = U42, V42, W42
        else:
            # Ultimate fallback: Schoolbook Rank 48
            U_fb = np.zeros((n * m, 48))
            V_fb = np.zeros((m * p, 48))
            W_fb = np.zeros((p * n, 48))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        U_fb[a, col] = 1
                        V_fb[b, col] = 1
                        W_fb[c, col] = 1
                        col += 1
            best_U, best_V, best_W = U_fb, V_fb, W_fb

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
