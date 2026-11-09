"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-97 tensor decomposition for 4x4 * 4x7 matrix multiplication.

    Strategy:
      - Left part: 4x4 * 4x4 (B columns 0..3) using recursive Strassen, rank 49.
      - Right part: 4x4 * 4x3 (B columns 4..6) using schoolbook, rank 48.
      - Total rank = 97.

    Index conventions:
      - U shape (n*m, rank) = (16, rank); U[i*m+j, r]
      - V shape (m*p, rank) = (28, rank); V[j*p+k, r]
      - W shape (p*n, rank) = (28, rank); W[k*n+i, r]
      - Tensor: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    deadline = start_time + 110  # safety margin under 150s wall clock

    # ------------------------------------------------------------------
    # Strassen rank-7 decomposition of the 2x2x2 matmul tensor.
    # ------------------------------------------------------------------
    s2x2 = [
        ((1,0,0,1), (1,0,0,1), (1,0,0,-1)),  # p1
        ((0,0,1,1), (1,0,0,0), (0,0,1,-1)),  # p2
        ((1,0,0,0), (0,1,0,-1), (0,1,0,1)),  # p3
        ((0,0,0,1), (0,0,1,-1), (-1,0,1,0)), # p4
        ((1,1,0,0), (0,0,0,1), (0,-1,0,1)),  # p5
        ((-1,0,1,0), (1,1,0,0), (1,0,0,0)),  # p6
        ((0,1,0,-1), (0,0,1,1), (-1,0,0,0)), # p7
    ]

    # ------------------------------------------------------------------
    # Recursive Strassen for 4x4 * 4x4 -> 4x4.
    # ------------------------------------------------------------------
    # Block-level Strassen definitions
    block_defs = [
        # (X_blocks, Y_blocks)
        # M1 = (A00+A11)*(B00+B11)
        ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
        # M2 = (A10+A11)*B00
        ([((1,0),1), ((1,1),1)], [((0,0),1)]),
        # M3 = A00*(B01-B11)
        ([((0,0),1)], [((0,1),1), ((1,1),-1)]),
        # M4 = A11*(B10-B00)
        ([((1,1),1)], [((1,0),1), ((0,0),-1)]),
        # M5 = (A00+A01)*B11
        ([((0,0),1), ((0,1),1)], [((1,1),1)]),
        # M6 = (A10-A00)*(B00+B01)
        ([((1,0),1), ((0,0),-1)], [((0,0),1), ((0,1),1)]),
        # M7 = (A01-A11)*(B10+B11)
        ([((0,1),1), ((1,1),-1)], [((1,0),1), ((1,1),1)]),
    ]

    # Output reconstruction: for each output block (ob), sum over M_k with signs.
    # C00 = M1 - M4 + M6 - M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M3 + M2 - M5
    out_recon = {
        (0,0): [(0,1), (3,-1), (5,1), (6,-1)],
        (1,0): [(1,1), (3,1)],
        (0,1): [(2,1), (4,1)],
        (1,1): [(0,1), (2,-1), (1,1), (4,-1)],
    }

    # Generate all 49 element-level rank-1 terms for 4x4x4.
    terms_444 = []
    for k, (x_blocks, y_blocks) in enumerate(block_defs):
        # Build X and Y local-to-global mappings
        X = {}  # (lr,lc) -> { (ar,ac): coeff }
        for (br, bc), coeff in x_blocks:
            for lr in range(2):
                for lc in range(2):
                    ar, ac = br*2+lr, bc*2+lc
                    if (lr,lc) not in X:
                        X[(lr,lc)] = {}
                    X[(lr,lc)][(ar,ac)] = X[(lr,lc)].get((ar,ac), 0) + coeff

        Y = {}  # (lr,lc) -> { (br_,bc_): coeff }
        for (br, bc), coeff in y_blocks:
            for lr in range(2):
                for lc in range(2):
                    br_, bc_ = br*2+lr, bc*2+lc
                    if (lr,lc) not in Y:
                        Y[(lr,lc)] = {}
                    Y[(lr,lc)][(br_,bc_)] = Y[(lr,lc)].get((br_,bc_), 0) + coeff

        # Apply the 7 element-level Strassen terms to X and Y
        for t in range(7):
            xvec, yvec, zvec = s2x2[t]
            a_vec = {}
            for lr in range(2):
                for lc in range(2):
                    xc = xvec[lr*2+lc]
                    if xc == 0:
                        continue
                    if (lr,lc) not in X:
                        continue
                    for (ar,ac), ac2 in X[(lr,lc)].items():
                        val = xc * ac2
                        if val != 0:
                            a_vec[(ar,ac)] = a_vec.get((ar,ac), 0) + val

            b_vec = {}
            for lr in range(2):
                for lc in range(2):
                    yc = yvec[lr*2+lc]
                    if yc == 0:
                        continue
                    if (lr,lc) not in Y:
                        continue
                    for (br_,bc_), bc2 in Y[(lr,lc)].items():
                        val = yc * bc2
                        if val != 0:
                            b_vec[(br_,bc_)] = b_vec.get((br_,bc_), 0) + val

            c_vec = {}
            for (ob_r, ob_c), sign_list in out_recon.items():
                for (mk, sign) in sign_list:
                    if mk != k:
                        continue
                    for lr in range(2):
                        for lc in range(2):
                            zc = zvec[lr*2+lc]
                            if zc == 0:
                                continue
                            cr, cc = ob_r*2+lr, ob_c*2+lc
                            val = sign * zc
                            if val != 0:
                                c_vec[(cr,cc)] = c_vec.get((cr,cc), 0) + val

            if a_vec and b_vec and c_vec:
                terms_444.append((a_vec, b_vec, c_vec))

    # ------------------------------------------------------------------
    # Schoolbook terms for 4x4 * 4x3 (B columns 4..6).
    # ------------------------------------------------------------------
    terms_443 = []
    for i in range(n):
        for j in range(m):
            for k in range(4, p):  # columns 4,5,6
                a_vec = {((i, j)): 1}
                b_vec = {((j, k)): 1}
                c_vec = {((i, k)): 1}
                terms_443.append((a_vec, b_vec, c_vec))

    all_terms = terms_444 + terms_443
    rank = len(all_terms)

    # ------------------------------------------------------------------
    # Build U, V, W matrices.
    # ------------------------------------------------------------------
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    for r, (a_vec, b_vec, c_vec) in enumerate(all_terms):
        for (ar, ac), val in a_vec.items():
            U[ar * m + ac][r] = val
        for (br, bc), val in b_vec.items():
            V[br * p + bc][r] = val
        for (cr, cc), val in c_vec.items():
            W[cc * n + cr][r] = val

    # ------------------------------------------------------------------
    # Verification and Fallback
    # ------------------------------------------------------------------
    try:
        if time.time() < deadline:
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            
            # Verify tensor identity: T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
            # T should equal the standard matmul tensor, which is 1 if i,j,k form a valid path?
            # No, the tensor of matrix multiplication is defined such that
            # C[i,k] = sum_j A[i,j] B[j,k].
            # The rank-1 decomposition represents the bilinear map.
            # We can verify by checking if the decomposition computes A@B correctly.
            
            np.random.seed(42)
            # Use integer values to avoid floating point precision issues in check
            A_test = np.random.randint(-10, 10, size=(n, m))
            B_test = np.random.randint(-10, 10, size=(m, p))
            C_expected = A_test @ B_test
            
            C_comp = np.zeros((n, p), dtype=float)
            
            # Efficient vectorized check
            # For each rank term r, we have vectors u_r, v_r, w_r.
            # C += sum_r (u_r reshaped to n,m) * (v_r reshaped to m,p) * (w_r reshaped to p,n) ... wait.
            # The formula is C[i,k] += sum_j U[i,j,r] V[j,k,r] W[k,i,r]
            
            # We can reshape Ua to (n, m, rank)
            # Va to (m, p, rank)
            # Wa to (p, n, rank)
            
            U_reshaped = Ua.reshape(n, m, rank)
            V_reshaped = Va.reshape(m, p, rank)
            W_reshaped = Wa.reshape(p, n, rank)
            
            # C[i,k] = sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # This is a tensor contraction.
            # Let's just do it in a loop over r for clarity and safety, rank is small (97)
            for r in range(rank):
                u_r = U_reshaped[:, :, r] # (n, m)
                v_r = V_reshaped[:, :, r] # (m, p)
                w_r = W_reshaped[:, :, r] # (p, n)
                
                # C[i,k] += sum_j u_r[i,j] * v_r[j,k] * w_r[k,i]
                # We can compute this as:
                # Temp[i,k] = sum_j u_r[i,j] v_r[j,k] = (u_r @ v_r)[i,k]
                # Then C[i,k] += Temp[i,k] * w_r[k,i]
                
                temp = u_r @ v_r # (n, p)
                C_comp += temp * w_r.T # w_r is (p,n), so w_r.T is (n,p). w_r[k,i] is w_r.T[i,k]. Correct.
                
            if not np.allclose(C_comp, C_expected, atol=1e-9):
                # Fallback to schoolbook rank 112
                rank_fb = n * m * p
                U_fb = np.zeros((n * m, rank_fb))
                V_fb = np.zeros((m * p, rank_fb))
                W_fb = np.zeros((p * n, rank_fb))
                
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_fb[i * m + j, col] = 1.0
                            V_fb[j * p + k, col] = 1.0
                            W_fb[k * n + i, col] = 1.0
                            col += 1
                            
                return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
                
    except Exception:
        # If verification fails for any reason (including timeout or import error),
        # return the schoolbook decomposition to guarantee a valid score (112) 
        # rather than an invalid result (score 0).
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
                    
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
