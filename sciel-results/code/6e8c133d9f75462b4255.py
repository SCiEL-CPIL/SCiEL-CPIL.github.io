import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin before 150s kill
    
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = make_schoolbook()
    best_rank = n * m * p  # 120
    
    # ------------------------------------------------------------------
    # Direct block construction: combine Strassen 2x2x2 and 2x3x2 blocks
    # to get a low-rank exact decomposition.
    # ------------------------------------------------------------------
    
    def _build_strassen_2x2():
        """Standard Strassen 2x2x2, rank 7."""
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        # P1=(a11+a22)(b11+b22)
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        # P2=(a21+a22)b11
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
        # P3=a11(b12-b22)
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
        # P4=a22(b21-b11)
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
        # P5=(a11+a12)b22
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
        # P6=(a21-a11)(b11+b12)
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        # P7=(a12-a22)(b21+b22)
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W
    
    def _build_strassen_2x3x2():
        """
        2x3x2 tensor decomposition.
        A is 2x3, B is 3x2, C is 2x2.
        We can use: split B into two 3x1 columns.
        C = A @ B = A @ [B1 | B2]
        
        Use Strassen-like:
        Let A = [[a11, a12, a13], [a21, a22, a23]]
        B = [[b11, b12], [b21, b22], [b31, b32]]
        
        C[0,0] = a11*b11 + a12*b21 + a13*b31
        C[0,1] = a11*b12 + a12*b22 + a13*b32
        C[1,0] = a21*b11 + a22*b21 + a23*b31
        C[1,1] = a21*b12 + a22*b22 + a23*b32
        
        Schoolbook rank = 12.
        We can try to reduce. One approach: group terms.
        
        Actually, for 2x3x2, the border rank is known to be 10 or 11.
        Let's try a rank-10 construction:
        
        P1 = (a11+a21)(b11+b12)  -- but this mixes rows of A and cols of B
        This is getting complex. Let's use schoolbook for now and optimize later.
        """
        # Use schoolbook for 2x3x2: rank 12
        U = np.zeros((6, 12), dtype=int)  # 2*3 = 6
        V = np.zeros((6, 12), dtype=int)  # 3*2 = 6
        W = np.zeros((4, 12), dtype=int)  # 2*2 = 4
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U[i*3+j, col] = 1
                    V[j*2+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
        return U, V, W
    
    def _build_strassen_4x4x2():
        """
        4x4x2 tensor: A is 4x4, B is 4x2, C is 4x2.
        Split into four 2x2x2 Strassen blocks: rank 4*7 = 28.
        But we can do better by sharing.
        
        Actually, 4x4x2 = two 4x4x1 tensors, or four 2x2x2.
        Let's just use four Strassen 2x2x2 blocks: rank 28.
        """
        U2, V2, W2 = _build_strassen_2x2()
        rank = 28
        U = np.zeros((16, rank), dtype=int)  # 4*4
        V = np.zeros((8, rank), dtype=int)   # 4*2
        W = np.zeros((8, rank), dtype=int)   # 4*2
        
        block_starts = [(0, 0, 0), (0, 2, 0), (2, 0, 0), (2, 2, 0)]
        for bi, (i_off, j_off, k_off) in enumerate(block_starts):
            base = bi * 7
            for t in range(7):
                col = base + t
                for li in range(2):
                    for lj in range(2):
                        coeff = U2[li*2+lj, t]
                        if coeff != 0:
                            U[(i_off+li)*4 + (j_off+lj), col] += coeff
                for lj in range(2):
                    for lk in range(2):
                        coeff = V2[lj*2+lk, t]
                        if coeff != 0:
                            V[(j_off+lj)*2 + (k_off+lk), col] += coeff
                for lk in range(2):
                    for li in range(2):
                        coeff = W2[lk*2+li, t]
                        if coeff != 0:
                            W[(k_off+lk)*4 + (i_off+li), col] += coeff
        return U, V, W
    
    def _build_4x5x6_direct():
        """
        Direct construction for 4x5x6:
        Split into:
          - 4x4x6 part: use three 4x4x2 Strassen blocks (rank 3*28 = 84)
          - 4x1x6 part: schoolbook (rank 24)
        Total: 108
        
        Then try to reduce by optimizing.
        """
        U442, V442, W442 = _build_strassen_4x4x2()
        rank = 84 + 24  # 108
        dim_U = n * m  # 20
        dim_V = m * p  # 30
        dim_W = p * n  # 24
        
        U = np.zeros((dim_U, rank), dtype=int)
        V = np.zeros((dim_V, rank), dtype=int)
        W = np.zeros((dim_W, rank), dtype=int)
        
        # Place three 4x4x2 blocks for k=0..5 (j=0..3)
        # Block 1: j=0..3, k=0..1
        for t in range(28):
            col = t
            for i in range(4):
                for j in range(4):
                    coeff = U442[i*4+j, t]
                    if coeff != 0:
                        U[i*5+j, col] += coeff
            for j in range(4):
                for k in range(2):
                    coeff = V442[j*2+k, t]
                    if coeff != 0:
                        V[j*6+k, col] += coeff
            for k in range(2):
                for i in range(4):
                    coeff = W442[k*4+i, t]
                    if coeff != 0:
                        W[k*4+i, col] += coeff
        
        # Block 2: j=0..3, k=2..3
        for t in range(28):
            col = 28 + t
            for i in range(4):
                for j in range(4):
                    coeff = U442[i*4+j, t]
                    if coeff != 0:
                        U[i*5+j, col] += coeff
            for j in range(4):
                for k in range(2):
                    coeff = V442[j*2+k, t]
                    if coeff != 0:
                        V[j*6+2+k, col] += coeff
            for k in range(2):
                for i in range(4):
                    coeff = W442[k*4+i, t]
                    if coeff != 0:
                        W[(2+k)*4+i, col] += coeff
        
        # Block 3: j=0..3, k=4..5
        for t in range(28):
            col = 56 + t
            for i in range(4):
                for j in range(4):
                    coeff = U442[i*4+j, t]
                    if coeff != 0:
                        U[i*5+j, col] += coeff
            for j in range(4):
                for k in range(2):
                    coeff = V442[j*2+k, t]
                    if coeff != 0:
                        V[j*6+4+k, col] += coeff
            for k in range(2):
                for i in range(4):
                    coeff = W442[k*4+i, t]
                    if coeff != 0:
                        W[(4+k)*4+i, col] += coeff
        
        # Schoolbook for j=4, all k (4x1x6 = 24 terms)
        col = 84
        for i in range(4):
            for k in range(6):
                U[i*5+4, col] = 1
                V[4*6+k, col] = 1
                W[k*4+i, col] = 1
                col += 1
        
        return U, V, W
    
    # Build the direct construction
    U_direct, V_direct, W_direct = _build_4x5x6_direct()
    direct_rank = U_direct.shape[1]
    
    # Verify
    U3 = U_direct.astype(np.float64).reshape(n, m, direct_rank)
    V3 = V_direct.astype(np.float64).reshape(m, p, direct_rank)
    W3 = W_direct.astype(np.float64).reshape(p, n, direct_rank)
    T_check = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    error = np.max(np.abs(T_check - 1.0))
    
    if error < 1e-9:
        best_U = U_direct.tolist()
        best_V = V_direct.tolist()
        best_W = W_direct.tolist()
        best_rank = direct_rank
    
    # ------------------------------------------------------------------
    # Try to reduce rank by 1, 2, 3 using local optimization
    # ------------------------------------------------------------------
    
    T = np.ones((n, m, p), dtype=np.float64)
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    def residual(x, R):
        U_flat = x[0 : R * dim_U]
        V_flat = x[R * dim_U : R * dim_U + R * dim_V]
        W_flat = x[R * dim_U + R * dim_V : ]
        
        U = U_flat.reshape(R, dim_U)
        V = V_flat.reshape(R, dim_V)
        W = W_flat.reshape(R, dim_W)
        
        U_3d = U.T.reshape(n, m, R)
        V_3d = V.T.reshape(m, p, R)
        W_3d = W.T.reshape(p, n, R)
        
        tensor_sum = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        res = tensor_sum - T
        return res.flatten()
    
    def try_reduce_rank(seed_U, seed_V, seed_W, target_rank, max_nfev=100):
        """Try to reduce rank from seed to target_rank."""
        if target_rank >= seed_U.shape[1]:
            return None
        
        num_params = target_rank * (dim_U + dim_V + dim_W)
        
        # Initialize from first target_rank columns of seed
        x0_parts = []
        for r in range(target_rank):
            x0_parts.append(seed_U[:, r].flatten())
        for r in range(target_rank):
            x0_parts.append(seed_V[:, r].flatten())
        for r in range(target_rank):
            x0_parts.append(seed_W[:, r].flatten())
        x0 = np.concatenate(x0_parts).astype(np.float64)
        
        # Small perturbation
        x0 += np.random.normal(0, 0.001, num_params)
        
        try:
            from scipy.optimize import least_squares
            res = least_squares(residual, x0, args=(target_rank,), method='lm',
                               max_nfev=max_nfev, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            
            if res.cost < 1e-10:
                U_flat = res.x[0 : target_rank * dim_U]
                V_flat = res.x[target_rank * dim_U : target_rank * dim_U + target_rank * dim_V]
                W_flat = res.x[target_rank * dim_U + target_rank * dim_V : ]
                
                U = U_flat.reshape(dim_U, target_rank)
                V = V_flat.reshape(dim_V, target_rank)
                W = W_flat.reshape(dim_W, target_rank)
                
                # Verify
                U3 = U.T.reshape(n, m, target_rank)
                V3 = V.T.reshape(m, p, target_rank)
                W3 = W.T.reshape(p, n, target_rank)
                T_out = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                err = np.max(np.abs(T_out - 1.0))
                
                if err < 1e-10:
                    # Try rounding to half-integers
                    U_half = np.round(U * 2) / 2
                    V_half = np.round(V * 2) / 2
                    W_half = np.round(W * 2) / 2
                    
                    U3h = U_half.T.reshape(n, m, target_rank)
                    V3h = V_half.T.reshape(m, p, target_rank)
                    W3h = W_half.T.reshape(p, n, target_rank)
                    T_out_h = np.einsum('ijr,jkr,kir->ijk', U3h, V3h, W3h)
                    err_h = np.max(np.abs(T_out_h - 1.0))
                    
                    if err_h < 1e-10:
                        return U_half, V_half, W_half
                    else:
                        return U, V, W
        except Exception:
            pass
        
        return None
    
    # Try to reduce from best_rank downward
    for target_rank in range(best_rank - 1, max(best_rank - 5, 100), -1):
        if time.time() > deadline:
            break
        
        np.random.seed(100 + target_rank)
        result = try_reduce_rank(np.array(best_U, dtype=np.float64),
                                 np.array(best_V, dtype=np.float64),
                                 np.array(best_W, dtype=np.float64),
                                 target_rank, max_nfev=80)
        
        if result is not None:
            U_r, V_r, W_r = result
            best_U = U_r.tolist()
            best_V = V_r.tolist()
            best_W = W_r.tolist()
            best_rank = target_rank
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
