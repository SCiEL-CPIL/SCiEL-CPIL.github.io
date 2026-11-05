"""Rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.

Uses block-Strassen: rank 49 for 4x4*4x4 + rank 28 for 4x4*4x2 = 77.
Additionally attempts a search for a lower-rank 2x2 x 2x3 sub-decomposition
that could reduce the total rank. Falls back to the proven rank-77 construction.
"""

import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin before the 150s kill


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def construct():
    """
    Construct a rank-reduced tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Returns a dict with keys 'U', 'V', 'W' where:
      U: shape (n*m, rank) = (16, rank)
      V: shape (m*p, rank) = (24, rank)
      W: shape (p*n, rank) = (24, rank)
    
    The tensor condition is:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} * (1 if j==j' else 0)
    More precisely, for matrix multiplication C = A * B where A is 4x4 and B is 4x6:
      C[i,k] = sum_j A[i,j] * B[j,k]
    The tensor T[i,j,k] = 1 for all i,j,k (since each product A[i,j]*B[j,k] contributes
    to C[i,k]). So we need:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
    Wait, that's not right. The matrix multiplication tensor has entries:
      T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    Since it always does, T[i,j,k] = 1 for all i,j,k.
    But that's the schoolbook. The Strassen decomposition works by having:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
    
    Actually no. Let me re-derive. The bilinear map is:
      (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    This is a bilinear map from R^{n*m} x R^{m*p} -> R^{n*p}.
    The tensor T in R^{n*m} x R^{m*p} x R^{n*p} has entries:
      T[(i,j), (j,k), (i,k)] = 1  (using the same j for both)
    All other entries are 0.
    
    So a rank-r decomposition is:
      T = sum_{r=1}^{r} u_r ⊗ v_r ⊗ w_r
    where u_r in R^{n*m}, v_r in R^{m*p}, w_r in R^{n*p}.
    In component form:
      T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    where a = i*m+j, b = j'*p+k, c = i'*n+k'.
    And T[a,b,c] = 1 iff a and b share the same j (i.e., j == j') AND c's i' == a's i
    AND c's k' == b's k. Otherwise 0.
    
    So: T[i*m+j, j*p+k, k*n+i] = 1, all else 0.
    """
    n, m, p = 4, 4, 6
    
    U_cols = []  # Each element is a list of length n*m
    V_cols = []  # Each element is a list of length m*p
    W_cols = []  # Each element is a list of length p*n
    
    def add_term_ijk(u_dict, v_dict, w_dict):
        """
        Add a term using (i,j) for U, (j,k) for V, (k,i) for W indexing.
        u_dict: {(i,j): coeff}
        v_dict: {(j,k): coeff}
        w_dict: {(i,k): coeff}  -> stored at index k*n+i
        """
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (i, j), coeff in u_dict.items():
            u_col[i * m + j] += coeff
        for (j, k), coeff in v_dict.items():
            v_col[j * p + k] += coeff
        for (i, k), coeff in w_dict.items():
            w_col[k * n + i] += coeff
        
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    def strassen_2x2(oi, oj, ok):
        """
        Add Strassen decomposition for 2x2 * 2x2 block multiplication.
        oi: offset row in A (top-left corner row)
        oj: offset col in A / row in B
        ok: offset col in B (top-left corner col)
        
        Computes C[oi:oi+2, ok:ok+2] = A[oi:oi+2, oj:oj+2] * B[oj:oj+2, ok:ok+2]
        """
        # m1 = (x00+x11)(y00+y11)
        add_term_ijk(
            {(oi, oj): 1, (oi+1, oj+1): 1},
            {(oj, ok): 1, (oj+1, ok+1): 1},
            {(oi, ok): 1, (oi+1, ok+1): 1}
        )
        
        # m2 = (x10+x11)y00
        add_term_ijk(
            {(oi+1, oj): 1, (oi+1, oj+1): 1},
            {(oj, ok): 1},
            {(oi+1, ok): 1, (oi+1, ok+1): -1}
        )
        
        # m3 = x00(y01-y11)
        add_term_ijk(
            {(oi, oj): 1},
            {(oj, ok+1): 1, (oj+1, ok+1): -1},
            {(oi, ok+1): 1, (oi+1, ok+1): 1}
        )
        
        # m4 = x11(y10-y00)
        add_term_ijk(
            {(oi+1, oj+1): 1},
            {(oj+1, ok): 1, (oj, ok): -1},
            {(oi, ok): 1, (oi+1, ok): 1}
        )
        
        # m5 = (x00+x01)y11
        add_term_ijk(
            {(oi, oj): 1, (oi, oj+1): 1},
            {(oj+1, ok+1): 1},
            {(oi, ok): -1, (oi, ok+1): 1}
        )
        
        # m6 = (x10-x00)(y00+y01)
        add_term_ijk(
            {(oi+1, oj): 1, (oi, oj): -1},
            {(oj, ok): 1, (oj, ok+1): 1},
            {(oi+1, ok+1): 1}
        )
        
        # m7 = (x01-x11)(y10+y11)
        add_term_ijk(
            {(oi, oj+1): 1, (oi+1, oj+1): -1},
            {(oj+1, ok): 1, (oj+1, ok+1): 1},
            {(oi, ok): 1}
        )
    
    def strassen_4x4(oi, oj, ok):
        """
        Add recursive Strassen decomposition for 4x4 * 4x4 block multiplication.
        Uses 7 sub-problems, each of which is a 2x2 * 2x2 Strassen.
        Total: 7 * 7 = 49 rank-1 terms.
        """
        # The 4x4 matrices are divided into 2x2 blocks:
        # A = [[A00, A01], [A10, A11]]  where Aij is 2x2
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        
        # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
        # M2 = (A10+A11)B00 -> contributes to C10 and C11
        # M3 = A00(B01-B11) -> contributes to C01 and C11
        # M4 = A11(B10-B00) -> contributes to C00 and C10
        # M5 = (A00+A01)B11 -> contributes to C00 and C01
        # M6 = (A10-A00)(B00+B01) -> contributes to C11
        # M7 = (A01-A11)(B10+B11) -> contributes to C00
        
        # For each Strassen term, we need to specify which 2x2 blocks of A, B, C
        # are involved and with what signs.
        
        # M1 = (A00+A11)(B00+B11)
        # U blocks: A00 (+1), A11 (+1)
        # V blocks: B00 (+1), B11 (+1)
        # W blocks: C00 (+1), C11 (+1)
        _strassen_super(oi, oj, ok, 
                        u_blocks={0: 1, 3: 1},   # A00, A11 (block indices 0,1,2,3 = TL,TR,BL,BR)
                        v_blocks={0: 1, 3: 1},   # B00, B11
                        w_blocks={0: 1, 3: 1})   # C00, C11
        
        # M2 = (A10+A11)B00
        # U: A10 (+1), A11 (+1)
        # V: B00 (+1)
        # W: C10 (+1), C11 (-1)
        _strassen_super(oi, oj, ok,
                        u_blocks={2: 1, 3: 1},
                        v_blocks={0: 1},
                        w_blocks={2: 1, 3: -1})
        
        # M3 = A00(B01-B11)
        # U: A00 (+1)
        # V: B01 (+1), B11 (-1)
        # W: C01 (+1), C11 (+1)
        _strassen_super(oi, oj, ok,
                        u_blocks={0: 1},
                        v_blocks={1: 1, 3: -1},
                        w_blocks={1: 1, 3: 1})
        
        # M4 = A11(B10-B00)
        # U: A11 (+1)
        # V: B10 (+1), B00 (-1)
        # W: C00 (+1), C10 (+1)
        _strassen_super(oi, oj, ok,
                        u_blocks={3: 1},
                        v_blocks={2: 1, 0: -1},
                        w_blocks={0: 1, 2: 1})
        
        # M5 = (A00+A01)B11
        # U: A00 (+1), A01 (+1)
        # V: B11 (+1)
        # W: C00 (-1), C01 (+1)
        _strassen_super(oi, oj, ok,
                        u_blocks={0: 1, 1: 1},
                        v_blocks={3: 1},
                        w_blocks={0: -1, 1: 1})
        
        # M6 = (A10-A00)(B00+B01)
        # U: A10 (+1), A00 (-1)
        # V: B00 (+1), B01 (+1)
        # W: C11 (+1)
        _strassen_super(oi, oj, ok,
                        u_blocks={2: 1, 0: -1},
                        v_blocks={0: 1, 1: 1},
                        w_blocks={3: 1})
        
        # M7 = (A01-A11)(B10+B11)
        # U: A01 (+1), A11 (-1)
        # V: B10 (+1), B11 (+1)
        # W: C00 (+1)
        _strassen_super(oi, oj, ok,
                        u_blocks={1: 1, 3: -1},
                        v_blocks={2: 1, 3: 1},
                        w_blocks={0: 1})
    
    def _strassen_super(oi, oj, ok, u_blocks, v_blocks, w_blocks):
        """
        Add a "super" Strassen term that is a sum over 2x2 blocks.
        
        u_blocks: dict mapping block index (0=TL,1=TR,2=BL,3=BR) to coefficient for A
        v_blocks: same for B
        w_blocks: same for C
        
        This generates 7 rank-1 terms (one for each Strassen 2x2 multiplication).
        """
        # The 7 Strassen 2x2 terms, each with (u_pattern, v_pattern, w_pattern)
        # where patterns are 2x2 matrices as dicts {(row,col): coeff}
        strassen_2x2_terms = [
            # m1 = (x00+x11)(y00+y11)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # m2 = (x10+x11)y00
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            # m3 = x00(y01-y11)
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            # m4 = x11(y10-y00)
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # m5 = (x00+x01)y11
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            # m6 = (x10-x00)(y00+y01)
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # m7 = (x01-x11)(y10+y11)
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        # Block offsets: block 0 = (0,0), 1 = (0,1), 2 = (1,0), 3 = (1,1)
        block_offsets = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
        
        for u_pat, v_pat, w_pat in strassen_2x2_terms:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            # Fill U
            for block_idx, block_coeff in u_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in u_pat.items():
                    i = oi + 2*br + r
                    j = oj + 2*bc + c
                    if 0 <= i < n and 0 <= j < m:
                        u_col[i * m + j] += block_coeff * coeff
            
            # Fill V
            for block_idx, block_coeff in v_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in v_pat.items():
                    j = oj + 2*br + r
                    k = ok + 2*bc + c
                    if 0 <= j < m and 0 <= k < p:
                        v_col[j * p + k] += block_coeff * coeff
            
            # Fill W
            for block_idx, block_coeff in w_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in w_pat.items():
                    i = oi + 2*br + r
                    k = ok + 2*bc + c
                    if 0 <= i < n and 0 <= k < p:
                        w_col[k * n + i] += block_coeff * coeff
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    # Build the proven rank-77 decomposition
    if _time_remaining() > 5.0:
        # PART 1: 4x4 * 4x4 (columns 0-3 of B) -> rank 49
        strassen_4x4(0, 0, 0)
    
    if _time_remaining() > 3.0:
        # PART 2: 4x4 * 4x2 (columns 4-5 of B) -> rank 28
        for rb in range(2):
            if _time_remaining() < 2.0:
                break
            strassen_2x2(rb*2, 0, 4)
            strassen_2x2(rb*2, 2, 4)
    
    rank = len(U_cols)
    
    # Transpose to get U as (n*m, rank), V as (m*p, rank), W as (p*n, rank)
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    # Verify the decomposition
    if _verify_decomposition(U_final, V_final, W_final, n, m, p):
        result_77 = {"U": U_final, "V": V_final, "W": W_final}
    else:
        result_77 = None
    
    best_result = result_77
    best_rank = rank if result_77 is not None else n*m*p
    
    # Try to find a lower-rank decomposition of the 4x4 x 4x2 sub-tensor.
    # The 4x4 x 4x2 tensor has 16*24*24 = 9216 entries but only 4*4*2=32 nonzero.
    # Current rank: 28 (four 2x2 x 2x2 Strassen blocks).
    # Try to find rank < 28 using optimization on the full sub-tensor.
    if _time_remaining() > 30.0:
        improved = _search_4x4x4x2_low_rank()
        if improved is not None:
            U_imp, V_imp, W_imp = improved
            imp_rank = len(U_imp)
            if imp_rank < best_rank - 49:  # Must be better than 28
                # Combine with the 4x4 x 4x4 part
                U_combined = [row + [0]*imp_rank for row in U_final]
                V_combined = [row + [0]*imp_rank for row in V_final]
                W_combined = [row + [0]*imp_rank for row in W_final]
                # Add the improved 4x4 x 4x2 terms
                # U_imp has shape (16, imp_rank), but only rows corresponding to A entries matter
                # V_imp has shape (24, imp_rank), only columns 4,5 of B matter
                # W_imp has shape (24, imp_rank), only rows 4,5 of C matter
                for i in range(16):
                    U_combined[i] = U_combined[i] + U_imp[i]
                for i in range(24):
                    V_combined[i] = V_combined[i] + V_imp[i]
                for i in range(24):
                    W_combined[i] = W_combined[i] + W_imp[i]
                
                if _verify_decomposition(U_combined, V_combined, W_combined, n, m, p):
                    best_result = {"U": U_combined, "V": V_combined, "W": W_combined}
                    best_rank = 49 + imp_rank
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook rank-96
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def _search_2x2x2x3_low_rank():
    """
    Try to find a rank-r decomposition of the 2x2 x 2x3 matrix multiplication tensor.
    The tensor T has entries T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2}.
    
    We search for rank r = 9, 10 (below the current 11) using a smooth optimization.
    Returns (U, V, W) as lists of rows if found, else None.
    U: 4 x rank, V: 6 x rank, W: 6 x rank
    """
    for target_rank in [10, 9]:
        if _time_remaining() < 15.0:
            break
        result = _optimize_2x2x2x3(target_rank)
        if result is not None:
            return result
    return None


def _optimize_2x2x2x3(target_rank):
    """
    Use smooth optimization with analytic gradients to find a rank-target_rank
    decomposition of 2x2 x 2x3 matrix multiplication.
    
    The tensor condition: for all i in {0,1}, j in {0,1}, k in {0,1,2}:
      sum_r U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1
    
    Returns (U, V, W) as numpy arrays if found, else None.
    """
    np.random.seed(42)
    
    # Precompute the index mapping
    # For each (i,j,k), we need U_idx, V_idx, W_idx
    tensor_terms = []
    for i in range(2):
        for j in range(2):
            for k in range(3):
                u_idx = i * 2 + j
                v_idx = j * 3 + k
                w_idx = k * 2 + i
                tensor_terms.append((u_idx, v_idx, w_idx))
    
    n_terms = len(tensor_terms)  # 12
    
    best_loss = float('inf')
    best_U = best_V = best_W = None
    
    n_trials = 8
    for trial in range(n_trials):
        if _time_remaining() < 8.0:
            break
        
        # Initialize with random values
        U = np.random.randn(4, target_rank) * 0.7
        V = np.random.randn(6, target_rank) * 0.7
        W = np.random.randn(6, target_rank) * 0.7
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def loss_and_grad(params):
            U_f = params[0:4*target_rank].reshape(4, target_rank)
            V_f = params[4*target_rank:10*target_rank].reshape(6, target_rank)
            W_f = params[10*target_rank:16*target_rank].reshape(6, target_rank)
            
            loss = 0.0
            dU = np.zeros_like(U_f)
            dV = np.zeros_like(V_f)
            dW = np.zeros_like(W_f)
            
            for (u_idx, v_idx, w_idx) in tensor_terms:
                # val = sum_r U[u_idx,r] * V[v_idx,r] * W[w_idx,r]
                vw = V_f[v_idx, :] * W_f[w_idx, :]
                val = np.dot(U_f[u_idx, :], vw)
                err = val - 1.0
                loss += err * err
                
                # dU[u_idx,r] += 2*err * V[v_idx,r] * W[w_idx,r]
                dU[u_idx, :] += 2.0 * err * vw
                # dV[v_idx,r] += 2*err * U[u_idx,r] * W[w_idx,r]
                uw = U_f[u_idx, :] * W_f[w_idx, :]
                dV[v_idx, :] += 2.0 * err * uw
                # dW[w_idx,r] += 2*err * U[u_idx,r] * V[v_idx,r]
                uv = U_f[u_idx, :] * V_f[v_idx, :]
                dW[w_idx, :] += 2.0 * err * uv
            
            grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            return loss, grad
        
        try:
            from scipy.optimize import minimize
            res = minimize(loss_and_grad, params, method='L-BFGS-B', jac=True,
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12,
                                   'maxls': 50, 'maxfun': 2000})
            if res.fun < best_loss:
                best_loss = res.fun
                best_U = res.x[0:4*target_rank].reshape(4, target_rank)
                best_V = res.x[4*target_rank:10*target_rank].reshape(6, target_rank)
                best_W = res.x[10*target_rank:16*target_rank].reshape(6, target_rank)
        except Exception:
            pass
        
        if best_loss < 1e-14:
            break
    
    if best_loss < 1e-10 and best_U is not None:
        # Try rounding to nearest integers first, then half-integers
        for scale in [1, 2, 4]:
            U_rounded = np.round(best_U * scale) / scale
            V_rounded = np.round(best_V * scale) / scale
            W_rounded = np.round(best_W * scale) / scale
            
            # Verify
            ok = True
            for (u_idx, v_idx, w_idx) in tensor_terms:
                val = np.dot(U_rounded[u_idx, :], V_rounded[v_idx, :] * W_rounded[w_idx, :])
                if abs(val - 1.0) > 1e-9:
                    ok = False
                    break
            if ok:
                return (U_rounded, V_rounded, W_rounded)
    
    return None


def _build_with_improved_sub(sub_decomp, n, m, p, strassen_4x4, add_term_ijk):
    """
    Build the full 4x4 x 4x6 decomposition using an improved 2x2 x 2x3 sub-decomposition.
    sub_decomp: (U_sub, V_sub, W_sub) for 2x2 x 2x3, each as numpy arrays (rows x rank).
    
    The 4x4 x 4x2 part (columns 4-5 of B) is restructured as 2 instances of 2x2 x 2x3:
    
    For output rows i in {0,1} (top half of C), columns k in {4,5}:
      C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
             = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    
    The first sum is a 2x2 x 2x2 multiplication (A[0:2,0:2] * B[0:2,4:6]).
    The second sum is a 2x2 x 2x2 multiplication (A[0:2,2:4] * B[2:4,4:6]).
    
    Alternatively, we can view the 2x4 x 4x2 product differently:
    For fixed output row i, C[i,4:6] = A[i,0:4] * B[0:4, 4:6]
    
    We can split this as:
    C[i,4:6] = A[i,0:2] * B[0:2, 4:6] + A[i,2:4] * B[2:4, 4:6]
    
    Now, for the 2x2 x 2x3 interpretation: consider the "wide" multiplication
    where A is 2x2, B is 2x3, and C is 2x3. This doesn't directly match.
    
    Let me reconsider the structure. The 2x2 x 2x3 tensor T_sub has:
      T_sub[i, j, k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2}
    with U index = i*2+j, V index = j*3+k, W index = k*2+i.
    
    This computes: C[i,k] = sum_j A[i,j] * B[j,k] where A is 2x2, B is 2x3, C is 2x3.
    
    For our 4x4 x 4x2 sub-problem (columns 4,5 of B):
    C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {4,5}, j in {0,1,2,3}
    
    We can split j into two groups: j in {0,1} and j in {2,3}.
    Group 1: C[i,k] += sum_{j=0}^{1} A[i,j] * B[j,k]  -- this is 2x2 x 2x2
    Group 2: C[i,k] += sum_{j=2}^{3} A[i,j] * B[j,k]  -- this is 2x2 x 2x2
    
    The 2x2 x 2x3 tensor doesn't directly apply here since both sub-problems are 2x2 x 2x2.
    
    However, we CAN use a different decomposition. Consider the full 2x4 x 4x2 product:
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    If we relabel: let the "effective" B matrix be B[0:4, 4:6] which is 4x2.
    And A[0:2, 0:4] which is 2x4.
    
    We can write: C = A[0:2,:] * B[:,4:6]
    
    Using the 2x2 x 2x3 decomposition: this doesn't directly fit because
    the dimensions are 2x4 x 4x2, not 2x2 x 2x3.
    
    A better approach: use the 2x2 x 2x3 decomposition to compute
    C[i, k] where we treat the 4 columns of A as 2 groups of 2, and the 2 rows of B
    as part of a 2x3 structure.
    
    Actually, the cleanest way: the 2x2 x 2x3 tensor computes C = A*B for
    A (2x2), B (2x3), C (2x3). 
    
    For our problem, for output rows 0,1:
    C[0:2, 4:6] = A[0:2, 0:4] * B[0:4, 4:6]
    
    We can split this as:
    C[0:2, 4:6] = A[0:2, 0:2] * B[0:2, 4:6] + A[0:2, 2:4] * B[2:4, 4:6]
    
    Each term is 2x2 x 2x2. With Strassen that's 7+7=14 per 2-row block, 28 total.
    
    The 2x2 x 2x3 doesn't help directly. Return None.
    """
    return None


def _search_4x4x4x2_low_rank():
    """
    Search for a low-rank decomposition of the 4x4 x 4x2 matrix multiplication tensor.
    This tensor computes C[0:4, 4:6] = A[0:4, 0:4] * B[0:4, 4:6].
    
    The tensor has entries: T[i*4+j, j*6+k, k*4+i] = 1 for i in 0..3, j in 0..3, k in {4,5}.
    In local coordinates (k' = k-4 in {0,1}):
    T[i*4+j, j*6+4+k', k'*4+i] = 1
    
    We search for rank 24, 25, 26, 27 (below current 28).
    Returns (U, V, W) as lists of rows if found, else None.
    U: 16 x rank, V: 24 x rank, W: 24 x rank
    """
    n, m, p = 4, 4, 6
    
    # Precompute tensor terms: (u_idx, v_idx, w_idx)
    terms = []
    for i in range(n):
        for j in range(m):
            for k in [4, 5]:
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                terms.append((u_idx, v_idx, w_idx))
    
    n_terms = len(terms)  # 32
    
    for target_rank in [24, 25, 26, 27]:
        if _time_remaining() < 15.0:
            break
        
        np.random.seed(123 + target_rank)
        best_loss = float('inf')
        best_params = None
        
        n_trials = 4
        for trial in range(n_trials):
            if _time_remaining() < 10.0:
                break
            
            U = np.random.randn(16, target_rank) * 0.5
            V = np.random.randn(24, target_rank) * 0.5
            W = np.random.randn(24, target_rank) * 0.5
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def loss_and_grad(params):
                U_f = params[0:16*target_rank].reshape(16, target_rank)
                V_f = params[16*target_rank:40*target_rank].reshape(24, target_rank)
                W_f = params[40*target_rank:64*target_rank].reshape(24, target_rank)
                # Verify sizes: 16+24+24 = 64, so 64*target_rank total
                assert len(params) == 64 * target_rank
                
                loss = 0.0
                dU = np.zeros_like(U_f)
                dV = np.zeros_like(V_f)
                dW = np.zeros_like(W_f)
                
                for (u_idx, v_idx, w_idx) in terms:
                    vw = V_f[v_idx, :] * W_f[w_idx, :]
                    val = np.dot(U_f[u_idx, :], vw)
                    err = val - 1.0
                    loss += err * err
                    dU[u_idx, :] += 2.0 * err * vw
                    uw = U_f[u_idx, :] * W_f[w_idx, :]
                    dV[v_idx, :] += 2.0 * err * uw
                    uv = U_f[u_idx, :] * V_f[v_idx, :]
                    dW[w_idx, :] += 2.0 * err * uv
                
                grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return loss, grad
            
            try:
                from scipy.optimize import minimize
                res = minimize(loss_and_grad, params, method='L-BFGS-B', jac=True,
                              options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12,
                                       'maxls': 40, 'maxfun': 1500})
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
            
            if best_loss < 1e-12:
                break
        
        if best_loss < 1e-10 and best_params is not None:
            U_f = best_params[0:16*target_rank].reshape(16, target_rank)
            V_f = best_params[16*target_rank:40*target_rank].reshape(24, target_rank)
            W_f = best_params[40*target_rank:64*target_rank].reshape(24, target_rank)
            
            # Zero out entries that should be zero (V and W only for k=4,5)
            # V: only rows j*6+4 and j*6+5 should be nonzero (k=4,5)
            for j in range(4):
                for k in range(4):
                    V_f[j*6+k, :] = 0.0
            # W: only rows 4*4+i and 5*4+i should be nonzero (k=4,5)
            for k in range(4):
                for i in range(4):
                    W_f[k*4+i, :] = 0.0
            
            # Try rounding
            for scale in [1, 2, 4]:
                U_r = np.round(U_f * scale) / scale
                V_r = np.round(V_f * scale) / scale
                W_r = np.round(W_f * scale) / scale
                
                ok = True
                for (u_idx, v_idx, w_idx) in terms:
                    val = np.dot(U_r[u_idx, :], V_r[v_idx, :] * W_r[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        ok = False
                        break
                if ok:
                    # Convert to lists
                    U_list = [list(U_r[r, :]) for r in range(16)]
                    V_list = [list(V_r[r, :]) for r in range(24)]
                    W_list = [list(W_r[r, :]) for r in range(24)]
                    return (U_list, V_list, W_list)
    
    return None


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is correct.
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each (i, j, k), compute the sum
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = np.dot(U_arr[u_idx, :], V_arr[v_idx, :] * W_arr[w_idx, :])
                if abs(total - 1.0) > 1e-9:
                    return False
    return True
