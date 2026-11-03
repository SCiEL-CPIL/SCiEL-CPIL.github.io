"""Matrix multiplication 4x4 * 4x9 rank-minimal exact tensor decomposition.

Strategy: split B into [B1 (4x4), B2 (4x4), B3 (4x1)].
  C = [A*B1, A*B2, A*B3]
  A*B1: Strassen rank 49
  A*B2: Strassen rank 49
  A*B3: schoolbook rank 16
  Total rank: 114

Strassen for 4x4*4x4:
  Split A and B into 2x2 blocks.
  7 block products M1..M7, each a 2x2*2x2 multiplication.
  Each 2x2*2x2 uses Strassen (rank 7).
  Total: 7 * 7 = 49 terms.
"""
import time

_START = time.time()
_DEADLINE = 130.0  # safety margin below 150s kill


def _time_ok():
    return time.time() - _START < _DEADLINE


def _build_tensor(n, m, p):
    """T[io, u, v] = 1 iff output (i,k) uses a[i,j]*b[j,k]; io=k*n+i, u=i*m+j, v=j*p+k."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(k * n + i, i * m + j, j * p + k)] = 1
    return T


def _exact_check(U, V, W, T, n, m, p):
    """Verify sum_c U[u,c]*V[v,c]*W[io,c] == T[io,u,v] for all (io,u,v)."""
    rank = len(U[0])
    for io in range(p * n):
        for u in range(n * m):
            for v in range(m * p):
                s = 0
                for c in range(rank):
                    s += U[u][c] * V[v][c] * W[io][c]
                if s != T.get((io, u, v), 0):
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2_terms():
    """Return 7 tuples (u_pattern, v_pattern, w_pattern) for 2x2*2x2 Strassen."""
    terms = []
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    ))
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    ))
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    ))
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ))
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    return terms


def _strassen_4x4_4x4_terms():
    """Return 49 tuples (u, v, w) for 4x4*4x4 Strassen decomposition."""
    inner_terms = _strassen_2x2_terms()
    
    top_products = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    all_terms = []
    
    for (a_blocks, b_blocks, c_blocks) in top_products:
        for (u_inner, v_inner, w_inner) in inner_terms:
            u = {}
            v = {}
            w = {}
            
            for (bi, bj), ca in a_blocks.items():
                for (li, lj), ci in u_inner.items():
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    u[(i, j)] = u.get((i, j), 0) + ca * ci
            
            for (bi, bj), cb in b_blocks.items():
                for (li, lj), ci in v_inner.items():
                    j = 2 * bi + li
                    k = 2 * bj + lj
                    v[(j, k)] = v.get((j, k), 0) + cb * ci
            
            for (bi, bj), cc in c_blocks.items():
                for (li, lj), ci in w_inner.items():
                    i = 2 * bi + li
                    k = 2 * bj + lj
                    w[(i, k)] = w.get((i, k), 0) + cc * ci
            
            all_terms.append((u, v, w))
    
    return all_terms


def _rank114():
    """Build rank-114 decomposition for 4x4 * 4x9.
    
    Splits B into B1 (4x4, cols 0-3), B2 (4x4, cols 4-7), B3 (4x1, col 8).
    C = [A*B1, A*B2, A*B3]
    
    A*B1: Strassen rank 49
    A*B2: Strassen rank 49  
    A*B3: schoolbook rank 16
    Total: 114
    """
    n, m, p = 4, 4, 9
    rank = 114
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = _strassen_4x4_4x4_terms()
    
    # Part 1: A*B1 (B1 = B[:, 0:4])
    # Output C[:, 0:4]
    for t_idx, (u_pat, v_pat, w_pat) in enumerate(strassen_terms):
        col = t_idx
        for (i, j), coeff in u_pat.items():
            U[i * m + j][col] += coeff
        for (j, k), coeff in v_pat.items():
            V[j * p + k][col] += coeff
        for (i, k), coeff in w_pat.items():
            W[k * n + i][col] += coeff
    
    # Part 2: A*B2 (B2 = B[:, 4:8])
    # Output C[:, 4:8]
    for t_idx, (u_pat, v_pat, w_pat) in enumerate(strassen_terms):
        col = 49 + t_idx
        for (i, j), coeff in u_pat.items():
            U[i * m + j][col] += coeff
        for (j, k), coeff in v_pat.items():
            # k here is local to B2, so actual k = k + 4
            V[j * p + (k + 4)][col] += coeff
        for (i, k), coeff in w_pat.items():
            # k here is local to B2, so actual k = k + 4
            W[(k + 4) * n + i][col] += coeff
    
    # Part 3: A*B3 (B3 = B[:, 8:9])
    # Output C[:, 8:9]
    col = 98
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 8][col] = 1
            W[8 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s limit

    n, m, p = 4, 4, 9
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36

    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 144
    U_fb = [[0] * rank_schoolbook for _ in range(nm)]
    V_fb = [[0] * rank_schoolbook for _ in range(mp)]
    W_fb = [[0] * rank_schoolbook for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U = [row[:] for row in U_fb]
    best_V = [row[:] for row in V_fb]
    best_W = [row[:] for row in W_fb]

    # Try rank-114 Strassen-based decomposition
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}

    try:
        result114 = _rank114()
        T = _build_tensor(n, m, p)
        if _exact_check(result114["U"], result114["V"], result114["W"], T, n, m, p):
            best_rank = 114
            best_U = result114["U"]
            best_V = result114["V"]
            best_W = result114["W"]
    except Exception:
        pass

    # Try to improve further using rank reduction
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}

    import numpy as np
    
    def tensor_residual(U, V, W):
        """Compute the residual tensor R[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1"""
        rank = len(U[0])
        U_3d = np.array(U).reshape(n, m, rank)
        V_3d = np.array(V).reshape(m, p, rank)
        W_3d = np.array(W).reshape(p, n, rank)

        UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
        T = np.einsum('ijkr,kir->ijk', UV, W_3d)
        return T - 1.0

    def verify_decomposition(U, V, W, tol=1e-8):
        """Verify that the decomposition is exact."""
        T = tensor_residual(U, V, W)
        return np.max(np.abs(T)) < tol

    # Try to find better decompositions using optimization
    target_ranks = [100, 105, 110]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(2):
            if time.time() > deadline or found:
                break
            
            # Initialize from rank-114 if possible
            if best_rank == 114 and target_rank <= 114:
                # Try to reduce rank by zeroing some columns
                U = [row[:] for row in best_U]
                V = [row[:] for row in best_V]
                W = [row[:] for row in best_W]
                # Keep first target_rank columns
                U = [row[:target_rank] for row in U]
                V = [row[:target_rank] for row in V]
                W = [row[:target_rank] for row in W]
            else:
                # Random initialization
                scale = 0.3
                U = [[np.random.randn() * scale for _ in range(target_rank)] for _ in range(nm)]
                V = [[np.random.randn() * scale for _ in range(target_rank)] for _ in range(mp)]
                W = [[np.random.randn() * scale for _ in range(target_rank)] for _ in range(pn)]
            
            # Convert to numpy for optimization
            U_np = np.array(U, dtype=np.float64)
            V_np = np.array(V, dtype=np.float64)
            W_np = np.array(W, dtype=np.float64)
            
            # Adam optimizer state
            m_U = np.zeros_like(U_np)
            v_U = np.zeros_like(U_np)
            m_V = np.zeros_like(V_np)
            v_V = np.zeros_like(V_np)
            m_W = np.zeros_like(W_np)
            v_W = np.zeros_like(W_np)
            t = 0
            lr = 0.05
            
            max_steps = 1000
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                t += 1
                U_3d = U_np.reshape(n, m, target_rank)
                V_3d = V_np.reshape(m, p, target_rank)
                W_3d = W_np.reshape(p, n, target_rank)
                
                UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
                T = np.einsum('ijkr,kir->ijk', UV, W_3d)
                err = T - 1.0
                
                max_err = np.max(np.abs(err))
                if max_err < 1e-8:
                    found = True
                    break
                
                # Adaptive learning rate
                if max_err > 1.0:
                    lr = 0.01
                elif max_err > 0.1:
                    lr = 0.05
                else:
                    lr = 0.001
                
                grad_U = 2.0 * np.einsum('ijk,jkr,kir->ijr', err, V_3d, W_3d)
                grad_V = 2.0 * np.einsum('ijk,ijr,kir->jkr', err, U_3d, W_3d)
                grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kir', err, U_3d, V_3d)
                
                m_U = 0.9 * m_U + 0.1 * grad_U
                v_U = 0.999 * v_U + 0.001 * grad_U**2
                U_np -= lr * (m_U / (1 - 0.9**t)) / (np.sqrt(v_U / (1 - 0.999**t)) + 1e-8)
                
                m_V = 0.9 * m_V + 0.1 * grad_V
                v_V = 0.999 * v_V + 0.001 * grad_V**2
                V_np -= lr * (m_V / (1 - 0.9**t)) / (np.sqrt(v_V / (1 - 0.999**t)) + 1e-8)
                
                m_W = 0.9 * m_W + 0.1 * grad_W
                v_W = 0.999 * v_W + 0.001 * grad_W**2
                W_np -= lr * (m_W / (1 - 0.9**t)) / (np.sqrt(v_W / (1 - 0.999**t)) + 1e-8)
            
            # Final check
            if not found:
                if time.time() > deadline:
                    break
                U_3d = U_np.reshape(n, m, target_rank)
                V_3d = V_np.reshape(m, p, target_rank)
                W_3d = W_np.reshape(p, n, target_rank)
                UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
                T = np.einsum('ijkr,kir->ijk', UV, W_3d)
                if np.max(np.abs(T - 1.0)) < 1e-8:
                    found = True
            
            if found:
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_U = U_np.tolist()
                    best_V = V_np.tolist()
                    best_W = W_np.tolist()
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
