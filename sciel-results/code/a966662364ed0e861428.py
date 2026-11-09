import numpy as np
import time
import sys

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS

    # Try to find a lower-rank decomposition
    # Start with known good solutions and try to improve
    
    # First, build the Strassen-based rank-84 solution as fallback
    strassen_result = _build_strassen_84(n, m, p)
    best_rank = 84
    best_result = strassen_result
    
    # Try to find lower rank decompositions using structured search
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(16, 24, 24) = 24
    # But actual minimum is likely higher
    
    # Try various structured approaches
    if time.time() < deadline - 5:
        # Approach 1: Try rank-48 using 2x2 Strassen blocks with sharing
        result = _try_rank_48(n, m, p, deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_rank = 48
                best_result = {"U": U, "V": V, "W": W}
    
    if time.time() < deadline - 5:
        # Approach 2: Try rank-56
        result = _try_rank_56(n, m, p, deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_rank = 56
                best_result = {"U": U, "V": V, "W": W}
    
    if time.time() < deadline - 5:
        # Approach 3: Try rank-64
        result = _try_rank_64(n, m, p, deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_rank = 64
                best_result = {"U": U, "V": V, "W": W}
    
    if time.time() < deadline - 5:
        # Approach 4: Try rank-72
        result = _try_rank_72(n, m, p, deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_rank = 72
                best_result = {"U": U, "V": V, "W": W}
    
    if time.time() < deadline - 5:
        # Approach 5: Try rank-80
        result = _try_rank_80(n, m, p, deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_rank = 80
                best_result = {"U": U, "V": V, "W": W}
    
    if time.time() < deadline - 3:
        # Approach 6: Random search for lower ranks
        for target_rank in range(40, best_rank, 2):
            if time.time() > deadline - 3:
                break
            result = _random_search(n, m, p, target_rank, deadline - time.time())
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    best_rank = target_rank
                    best_result = {"U": U, "V": V, "W": W}
                    break  # Found lower rank, continue trying even lower
    
    return best_result

def _build_strassen_84(n, m, p):
    """Build the Strassen-based rank-84 decomposition."""
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(2):
        for k in range(3):
            for j in range(2):
                a_idx = 2*i * m + 2*j
                b_idx = 2*i * m + 2*j + 1
                c_idx = (2*i+1) * m + 2*j
                d_idx = (2*i+1) * m + 2*j + 1

                e_idx = 2*j * p + 2*k
                f_idx = 2*j * p + 2*k + 1
                g_idx = (2*j+1) * p + 2*k
                h_idx = (2*j+1) * p + 2*k + 1

                c11_idx = (2*k) * n + 2*i
                c12_idx = (2*k+1) * n + 2*i
                c21_idx = (2*k) * n + 2*i + 1
                c22_idx = (2*k+1) * n + 2*i + 1

                # M1 = (a+d)(e+h)
                U[a_idx][col] = 1; U[d_idx][col] = 1
                V[e_idx][col] = 1; V[h_idx][col] = 1
                W[c11_idx][col] = 1; W[c22_idx][col] = 1
                col += 1

                # M2 = (c+d)e
                U[c_idx][col] = 1; U[d_idx][col] = 1
                V[e_idx][col] = 1
                W[c21_idx][col] = 1; W[c22_idx][col] = -1
                col += 1

                # M3 = a(f-h)
                U[a_idx][col] = 1
                V[f_idx][col] = 1; V[h_idx][col] = -1
                W[c12_idx][col] = 1; W[c22_idx][col] = 1
                col += 1

                # M4 = d(g-e)
                U[d_idx][col] = 1
                V[g_idx][col] = 1; V[e_idx][col] = -1
                W[c11_idx][col] = 1; W[c21_idx][col] = 1
                col += 1

                # M5 = (a+b)h
                U[a_idx][col] = 1; U[b_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = -1; W[c12_idx][col] = 1
                col += 1

                # M6 = (c-a)(e+f)
                U[c_idx][col] = 1; U[a_idx][col] = -1
                V[e_idx][col] = 1; V[f_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1

                # M7 = (b-d)(g+h)
                U[b_idx][col] = 1; U[d_idx][col] = -1
                V[g_idx][col] = 1; V[h_idx][col] = 1
                W[c11_idx][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}

def _try_rank_48(n, m, p, deadline):
    """
    Try rank-48 decomposition.
    Partition A (4x4) into 2x2 blocks of 2x2.
    Partition B (4x6) into 2x3 blocks of 2x2.
    For each output block C_ik = A_i0*B_0k + A_i1*B_1k,
    use a 4-multiplication scheme (if possible) instead of 7.
    
    Actually, let's try: for each (i,k), use 4 multiplications
    Total: 2 * 3 * 4 = 24... that's too optimistic.
    
    Let me try: use 8 multiplications per (i,k) pair
    Total: 2 * 3 * 8 = 48
    """
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(2):
        for k in range(3):
            # C_ik = A_i0 * B_0k + A_i1 * B_1k
            # A_i0 = [[a,b],[c,d]], A_i1 = [[e,f],[g,h]]
            # B_0k = [[p,q],[r,s]], B_1k = [[t,u],[v,w]]
            # C_ik = [[a*p+b*t, a*q+b*u], [c*p+d*t, c*q+d*u]] + [[e*r+f*v, e*s+f*w], [g*r+h*v, g*s+h*w]]
            #      = [[ap+et+br+fv... wait, this is getting complicated.
            
            # Let me use a different approach: treat A_i0*B_0k + A_i1*B_1k as a single
            # 2x2 x 2x2 + 2x2 x 2x2 problem.
            # We can combine: let X = [[a,b],[c,d]], Y = [[p,q],[r,s]], 
            #                    Z = [[e,f],[g,h]], T = [[t,u],[v,w]]
            # Compute XY + ZT using 8 multiplications (known result)
            
            a_idx = 2*i * m + 0
            b_idx = 2*i * m + 1
            c_idx = (2*i+1) * m + 0
            d_idx = (2*i+1) * m + 1
            
            e_idx = 2*i * m + 2
            f_idx = 2*i * m + 3
            g_idx = (2*i+1) * m + 2
            h_idx = (2*i+1) * m + 3
            
            p_idx = 0 * p + 2*k
            q_idx = 0 * p + 2*k + 1
            r_idx = 1 * p + 2*k
            s_idx = 1 * p + 2*k + 1
            
            t_idx = 2 * p + 2*k
            u_idx = 2 * p + 2*k + 1
            v_idx = 3 * p + 2*k
            w_idx = 3 * p + 2*k + 1
            
            c11_idx = (2*k) * n + 2*i
            c12_idx = (2*k+1) * n + 2*i
            c21_idx = (2*k) * n + 2*i + 1
            c22_idx = (2*k+1) * n + 2*i + 1
            
            # Use 8 multiplications for XY + ZT:
            # M1 = (a+e)(p+t)
            # M2 = (c+g)(p+t)  
            # M3 = (b+f)(u+w)
            # M4 = (d+h)(u+w)
            # M5 = (a+c)(r+s)
            # M6 = (e+g)(r+s)
            # M7 = (b+d)(q+v)
            # M8 = (f+h)(q+v)
            
            # C11 = M1 - M2 - M5 + M6 ... this doesn't seem right.
            
            # Let me use a known 8-multiplication scheme for 2x2 + 2x2.
            # Actually, the rank of the sum of two 2x2x2x2 tensors is at most 14.
            # Let me just use 14 per block: 2*3*14 = 84, same as before.
            
            # For rank 48, I need 8 per block. Let me try a different structure.
            pass
    
    # This approach is too complex without a known scheme. Return None.
    return None

def _try_rank_56(n, m, p, deadline):
    """Try rank-56: use ~9.33 multiplications per block average."""
    return None

def _try_rank_64(n, m, p, deadline):
    """Try rank-64: use ~10.67 multiplications per block average."""
    return None

def _try_rank_72(n, m, p, deadline):
    """Try rank-72: use 12 multiplications per block."""
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(2):
        for k in range(3):
            for j in range(2):
                a_idx = 2*i * m + 2*j
                b_idx = 2*i * m + 2*j + 1
                c_idx = (2*i+1) * m + 2*j
                d_idx = (2*i+1) * m + 2*j + 1

                e_idx = 2*j * p + 2*k
                f_idx = 2*j * p + 2*k + 1
                g_idx = (2*j+1) * p + 2*k
                h_idx = (2*j+1) * p + 2*k + 1

                c11_idx = (2*k) * n + 2*i
                c12_idx = (2*k+1) * n + 2*i
                c21_idx = (2*k) * n + 2*i + 1
                c22_idx = (2*k+1) * n + 2*i + 1

                # Use 12 multiplications (less than 7+7=14, more than 7)
                # M1 = (a+d)(e+h)
                U[a_idx][col] = 1; U[d_idx][col] = 1
                V[e_idx][col] = 1; V[h_idx][col] = 1
                W[c11_idx][col] = 1; W[c22_idx][col] = 1
                col += 1

                # M2 = (c+d)e
                U[c_idx][col] = 1; U[d_idx][col] = 1
                V[e_idx][col] = 1
                W[c21_idx][col] = 1; W[c22_idx][col] = -1
                col += 1

                # M3 = a(f-h)
                U[a_idx][col] = 1
                V[f_idx][col] = 1; V[h_idx][col] = -1
                W[c12_idx][col] = 1; W[c22_idx][col] = 1
                col += 1

                # M4 = d(g-e)
                U[d_idx][col] = 1
                V[g_idx][col] = 1; V[e_idx][col] = -1
                W[c11_idx][col] = 1; W[c21_idx][col] = 1
                col += 1

                # M5 = (a+b)h
                U[a_idx][col] = 1; U[b_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = -1; W[c12_idx][col] = 1
                col += 1

                # M6 = (c-a)(e+f)
                U[c_idx][col] = 1; U[a_idx][col] = -1
                V[e_idx][col] = 1; V[f_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1

                # M7 = (b-d)(g+h)
                U[b_idx][col] = 1; U[d_idx][col] = -1
                V[g_idx][col] = 1; V[h_idx][col] = 1
                W[c11_idx][col] = 1
                col += 1

                # M8 = (a+c)(e+g)
                U[a_idx][col] = 1; U[c_idx][col] = 1
                V[e_idx][col] = 1; V[g_idx][col] = 1
                W[c11_idx][col] = 1; W[c21_idx][col] = -1
                col += 1

                # M9 = (b+d)(f+h)
                U[b_idx][col] = 1; U[d_idx][col] = 1
                V[f_idx][col] = 1; V[h_idx][col] = 1
                W[c12_idx][col] = 1; W[c22_idx][col] = -1
                col += 1

                # M10 = (a-b)(e-g)
                U[a_idx][col] = 1; U[b_idx][col] = -1
                V[e_idx][col] = 1; V[g_idx][col] = -1
                W[c11_idx][col] = 1; W[c22_idx][col] = -1
                col += 1

                # M11 = (c-d)(f-h)
                U[c_idx][col] = 1; U[d_idx][col] = -1
                V[f_idx][col] = 1; V[h_idx][col] = -1
                W[c12_idx][col] = 1; W[c21_idx][col] = -1
                col += 1

                # M12 = (a-d)(e-h)
                U[a_idx][col] = 1; U[d_idx][col] = -1
                V[e_idx][col] = 1; V[h_idx][col] = -1
                W[c11_idx][col] = 1; W[c22_idx][col] = -1
                col += 1

    return {"U": U, "V": V, "W": W}

def _try_rank_80(n, m, p, deadline):
    """Try rank-80."""
    return None

def _random_search(n, m, p, target_rank, time_budget):
    """Random search for a decomposition of given rank."""
    start_time = time.time()
    np.random.seed(42)
    
    for attempt in range(20):
        if time.time() - start_time > time_budget:
            break
        
        U_arr = np.random.randn(n * m, target_rank) * 0.1
        V_arr = np.random.randn(m * p, target_rank) * 0.1
        W_arr = np.random.randn(p * n, target_rank) * 0.1
        
        for iteration in range(50):
            if time.time() - start_time > time_budget:
                break
            
            # Simple gradient step
            for idx in range(n * m):
                i, j = divmod(idx, m)
                for k in range(p):
                    v_row = V_arr[j * p + k]
                    w_row = W_arr[k * n + i]
                    contrib = np.sum(U_arr[idx] * v_row * w_row)
                    grad = (contrib - 1.0) * v_row * w_row
                    U_arr[idx] -= 0.005 * grad
            
            for idx in range(m * p):
                j, k = divmod(idx, p)
                for i in range(n):
                    u_row = U_arr[i * m + j]
                    w_row = W_arr[k * n + i]
                    contrib = np.sum(V_arr[idx] * u_row * w_row)
                    grad = (contrib - 1.0) * u_row * w_row
                    V_arr[idx] -= 0.005 * grad
            
            for idx in range(p * n):
                k, i = divmod(idx, n)
                for j in range(m):
                    u_row = U_arr[i * m + j]
                    v_row = V_arr[j * p + k]
                    contrib = np.sum(W_arr[idx] * u_row * v_row)
                    grad = (contrib - 1.0) * u_row * v_row
                    W_arr[idx] -= 0.005 * grad
            
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-6:
                U_list = [[float(x) for x in row] for row in U_arr]
                V_list = [[float(x) for x in row] for row in V_arr]
                W_list = [[float(x) for x in row] for row in W_arr]
                return U_list, V_list, W_list
    
    return None

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition."""
    rank = len(U[0]) if U else 0
    U_mat = np.array(U, dtype=float).T
    V_mat = np.array(V, dtype=float).T
    W_mat = np.array(W, dtype=float).T

    recon = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u_vec = U_mat[c].reshape(n, m)
        v_vec = V_mat[c].reshape(m, p)
        w_vec = W_mat[c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]

    if not np.allclose(recon, np.ones((n, m, p)), atol=1e-8):
        return False
    return True

def schoolbook(n, m, p):
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

def verify(U, V, W, n, m, p):
    return _verify(U, V, W, n, m, p)
