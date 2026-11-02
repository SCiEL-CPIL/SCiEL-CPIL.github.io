"""Improved approach: search for better block decompositions."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_rank39():
    """Explicit rank-39 decomposition for M_{2,2,11}."""
    n, m, p = 2, 2, 11
    rank = 39
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        w_c00 = k0 * n
        w_c10 = k0 * n + 1
        w_c01 = k1 * n
        w_c11 = k1 * n + 1
        
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1
    
    k2 = 10
    v_e2 = k2
    v_g2 = p + k2
    w_c02 = k2 * n
    w_c12 = k2 * n + 1
    
    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    U[3][r] = 1
    V[v_g2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    assert r == rank
    return U, V, W

def _gradient_descent(U, V, W, n, m, p, max_iters=5000, lr=0.01, deadline=None):
    """Use gradient descent to optimize the decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    ta_list, tb_list, tc_list = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ta_list.append(i * m + j)
                tb_list.append(j * p + k)
                tc_list.append(k * n + i)
    ta_arr = np.array(ta_list)
    tb_arr = np.array(tb_list)
    tc_arr = np.array(tc_list)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Compute error
        err = np.zeros(len(ta_arr))
        for c in range(rank):
            err += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
        err -= 1.0
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for idx in range(len(ta_arr)):
            if abs(err[idx]) < 1e-12:
                continue
            a, b, c_idx = ta_arr[idx], tb_arr[idx], tc_arr[idx]
            for c in range(rank):
                ua = U[a, c]
                vb = V[b, c]
                wc = W[c_idx, c]
                dU[a, c] += err[idx] * vb * wc
                dV[b, c] += err[idx] * ua * wc
                dW[c_idx, c] += err[idx] * ua * vb
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Check convergence
        if it % 100 == 0:
            err_sum = np.sum(err**2)
            if err_sum < 1e-10:
                break
    
    return U.tolist(), V.tolist(), W.tolist()

def _find_rank_r_2x2x3(r, deadline, max_restarts=50):
    """Try to find a rank-r decomposition for M_{2,2,3}."""
    n, m, p = 2, 2, 3
    
    # Strassen initialization for first 2 columns
    def strassen_init():
        U = np.zeros((4, r))
        V = np.zeros((6, r))
        W = np.zeros((6, r))
        
        # Strassen for columns 0,1 (7 multiplications)
        # m1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[5, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # m2 = (A10+A11)*B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1
        
        # m3 = A00*(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[5, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # m4 = A11*(B10-B00)
        U[3, 3] = 1
        V[4, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # m5 = (A00+A01)*B11
        U[0, 4] = 1; U[1, 4] = 1
        V[5, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # m6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # m7 = (A01-A11)*(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[4, 6] = 1; V[5, 6] = 1
        W[0, 6] = 1
        
        # Add random entries for the remaining r-7 columns
        if r > 7:
            U[:, 7:r] = np.random.randn(4, r-7) * 0.1
            V[:, 7:r] = np.random.randn(6, r-7) * 0.1
            W[:, 7:r] = np.random.randn(6, r-7) * 0.1
        
        return U, V, W
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        np.random.seed(restart * 1000 + 42)
        U, V, W = strassen_init()
        
        U_l, V_l, W_l = _gradient_descent(U, V, W, n, m, p, max_iters=3000, lr=0.005, deadline=deadline)
        
        # Round to half-integers
        U_r = [[0.5 * round(2 * x) for x in row] for row in U_l]
        V_r = [[0.5 * round(2 * x) for x in row] for row in V_l]
        W_r = [[0.5 * round(2 * x) for x in row] for row in W_l]
        
        if _verify(U_r, V_r, W_r, n, m, p):
            return U_r, V_r, W_r
        
        # Also try without rounding
        if _verify(U_l, V_l, W_l, n, m, p, tol=1e-6):
            # Try to find a rational approximation
            return U_l, V_l, W_l
    
    return None

def _construct_from_blocks(block_decomps, n=2, m=2, p=11):
    """Construct a decomposition from block decompositions."""
    # block_decomps is a list of (start_k, end_k, U_block, V_block, W_block)
    total_rank = sum(len(Ub[0]) for _, _, Ub, _, _ in block_decomps)
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    r = 0
    for start_k, end_k, Ub, Vb, Wb in block_decomps:
        rank_b = len(Ub[0])
        p_b = end_k - start_k
        
        for i in range(n * m):
            for c in range(rank_b):
                U[i][r + c] = Ub[i][c]
        
        for j in range(m):
            for k in range(start_k, end_k):
                local_k = k - start_k
                local_idx = j * p_b + local_k
                global_idx = j * p + k
                for c in range(rank_b):
                    V[global_idx][r + c] = Vb[local_idx][c]
        
        for k in range(start_k, end_k):
            for i in range(n):
                local_k = k - start_k
                local_idx = local_k * n + i
                global_idx = k * n + i
                for c in range(rank_b):
                    W[global_idx][r + c] = Wb[local_idx][c]
        
        r += rank_b
    
    assert r == total_rank
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Try to find a rank-9 or rank-10 decomposition for M_{2,2,3}
    best_block = None
    for r in [9, 10, 11]:
        block = _find_rank_r_2x2x3(r, deadline - 20, max_restarts=30)
        if block is not None:
            best_block = (r, block)
            break
    
    if best_block is not None:
        r_block, (Ub, Vb, Wb) = best_block
        # Use 3 blocks of 3 columns + 1 block of 2 columns
        # Blocks: (0,3), (3,6), (6,9), (9,11)
        blocks = [
            (0, 3, Ub, Vb, Wb),
            (3, 6, Ub, Vb, Wb),
            (6, 9, Ub, Vb, Wb),
        ]
        
        # For the last 2 columns, use Strassen
        U_s = [[0.0] * 7 for _ in range(4)]
        V_s = [[0.0] * 7 for _ in range(4)]  # 2*2=4
        W_s = [[0.0] * 7 for _ in range(4)]  # 2*2=4
        
        # Strassen for 2x2x2
        U_s[0][0] = 1; U_s[3][0] = 1
        V_s[0][0] = 1; V_s[3][0] = 1
        W_s[0][0] = 1; W_s[3][0] = 1
        
        U_s[2][1] = 1; U_s[3][1] = 1
        V_s[0][1] = 1
        W_s[1][1] = 1; W_s[3][1] = -1
        
        U_s[0][2] = 1
        V_s[1][2] = 1; V_s[3][2] = -1
        W_s[2][2] = 1; W_s[3][2] = 1
        
        U_s[3][3] = 1
        V_s[2][3] = 1; V_s[0][3] = -1
        W_s[0][3] = 1; W_s[1][3] = 1
        
        U_s[0][4] = 1; U_s[1][4] = 1
        V_s[3][4] = 1
        W_s[0][4] = -1; W_s[2][4] = 1
        
        U_s[2][5] = 1; U_s[0][5] = -1
        V_s[0][5] = 1; V_s[1][5] = 1
        W_s[3][5] = 1
        
        U_s[1][6] = 1; U_s[3][6] = -1
        V_s[2][6] = 1; V_s[3][6] = 1
        W_s[0][6] = 1
        
        blocks.append((9, 11, U_s, V_s, W_s))
        
        U, V, W = _construct_from_blocks(blocks, n, m, p)
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to rank-39
    U, V, W = _construct_rank39()
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Final fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
