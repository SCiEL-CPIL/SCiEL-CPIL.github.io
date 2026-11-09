"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_3x3x3():
    """
    Return (U, V, W) for 3x3 x 3x3 matrix multiplication, rank 7.
    U: (9, 7) - linear combos of A entries, indexed by (i,j) -> row i*3+j
    V: (9, 7) - linear combos of B entries, indexed by (j,k) -> row j*3+k
    W: (9, 7) - linear combos giving C entries, indexed by (k,i) -> row k*3+i (trace convention W[k*n+i])
    """
    n = 3
    U = [[0]*7 for _ in range(9)]
    V = [[0]*7 for _ in range(9)]
    W = [[0]*7 for _ in range(9)]
    
    # M1: (A00+A22)(B00+B22)
    U[0][0] = 1; U[8][0] = 1
    V[0][0] = 1; V[8][0] = 1
    # M2: (A20+A22)B00
    U[6][1] = 1; U[8][1] = 1
    V[0][1] = 1
    # M3: A00(B01-B21)
    U[0][2] = 1
    V[1][2] = 1; V[7][2] = -1
    # M4: A22(B21-B01)
    U[8][3] = 1
    V[7][3] = 1; V[1][3] = -1
    # M5: (A00+A01)B21
    U[0][4] = 1; U[1][4] = 1
    V[7][4] = 1
    # M6: (A10+A20)(B00+B10)
    U[3][5] = 1; U[6][5] = 1
    V[0][5] = 1; V[3][5] = 1
    # M7: (A01+A11)(B10+B20)
    U[1][6] = 1; U[4][6] = 1
    V[3][6] = 1; V[6][6] = 1
    
    # W: C[i][k], W row = k*3+i
    W[0][0] = 1; W[0][1] = -1; W[0][2] = 1; W[0][3] = 1
    W[3][2] = 1; W[3][4] = 1
    W[6][6] = 1
    W[1][5] = 1; W[1][6] = 1
    W[4][0] = 1; W[4][2] = -1; W[4][4] = 1; W[4][6] = -1
    W[7][3] = 1; W[7][5] = 1
    W[2][1] = 1; W[2][3] = 1
    W[5][0] = 1; W[5][1] = -1; W[5][3] = 1
    W[8][0] = 1; W[8][4] = -1; W[8][5] = 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, verbose=False):
    """Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k} for all i,j,k."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                expected = 1.0 if i == k else 0.0
                err = abs(s - expected)
                if err > max_err:
                    max_err = err
                if verbose and err > 1e-10:
                    print(f"  ERROR: i={i},j={j},k={k}: got {s}, expected {expected}")
    return max_err


def _construct_rank35():
    """Construct rank-35 decomposition using 5 independent Strassen blocks."""
    n, m, p = 3, 3, 5
    U_str, V_str, W_str = _strassen_3x3x3()
    rank = 5 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col_out in range(p):
        base = col_out * 7
        for c in range(7):
            for i in range(n):
                for j in range(m):
                    U[i*m + j][base + c] = U_str[i*m + j][c]
            for j in range(m):
                val = V_str[j*3 + 0][c]
                if val != 0:
                    V[j*p + col_out][base + c] += val
            for i in range(n):
                val = W_str[0*3 + i][c]
                if val != 0:
                    W[col_out*n + i][base + c] += val
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    t0 = time.time()
    deadline = t0 + 120.0
    
    # Start with rank-35 baseline
    U, V, W = _construct_rank35()
    max_err = _verify_decomposition(U, V, W, n, m, p)
    if max_err > 1e-10:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    best_rank = rank
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find lower-rank decomposition via numerical search
    # The tensor T[i,j,k] = delta_{i,k} lives in R^{n*m} x R^{m*p} x R^{p*n}
    # We seek rank r such that T = sum_c u_c v_c w_c (outer products)
    
    # Strategy: use alternating least squares / tensor train style approach
    # For small ranks, try random initialization and iterative refinement
    
    target_ranks = [20, 25, 30]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 30.0:
            break
        
        found = False
        for trial in range(20):
            if time.time() > deadline - 30.0:
                break
            
            # Initialize with random rank-1 components
            np.random.seed(42 + trial * 1000 + target_rank)
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)
            
            # Alternating minimization: fix two, solve for third
            converged = False
            for it in range(200):
                if time.time() > deadline - 30.0:
                    break
                
                # Solve for U: for each c, U[:,c] should satisfy
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{i,k}
                # For fixed c, this is a linear system in U[:,c]
                # Actually, let's use a different approach: gradient descent on the loss
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U_rand[i*m+j, c] * V_rand[j*p+k, c] * W_rand[k*n+i, c]
                            expected = 1.0 if i == k else 0.0
                            loss += (s - expected) ** 2
                
                if loss < 1e-12:
                    converged = True
                    break
                
                # Gradient step
                lr = 0.01
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U_rand[i*m+j, c] * V_rand[j*p+k, c] * W_rand[k*n+i, c]
                            expected = 1.0 if i == k else 0.0
                            diff = s - expected
                            for c in range(target_rank):
                                U_rand[i*m+j, c] -= lr * diff * V_rand[j*p+k, c] * W_rand[k*n+i, c]
                                V_rand[j*p+k, c] -= lr * diff * U_rand[i*m+j, c] * W_rand[k*n+i, c]
                                W_rand[k*n+i, c] -= lr * diff * U_rand[i*m+j, c] * V_rand[j*p+k, c]
            
            if converged:
                # Verify with exact check
                max_err_check = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U_rand[i*m+j, c] * V_rand[j*p+k, c] * W_rand[k*n+i, c]
                            expected = 1.0 if i == k else 0.0
                            max_err_check = max(max_err_check, abs(s - expected))
                if max_err_check < 1e-10:
                    # Round to nearest half-integer if possible
                    U_int = np.round(U_rand * 2) / 2
                    V_int = np.round(V_rand * 2) / 2
                    W_int = np.round(W_rand * 2) / 2
                    
                    # Verify rounded version
                    max_err_rounded = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for c in range(target_rank):
                                    s += U_int[i*m+j, c] * V_int[j*p+k, c] * W_int[k*n+i, c]
                                expected = 1.0 if i == k else 0.0
                                max_err_rounded = max(max_err_rounded, abs(s - expected))
                    
                    if max_err_rounded < 1e-10:
                        U_list = U_int.tolist()
                        V_list = V_int.tolist()
                        W_list = W_int.tolist()
                        best_rank = target_rank
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        found = True
                        break
                    else:
                        # Use unrounded
                        U_list = U_rand.tolist()
                        V_list = V_rand.tolist()
                        W_list = W_rand.tolist()
                        best_rank = target_rank
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        found = True
                        break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
