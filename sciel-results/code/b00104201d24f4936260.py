import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 9]
_start_time = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T[i*m+j, j*p+k, k*n+i] = 1
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if valid index, else 0
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Expected: 1 if this is a valid product term
                # C[k*n+i] gets contribution A[i*m+j] * B[j*p+k]
                expected = 1.0
                actual = np.dot(U_arr[i*m+j], V_arr[j*p+k])
                # Wait, that's not right. The tensor is:
                # T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
                # T[a,b,c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
                # Otherwise T[a,b,c] = 0
                # The decomposition is: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                
                # Let's check differently: for each (i,j,k), the product A[i,j]*B[j,k] contributes to C[i,k]
                # So T[i*m+j, j*p+k, k*n+i] = 1
                # And T[a,b,c] = 0 for all other (a,b,c)
                
                # Actually, let me just verify by checking all entries
                pass
    
    # More efficient verification:
    # For each r, the rank-1 tensor is U[:,r] outer V[:,r] outer W[:,r]
    # Sum over r should equal the target tensor
    
    # Target tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # i.e., a % m == b // p and b % p == c // n and c % n == a // m
    # and a // m == c // n and b // p == a % m
    
    errors = 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Check if this is a valid triple
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                
                is_valid = (i_a == i_c) and (j_a == j_b) and (k_b == k_c)
                expected = 1.0 if is_valid else 0.0
                
                actual = 0.0
                for r in range(rank):
                    actual += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
                
                if abs(actual - expected) > 1e-9:
                    errors += 1
                    if errors > 10:
                        return False
    
    return errors == 0

def _rank50_baseline():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 4 blocks of 2x2
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
    
    k8 = 8
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    
    u_a = 0 * m + 0
    u_b = 0 * m + 1
    u_c = 1 * m + 0
    u_d = 1 * m + 1
    
    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    u_ar0 = 0 * m + 2
    u_ar1 = 1 * m + 2
    
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2
    
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U, V, W, n, m, p, max_iter=1000):
    """Try to reduce rank by finding zero columns or combining."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Try to find columns that can be removed (zero or redundant)
    # This is complex; instead try a simple approach: check if any column is all zeros
    for r in range(rank):
        if np.allclose(U_arr[:, r], 0) and np.allclose(V_arr[:, r], 0) and np.allclose(W_arr[:, r], 0):
            # Remove this column
            U_arr = np.delete(U_arr, r, axis=1)
            V_arr = np.delete(V_arr, r, axis=1)
            W_arr = np.delete(W_arr, r, axis=1)
            rank -= 1
            r -= 1
    
    # Try random perturbations and rounding
    best_rank = rank
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    for _ in range(max_iter):
        if time.time() - _start_time > TIME_LIMIT - 5:
            break
        
        # Try small random perturbations
        U_pert = best_U.copy()
        V_pert = best_V.copy()
        W_pert = best_W.copy()
        
        # Randomly perturb a few entries
        for _ in range(3):
            mat = random.choice([U_pert, V_pert, W_pert])
            i, j = random.randint(0, mat.shape[0]-1), random.randint(0, mat.shape[1]-1)
            mat[i, j] += random.choice([-1, 1]) * 0.5
        
        # Round to nearest half-integer
        U_round = np.round(U_pert * 2) / 2
        V_round = np.round(V_pert * 2) / 2
        W_round = np.round(W_pert * 2) / 2
        
        # Verify
        if _verify_fast(U_round, V_round, W_round, n, m, p):
            # Check if we can reduce rank by removing zero columns
            new_rank = U_round.shape[1]
            for r in range(new_rank):
                if np.allclose(U_round[:, r], 0) and np.allclose(V_round[:, r], 0) and np.allclose(W_round[:, r], 0):
                    U_round = np.delete(U_round, r, axis=1)
                    V_round = np.delete(V_round, r, axis=1)
                    W_round = np.delete(W_round, r, axis=1)
                    new_rank -= 1
                    r -= 1
            
            if new_rank < best_rank:
                best_rank = new_rank
                best_U, best_V, best_W = U_round.copy(), V_round.copy(), W_round.copy()
                if best_rank <= 45:  # Early stop if we've made good progress
                    break
    
    return best_U, best_V, best_W, best_rank

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using tensor operations."""
    rank = U.shape[1]
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Target: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_target[a, b, c] = 1.0
    
    # Compute decomposition
    # T_decomp[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # This is a tensor contraction
    
    # Efficient way: for each r, add outer product
    T_decomp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    return np.allclose(T_decomp, T_target, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    result = _rank50_baseline()
    
    if time.time() - _start_time > TIME_LIMIT - 10:
        return result
    
    # Try to reduce rank
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    U_red, V_red, W_red, new_rank = _try_reduce_rank(U, V, W, n, m, p, max_iter=500)
    
    if new_rank < 50:
        return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
    
    return result
    n, m, p = DIMENSIONS
    # Rank 50 decomposition
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: i*m+j
    # B indices: j*p+k
    # C indices: k*n+i
    
    # Part 1: A_L (2x2) * B_T (2x9)
    # A_L uses A_{00}, A_{01}, A_{10}, A_{11} -> U indices 0, 1, 3, 4
    # B_T uses B_{0k}, B_{1k} -> V indices 0..8, 9..17
    
    col = 0
    # 4 blocks of 2x2
    for b in range(4):
        # B columns 2b, 2b+1
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # B indices
        v_e = 0 * p + k0 # B_{0, k0}
        v_f = 0 * p + k1 # B_{0, k1}
        v_g = 1 * p + k0 # B_{1, k0}
        v_h = 1 * p + k1 # B_{1, k1}
        
        # A indices
        u_a = 0 * m + 0 # A_{00}
        u_b = 0 * m + 1 # A_{01}
        u_c = 1 * m + 0 # A_{10}
        u_d = 1 * m + 1 # A_{11}
        
        # Strassen Products
        # M1 = (a+d)(e+h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        # M2 = (c+d)e
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        # M3 = a(f-h)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        # M4 = d(g-e)
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        # M5 = (a+b)h
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        # M6 = (c-a)(e+f)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        # M7 = (b-d)(g+h)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        # Output C_{0, k0}, C_{0, k1}, C_{1, k0}, C_{1, k1}
        # W indices: k*n+i
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # C00 = M1 + M4 - M5 + M7
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        # C01 = M3 + M5
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        # C10 = M2 + M4
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        # C11 = M1 - M2 + M3 + M6
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
        
    # 1 block of 2x1 (last column k=8)
    k8 = 8
    v_e = 0 * p + k8 # B_{0, 8}
    v_g = 1 * p + k8 # B_{1, 8}
    
    u_a = 0 * m + 0
    u_b = 0 * m + 1
    u_c = 1 * m + 0
    u_d = 1 * m + 1
    
    # P1 = a e
    U[u_a][col] += 1
    V[v_e][col] += 1
    # P2 = b g
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    # P3 = c e
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    # P4 = d g
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    # C08 = P1 + P2
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    
    # C18 = P3 + P4
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    # Part 2: A_R (2x1) * B_B (1x9)
    # A_R uses A_{02}, A_{12} -> U indices 2, 5
    # B_B uses B_{2k} -> V indices 18..26
    
    u_ar0 = 0 * m + 2 # A_{02}
    u_ar1 = 1 * m + 2 # A_{12}
    
    for k in range(9):
        v_b = 2 * p + k # B_{2, k}
        
        # Q_k = A_{02} B_{2k}
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        # R_k = A_{12} B_{2k}
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2
        
    return {"U": U, "V": V, "W": W}
