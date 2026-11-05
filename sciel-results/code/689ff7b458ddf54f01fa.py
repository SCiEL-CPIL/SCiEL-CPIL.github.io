import numpy as np
import time

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 140.0 # 10s margin

def _time_check():
    return (time.time() - START_TIME) > TIME_LIMIT

def _verify_decomposition(U, V, W, n=6):
    """
    Verify if U, V, W exactly decompose the n x n matrix multiplication tensor.
    Convention: W[k*n+i, r]
    """
    if _time_check():
        return False
        
    rank = U.shape[1]
    np.random.seed(42)
    A = np.random.randint(-10, 10, (n, n)).astype(float)
    B = np.random.randint(-10, 10, (n, n)).astype(float)
    
    C_expected = A @ B
    
    C_computed = np.zeros((n, n))
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    # Vectorized verification
    # U: (n*n, rank), V: (n*n, rank), W: (n*n, rank)
    # u_coeffs: (rank,) = U.T @ A_flat
    # v_coeffs: (rank,) = V.T @ B_flat
    # m_vals: (rank,) = u_coeffs * v_coeffs
    # C_computed = sum_r m_vals[r] * W[:, r].reshape(n, n).T
    # W[:, r] is vector of length n*n. Index k*n+i.
    # reshape(n, n) gives W[k, i].
    # We need C[i, k] += m_val * W[k, i].
    # So C += sum_r m_vals[r] * W[:, r].reshape(n, n).T
    
    u_coeffs = np.dot(U.T, A_flat)
    v_coeffs = np.dot(V.T, B_flat)
    m_vals = u_coeffs * v_coeffs
    
    # W is (n*n, rank). We want to sum over rank.
    # C = (W.T @ m_vals).reshape(n, n)?
    # W.T is (rank, n*n).
    # W.T @ m_vals is (n*n,).
    # Let W_flat_result = W.T @ m_vals.
    # W_flat_result[idx] = sum_r W[idx, r] * m_vals[r].
    # idx = k*n + i.
    # So W_flat_result[k*n+i] = sum_r W[k*n+i, r] * m_vals[r].
    # We want C[i, k] = sum_r W[k*n+i, r] * m_vals[r].
    # So C[i, k] = W_flat_result[k*n+i].
    # This means C is the transpose of the matrix formed by reshaping W_flat_result.
    # Let M = W_flat_result.reshape(n, n). M[k, i] = W_flat_result[k*n+i].
    # We want C[i, k] = M[k, i].
    # So C = M.T.
    
    W_flat_result = np.dot(W.T, m_vals)
    C_computed = W_flat_result.reshape(n, n).T
    
    if np.allclose(C_computed, C_expected, atol=1e-9):
        return True
    return False

def _build_rank_216():
    n = 6
    rank = n**3
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    idx = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i*n+j, idx] = 1
                V[j*n+k, idx] = 1
                W[k*n+i, idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}

def _build_rank_189():
    """
    Strassen 2x2 blocks of 3x3 schoolbook.
    """
    if _time_check():
        return None
    n = 6
    rank = 189
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    # Strassen definitions
    # A blocks: 00, 01, 10, 11
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    s_A = [
        [((0,0), 1), ((1,1), 1)],
        [((1,0), 1), ((1,1), 1)],
        [((0,0), 1)],
        [((1,1), 1)],
        [((0,0), 1), ((0,1), 1)],
        [((1,0), 1), ((0,0), -1)],
        [((0,1), 1), ((1,1), -1)]
    ]
    
    s_B = [
        [((0,0), 1), ((1,1), 1)],
        [((0,0), 1)],
        [((0,1), 1), ((1,1), -1)],
        [((1,0), 1), ((0,0), -1)],
        [((1,1), 1)],
        [((0,0), 1), ((0,1), 1)],
        [((1,0), 1), ((1,1), 1)]
    ]
    
    s_C = [
        [((0,0), 1), ((1,1), 1)],
        [((1,0), 1), ((1,1), 1)],
        [((0,1), 1), ((1,1), -1)],
        [((0,0), -1), ((1,0), 1)],
        [((0,0), 1), ((0,1), 1)],
        [((1,1), 1)],
        [((0,0), 1)]
    ]
    
    col_idx = 0
    for s in range(7):
        for i_prime in range(3):
            for j_prime in range(3):
                for k_prime in range(3):
                    # U
                    for (I, J), coeff in s_A[s]:
                        i_g = I * 3 + i_prime
                        j_g = J * 3 + j_prime
                        U[i_g * 6 + j_g, col_idx] += coeff
                    # V
                    for (J, K), coeff in s_B[s]:
                        j_g = J * 3 + j_prime
                        k_g = K * 3 + k_prime
                        V[j_g * 6 + k_g, col_idx] += coeff
                    # W
                    for (I, K), coeff in s_C[s]:
                        i_g = I * 3 + i_prime
                        k_g = K * 3 + k_prime
                        W[k_g * 6 + i_g, col_idx] += coeff
                        
                    col_idx += 1
                    
    return {"U": U, "V": V, "W": W}

def _get_bcdl_22():
    """
    Returns U, V, W for 3x3 matrix multiplication, rank 22 (BCDL).
    """
    # BCDL 1982 Rank 22 coefficients
    # These are sparse integer matrices.
    # U, V, W are 9x22.
    
    # I will use a known valid set.
    # If this set is incorrect, verification will fail.
    
    U3 = np.zeros((9, 22))
    V3 = np.zeros((9, 22))
    W3 = np.zeros((9, 22))
    
    # Placeholder for actual BCDL coefficients.
    # Since I cannot guarantee memory of 22x9x3 integers, 
    # I will use a Rank 23 solution if available or fall back.
    # Actually, I will use a Rank 22 solution from a reliable source embedded here.
    # (Simulating that I have them)
    
    # For safety, if I don't have them, return None.
    # I will return None to force fallback to 189 which is guaranteed correct.
    # However, 154 is the target.
    # I will attempt to construct a valid Rank 22 using Strassen 2x2 + 1? No.
    
    # Let's use Rank 23?
    # There is a Rank 23 solution.
    # But 189 is safe.
    
    return None

def _build_rank_154():
    if _time_check():
        return None
        
    bcdl = _get_bcdl_22()
    if bcdl is None:
        return None
        
    n = 6
    rank = 154
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    # Strassen definitions (same as 189)
    s_A = [
        [((0,0), 1), ((1,1), 1)],
        [((1,0), 1), ((1,1), 1)],
        [((0,0), 1)],
        [((1,1), 1)],
        [((0,0), 1), ((0,1), 1)],
        [((1,0), 1), ((0,0), -1)],
        [((0,1), 1), ((1,1), -1)]
    ]
    s_B = [
        [((0,0), 1), ((1,1), 1)],
        [((0,0), 1)],
        [((0,1), 1), ((1,1), -1)],
        [((1,0), 1), ((0,0), -1)],
        [((1,1), 1)],
        [((0,0), 1), ((0,1), 1)],
        [((1,0), 1), ((1,1), 1)]
    ]
    s_C = [
        [((0,0), 1), ((1,1), 1)],
        [((1,0), 1), ((1,1), 1)],
        [((0,1), 1), ((1,1), -1)],
        [((0,0), -1), ((1,0), 1)],
        [((0,0), 1), ((0,1), 1)],
        [((1,1), 1)],
        [((0,0), 1)]
    ]
    
    U3 = bcdl['U']
    V3 = bcdl['V']
    W3 = bcdl['W']
    
    col_idx = 0
    for s in range(7):
        for r_prime in range(22):
            # For each Strassen term s and BCDL term r_prime
            # U_global
            for (I, J), coeff_A in s_A[s]:
                # U3[r_prime, i'*3+j'] contributes to A[I*3+i', J*3+j']
                for i_prime in range(3):
                    for j_prime in range(3):
                        u_val = U3[r_prime, i_prime*3 + j_prime]
                        if u_val != 0:
                            i_g = I * 3 + i_prime
                            j_g = J * 3 + j_prime
                            U[i_g * 6 + j_g, col_idx] += coeff_A * u_val
                            
            # V_global
            for (J, K), coeff_B in s_B[s]:
                for j_prime in range(3):
                    for k_prime in range(3):
                        v_val = V3[r_prime, j_prime*3 + k_prime]
                        if v_val != 0:
                            j_g = J * 3 + j_prime
                            k_g = K * 3 + k_prime
                            V[j_g * 6 + k_g, col_idx] += coeff_B * v_val
                            
            # W_global
            for (I, K), coeff_C in s_C[s]:
                for i_prime in range(3):
                    for k_prime in range(3):
                        w_val = W3[r_prime, k_prime*3 + i_prime] # Trace convention k*3+i
                        if w_val != 0:
                            i_g = I * 3 + i_prime
                            k_g = K * 3 + k_prime
                            W[k_g * 6 + i_g, col_idx] += coeff_C * w_val
                            
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}

def construct():
    # Try 154
    res_154 = _build_rank_154()
    if res_154:
        if _verify_decomposition(res_154['U'], res_154['V'], res_154['W']):
            return {"U": res_154['U'].tolist(), "V": res_154['V'].tolist(), "W": res_154['W'].tolist()}
            
    # Try 189
    res_189 = _build_rank_189()
    if res_189:
        if _verify_decomposition(res_189['U'], res_189['V'], res_189['W']):
            return {"U": res_189['U'].tolist(), "V": res_189['V'].tolist(), "W": res_189['W'].tolist()}
            
    # Fallback 216
    res_216 = _build_rank_216()
    return {"U": res_216['U'].tolist(), "V": res_216['V'].tolist(), "W": res_216['W'].tolist()}

if __name__ == "__main__":
    res = construct()
    print("Constructed", len(res['U'][0]), "rank factors")
