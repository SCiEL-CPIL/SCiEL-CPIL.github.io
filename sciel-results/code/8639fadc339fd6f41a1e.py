import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Rank 56 decomposition using Strassen for 2x2x2 parts
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def get_strassen_terms():
        # Returns list of 7 terms for 2x2x2 Strassen.
        # U: {(i, j_low): coeff} for A[i, j]
        # V: {(j_low, k_low): coeff} for B[j, k]  
        # W: {(k_low, i): coeff} for C[i, k] (trace convention W[k*n+i])
        
        # Standard Strassen for 2x2:
        # M0 = (A00+A11)(B00+B11)
        # M1 = (A10+A11)B00
        # M2 = A00(B01-B11)
        # M3 = A11(B10-B00)
        # M4 = (A00+A01)B11
        # M5 = (A10-A00)(B00+B01)
        # M6 = (A01-A11)(B10+B11)
        # C00 = M0 + M3 - M4 + M6
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M1 + M2 + M5
        
        terms = []
        
        # M0 = (A00 + A11)(B00 + B11)
        terms.append({'U': {(0,0):1, (1,1):1}, 'V': {(0,0):1, (1,1):1}, 'W': {(0,0):1, (1,1):1}})
        
        # M1 = (A10 + A11)B00
        terms.append({'U': {(1,0):1, (1,1):1}, 'V': {(0,0):1}, 'W': {(0,1):1, (1,1):-1}})
        
        # M2 = A00(B01 - B11)
        terms.append({'U': {(0,0):1}, 'V': {(0,1):1, (1,1):-1}, 'W': {(1,0):1, (1,1):1}})
        
        # M3 = A11(B10 - B00)
        terms.append({'U': {(1,1):1}, 'V': {(1,0):1, (0,0):-1}, 'W': {(0,0):1, (0,1):1}})
        
        # M4 = (A00 + A01)B11
        terms.append({'U': {(0,0):1, (0,1):1}, 'V': {(1,1):1}, 'W': {(0,0):-1, (1,0):1}})
        
        # M5 = (A10 - A00)(B00 + B01)
        terms.append({'U': {(1,0):1, (0,0):-1}, 'V': {(0,0):1, (0,1):1}, 'W': {(1,1):1}})
        
        # M6 = (A01 - A11)(B10 + B11)
        terms.append({'U': {(0,1):1, (1,1):-1}, 'V': {(1,0):1, (1,1):1}, 'W': {(0,0):1}})
        
        return terms

    terms = get_strassen_terms()
    
    col = 0
    for j_high in range(2):
        for l in range(4):
            if time.time() > deadline:
                break
            for t_idx in range(7):
                term = terms[t_idx]
                
                for (i, j_low), coeff in term['U'].items():
                    j = j_high * 2 + j_low
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                for (j_low, k_low), coeff in term['V'].items():
                    j = j_high * 2 + j_low
                    k = l * 2 + k_low
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                for (k_low, i), coeff in term['W'].items():
                    k = l * 2 + k_low
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                    
                col += 1
                
    # Verify the decomposition
    if time.time() < deadline:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal delta(i,j,k)
        # where delta is 1 if (i,j,k) is a valid index in the tensor
        # The tensor T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p)
        # Actually for matrix mult: T[i,j,k] = 1 for all valid (i,j,k)
        # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        # Sample check with random indices
        np.random.seed(42)
        test_ok = True
        for _ in range(100):
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            k = np.random.randint(0, p)
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            val = np.sum(U_arr[u_idx, :] * V_arr[v_idx, :] * W_arr[w_idx, :])
            if abs(val - 1.0) > 1e-9:
                test_ok = False
                break
        
        if not test_ok:
            # Fallback: return naive rank 64
            rank_naive = n * m * p
            U_naive = [[0] * rank_naive for _ in range(n * m)]
            V_naive = [[0] * rank_naive for _ in range(m * p)]
            W_naive = [[0] * rank_naive for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i * m * p + j * p + k
                        U_naive[i * m + j][r] = 1
                        V_naive[j * p + k][r] = 1
                        W_naive[k * n + i][r] = 1
            return {"U": U_naive, "V": V_naive, "W": W_naive}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
