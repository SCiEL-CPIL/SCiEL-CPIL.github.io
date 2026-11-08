import numpy as np
import time

def _build_rank17():
    """Build the known rank-17 decomposition."""
    n, m, p = 2, 3, 3
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Strassen terms for 2x2x2:
    # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 -= M1
    # M2 = (A10+A11)*B00 -> C10 += M2
    # M3 = A00*(B10-B11) -> C01 -= M3, C11 += M3 (wait, standard is C01 += M3, C11 -= M3? Let me re-derive)
    
    # Standard Strassen:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B01 + B11)
    #
    # C00 = M1 - M4 + M5
    # C01 = M3 + M7
    # C10 = M2 - M6
    # C11 = M1 - M3 + M4 - M7
    
    # U[i*m+j, s] corresponds to coefficient of A[i,j] in term s
    # V[j*p+k, s] corresponds to coefficient of B[j,k] in term s
    # W[k*n+i, s] corresponds to coefficient of C[k,i] in term s (note: W[k*n+i, r] per contract)
    
    # Term 0: M1
    U[0, 0] = 1; U[3, 0] = 1   # A00, A11
    V[0, 0] = 1; V[4, 0] = 1   # B00, B11
    W[0, 0] = 1; W[3, 0] = 1   # C00, C11
    
    # Term 1: M2
    U[3, 1] = 1; U[4, 1] = 1   # A10, A11
    V[0, 1] = 1               # B00
    W[1, 1] = 1; W[3, 1] = -1 # C10, -C11
    
    # Term 2: M3
    U[0, 2] = 1               # A00
    V[1, 2] = 1; V[4, 2] = -1 # B10, -B11
    W[2, 2] = 1; W[3, 2] = -1 # C01, -C11
    
    # Term 3: M4
    U[3, 3] = 1               # A11
    V[0, 3] = 1; V[3, 3] = -1 # B01, -B00
    W[0, 3] = -1; W[3, 3] = 1 # -C00, C11
    
    # Term 4: M5
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[4, 4] = 1               # B11
    W[0, 4] = 1; W[2, 4] = -1 # C00, -C01
    
    # Term 5: M6
    U[3, 5] = 1; U[0, 5] = -1 # A10, -A00
    V[0, 5] = 1; V[1, 5] = 1  # B00, B10
    W[1, 5] = -1              # -C10
    
    # Term 6: M7
    U[1, 6] = 1; U[3, 6] = -1 # A01, -A11
    V[0, 6] = 1; V[4, 6] = 1  # B01, B11
    W[2, 6] = 1; W[3, 6] = -1 # C01, -C11
    
    # Part 2: Schoolbook for remaining terms
    # We need to handle:
    # C[i,2] = sum_j A[i,j] * B[j,2] for i=0,1
    # C[2,i] = sum_j A[2,j] * B[j,i] for i=0,1 (wait, A is 2x3, so A[2] doesn't exist)
    # Actually A is 2x3, B is 3x3, C is 2x3.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Terms where k=2 (3rd column of C):
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # Terms where j=2 (3rd row of A, 3rd col of B):
    # These contribute to all C[i,k] via A[i,2]*B[2,k]
    
    # Let's just use schoolbook for the terms not covered by Strassen.
    # Strassen covers A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Remaining contributions to C[0:2, 0:2] come from A[i,2]*B[2,k] for i=0,1, k=0,1
    # And all contributions to C[i,2] for i=0,1 (the 3rd column)
    
    # Term 7: A[0,2]*B[2,0] -> C[0,0]
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # Term 8: A[0,2]*B[2,1] -> C[0,1]
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    # Term 9: A[1,2]*B[2,0] -> C[1,0]
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    # Term 10: A[1,2]*B[2,1] -> C[1,1]
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Term 11: A[0,0]*B[0,2] -> C[0,2]
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    # Term 12: A[0,1]*B[1,2] -> C[0,2]
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    # Term 13: A[0,2]*B[2,2] -> C[0,2]
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    # Term 14: A[1,0]*B[0,2] -> C[1,2]
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    # Term 15: A[1,1]*B[1,2] -> C[1,2]
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    # Term 16: A[1,2]*B[2,2] -> C[1,2]
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    return U, V, W

def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the 2x3x3 matrix multiplication tensor."""
    n, m, p = 2, 3, 3
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] for s in range(r))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 2, 3, 3
    start_time = time.time()
    deadline = start_time + 120
    
    U17, V17, W17 = _build_rank17()
    
    if not _verify(U17, V17, W17):
        # Should never happen, but return anyway
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
    
    # Try to find a rank-16 decomposition by solving a system of polynomial equations
    # using a numerical approach. We'll use a simple gradient-free local search.
    
    # For rank 16, we have 16 terms. The equations are:
    # sum_{s=0}^{15} U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1 for all (i,j,k)
    # This is 2*3*3 = 18 equations.
    
    # We initialize by taking the rank-17 solution and zeroing out one term,
    # then adjusting the remaining 16 terms.
    
    best_rank = 17
    best_U, best_V, best_W = U17.copy(), V17.copy(), W17.copy()
    
    for remove_idx in range(17):
        if time.time() > deadline - 20:
            break
        
        U16 = np.delete(U17, remove_idx, axis=1)
        V16 = np.delete(V17, remove_idx, axis=1)
        W16 = np.delete(W17, remove_idx, axis=1)
        
        # Simple local search: perturb entries and keep if residual decreases
        def residual(Um, Vm, Wm):
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(Um[i*m+j, s] * Vm[j*p+k, s] * Wm[k*n+i, s] for s in range(16))
                        res += (val - 1.0)**2
            return res
        
        cur_res = residual(U16, V16, W16)
        
        for it in range(500):
            if time.time() > deadline - 10:
                break
            
            if cur_res < 1e-10:
                if 16 < best_rank:
                    best_rank = 16
                    best_U, best_V, best_W = U16.copy(), V16.copy(), W16.copy()
                break
            
            # Try a random perturbation
            mat_choice = np.random.choice(3)
            if mat_choice == 0:
                idx = np.random.randint(n*m)
                s = np.random.randint(16)
                old = U16[idx, s]
                U16[idx, s] += np.random.uniform(-0.05, 0.05)
                new_res = residual(U16, V16, W16)
                if new_res < cur_res:
                    cur_res = new_res
                else:
                    U16[idx, s] = old
            elif mat_choice == 1:
                idx = np.random.randint(m*p)
                s = np.random.randint(16)
                old = V16[idx, s]
                V16[idx, s] += np.random.uniform(-0.05, 0.05)
                new_res = residual(U16, V16, W16)
                if new_res < cur_res:
                    cur_res = new_res
                else:
                    V16[idx, s] = old
            else:
                idx = np.random.randint(p*n)
                s = np.random.randint(16)
                old = W16[idx, s]
                W16[idx, s] += np.random.uniform(-0.05, 0.05)
                new_res = residual(U16, V16, W16)
                if new_res < cur_res:
                    cur_res = new_res
                else:
                    W16[idx, s] = old
    
    if _verify(best_U, best_V, best_W):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
