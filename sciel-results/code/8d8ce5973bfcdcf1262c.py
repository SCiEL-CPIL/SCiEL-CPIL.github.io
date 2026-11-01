import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank_schoolbook

    def calculate_error(U, V, W):
        """
        Calculate the sum of squared errors between the reconstructed tensor and the target tensor (all 1s).
        U: (n*m, R), V: (m*p, R), W: (p*n, R)
        """
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        try:
            R = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
            error = np.sum((R - 1.0) ** 2)
        except:
            R = np.zeros((n, m, p))
            for c in range(U.shape[1]):
                uc = U_reshaped[:, :, c]
                vc = V_reshaped[:, :, c]
                wc = W_reshaped[:, :, c]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            R[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
            error = np.sum((R - 1.0) ** 2)
        return error

    def verify_exact(U, V, W):
        """
        Verify that the decomposition is exactly correct using integer/half-integer logic.
        Returns True if exact, False otherwise.
        """
        # Convert to Fractions for exact arithmetic
        U_frac = [[Fraction(U[i, j]).limit_denominator(10) for j in range(U.shape[1])] for i in range(U.shape[0])]
        V_frac = [[Fraction(V[i, j]).limit_denominator(10) for j in range(V.shape[1])] for i in range(V.shape[0])]
        W_frac = [[Fraction(W[i, j]).limit_denominator(10) for j in range(W.shape[0])] for i in range(W.shape[0])]
        
        U_r = np.array(U_frac).reshape(n, m, -1)
        V_r = np.array(V_frac).reshape(m, p, -1)
        W_r = np.array(W_frac).reshape(p, n, -1)
        
        R = np.zeros((n, m, p), dtype=object)
        for c in range(U.shape[1]):
            uc = U_r[:, :, c]
            vc = V_r[:, :, c]
            wc = W_r[:, :, c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        R[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
                        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if R[i, j, k] != 1:
                        return False
        return True

    def find_decomposition_for_p(p_val, max_rank, time_limit=10.0):
        """
        Attempt to find a rank-minimal decomposition for M_{2, 2, p_val} using a structured search.
        We exploit the structure:
        U[i,j,c] depends on i,j.
        V[j,k,c] depends on j,k.
        W[k,i,c] depends on k,i.
        
        For small p, we can use a heuristic grid search or specific algebraic patterns.
        Here we try a simple pattern based on Strassen-like additions for the k dimension.
        """
        if time.time() > time_limit:
            return None
            
        # Schoolbook rank for this block
        rank_sb = n * m * p_val
        
        # Try ranks from theoretical lower bound up to schoolbook
        # Lower bound is max(n*m, m*p, p*n) = max(4, 2*p, 2*p) = 2*p
        # For p=8, LB=16. For p=7, LB=14.
        # Known: M_{2,2,2} rank 7. M_{2,2,4} rank 13? M_{2,2,8} rank 21?
        # Let's just try to find *any* rank lower than schoolbook.
        
        # Strategy: Use random initialization + local search with rational snapping
        # But for small p, we can be more exhaustive or use specific templates.
        
        # Template: 
        # Let's try to construct a solution with rank R.
        # We will use a simple local search on integer/half-integer values.
        
        best_rank_found = rank_sb
        best_U = None
        best_V = None
        best_W = None
        
        # We'll try ranks from min(2*p_val + 2, rank_sb) down to 2*p_val
        # Actually, let's just try a few specific ranks known to be promising or just search for improvement.
        
        # For p=8, schoolbook is 32.
        # For p=7, schoolbook is 28.
        
        # Let's try a simple optimization:
        # Initialize with random half-integers.
        # Perform simulated annealing on the discrete space of {-1, -0.5, 0, 0.5, 1}.
        
        candidates = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        for target_rank in range(max(2 * p_val, 10), rank_sb + 1):
            if time.time() > time_limit:
                break
                
            # Try a few random restarts for this rank
            for restart in range(5):
                if time.time() > time_limit:
                    break
                    
                U = np.random.choice(candidates, size=(n * m, target_rank))
                V = np.random.choice(candidates, size=(m * p_val, target_rank))
                W = np.random.choice(candidates, size=(p_val * n, target_rank))
                
                # Local search: flip random entries
                current_error = calculate_error(U, V, W)
                best_local_error = current_error
                best_local_U = U.copy()
                best_local_V = V.copy()
                best_local_W = W.copy()
                
                steps = 200
                for s in range(steps):
                    if time.time() > time_limit:
                        break
                    
                    # Pick random matrix and entry
                    mat_idx = np.random.randint(3)
                    if mat_idx == 0:
                        mat = U
                        rows, cols = n * m, target_rank
                    elif mat_idx == 1:
                        mat = V
                        rows, cols = m * p_val, target_rank
                    else:
                        mat = W
                        rows, cols = p_val * n, target_rank
                        
                    r = np.random.randint(rows)
                    c = np.random.randint(cols)
                    
                    old_val = mat[r, c]
                    # Try a different candidate
                    new_val = np.random.choice(candidates)
                    if new_val == old_val:
                        continue
                        
                    mat[r, c] = new_val
                    new_error = calculate_error(U, V, W)
                    
                    if new_error < best_local_error:
                        best_local_error = new_error
                        best_local_U = U.copy()
                        best_local_V = V.copy()
                        best_local_W = W.copy()
                        
                    if new_error < current_error:
                        current_error = new_error
                    else:
                        mat[r, c] = old_val
                        
                if best_local_error < 1e-9:
                    # Check exactness
                    if verify_exact(best_local_U, best_local_V, best_local_W):
                        return {
                            "U": best_local_U,
                            "V": best_local_V,
                            "W": best_local_W,
                            "rank": target_rank
                        }
                        
        return None

    # Decompose 2x15 into 2x8 and 2x7
    p1 = 8
    p2 = 7
    
    # Find decomposition for 2x8
    res1 = find_decomposition_for_p(p1, 40, time_limit=60.0)
    if res1 is None:
        # Fallback to schoolbook for p1
        U1 = np.zeros((n * m, n * m * p1))
        V1 = np.zeros((m * p1, n * m * p1))
        W1 = np.zeros((p1 * n, n * m * p1))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p1):
                    U1[i * m + j, col] = 1
                    V1[j * p1 + k, col] = 1
                    W1[k * n + i, col] = 1
                    col += 1
        res1 = {"U": U1, "V": V1, "W": W1, "rank": n * m * p1}
        
    # Find decomposition for 2x7
    res2 = find_decomposition_for_p(p2, 40, time_limit=60.0)
    if res2 is None:
        # Fallback to schoolbook for p2
        U2 = np.zeros((n * m, n * m * p2))
        V2 = np.zeros((m * p2, n * m * p2))
        W2 = np.zeros((p2 * n, n * m * p2))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p2):
                    U2[i * m + j, col] = 1
                    V2[j * p2 + k, col] = 1
                    W2[k * n + i, col] = 1
                    col += 1
        res2 = {"U": U2, "V": V2, "W": W2, "rank": n * m * p2}
        
    # Combine
    r1 = res1["rank"]
    r2 = res2["rank"]
    total_rank = r1 + r2
    
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    # Block 1: k in [0, 8)
    U_full[:, 0:r1] = res1["U"]
    V_full[:, 0:r1] = res1["V"]
    W_full[0:p1*n, 0:r1] = res1["W"]
    
    # Block 2: k in [8, 15)
    U_full[:, r1:total_rank] = res2["U"]
    V_full[:, r1:total_rank] = res2["V"]
    W_full[p1*n:p*n, r1:total_rank] = res2["W"]
    
    # Verify combined result
    if verify_exact(U_full, V_full, W_full):
        if total_rank < best_rank:
            best_rank = total_rank
            best_result = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
            
    return best_result
