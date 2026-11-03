"""Rewritten program for 3x3x7 matrix multiplication rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _build_target(n, m, p):
    """Build list of (a,b,c,expected) entries for the tensor."""
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                entries.append((a, b, c, 1.0))
    return entries

def _build_all_nonzero(n, m, p):
    """Set of all (a,b,c) that should be nonzero."""
    s = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s.add((a, b, c))
    return s

def _full_verify(U, V, W, target, nonzero_set, n, m, p, tol=1e-6):
    """Full verification of all target and non-target entries."""
    for a, b, c, exp in target:
        val = np.dot(U[a], V[b] * W[c])
        if abs(val - exp) > tol:
            return False
    # Check ALL non-target entries (small enough for 9x21x21=3969 total)
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in nonzero_set:
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val) > tol:
                        return False
    return True

def _compute_full_error(U, V, W, target, nonzero_set, n, m, p):
    """Compute full error over all entries."""
    err = 0.0
    for a, b, c, exp in target:
        val = np.dot(U[a], V[b] * W[c])
        err += (val - exp) ** 2
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in nonzero_set:
                    val = np.dot(U[a], V[b] * W[c])
                    err += val ** 2
    return err

def _try_field_extension(n, m, p, deadline, start_time):
    """
    Try field extension approach: represent 7 output columns as coefficients
    of an element in F_q(alpha) where alpha^7 = 1 (or similar).
    The idea: if we can find a cyclic or structured representation, the
    multiplication tensor decomposes more efficiently.
    """
    # For 3x3x7, we have 9 input pairs (i,j) and 21 output positions (k,i)
    # Key insight: group outputs by the i-index (3 groups of 7)
    # For each i, the 7 outputs C[i][*] = sum_j A[i][j] * B[j][*]
    # This is like multiplying a 1x3 row by a 3x7 matrix for each i
    
    # Try a structured approach: use the fact that for fixed i,
    # the sub-problem is 1x3x7 which has rank 7 (schoolbook)
    # But with 3 such sub-problems sharing structure, we can do better
    
    # Approach: try rank values and use structured initialization
    best_result = None
    best_rank = n * m * p  # 63
    
    target = _build_target(n, m, p)
    nonzero_set = _build_all_nonzero(n, m, p)
    
    # Try various rank values with structured initialization
    for rank in range(30, 63):
        if time.time() - start_time > deadline:
            break
        
        for restart in range(5):
            if time.time() - start_time > deadline:
                break
            
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Structured initialization based on field extension idea
            # Divide rank into 3 groups of ~rank/3 for each output row i
            group_size = rank // 3
            remainder = rank % 3
            
            np_rng = np.random.RandomState(restart * 7919 + 12345)
            
            for i in range(n):
                base = i * group_size
                for j in range(m):
                    # U entries for input (i,j)
                    U[i * m + j, base:(base + group_size)] = np_rng.randn(group_size) * 0.5
                    # V entries for input (j,*)
                    V[j * p:(j + 1) * p, base:(base + group_size)] = np_rng.randn(p, group_size) * 0.5
                    # W entries for output (*, i)
                    W[:, base:(base + group_size)] = np_rng.randn(p * n, group_size) * 0.5
            
            # Add small random components to all
            U += np_rng.randn(n * m, rank) * 0.01
            V += np_rng.randn(m * p, rank) * 0.01
            W += np_rng.randn(p * n, rank) * 0.01
            
            # Gradient descent
            lr = 0.005
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            mU, mV, mW = np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)
            vU, vV, vW = np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)
            t = 0
            best_err = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Precompute indices for efficiency
            target_arr = np.array(target, dtype=np.int32)
            
            for iteration in range(200):
                if time.time() - start_time > deadline:
                    break
                t += 1
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                # Target entries
                for idx in range(len(target)):
                    a, b, c, exp = target[idx]
                    T_val = np.dot(U[a], V[b] * W[c])
                    E = T_val - exp
                    dU[a] += 2 * E * (V[b] * W[c])
                    dV[b] += 2 * E * (U[a] * W[c])
                    dW[c] += 2 * E * (U[a] * V[b])
                
                # Sample non-target entries
                rng_iter = np.random.RandomState(iteration * 13 + restart * 100)
                for _ in range(20):
                    a = rng_iter.randint(0, n * m)
                    b = rng_iter.randint(0, m * p)
                    c = rng_iter.randint(0, p * n)
                    if (a, b, c) not in nonzero_set:
                        T_val = np.dot(U[a], V[b] * W[c])
                        dU[a] += 2 * T_val * (V[b] * W[c])
                        dV[b] += 2 * T_val * (U[a] * W[c])
                        dW[c] += 2 * T_val * (U[a] * V[b])
                
                # Adam update
                mU = beta1 * mU + (1 - beta1) * dU
                mV = beta1 * mV + (1 - beta1) * dV
                mW = beta1 * mW + (1 - beta1) * dW
                vU = beta2 * vU + (1 - beta2) * (dU ** 2)
                vV = beta2 * vV + (1 - beta2) * (dV ** 2)
                vW = beta2 * vW + (1 - beta2) * (dW ** 2)
                mU_hat = mU / (1 - beta1 ** t)
                mV_hat = mV / (1 - beta1 ** t)
                mW_hat = mW / (1 - beta1 ** t)
                vU_hat = vU / (1 - beta2 ** t)
                vV_hat = vV / (1 - beta2 ** t)
                vW_hat = vW / (1 - beta2 ** t)
                
                U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
                V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
                W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
                
                if iteration % 25 == 0:
                    err = _compute_full_error(U, V, W, target, nonzero_set, n, m, p)
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if _full_verify(U, V, W, target, nonzero_set, n, m, p):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = rank
                    return best_result, best_rank
            
            # Check best found at this rank
            if _full_verify(best_U, best_V, best_W, target, nonzero_set, n, m, p):
                best_result = {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                best_rank = rank
                return best_result, best_rank
    
    return best_result, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 140  # Leave 10s safety margin
    
    target = _build_target(n, m, p)
    nonzero_set = _build_all_nonzero(n, m, p)
    
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 63
    
    # Build schoolbook solution as fallback
    U_fb = np.zeros((n * m, rank_schoolbook))
    V_fb = np.zeros((m * p, rank_schoolbook))
    W_fb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
    
    best_result = {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    best_rank = rank_schoolbook
    
    # Try field extension approach
    fe_result, fe_rank = _try_field_extension(n, m, p, DEADLINE, start_time)
    if fe_result is not None and fe_rank < best_rank:
        best_result = fe_result
        best_rank = fe_rank
    
    return best_result
# EVOLVE-BLOCK-END
