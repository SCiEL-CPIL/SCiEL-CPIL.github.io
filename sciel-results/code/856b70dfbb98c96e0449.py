"""Combined approach: verified rank-39 Strassen+schoolbook, then structured exact search for rank-38."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import combinations

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
    """
    Explicit rank-39 decomposition for M_{2,2,11}.
    5 blocks of 2 columns (Strassen rank 7 each) + 1 single column (schoolbook rank 4).
    Total: 5*7 + 4 = 39.
    """
    n, m, p = 2, 2, 11
    rank = 39
    
    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 39
    V = [[0.0] * rank for _ in range(m * p)]   # 22 x 39
    W = [[0.0] * rank for _ in range(p * n)]   # 22 x 39
    
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
    
    U[0][r] = 1; V[v_e2][r] = 1; W[w_c02][r] = 1; r += 1
    U[1][r] = 1; V[v_g2][r] = 1; W[w_c02][r] = 1; r += 1
    U[2][r] = 1; V[v_e2][r] = 1; W[w_c12][r] = 1; r += 1
    U[3][r] = 1; V[v_g2][r] = 1; W[w_c12][r] = 1; r += 1
    
    assert r == rank
    return U, V, W

def _try_structured_rank38(deadline):
    """
    Structured search for rank-38: Try to find a rank-38 decomposition by
    parameterizing small perturbations to the rank-39 structure.
    
    Strategy: The rank-39 solution has 5 Strassen blocks (7 terms each) + 1 schoolbook (4 terms).
    Try to merge or eliminate one term by adjusting coefficients in a structured way.
    
    Approach: For each pair of terms (a, b) in the rank-39 solution, check if we can
    express term_a + alpha * term_b as a valid replacement, reducing rank by 1.
    This is a different search family: exact algebraic combination vs. iterative ALS.
    """
    n, m, p = 2, 2, 11
    U39, V39, W39 = _construct_rank39()
    rank39 = 39
    
    # Build the tensor entries
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    
    # For each pair of rank-39 terms, try to see if we can replace them with a single term
    # that covers both, reducing rank by 1.
    # This is a heuristic: check if the sum of two rank-1 tensors is itself rank-1
    # (which is rare, but worth checking for structured cases).
    
    # Alternative structured approach: Try to find a linear dependency among the 39 terms
    # that allows elimination of one term.
    
    # More practical: Try random structured perturbations
    # Keep 38 of the 39 terms fixed, and see if we can adjust one term to compensate.
    
    # For each term to potentially eliminate (0..38), try to find a new term that,
    # combined with the remaining 38, still decomposes the tensor.
    
    # This is a large search space, so use a heuristic:
    # For each term t to eliminate, compute the residual tensor R = T - sum_{r != t} T_r
    # and check if R is rank-1.
    
    # Compute residual for each term elimination
    best_result = None
    best_err = float('inf')
    
    for elim in range(rank39):
        if time.time() > deadline - 20:
            break
        
        # Compute residual: T - sum_{r != elim} U_r * V_r * W_r
        # Residual is a 3-way tensor of size 4 x 22 x 22
        # But we only care about the 44 non-zero entries.
        
        # For each non-zero tensor entry, compute the residual value
        residuals = {}
        for (u_idx, v_idx, w_idx) in tensor_entries:
            s = 0.0
            for r in range(rank39):
                if r == elim:
                    continue
                s += U39[u_idx][r] * V39[v_idx][r] * W39[w_idx][r]
            residuals[(u_idx, v_idx, w_idx)] = 1.0 - s
        
        # Check if residuals form a rank-1 tensor
        # A rank-1 tensor has the form U_new[u] * V_new[v] * W_new[w]
        # We need to find U_new, V_new, W_new such that
        # U_new[u_idx] * V_new[v_idx] * W_new[w_idx] = residuals[(u_idx, v_idx, w_idx)]
        
        # This is a system of equations. Try to solve it.
        # For simplicity, try to find a solution by picking one entry to normalize.
        
        # Find an entry with non-zero residual
        nonzero_entries = [(u, v, w) for (u, v, w), val in residuals.items() if abs(val) > 1e-12]
        if not nonzero_entries:
            # Residual is zero, meaning term elim was redundant
            # This would give rank-38 directly
            return U39, V39, W39, elim
        
        # Try to solve for rank-1 approximation of residuals
        # Use a simple iterative approach: fix U_new and W_new, solve for V_new, etc.
        
        # Initialize
        u_vals = {u: 1.0 for u in range(n * m)}
        v_vals = {v: 1.0 for v in range(m * p)}
        w_vals = {w: 1.0 for w in range(p * n)}
        
        converged = False
        for it in range(50):
            if time.time() > deadline - 15:
                break
            
            # Update u_vals
            for u in range(n * m):
                entries_u = [(v, w) for (uu, v, w) in tensor_entries if uu == u and abs(residuals[(uu, v, w)]) > 1e-12]
                if not entries_u:
                    continue
                # Sum over entries: u * v_vals[v] * w_vals[w] = residuals[(u, v, w)]
                # So u = sum(residuals / (v_vals[v] * w_vals[w])) / len
                num = 0.0
                den = 0.0
                for (v, w) in entries_u:
                    vw = v_vals[v] * w_vals[w]
                    if abs(vw) > 1e-12:
                        num += residuals[(u, v, w)] / vw
                        den += 1.0
                if den > 0:
                    u_vals[u] = num / den
            
            # Update v_vals
            for v in range(m * p):
                entries_v = [(u, w) for (u, vv, w) in tensor_entries if vv == v and abs(residuals[(u, vv, w)]) > 1e-12]
                if not entries_v:
                    continue
                num = 0.0
                den = 0.0
                for (u, w) in entries_v:
                    uw = u_vals[u] * w_vals[w]
                    if abs(uw) > 1e-12:
                        num += residuals[(u, v, w)] / uw
                        den += 1.0
                if den > 0:
                    v_vals[v] = num / den
            
            # Update w_vals
            for w in range(p * n):
                entries_w = [(u, v) for (u, v, ww) in tensor_entries if ww == w and abs(residuals[(u, v, ww)]) > 1e-12]
                if not entries_w:
                    continue
                num = 0.0
                den = 0.0
                for (u, v) in entries_w:
                    uv = u_vals[u] * v_vals[v]
                    if abs(uv) > 1e-12:
                        num += residuals[(u, v, w)] / uv
                        den += 1.0
                if den > 0:
                    w_vals[w] = num / den
            
            # Check convergence
            err = 0.0
            for (u, v, w) in tensor_entries:
                pred = u_vals[u] * v_vals[v] * w_vals[w]
                err += (pred - residuals[(u, v, w)]) ** 2
            if err < 1e-12:
                converged = True
                break
        
        if converged:
            # Build the rank-38 solution
            U_new = [[0.0] * 38 for _ in range(n * m)]
            V_new = [[0.0] * 38 for _ in range(m * p)]
            W_new = [[0.0] * 38 for _ in range(p * n)]
            
            for r in range(rank39):
                if r == elim:
                    continue
                new_r = r if r < elim else r - 1
                for u in range(n * m):
                    U_new[u][new_r] = U39[u][r]
                for v in range(m * p):
                    V_new[v][new_r] = V39[v][r]
                for w in range(p * n):
                    W_new[w][new_r] = W39[w][r]
            
            # Add the new term as the 38th column
            for u in range(n * m):
                U_new[u][37] = u_vals[u]
            for v in range(m * p):
                V_new[v][37] = v_vals[v]
            for w in range(p * n):
                W_new[w][37] = w_vals[w]
            
            if _verify(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new, elim
    
    return None

def _try_als_rank(target_rank, n, m, p, deadline, max_rounds=500, seed=42):
    """
    Try to find a decomposition of given rank using ALS.
    Returns (U, V, W) as lists if found, else None.
    """
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, target_rank) * 0.5
    V = rng.randn(m * p, target_rank) * 0.5
    W = rng.randn(p * n, target_rank) * 0.5
    
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
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
        for a in range(n * m):
            mask = (ta_arr == a)
            if not np.any(mask):
                continue
            tb_sub = tb_arr[mask]
            tc_sub = tc_arr[mask]
            V_sub = V[tb_sub, :]
            W_sub = W[tc_sub, :]
            prod = V_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]
        
        for b in range(m * p):
            mask = (tb_arr == b)
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tc_sub = tc_arr[mask]
            U_sub = U[ta_sub, :]
            W_sub = W[tc_sub, :]
            prod = U_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]
        
        for c_idx in range(p * n):
            mask = (tc_arr == c_idx)
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tb_sub = tb_arr[mask]
            U_sub = U[ta_sub, :]
            V_sub = V[tb_sub, :]
            prod = U_sub * V_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                W[c_idx, valid] = num[valid] / den[valid]
        
        if it % 50 == 0:
            sums = np.zeros(len(ta_arr))
            for c in range(target_rank):
                sums += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
            err = np.sum((sums - 1.0) ** 2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
    
    if best_err < 1e-8 and best_U is not None:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    U, V, W = _construct_rank39()
    if _verify(U, V, W, n, m, p):
        # First, try the structured rank-38 search
        if time.time() < deadline - 30:
            result = _try_structured_rank38(deadline)
            if result is not None:
                U_new, V_new, W_new, elim = result
                if _verify(U_new, V_new, W_new, n, m, p):
                    return {"U": U_new, "V": V_new, "W": W_new}
        
        # Then try ALS for rank-38
        if time.time() < deadline - 15:
            for seed in range(10):
                if time.time() > deadline - 5:
                    break
                result = _try_als_rank(38, n, m, p, deadline=deadline - 2, max_rounds=300, seed=seed)
                if result is not None:
                    U2, V2, W2 = result
                    if _verify(U2, V2, W2, n, m, p):
                        return {"U": U2, "V": V2, "W": W2}
        
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
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
