import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Precompute the valid tensor indices
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_indices.append((i * m + j, j * p + k, k * n + i))
    valid_indices = np.array(valid_indices)  # shape (48, 3)
    valid_set = set((a, b, c) for a, b, c in valid_indices)

    # ---- Build Strassen rank-42 decomposition ----
    def add_strassen_2x2x2(U, V, W, i_start, j_start, k_start, offset):
        a00 = i_start * m + j_start
        a01 = i_start * m + j_start + 1
        a10 = (i_start + 1) * m + j_start
        a11 = (i_start + 1) * m + j_start + 1
        b00 = j_start * p + k_start
        b01 = j_start * p + k_start + 1
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + k_start + 1
        w00 = k_start * n + i_start
        w01 = (k_start + 1) * n + i_start
        w10 = k_start * n + i_start + 1
        w11 = (k_start + 1) * n + i_start + 1

        # M1 = (A00+A11)(B00+B11)
        U[a00, offset] += 1.0; U[a11, offset] += 1.0
        V[b00, offset] += 1.0; V[b11, offset] += 1.0
        W[w00, offset] += 1.0; W[w11, offset] += 1.0
        # M2 = (A10+A11)B00
        U[a10, offset+1] += 1.0; U[a11, offset+1] += 1.0
        V[b00, offset+1] += 1.0
        W[w10, offset+1] += 1.0; W[w11, offset+1] -= 1.0
        # M3 = A00(B01-B11)
        U[a00, offset+2] += 1.0
        V[b01, offset+2] += 1.0; V[b11, offset+2] -= 1.0
        W[w01, offset+2] += 1.0; W[w11, offset+2] += 1.0
        # M4 = A11(B10-B00)
        U[a11, offset+3] += 1.0
        V[b10, offset+3] += 1.0; V[b00, offset+3] -= 1.0
        W[w00, offset+3] += 1.0; W[w10, offset+3] += 1.0
        # M5 = (A00+A01)B11
        U[a00, offset+4] += 1.0; U[a01, offset+4] += 1.0
        V[b11, offset+4] += 1.0
        W[w00, offset+4] -= 1.0; W[w01, offset+4] += 1.0
        # M6 = (A10-A00)(B00+B01)
        U[a10, offset+5] += 1.0; U[a00, offset+5] -= 1.0
        V[b00, offset+5] += 1.0; V[b01, offset+5] += 1.0
        W[w11, offset+5] += 1.0
        # M7 = (A01-A11)(B10+B11)
        U[a01, offset+6] += 1.0; U[a11, offset+6] -= 1.0
        V[b10, offset+6] += 1.0; V[b11, offset+6] += 1.0
        W[w00, offset+6] += 1.0

    RANK_INIT = 42
    U = np.zeros((nm, RANK_INIT), dtype=np.float64)
    V = np.zeros((mp, RANK_INIT), dtype=np.float64)
    W = np.zeros((pn, RANK_INIT), dtype=np.float64)

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_2x2x2(U, V, W, i_start, j_start, k_start, offset)
                offset += 7

    # ---- Verification helpers ----
    def verify_all_entries(Um, Vm, Wm, tol=1e-9):
        """Check ALL entries: valid ones should be 1, invalid ones should be 0."""
        T = np.einsum('ar,br,cr->abc', Um, Vm, Wm, optimize=True)
        
        valid_a = valid_indices[:, 0]
        valid_b = valid_indices[:, 1]
        valid_c = valid_indices[:, 2]
        valid_vals = T[valid_a, valid_b, valid_c]
        
        if not np.all(np.abs(valid_vals - 1.0) < tol):
            return False
            
        mask = np.zeros((nm, mp, pn), dtype=bool)
        mask[valid_a, valid_b, valid_c] = True
        
        invalid_vals = T[~mask]
        if not np.all(np.abs(invalid_vals) < tol):
            return False
            
        return True

    # Sanity check base decomposition
    if not verify_all_entries(U, V, W):
        # Fallback to schoolbook if something went wrong
        rank0 = n * m * p
        U = np.zeros((nm, rank0), dtype=np.float64)
        V = np.zeros((mp, rank0), dtype=np.float64)
        W = np.zeros((pn, rank0), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * (m * p) + j * p + k
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = best_U.shape[1]

    # ---- Enhanced Column Removal with Full Tensor Check ----
    # We will use a randomized restart strategy.
    # In each restart, we shuffle the columns and try to remove them greedily.
    # This helps escape local minima where a specific column order prevents removal.
    
    np.random.seed(42)
    
    # Correct Randomized Greedy Removal
    for restart in range(100):
        if time.time() > deadline - 15:
            break
            
        cur_U, cur_V, cur_W = best_U.copy(), best_V.copy(), best_W.copy()
        cur_rank = cur_U.shape[1]
        
        # Permute columns to change the order in which they are considered
        perm = np.random.permutation(cur_rank)
        cur_U = cur_U[:, perm]
        cur_V = cur_V[:, perm]
        cur_W = cur_W[:, perm]
        
        # Greedy removal: try to remove column 0, then 0, then 0...
        # If we can't remove 0, try 1, etc.
        # Actually, just scan 0 to R-1 repeatedly.
        
        changed = True
        while changed and time.time() < deadline - 15:
            changed = False
            cur_rank = cur_U.shape[1]
            for col in range(cur_rank):
                if time.time() > deadline - 15:
                    break
                U2 = np.delete(cur_U, col, axis=1)
                V2 = np.delete(cur_V, col, axis=1)
                W2 = np.delete(cur_W, col, axis=1)
                
                if verify_all_entries(U2, V2, W2):
                    cur_U, cur_V, cur_W = U2, V2, W2
                    changed = True
                    break # Restart scan from 0
                    
        if cur_U.shape[1] < best_rank:
            best_U, best_V, best_W = cur_U, cur_V, cur_W
            best_rank = cur_U.shape[1]
            # If we found a new best, we can continue searching from here.
            # The next restarts will use this new best.

    # ---- Final exact verification ----
    if time.time() < deadline - 2:
        # Convert to Fractions for exactness
        r = best_U.shape[1]
        Uf = np.array([[Fraction(int(round(best_U[i, j]))) for j in range(r)] for i in range(nm)])
        Vf = np.array([[Fraction(int(round(best_V[i, j]))) for j in range(r)] for i in range(mp)])
        Wf = np.array([[Fraction(int(round(best_W[i, j]))) for j in range(r)] for i in range(pn)])
        
        exact_ok = True
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    s = Fraction(0)
                    for r_idx in range(r):
                        s += Uf[a, r_idx] * Vf[b, r_idx] * Wf[c, r_idx]
                    want = Fraction(1) if (a, b, c) in valid_set else Fraction(0)
                    if s != want:
                        exact_ok = False
                        break
                if not exact_ok:
                    break
            if not exact_ok:
                break
        
        if exact_ok:
            final_U, final_V, final_W = best_U, best_V, best_W
        else:
            # Fallback to original if exact check fails (shouldn't happen if float check passed)
            final_U, final_V, final_W = U, V, W
    else:
        final_U, final_V, final_W = U, V, W

    final_U = np.round(final_U)
    final_V = np.round(final_V)
    final_W = np.round(final_W)

    return {"U": final_U.tolist(), "V": final_V.tolist(), "W": final_W.tolist()}
