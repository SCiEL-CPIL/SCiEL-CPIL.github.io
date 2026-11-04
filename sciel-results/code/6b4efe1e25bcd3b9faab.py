import numpy as np
import time
import itertools
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

    # ---- Vectorized fast verification ----
    def verify_fast(Um, Vm, Wm, tol=1e-9):
        """Check all 48 valid entries vectorized."""
        r = Um.shape[1]
        U_rows = Um[valid_indices[:, 0]]
        V_rows = Vm[valid_indices[:, 1]]
        W_rows = Wm[valid_indices[:, 2]]
        products = U_rows * V_rows * W_rows
        sums = products.sum(axis=1)
        return np.all(np.abs(sums - 1.0) < tol)

    def verify_all_entries(Um, Vm, Wm, tol=1e-9):
        """Check ALL entries: valid ones should be 1, invalid ones should be 0."""
        r = Um.shape[1]
        # Efficient check: Compute the full tensor contraction
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We can do this via matrix multiplications if we reshape, but for small sizes, 
        # direct loop or vectorized broadcast is fine.
        # Since nm=8, mp=24, pn=12, total 2304 entries.
        # r is at most 42.
        # 2304 * 42 operations is trivial.
        
        # Vectorized approach:
        # Create a tensor of shape (nm, mp, r) and (pn, r) ? No.
        # Let's stick to the loop but optimized with numpy inner products?
        # Actually, the previous implementation was a triple loop in Python.
        # Let's use numpy broadcasting for speed.
        
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # We want T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is the sum of outer products of (U[:,r] * V[:,r]) with W[:,r]?
        # No, it's sum_r (U[:,r] * V[:,r])[:, None] * W[:,r][None, :] ... no.
        
        # T = np.einsum('ar,br,cr->abc', U, V, W)
        # This creates an 8x24x12 tensor.
        T = np.einsum('ar,br,cr->abc', Um, Vm, Wm)
        
        # Check valid entries
        valid_a = valid_indices[:, 0]
        valid_b = valid_indices[:, 1]
        valid_c = valid_indices[:, 2]
        valid_vals = T[valid_a, valid_b, valid_c]
        
        if not np.all(np.abs(valid_vals - 1.0) < tol):
            return False
            
        # Check invalid entries
        # Total entries: nm*mp*pn = 2304
        # We need to check that all non-valid entries are 0.
        # Create a mask
        mask = np.zeros((nm, mp, pn), dtype=bool)
        mask[valid_a, valid_b, valid_c] = True
        
        invalid_vals = T[~mask]
        if not np.all(np.abs(invalid_vals) < tol):
            return False
            
        return True

    # ---- Exact verification using Fraction ----
    def verify_exact(Um, Vm, Wm):
        r = Um.shape[1]
        # Convert to Fractions for exactness
        Uf = np.array([[Fraction(int(round(Um[i, j]))) for j in range(r)] for i in range(nm)])
        Vf = np.array([[Fraction(int(round(Vm[i, j]))) for j in range(r)] for i in range(mp)])
        Wf = np.array([[Fraction(int(round(Wm[i, j]))) for j in range(r)] for i in range(pn)])
        
        # Use einsum with objects? No, just loop for exactness, it's slow but r is small.
        # Actually, we can just check the valid entries and assume the rest if they were 0.0 in float, they are likely 0.
        # But to be safe, we check all.
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    s = Fraction(0)
                    for r_idx in range(r):
                        s += Uf[a, r_idx] * Vf[b, r_idx] * Wf[c, r_idx]
                    want = Fraction(1) if (a, b, c) in valid_set else Fraction(0)
                    if s != want:
                        return False
        return True

    # Sanity check base decomposition
    if not verify_fast(U, V, W):
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
    
    # Try multiple random orderings of column removal
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        cur_U, cur_V, cur_W = best_U.copy(), best_V.copy(), best_W.copy()
        cur_rank = cur_U.shape[1]
        
        # Shuffle column indices to try different removal orders
        cols = list(range(cur_rank))
        np.random.shuffle(cols)
        
        # Greedy removal in this order
        for col_idx in cols:
            if time.time() > deadline - 10:
                break
            if col_idx >= cur_U.shape[1]:
                continue
                
            # Try to remove column col_idx
            U2 = np.delete(cur_U, col_idx, axis=1)
            V2 = np.delete(cur_V, col_idx, axis=1)
            W2 = np.delete(cur_W, col_idx, axis=1)
            
            if verify_all_entries(U2, V2, W2):
                cur_U, cur_V, cur_W = U2, V2, W2
                cur_rank -= 1
                # After a removal, the indices of subsequent columns shift.
                # But we are iterating over a pre-shuffled list of *original* indices?
                # No, `cols` contains indices 0..R-1.
                # If we remove index `k`, all indices `> k` in the *current* matrix shift left.
                # My `cols` list is static. This logic is flawed for dynamic removal.
                
                # Correct approach: Iterate while possible.
                # Let's restart the greedy loop properly.
                break # Restart greedy loop with new state
        
        # The above loop logic was slightly off. Let's do a proper greedy pass.
        # We'll just run a standard greedy pass, but the *initial* state is the best found so far.
        # To add randomness, we can randomly swap columns before the greedy pass.
        
        # Let's refine:
        # 1. Copy best
        # 2. Randomly permute columns
        # 3. Run greedy removal (try remove 0, then 0, then 0...)
        # 4. If better, update best.
        
        pass # The previous block was a draft. Implementing below.

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
        if verify_exact(best_U, best_V, best_W):
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
