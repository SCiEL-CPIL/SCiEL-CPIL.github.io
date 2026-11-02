# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    nm = n * m  # 9
    mp = m * p  # 12
    pn = p * n  # 12
    
    # Active triples: (a, b, c) where a=i*m+j, b=j*p+k, c=k*n+i
    active_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                active_triples.add((a, b, c))
    
    active_set = active_triples
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify if the decomposition is exact."""
        r = len(U[0])
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    s = 0
                    for t in range(r):
                        s += U[a][t] * V[b][t] * W[c][t]
                    expected = 1 if (a, b, c) in active_set else 0
                    if s != expected:
                        return False
        return True
    
    def count_errors(U, V, W):
        """Count number of violated triples."""
        r = len(U[0])
        err_count = 0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    s = 0
                    for t in range(r):
                        s += U[a][t] * V[b][t] * W[c][t]
                    expected = 1 if (a, b, c) in active_set else 0
                    if s != expected:
                        err_count += 1
        return err_count
    
    def local_search_numpy(U, V, W, r, max_iters=30000, time_limit=8.0):
        """
        Local search using NumPy for fast sum updates.
        """
        start_ls = time.time()
        
        U_arr = np.array(U, dtype=int)
        V_arr = np.array(V, dtype=int)
        W_arr = np.array(W, dtype=int)
        
        # Compute all sums: sums[a][b][c] = sum_t U[a,t]*V[b,t]*W[c,t]
        # Use einsum for speed
        sums = np.einsum('at,bt,ct->abc', U_arr, V_arr, W_arr)
        
        expected_arr = np.zeros((nm, mp, pn), dtype=int)
        for (a, b, c) in active_set:
            expected_arr[a, b, c] = 1
        
        def get_err_count():
            return int(np.sum(sums != expected_arr))
        
        curr_err = get_err_count()
        best_err = curr_err
        best_U = U[:]
        best_V = V[:]
        best_W = W[:]
        
        if curr_err == 0:
            return U, V, W, 0
        
        for it in range(max_iters):
            if time.time() > time_limit or time.time() > deadline:
                break
            
            # Random move
            mat_idx = random.randint(0, 2)
            delta = random.choice([-2, -1, 1, 2])
            
            if mat_idx == 0:  # Modify U
                row = random.randint(0, nm - 1)
                col = random.randint(0, r - 1)
                old_val = U_arr[row, col]
                new_val = old_val + delta
                if abs(new_val) > 10:
                    continue
                
                # Compute change in sums
                # sums[row, b, c] += delta * V[b, col] * W[c, col]
                old_sums_row = sums[row, :, :].copy()
                change = delta * np.outer(V_arr[:, col], W_arr[:, col])
                new_sums_row = old_sums_row + change
                
                old_err_row = int(np.sum(old_sums_row != expected_arr[row, :, :]))
                new_err_row = int(np.sum(new_sums_row != expected_arr[row, :, :]))
                improvement = old_err_row - new_err_row
                
                if improvement > 0:
                    U_arr[row, col] = new_val
                    sums[row, :, :] = new_sums_row
                    curr_err -= improvement
                    if curr_err < best_err:
                        best_err = curr_err
                        best_U = U_arr.tolist()
                        best_V = V_arr.tolist()
                        best_W = W_arr.tolist()
                        if best_err == 0:
                            return best_U, best_V, best_W, 0
                else:
                    # No change, sums remain the same
                    pass

            elif mat_idx == 1:  # Modify V
                row = random.randint(0, mp - 1)
                col = random.randint(0, r - 1)
                old_val = V_arr[row, col]
                new_val = old_val + delta
                if abs(new_val) > 10:
                    continue
                
                old_sums_col = sums[:, row, :].copy()
                change = delta * np.outer(U_arr[:, col], W_arr[:, col])
                new_sums_col = old_sums_col + change
                
                old_err_col = int(np.sum(old_sums_col != expected_arr[:, row, :]))
                new_err_col = int(np.sum(new_sums_col != expected_arr[:, row, :]))
                improvement = old_err_col - new_err_col
                
                if improvement > 0:
                    V_arr[row, col] = new_val
                    sums[:, row, :] = new_sums_col
                    curr_err -= improvement
                    if curr_err < best_err:
                        best_err = curr_err
                        best_U = U_arr.tolist()
                        best_V = V_arr.tolist()
                        best_W = W_arr.tolist()
                        if best_err == 0:
                            return best_U, best_V, best_W, 0
                else:
                    pass
                    
            else:  # Modify W
                row = random.randint(0, pn - 1)
                col = random.randint(0, r - 1)
                old_val = W_arr[row, col]
                new_val = old_val + delta
                if abs(new_val) > 10:
                    continue
                
                old_sums_w = sums[:, :, row].copy()
                change = delta * np.outer(U_arr[:, col], V_arr[:, col])
                new_sums_w = old_sums_w + change
                
                old_err_w = int(np.sum(old_sums_w != expected_arr[:, :, row]))
                new_err_w = int(np.sum(new_sums_w != expected_arr[:, :, row]))
                improvement = old_err_w - new_err_w
                
                if improvement > 0:
                    W_arr[row, col] = new_val
                    sums[:, :, row] = new_sums_w
                    curr_err -= improvement
                    if curr_err < best_err:
                        best_err = curr_err
                        best_U = U_arr.tolist()
                        best_V = V_arr.tolist()
                        best_W = W_arr.tolist()
                        if best_err == 0:
                            return best_U, best_V, best_W, 0
                else:
                    pass
        
        return best_U, best_V, best_W, best_err
    
    # Strategy: Try to find a decomposition with rank < 36
    # Start with schoolbook, then try to reduce rank by merging/eliminating terms
    
    best_result = schoolbook()
    best_rank = n * m * p  # 36
    
    # Try ranks from a reasonable lower bound up to 35
    # Lower bound: at least max(n*m, m*p, p*n) / something... 
    # For 3x3x4, known results suggest rank around 22-25
    
    # Approach 1: Start from schoolbook and try to eliminate redundant terms
    # Approach 2: Random initialization + local search for each rank
    
    def try_rank_random(r, max_attempts=20, time_per_attempt=3.0):
        """Try to find a rank-r decomposition via random init + local search."""
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Sparse random initialization
            U = [[0] * r for _ in range(nm)]
            V = [[0] * r for _ in range(mp)]
            W = [[0] * r for _ in range(pn)]
            
            # Initialize with sparse random ±1 entries
            for a in range(nm):
                k = random.randint(1, min(3, r))
                cols = random.sample(range(r), k)
                for c in cols:
                    U[a][c] = random.choice([-1, 1])
            for b in range(mp):
                k = random.randint(1, min(3, r))
                cols = random.sample(range(r), k)
                for c in cols:
                    V[b][c] = random.choice([-1, 1])
            for c in range(pn):
                k = random.randint(1, min(3, r))
                cols = random.sample(range(r), k)
                for c2 in cols:
                    W[c][c2] = random.choice([-1, 1])
            
            # Local search
            U, V, W, err = local_search_numpy(U, V, W, r, max_iters=15000, time_limit=time_per_attempt)
            
            if err == 0:
                return {"U": U, "V": V, "W": W}
        
        return None
    
    def try_rank_from_schoolbook(r, time_limit=5.0):
        """
        Try to reduce schoolbook to rank r by zeroing out entries.
        Start with schoolbook (rank 36), try to set entries to 0 to reduce effective rank.
        """
        if time.time() > deadline:
            return None
        
        # Start with schoolbook
        sb = schoolbook()
        U = [row[:] for row in sb["U"]]
        V = [row[:] for row in sb["V"]]
        W = [row[:] for row in sb["W"]]
        
        # We want to find a subset of columns that still works
        # This is equivalent to finding which schoolbook terms are redundant
        
        # Alternative: start with rank-r random init but biased toward schoolbook structure
        # For each of the r "slots", assign it to cover some (i,j,k) triple
        
        # Let's try: assign r slots to cover the 36 triples, with some slots covering multiple
        if r < 36:
            # We need to merge 36-r pairs of triples
            # This is complex; let's just do random init with schoolbook bias
            
            U = [[0] * r for _ in range(nm)]
            V = [[0] * r for _ in range(mp)]
            W = [[0] * r for _ in range(pn)]
            
            # Assign each triple to a random slot, allowing multiple triples per slot
            triple_list = list(active_set)
            random.shuffle(triple_list)
            
            for idx, (a, b, c) in enumerate(triple_list):
                slot = idx % r
                # Add contribution to this slot
                # This doesn't directly give us valid U,V,W, but it's a starting point
                # For now, just mark that this slot should cover this triple
                pass
            
            # Better: for each slot, pick which (i,j,k) it covers
            # U[a][slot] = 1 if a = i*m+j for the assigned (i,j,k)
            # V[b][slot] = 1 if b = j*p+k
            # W[c][slot] = 1 if c = k*n+i
            
            # Assign triples to slots
            slot_triples = {s: [] for s in range(r)}
            for idx, (a, b, c) in enumerate(triple_list):
                slot = idx % r
                slot_triples[slot].append((a, b, c))
            
            for slot in range(r):
                for (a, b, c) in slot_triples[slot]:
                    U[a][slot] += 1
                    V[b][slot] += 1
                    W[c][slot] += 1
            
            # Local search to fix errors
            U, V, W, err = local_search_numpy(U, V, W, r, max_iters=20000, time_limit=time_limit)
            
            if err == 0:
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try ranks from 22 upward (known lower bound for related problems)
    for r in range(22, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try from schoolbook structure first
        result = try_rank_from_schoolbook(r, time_limit=3.0)
        if result:
            best_result = result
            best_rank = r
            break
        
        # Try random initialization
        result = try_rank_random(r, max_attempts=10, time_per_attempt=2.0)
        if result:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
