"""Rewritten program: search for a rank-minimal exact tensor decomposition of 2x3 * 3x7."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]
DEADLINE = 110.0  # safety margin under 120s limit (and 150s kill)


def _get_expected_tensor():
    """
    Pre-compute the expected tensor T as a dictionary of (a, b, c) -> value.
    a in [0, n*m), b in [0, m*p), c in [0, p*n).
    T[a, b, c] = 1 if j_a == j_b and c == k_b * n + i_a.
    Else 0.
    """
    n, m, p = DIMENSIONS
    expected = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected[(a, b, c)] = 1
    return expected


EXPECTED_TENSOR = _get_expected_tensor()


def _verify_fast(U, V, W):
    """
    Check exact tensor decomposition for all entries.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    nm = n * m
    mp = m * p
    np_ = p * n
    
    # Convert to numpy arrays for speed
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # We need to check sum_c U[a,c] V[b,c] W[c,c] == T[a,b,c]
    # This is equivalent to checking the tensor product.
    # To be safe and fast, we iterate over the non-zero entries of T first.
    # If any mismatch, return False.
    # Then check a sample of zero entries? No, must check all.
    # But checking all 1764 entries with rank 30 is fast.
    
    # Let's compute the full reconstructed tensor and compare.
    # Rec[a, b, c] = sum_c U[a, c] V[b, c] W[c, c]
    # This is a 3-way contraction.
    # We can do this by iterating over c and accumulating into a 3D array.
    
    rec = np.zeros((nm, mp, np_), dtype=int)
    for c in range(rank):
        # rec += U[:, c, None, None] * V[None, :, c, None] * W[None, None, :, c]
        # But U is 2D.
        u_col = U_np[:, c]
        v_col = V_np[:, c]
        w_col = W_np[:, c]
        
        # Outer product
        # rec[a, b, k] += u_col[a] * v_col[b] * w_col[k]
        # Using broadcasting
        rec += np.einsum('a,b,k->abk', u_col, v_col, w_col)
        
    # Now compare rec with expected tensor.
    # Expected tensor is sparse.
    # We can create a full expected array and subtract.
    exp = np.zeros((nm, mp, np_), dtype=int)
    for (a, b, c_idx), val in EXPECTED_TENSOR.items():
        exp[a, b, c_idx] = val
        
    # Check if rec == exp
    if not np.array_equal(rec, exp):
        return False
    return True


def _cost(U, V, W):
    """
    Calculate the number of mismatches between the reconstructed tensor and the expected tensor.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    nm = n * m
    mp = m * p
    np_ = p * n
    
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    rec = np.zeros((nm, mp, np_), dtype=int)
    for c in range(rank):
        u_col = U_np[:, c]
        v_col = V_np[:, c]
        w_col = W_np[:, c]
        rec += np.einsum('a,b,k->abk', u_col, v_col, w_col)
        
    # Count mismatches
    # We only care about the entries in EXPECTED_TENSOR and the rest.
    # Mismatches = sum(abs(rec - exp))
    # Since exp is 0 or 1, and rec is integer.
    # We can just sum the absolute differences.
    # But we don't have exp as a dense array in _cost to save memory/time?
    # Let's just compute the sum of squares or absolute diff against the sparse structure.
    
    # Simpler: compute rec, then for each expected entry, check.
    # And for non-expected entries, check if rec is 0.
    
    cost = 0
    # Check expected 1s
    for (a, b, c_idx), val in EXPECTED_TENSOR.items():
        if rec[a, b, c_idx] != val:
            cost += abs(rec[a, b, c_idx] - val)
            
    # Check non-expected entries (should be 0)
    # This is expensive if done naively.
    # We can compute the total sum of squares of rec, and subtract the correct 1s.
    # Sum of squares of rec = sum(rec^2)
    # Sum of squares of exp = number of 1s = n*m*p
    # If rec == exp, sum(rec^2) = n*m*p and all 1s are in the right place.
    # But we need exact match.
    
    # Let's just iterate over all entries. 1764 is small.
    for a in range(nm):
        for b in range(mp):
            for c_idx in range(np_):
                exp_val = EXPECTED_TENSOR.get((a, b, c_idx), 0)
                if rec[a, b, c_idx] != exp_val:
                    cost += abs(rec[a, b, c_idx] - exp_val)
                    
    return cost


def _local_search(U, V, W, max_iters=2000, timeout=5.0):
    """
    Local search to minimize cost.
    U, V, W are lists of lists of integers.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    start_time = time.time()
    
    current_cost = _cost(U, V, W)
    if current_cost == 0:
        return U, V, W, 0
    
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    best_cost = current_cost
    
    for it in range(max_iters):
        if time.time() - start_time > timeout:
            break
            
        # Pick a random matrix and entry to flip
        mat_choice = random.choice([0, 1, 2])
        if mat_choice == 0:
            mat = U
            mat_name = 'U'
        elif mat_choice == 1:
            mat = V
            mat_name = 'V'
        else:
            mat = W
            mat_name = 'W'
            
        r = random.randint(0, len(mat) - 1)
        c = random.randint(0, rank - 1)
        
        old_val = mat[r][c]
        # Flip to a new value in {-1, 0, 1}
        new_val = random.choice([-1, 0, 1])
        if new_val == old_val:
            continue
            
        mat[r][c] = new_val
        
        new_cost = _cost(U, V, W)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
            if best_cost == 0:
                return best_U, best_V, best_W, 0
        else:
            # Revert
            mat[r][c] = old_val
            
    return best_U, best_V, best_W, best_cost


def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x7."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook rank is n*m*p = 42
    schoolbook_rank = n * m * p
    
    # Create schoolbook decomposition
    U = [[0]*schoolbook_rank for _ in range(n*m)]
    V = [[0]*schoolbook_rank for _ in range(m*p)]
    W = [[0]*schoolbook_rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    # Verify schoolbook
    assert _verify_fast(U, V, W), "Schoolbook verification failed"
    
    best_rank = schoolbook_rank
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank solutions
    # Search from rank 41 down to 25
    for rank in range(schoolbook_rank - 1, 24, -1):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Try a few random restarts for this rank
        found = False
        for restart in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            # Initialize with random sparse matrices
            U_init = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n*m)]
            V_init = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m*p)]
            W_init = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p*n)]
            
            # Run local search
            U_sol, V_sol, W_sol, cost = _local_search(U_init, V_init, W_init, max_iters=1000, timeout=2.0)
            
            if cost == 0:
                # Verify one last time
                if _verify_fast(U_sol, V_sol, W_sol):
                    best_rank = rank
                    best_result = {"U": U_sol, "V": V_sol, "W": W_sol}
                    found = True
                    break
                    
        if found:
            # Continue to try lower ranks
            continue
            
    return best_result


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
