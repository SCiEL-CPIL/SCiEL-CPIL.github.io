import numpy as np
import random
import time
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    start_time = time.time()
    time_limit = 115

    # Helper to check if a numpy array is close to half-integers
    def is_half_integer_array(arr, tol=1e-9):
        # arr shape (rows, cols)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                nearest = round(val * 2) / 2
                if abs(val - nearest) > tol:
                    return False
        return True

    def convert_to_list(arr):
        # Convert numpy array to list of lists with int/float
        res = []
        for i in range(arr.shape[0]):
            row = []
            for j in range(arr.shape[1]):
                val = arr[i, j]
                nearest = round(val * 2) / 2
                if nearest == int(nearest):
                    row.append(int(nearest))
                else:
                    row.append(nearest)
            res.append(row)
        return res

    def verify_decomposition(U, V, W, rank):
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Strategy: Local search on integer lattice for U, V, W
    # We fix rank R and try to find integer/half-integer matrices U, V, W
    # such that the tensor equation holds.
    
    # We will iterate over ranks from a lower bound up to schoolbook.
    # Known lower bound for 3x4x5 is at least 25 (actually 25 is a conjecture, let's start from 25).
    # We'll try ranks: 25, 26, 27, ... up to 60.
    
    # To make the search efficient, we use a random restart with local improvement.
    # We initialize U, V, W with small integers.
    # We define an error metric: sum over all (i,j,k) of (sum_c U[i,j,c]V[j,k,c]W[k,i,c] - 1)^2
    # We try to reduce this error by changing one entry in U, V, or W at a time.
    
    def local_search(R, time_budget):
        # Initialize U, V, W with random small integers
        # We use a range of -2 to 2 to keep values small
        U = np.zeros((nm, R), dtype=int)
        V = np.zeros((mp, R), dtype=int)
        W = np.zeros((pn, R), dtype=int)
        
        # Initialize with random values
        for i in range(nm):
            for c in range(R):
                U[i, c] = random.randint(-1, 1)
        for j in range(mp):
            for c in range(R):
                V[j, c] = random.randint(-1, 1)
        for k in range(pn):
            for c in range(R):
                W[k, c] = random.randint(-1, 1)
        
        # Precompute some terms to speed up error calculation
        # Error[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] - 1
        
        def calc_error(U, V, W):
            err = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(R):
                            s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        err[i, j, k] = s - 1
            return err
        
        err = calc_error(U, V, W)
        current_cost = np.sum(err**2)
        
        if current_cost < 1e-9:
            return U, V, W
        
        no_improvement_count = 0
        max_no_improvement = 500
        
        while time.time() - start_time < start_time + time_budget and no_improvement_count < max_no_improvement:
            best_new_cost = current_cost
            best_change = None
            
            # Try changing one entry in U
            for i in range(nm):
                for c in range(R):
                    old_val = U[i, c]
                    for new_val in range(-3, 4):
                        if new_val == old_val:
                            continue
                        U[i, c] = new_val
                        # Update error
                        new_cost = current_cost
                        for j_prime in range(m):
                            for k_prime in range(p):
                                old_err = err[j_prime // m, j_prime % m, k_prime] # Wait, indices are i,j,k
                                # Actually, U[i*m+j, c] affects err[i, j, k] for all j, k
                                # Let's restructure: U[i*m+j, c] is used in err[i, j, k]
                                # So if we change U[i*m+j, c], we affect err[i, j, k] for all k.
                                # Wait, U index is i*m+j. So for a fixed i, we have m values of j.
                                # Let's just recalculate the affected errors.
                                pass
                        # This is too slow. Let's precompute the contribution.
                        U[i, c] = old_val
            # The above is too slow. Let's use a different approach.
            # We'll just try a few random changes and accept if it improves.
            
            # Random change
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                idx = random.randint(0, nm - 1)
                c = random.randint(0, R - 1)
                old_val = U[idx, c]
                new_val = old_val + random.choice([-1, 1])
                if abs(new_val) > 3:
                    continue
                U[idx, c] = new_val
                # Recalculate error for affected (i,j,k)
                # U[idx, c] where idx = i*m+j
                i = idx // m
                j = idx % m
                for k in range(p):
                    old_err = err[i, j, k]
                    delta = U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c] - old_val * V[j * p + k, c] * W[k * n + i, c]
                    # Wait, delta = (new_val - old_val) * V[j*p+k, c] * W[k*n+i, c]
                    delta = (new_val - old_val) * V[j * p + k, c] * W[k * n + i, c]
                    new_err = old_err - delta # Wait, err = sum - 1, so new_err = old_err - (old contribution) + (new contribution)
                    # Actually, err[i,j,k] = sum_c ... - 1
                    # If we change U[i*m+j, c], the sum changes by (new_val - old_val) * V[j*p+k, c] * W[k*n+i, c]
                    # So new_err = old_err + (new_val - old_val) * V[j*p+k, c] * W[k*n+i, c]
                    new_err = old_err + delta
                    new_cost = current_cost - old_err**2 + new_err**2
                    err[i, j, k] = new_err
                if new_cost < current_cost:
                    current_cost = new_cost
                    no_improvement_count = 0
                else:
                    # Revert
                    U[idx, c] = old_val
                    for k in range(p):
                        old_err = err[i, j, k]
                        delta = (new_val - old_val) * V[j * p + k, c] * W[k * n + i, c]
                        err[i, j, k] = old_err - delta
                    no_improvement_count += 1
            elif mat == 'V':
                idx = random.randint(0, mp - 1)
                c = random.randint(0, R - 1)
                old_val = V[idx, c]
                new_val = old_val + random.choice([-1, 1])
                if abs(new_val) > 3:
                    continue
                V[idx, c] = new_val
                j = idx // p
                k = idx % p
                for i in range(n):
                    old_err = err[i, j, k]
                    delta = (new_val - old_val) * U[i * m + j, c] * W[k * n + i, c]
                    new_err = old_err + delta
                    new_cost = current_cost - old_err**2 + new_err**2
                    err[i, j, k] = new_err
                if new_cost < current_cost:
                    current_cost = new_cost
                    no_improvement_count = 0
                else:
                    V[idx, c] = old_val
                    for i in range(n):
                        old_err = err[i, j, k]
                        delta = (new_val - old_val) * U[i * m + j, c] * W[k * n + i, c]
                        err[i, j, k] = old_err - delta
                    no_improvement_count += 1
            else: # W
                idx = random.randint(0, pn - 1)
                c = random.randint(0, R - 1)
                old_val = W[idx, c]
                new_val = old_val + random.choice([-1, 1])
                if abs(new_val) > 3:
                    continue
                W[idx, c] = new_val
                k = idx // n
                i = idx % n
                for j in range(m):
                    old_err = err[i, j, k]
                    delta = (new_val - old_val) * U[i * m + j, c] * V[j * p + k, c]
                    new_err = old_err + delta
                    new_cost = current_cost - old_err**2 + new_err**2
                    err[i, j, k] = new_err
                if new_cost < current_cost:
                    current_cost = new_cost
                    no_improvement_count = 0
                else:
                    W[idx, c] = old_val
                    for j in range(m):
                        old_err = err[i, j, k]
                        delta = (new_val - old_val) * U[i * m + j, c] * V[j * p + k, c]
                        err[i, j, k] = old_err - delta
                    no_improvement_count += 1
        
        if current_cost < 1e-9:
            return U, V, W
        return None

    # Try ranks from 25 to 60
    for R in range(25, 61):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Try a few restarts
        for restart in range(50):
            if time.time() - start_time > time_limit - 2:
                break
            
            res = local_search(R, (time_limit - 2) - (time.time() - start_time))
            if res is not None:
                U, V, W = res
                # Verify
                U_list = convert_to_list(U)
                V_list = convert_to_list(V)
                W_list = convert_to_list(W)
                if verify_decomposition(U_list, V_list, W_list, R):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
