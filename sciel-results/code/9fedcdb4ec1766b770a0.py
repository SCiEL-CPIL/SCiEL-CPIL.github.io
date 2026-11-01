"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check that no spurious terms exist
    # For all (a,b,c) not in the valid set, the sum should be 0
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) not in valid:
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    if abs(val) > 1e-9:
                        return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    if time.time() > deadline - 10:
        return best_result
    
    # Try to find lower rank solutions using random search + local optimization
    # The tensor has 64 constraints. Try ranks from 32 down to 16.
    for target_rank in [32, 24, 16]:
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            
            # Random initialization with integers/half-integers
            np.random.seed(attempt)
            U_r = np.random.randint(-2, 3, size=(n*m, target_rank)) * 0.5
            V_r = np.random.randint(-2, 3, size=(m*p, target_rank)) * 0.5
            W_r = np.random.randint(-2, 3, size=(p*n, target_rank)) * 0.5
            
            # Iterate: for each constraint, try to adjust
            # Use a simple gradient-free approach: try flipping individual entries
            improved = True
            max_iter = 200
            for it in range(max_iter):
                if time.time() > deadline - 20:
                    break
                progress = False
                # Compute current errors
                errors = {}
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m+j
                            b = j*p+k
                            c = k*n+i
                            val = sum(U_r[a,r]*V_r[b,r]*W_r[c,r] for r in range(target_rank))
                            errors[(a,b,c)] = val - 1.0
                
                # Try to fix the largest error
                if errors:
                    max_err_key = max(errors, key=lambda x: abs(errors[x]))
                    if abs(errors[max_err_key]) < 1e-6:
                        break
                    
                    a, b, c = max_err_key
                    err = errors[max_err_key]
                    
                    # Try adjusting U[a, r], V[b, r], or W[c, r] for some r
                    best_improvement = abs(err)
                    best_change = None
                    for r in range(target_rank):
                        # Try changing U[a, r]
                        for delta in [-0.5, 0.5, -1.0, 1.0]:
                            new_val = U_r[a, r] + delta
                            if new_val != U_r[a, r]:
                                # Estimate improvement
                                new_err = err - delta * V_r[b, r] * W_r[c, r]
                                if abs(new_err) < best_improvement:
                                    best_improvement = abs(new_err)
                                    best_change = ('U', a, r, new_val)
                        # Try changing V[b, r]
                        for delta in [-0.5, 0.5, -1.0, 1.0]:
                            new_val = V_r[b, r] + delta
                            if new_val != V_r[b, r]:
                                new_err = err - delta * U_r[a, r] * W_r[c, r]
                                if abs(new_err) < best_improvement:
                                    best_improvement = abs(new_err)
                                    best_change = ('V', b, r, new_val)
                        # Try changing W[c, r]
                        for delta in [-0.5, 0.5, -1.0, 1.0]:
                            new_val = W_r[c, r] + delta
                            if new_val != W_r[c, r]:
                                new_err = err - delta * U_r[a, r] * V_r[b, r]
                                if abs(new_err) < best_improvement:
                                    best_improvement = abs(new_err)
                                    best_change = ('W', c, r, new_val)
                    
                    if best_change is not None:
                        mat, idx, r, new_val = best_change
                        if mat == 'U':
                            U_r[idx, r] = new_val
                        elif mat == 'V':
                            V_r[idx, r] = new_val
                        else:
                            W_r[idx, r] = new_val
                        progress = True
                
                if not progress:
                    break
            
            # Check if solution is valid
            U_list = U_r.tolist()
            V_list = V_r.tolist()
            W_list = W_r.tolist()
            if _verify(U_list, V_list, W_list, n, m, p):
                best_rank = target_rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
        
        if best_rank == target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
