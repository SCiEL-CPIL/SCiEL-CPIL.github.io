"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import itertools
import random
import math

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for r in range(rank):
        for row in range(len(U)):
            if len(U[row]) != rank:
                return False
        for row in range(len(V)):
            if len(V[row]) != rank:
                return False
        for row in range(len(W)):
            if len(W[row]) != rank:
                return False
    # Verify: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # And for all (i,j,k) where we don't want a 1, it should be 0.
    # The tensor is T[i,j,k] = 1 if this is a valid matmul index, else 0.
    # Actually for 3x3 * 3x7, the tensor has entries T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,7)
    # because (AB)[i,k] = sum_j A[i,j]*B[j,k], so the tensor is T[i,j,k] = 1 for all valid i,j,k.
    # Wait, no. The matrix multiplication tensor T has T[i,j,k] = 1 if we're computing (AB)[i,k] from A[i,j] and B[j,k].
    # For multiplying n x m by m x p, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # So we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    # And there are no other constraints because every (i,j,k) triple corresponds to a term in the matmul.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _is_integer_or_half_integer(x):
    """Check if x is an integer or half-integer (multiple of 0.5)."""
    # Check if 2*x is close to an integer
    two_x = 2 * x
    nearest = round(two_x)
    return abs(two_x - nearest) < 1e-9

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with schoolbook
    schoolbook_rank = n * m * p
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions using random search + local optimization
    # The matmul tensor for 3x3 * 3x7 has 63 non-zero entries (all 1s)
    # We need rank r such that there exist U (9 x r), V (21 x r), W (21 x r) with
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Strategy: Use the fact that we can try ranks from known lower bounds up
    # For 3x3*3x7, the Strassen-style or other algorithms might give lower rank
    # Lower bound: at least max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Also, the tensor rank is at least the max of the three matrix dimensions products
    
    # Let's try a random initialization + gradient-free local search approach
    # For each candidate rank r, try to find a valid decomposition
    
    def try_rank(r, max_iterations=50000):
        """Try to find a rank-r decomposition using random restarts and local search."""
        nonlocal deadline
        for restart in range(200):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small values
            U = [[random.gauss(0, 0.5) for _ in range(r)] for _ in range(n*m)]
            V = [[random.gauss(0, 0.5) for _ in range(r)] for _ in range(m*p)]
            W = [[random.gauss(0, 0.5) for _ in range(r)] for _ in range(p*n)]
            
            # Local search: randomly perturb and keep if error decreases
            def compute_error():
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for rr in range(r):
                                s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                            err += (s - 1.0) ** 2
                return err
            
            current_err = compute_error()
            if current_err < 1e-6:
                return U, V, W
            
            step_size = 0.1
            for it in range(max_iterations):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random tensor to perturb
                tensor_choice = random.choice(['U', 'V', 'W'])
                if tensor_choice == 'U':
                    row = random.randint(0, n*m-1)
                    col = random.randint(0, r-1)
                    old_val = U[row][col]
                    U[row][col] += random.gauss(0, step_size)
                    new_err = compute_error()
                    if new_err < current_err:
                        current_err = new_err
                        if current_err < 1e-6:
                            return U, V, W
                    else:
                        U[row][col] = old_val
                elif tensor_choice == 'V':
                    row = random.randint(0, m*p-1)
                    col = random.randint(0, r-1)
                    old_val = V[row][col]
                    V[row][col] += random.gauss(0, step_size)
                    new_err = compute_error()
                    if new_err < current_err:
                        current_err = new_err
                        if current_err < 1e-6:
                            return U, V, W
                    else:
                        V[row][col] = old_val
                else:
                    row = random.randint(0, p*n-1)
                    col = random.randint(0, r-1)
                    old_val = W[row][col]
                    W[row][col] += random.gauss(0, step_size)
                    new_err = compute_error()
                    if new_err < current_err:
                        current_err = new_err
                        if current_err < 1e-6:
                            return U, V, W
                    else:
                        W[row][col] = old_val
                
                # Occasionally reduce step size
                if it % 1000 == 0:
                    step_size *= 0.95
            
            # If we got close, try to snap to half-integers
            if current_err < 0.1:
                # Try rounding to nearest half-integer
                U2 = [[round(2*x)/2 for x in row] for row in U]
                V2 = [[round(2*x)/2 for x in row] for row in V]
                W2 = [[round(2*x)/2 for x in row] for row in W]
                if _verify_decomposition(U2, V2, W2, n, m, p):
                    return U2, V2, W2
        
        return None
    
    # Try ranks from lower bound to schoolbook
    # Lower bound for 3x3*3x7: at least 21 (max dimension product)
    # Let's try a few promising ranks
    for r in range(21, min(schoolbook_rank, 45)):
        if time.time() > deadline - 10:
            break
        result = try_rank(r, max_iterations=10000)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                # Check half-integer constraint
                valid = True
                for mat in [U, V, W]:
                    for row in mat:
                        for x in row:
                            if not _is_integer_or_half_integer(x):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = r
                    break  # Found lowest rank, stop
    
    return best_result
# EVOLVE-BLOCK-END
