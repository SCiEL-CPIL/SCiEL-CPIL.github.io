import numpy as np
import random
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_solution = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Try to find a better decomposition using local search over F_7
    # We search for ranks from 50 up to 99
    # If we find one, we return it.
    
    field_size = 7
    field_elements = list(range(field_size))
    
    def tensor_error(U, V, W, rank):
        # Calculate the error tensor E[i,j,k] = T[i,j,k] - sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # T[i,j,k] is 1 if this is a valid index in the multiplication tensor, else 0.
        # The tensor T for matrix multiplication A(n,m) * B(m,p) -> C(p,n) is defined as:
        # T[i, j, k] = 1 if the term A[i,j] * B[j,k] contributes to C[k,i].
        # Wait, the standard tensor is T[i,j,k] where i is row of A, j is col of A/row of B, k is col of B.
        # The product is C[k,i] += A[i,j] B[j,k].
        # So the tensor entry is 1 for all i,j,k.
        # The decomposition is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
        
        # We check if sum_c ... == 1 for all i,j,k.
        # Actually, the tensor T has 1s at all positions (i,j,k) in the index space.
        # So we want sum_c ... == 1.
        
        # We return the number of mismatches.
        mismatches = 0
        # Precompute W transposed for efficiency? W is (p*n, rank). Index k*n+i.
        # U is (n*m, rank). Index i*m+j.
        # V is (m*p, rank). Index j*p+k.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0
                    for c in range(rank):
                        s += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                    s = s % field_size
                    if s != 1:
                        mismatches += 1
        return mismatches

    def tensor_error_vectorized(U, V, W, rank):
        # Vectorized error calculation using numpy
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U_np.reshape(n, m, rank)
        V_reshaped = V_np.reshape(m, p, rank)
        W_reshaped = W_np.reshape(p, n, rank)
        
        # We need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] == 1 for all i,j,k
        # This is a 3D tensor contraction
        # Let's compute it by iterating over i, j, k but using numpy for the sum over c
        
        total_errors = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum over c of U[i,j,c] * V[j,k,c] * W[k,i,c]
                    s = np.sum(U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :])
                    if s % field_size != 1:
                        total_errors += 1
        return total_errors

    start_time = time.time()
    max_time = 110  # Leave 10s buffer
    
    # Try ranks from 50 to best_rank - 1
    for R in range(50, best_rank):
        if time.time() - start_time > max_time:
            break
            
        # Initialize random decomposition
        U = [[random.choice(field_elements) for _ in range(R)] for _ in range(n * m)]
        V = [[random.choice(field_elements) for _ in range(R)] for _ in range(m * p)]
        W = [[random.choice(field_elements) for _ in range(R)] for _ in range(p * n)]
        
        # Local search
        current_err = tensor_error(U, V, W, R)
        no_improve_count = 0
        max_iter = 2000
        
        while current_err > 0 and no_improve_count < 50 and time.time() - start_time < max_time:
            # Pick a random matrix and entry to perturb
            mat = random.choice([0, 1, 2])
            if mat == 0:
                idx = random.randint(0, n * m - 1)
                col = random.randint(0, R - 1)
                old_val = U[idx][col]
                new_val = random.choice(field_elements)
                if new_val == old_val:
                    continue
                U[idx][col] = new_val
                new_err = tensor_error(U, V, W, R)
                if new_err < current_err:
                    current_err = new_err
                    no_improve_count = 0
                else:
                    U[idx][col] = old_val
                    no_improve_count += 1
            elif mat == 1:
                idx = random.randint(0, m * p - 1)
                col = random.randint(0, R - 1)
                old_val = V[idx][col]
                new_val = random.choice(field_elements)
                if new_val == old_val:
                    continue
                V[idx][col] = new_val
                new_err = tensor_error(U, V, W, R)
                if new_err < current_err:
                    current_err = new_err
                    no_improve_count = 0
                else:
                    V[idx][col] = old_val
                    no_improve_count += 1
            else:
                idx = random.randint(0, p * n - 1)
                col = random.randint(0, R - 1)
                old_val = W[idx][col]
                new_val = random.choice(field_elements)
                if new_val == old_val:
                    continue
                W[idx][col] = new_val
                new_err = tensor_error(U, V, W, R)
                if new_err < current_err:
                    current_err = new_err
                    no_improve_count = 0
                else:
                    W[idx][col] = old_val
                    no_improve_count += 1
            
            if current_err == 0:
                break
        
        if current_err == 0:
            # We found a valid decomposition of rank R
            # Convert to integers (they are already 0..6)
            # The problem requires integer/half-integer entries. 0..6 are integers.
            best_solution = {"U": U, "V": V, "W": W}
            best_rank = R
            break # Found the lowest rank we tried, stop searching lower? No, we iterate R upwards.
            # Actually, we want the MINIMAL rank. So we should try smaller R first.
            # My loop is `for R in range(50, best_rank)`. This tries 50, 51, ...
            # If we find R=50, we break and return 50.
            # If we don't find 50, we try 51, etc.
            # This is correct.

    return best_solution
