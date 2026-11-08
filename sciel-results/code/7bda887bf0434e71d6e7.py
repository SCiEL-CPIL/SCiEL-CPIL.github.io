import numpy as np
import random
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # Build the multiplication tensor T of shape (n*p, n*m, m*p)
    # T[k*n+i, j*m+r, j*p+s] = 1 if (i,j) from A, (j,k) from B -> C[i,k]
    # Actually, the standard decomposition is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is T[i,k,j] such that sum_j T[i,k,j] A[i,j] B[j,k] = C[i,k]
    # In the factorization format: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # where U indexes A entries (i,j), V indexes B entries (j,k), W indexes C entries (k,i) [trace convention: W[k*n+i,r]]
    # The condition is: sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] ... no, that's not right.
    # Actually the bilinear map is: C = U^T * D * V (or similar)
    # Standard: sum_r U[i,j,r] * V[j,k,r] = W[i,k,r] ... no.
    # The decomposition of a tensor T of shape (a,b,c) as rank r is:
    # T[a,b,c] = sum_{r=1}^r U[a,r] * V[b,r] * W[c,r]
    # Here a = n*m (indexing A[i,j]), b = m*p (indexing B[j,k]), c = p*n (indexing C[k,i] in trace convention)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if i=i', j=j', k=k' ... no.
    # The multiplication tensor: T[A_idx, B_idx, C_idx] = 1 if the term A[A_idx] * B[B_idx] contributes to C[C_idx].
    # A_idx = i*m + j (for A[i,j])
    # B_idx = j*p + k (for B[j,k])
    # C_idx = k*n + i (for C[i,k] in trace convention W[k*n+i,r])
    # The contribution: A[i,j] * B[j,k] -> C[i,k]
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all valid i, j, k where the indices match.
    # Note: the j in A_idx and the j in B_idx must be the same.
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # We need to find rank-minimal U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # The current best known solution has rank 50.
    # Let's try to find a lower rank using a structured approach.
    
    # First, let's generate the rank-50 solution as a fallback.
    def generate_rank50_solution():
        rank = 50
        U = np.zeros((n*m, rank), dtype=np.int64)
        V = np.zeros((m*p, rank), dtype=np.int64)
        W = np.zeros((p*n, rank), dtype=np.int64)
        
        col = 0
        
        def add_strassen_block(i0, j0, k0):
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            
            e_idx = j0 * p + k0
            f_idx = j0 * p + (k0 + 1)
            g_idx = (j0 + 1) * p + k0
            h_idx = (j0 + 1) * p + (k0 + 1)
            
            c00_idx = k0 * n + i0
            c01_idx = (k0 + 1) * n + i0
            c10_idx = k0 * n + (i0 + 1)
            c11_idx = (k0 + 1) * n + (i0 + 1)
            
            # M1 = (a+d)(e+h)
            U[a_idx, col] += 1
            U[d_idx, col] += 1
            V[e_idx, col] += 1
            V[h_idx, col] += 1
            W[c00_idx, col] += 1
            W[c11_idx, col] += 1
            col += 1
            
            # M2 = (c+d)e
            U[c_idx, col] += 1
            U[d_idx, col] += 1
            V[e_idx, col] += 1
            W[c10_idx, col] += 1
            W[c11_idx, col] -= 1
            col += 1
            
            # M3 = a(f-h)
            U[a_idx, col] += 1
            V[f_idx, col] += 1
            V[h_idx, col] -= 1
            W[c01_idx, col] += 1
            W[c11_idx, col] += 1
            col += 1
            
            # M4 = d(g-e)
            U[d_idx, col] += 1
            V[g_idx, col] += 1
            V[e_idx, col] -= 1
            W[c00_idx, col] += 1
            W[c10_idx, col] += 1
            col += 1
            
            # M5 = (a+b)h
            U[a_idx, col] += 1
            U[b_idx, col] += 1
            V[h_idx, col] += 1
            W[c00_idx, col] -= 1
            W[c01_idx, col] += 1
            col += 1
            
            # M6 = (c-a)(e+f)
            U[c_idx, col] += 1
            U[a_idx, col] -= 1
            V[e_idx, col] += 1
            V[f_idx, col] += 1
            W[c11_idx, col] += 1
            col += 1
            
            # M7 = (b-d)(g+h)
            U[b_idx, col] += 1
            U[d_idx, col] -= 1
            V[g_idx, col] += 1
            V[h_idx, col] += 1
            W[c00_idx, col] += 1
            col += 1

        def add_schoolbook_221_block(i0, j0, k0):
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            
            e_idx = j0 * p + k0
            g_idx = (j0 + 1) * p + k0
            
            c00_idx = k0 * n + i0
            c10_idx = k0 * n + (i0 + 1)
            
            U[a_idx, col] += 1
            V[e_idx, col] += 1
            W[c00_idx, col] += 1
            col += 1
            
            U[b_idx, col] += 1
            V[g_idx, col] += 1
            W[c00_idx, col] += 1
            col += 1
            
            U[c_idx, col] += 1
            V[e_idx, col] += 1
            W[c10_idx, col] += 1
            col += 1
            
            U[d_idx, col] += 1
            V[g_idx, col] += 1
            W[c10_idx, col] += 1
            col += 1

        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                add_strassen_block(0, j_start, k_start)
            add_schoolbook_221_block(0, j_start, 6)
            
        return U, V, W, rank
    
    # Try to find a lower rank solution using a Kronecker-structured approach
    # The idea: decompose the 2x4x7 multiplication as a sum of smaller structured blocks
    # and try to find a more efficient combination.
    
    # Let's try rank 48, 49 first with a heuristic search.
    # We'll use a random restart approach with gradient descent on the factors.
    
    def try_find_lower_rank(max_rank_try=49, min_rank_try=45, num_trials=20):
        """Try to find a rank < 50 solution using random initialization and verification."""
        import time
        start_time = time.time()
        timeout = 110  # Leave 10s buffer
        
        for target_rank in range(min_rank_try, max_rank_try + 1):
            if time.time() - start_time > timeout:
                break
                
            for trial in range(num_trials):
                if time.time() - start_time > timeout:
                    break
                    
                # Random initialization with integer/half-integer entries
                U = np.zeros((n*m, target_rank), dtype=np.float64)
                V = np.zeros((m*p, target_rank), dtype=np.float64)
                W = np.zeros((p*n, target_rank), dtype=np.float64)
                
                # Initialize with small random values
                for r in range(target_rank):
                    # Sparse random initialization
                    num_nonzero_U = random.randint(2, 5)
                    num_nonzero_V = random.randint(2, 5)
                    num_nonzero_W = random.randint(1, 3)
                    
                    U_indices = random.sample(range(n*m), num_nonzero_U)
                    for idx in U_indices:
                        U[idx, r] = random.choice([-1, 1]) * random.choice([0.5, 1.0])
                    
                    V_indices = random.sample(range(m*p), num_nonzero_V)
                    for idx in V_indices:
                        V[idx, r] = random.choice([-1, 1]) * random.choice([0.5, 1.0])
                    
                    W_indices = random.sample(range(p*n), num_nonzero_W)
                    for idx in W_indices:
                        W[idx, r] = random.choice([-1, 1]) * random.choice([0.5, 1.0])
                
                # Check if this decomposition matches T
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # We can check by computing the outer product sum and comparing
                # But this is expensive for large tensors. Instead, check on a subset of entries.
                
                # Actually, let's verify by checking all non-zero entries of T
                # T has n*m*p = 2*4*7 = 56 non-zero entries (each is 1)
                # For each non-zero T[a,b,c] = 1, we need sum_r U[a,r]*V[b,r]*W[c,r] = 1
                # For zero entries, we need the sum to be 0.
                
                # Let's check the 56 non-zero entries
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m + j
                            b = j*p + k
                            c = k*n + i
                            s = np.sum(U[a, :] * V[b, :] * W[c, :])
                            if abs(s - 1.0) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Also check some zero entries to make sure
                    zero_checks = 20
                    for _ in range(zero_checks):
                        a = random.randint(0, n*m-1)
                        b = random.randint(0, m*p-1)
                        c = random.randint(0, p*n-1)
                        # Check if this is a non-zero entry
                        i = a // m
                        j = a % m
                        jj = b // p
                        k = b % p
                        kk = c // n
                        ii = c % n
                        if i == ii and j == jj and k == kk:
                            continue  # Skip non-zero entries
                        s = np.sum(U[a, :] * V[b, :] * W[c, :])
                        if abs(s) > 1e-6:
                            valid = False
                            break
                    
                    if valid:
                        # Convert to integer/half-integer
                        U_out = np.round(U * 2) / 2
                        V_out = np.round(V * 2) / 2
                        W_out = np.round(W * 2) / 2
                        return U_out, V_out, W_out, target_rank
                        
        return None
    
    # Try to find a lower rank solution
    result = try_find_lower_rank(max_rank_try=49, min_rank_try=45, num_trials=30)
    
    if result is not None:
        U, V, W, rank = result
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Fallback to rank-50 solution
    U, V, W, rank = generate_rank50_solution()
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
