"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    start_time = time.time()
    time_limit = 130.0  # Leave 20 seconds margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
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
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact using the correct tensor structure."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rank = U_arr.shape[1]
        
        # Correct constraint:
        # sum_r W[k*n+i, r] * U[i'*m+j, r] * V[j'*p+k', r] = delta_{i',i} * delta_{j,j'} * delta_{k',k}
        # for all i, i', j, j', k, k'
        
        for i in range(n):
            for i_prime in range(n):
                for j in range(m):
                    for j_prime in range(m):
                        for k in range(p):
                            for k_prime in range(p):
                                expected = 1.0 if (i_prime == i and j == j_prime and k_prime == k) else 0.0
                                s = 0.0
                                for r in range(rank):
                                    s += W_arr[k*n+i, r] * U_arr[i_prime*m+j, r] * V_arr[j_prime*p+k_prime, r]
                                if abs(s - expected) > tol:
                                    return False
        return True
    
    def verify_fast(U, V, W, tol=1e-9):
        """Faster verification using numpy with correct tensor structure."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rank = U_arr.shape[1]
        
        # Reshape for easier manipulation
        # U_arr[i*m+j, r] -> U_reshaped[i, j, r]
        U_reshaped = U_arr.reshape(n, m, rank)
        # V_arr[j*p+k, r] -> V_reshaped[j, k, r]  
        V_reshaped = V_arr.reshape(m, p, rank)
        # W_arr[k*n+i, r] -> W_reshaped[k, i, r]
        W_reshaped = W_arr.reshape(p, n, rank)
        
        # For each (i, k), check:
        # sum_r W_reshaped[k, i, r] * U_reshaped[i_prime, j, r] * V_reshaped[j_prime, k_prime, r]
        # = delta_{i',i} * delta_{j,j'} * delta_{k',k}
        
        for i in range(n):
            for k in range(p):
                # Build the matrix M[i_prime, j, j_prime, k_prime] = sum_r W_reshaped[k,i,r] * U_reshaped[i_prime,j,r] * V_reshaped[j_prime,k_prime,r]
                # This should equal delta_{i',i} * delta_{j,j'} * delta_{k',k}
                
                # M[i_prime, j, j_prime, k_prime] = sum_r W_reshaped[k,i,r] * U_reshaped[i_prime,j,r] * V_reshaped[j_prime,k_prime,r]
                # = (U_reshaped[:, :, :] * W_reshaped[k,i,:][None,None,:]).sum(axis=2) outer (V_reshaped[:, :, :].T * W_reshaped[k,i,:])
                
                # Actually, let's compute it directly
                # For fixed (i,k), define:
                # A_r = U_reshaped[:, :, r] * W_reshaped[k, i, r]  shape (n, m)
                # B_r = V_reshaped[:, :, r] * W_reshaped[k, i, r]  shape (m, p)
                # Then M = sum_r A_r[:, :, None, None] * B_r[None, None, :, :]
                # M[i_prime, j, j_prime, k_prime] = sum_r A_r[i_prime, j] * B_r[j_prime, k_prime]
                
                M = np.zeros((n, m, m, p))
                for r in range(rank):
                    w_val = W_reshaped[k, i, r]
                    if w_val != 0:
                        A_r = U_reshaped[:, :, r] * w_val  # (n, m)
                        B_r = V_reshaped[:, :, r] * w_val  # (m, p)
                        M += A_r[:, :, None, None] * B_r[None, None, :, :]
                
                # Expected: delta_{i',i} * delta_{j,j'} * delta_{k',k}
                # This is 1 when i_prime=i, j=j_prime, k_prime=k, else 0
                # So M[i_prime, j, j_prime, k_prime] should be 1 iff i_prime==i and j==j_prime and k_prime==k
                
                for i_prime in range(n):
                    for j in range(m):
                        for j_prime in range(m):
                            for k_prime in range(p):
                                expected = 1.0 if (i_prime == i and j == j_prime and k_prime == k) else 0.0
                                if abs(M[i_prime, j, j_prime, k_prime] - expected) > tol:
                                    return False
        return True
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Strategy: Try to find lower-rank decompositions
    # The tensor we're decomposing is the all-ones tensor of size n x m x p
    # (because sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k)
    
    # Wait - I need to reconsider. The all-ones tensor has rank 1 (it's an outer product).
    # But matrix multiplication tensor is NOT the all-ones tensor.
    
    # Let me reconsider the problem. The matrix multiplication tensor is:
    # T = sum_{i,j,k} A_{i,j}^* ⊗ B_{j,k}^* ⊗ C_{i,k}^*
    # where the sum is over all valid (i,j,k) triples.
    
    # For A (n x m), B (m x p), C (n x p):
    # i ranges over [0,n), j over [0,m), k over [0,p)
    # And the coefficient is 1 for each such triple.
    
    # So yes, T[i,j,k] = 1 for all valid (i,j,k), which means T is the all-ones tensor.
    
    # But wait, that would give rank 1, which is clearly wrong.
    
    # I think the issue is that the tensor structure is more subtle. Let me reconsider.
    
    # Actually, I believe the correct formulation is:
    # The matrix multiplication tensor lives in (R^{n*m})^* ⊗ (R^{m*p})^* ⊗ (R^{n*p})^*
    # and is defined by:
    # T = sum_{j=0}^{m-1} (sum_{i=0}^{n-1} e_i ⊗ e_j) ⊗ (sum_{k=0}^{p-1} e_j ⊗ e_k) ⊗ ... 
    # No, that's not right either.
    
    # Let me think about this differently. The standard formulation:
    # We want to compute C = A * B where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T is an element of (R^{n*m})^* ⊗ (R^{m*p})^* ⊗ (R^{n*p})^*
    # such that for any A, B, C:
    # <T, A ⊗ B ⊗ C> = sum_{i,k} C[i,k] * (sum_j A[i,j] * B[j,k] - C[i,k])
    # Wait, that's not quite right either.
    
    # Actually, the correct formulation for the matrix multiplication tensor:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} (e_i^A ⊗ e_j^A) ⊗ (e_j^B ⊗ e_k^B) ⊗ (e_i^C ⊗ e_k^C)
    
    # Wait, I'm getting confused with the indexing. Let me use a cleaner formulation.
    
    # Let A have entries A_{i,j} for i in [n], j in [m]
    # Let B have entries B_{j,k} for j in [m], k in [p]  
    # Let C have entries C_{i,k} for i in [n], k in [p]
    
    # The relation is: C_{i,k} = sum_{j} A_{i,j} * B_{j,k}
    
    # The tensor T in (R^{n*m})^* ⊗ (R^{m*p})^* ⊗ (R^{n*p})^* is:
    # T = sum_{i,j,k} e_{i,j}^A ⊗ e_{j,k}^B ⊗ e_{i,k}^C
    # where e_{i,j}^A is the dual basis element picking out A_{i,j}, etc.
    
    # So T[i,j,k] = 1 for all valid (i,j,k), and 0 otherwise.
    # Since all (i,j,k) with i in [n], j in [m], k in [p] are valid, T is the all-ones tensor.
    
    # The all-ones tensor J_{n,m,p} = 1_n ⊗ 1_m ⊗ 1_p has tensor rank 1.
    
    # This is clearly wrong for matrix multiplication. There must be something I'm missing.
    
    # OH WAIT. I think I see the issue. The problem is that in the tensor decomposition framework,
    # the W matrix uses a different indexing convention. The task says "W[k*n+i, r]" which means
    # W is indexed by (k, i) not (i, k). This is the "trace convention".
    
    # Let me re-examine. If W[k*n+i, r] corresponds to C[i,k], then the row index of W is k*n+i.
    # So W row k*n+i corresponds to the coefficient of C[i,k].
    
    # Now, the condition for the decomposition to be correct:
    # For the algorithm to correctly compute C = A*B, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = coefficient of A[i,j]*B[j,k] in C[i,k]
    
    # The coefficient of A[i,j]*B[j,k] in C[i,k] is 1 (since C[i,k] = sum_j' A[i,j']*B[j',k], 
    # and the term with j'=j is A[i,j]*B[j,k]).
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    # This IS the all-ones tensor condition. But the all-ones tensor has rank 1, which is wrong.
    
    # I must be fundamentally misunderstanding the tensor structure. Let me look at this from 
    # a completely different angle.
    
    # Actually, I think the issue is that the tensor is NOT the all-ones tensor. The key insight 
    # is that the SAME indices appear in different positions. Specifically:
    # - U is indexed by (i,j) where i is row of A, j is column of A
    # - V is indexed by (j,k) where j is row of B, k is column of B
    # - W is indexed by (k,i) where k is column of C, i is row of C (using the trace convention)
    
    # The condition is:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    
    # But this is still the all-ones tensor!
    
    # Unless... the tensor is not simply the all-ones tensor because of the way the indices are 
    # contracted. Let me think about this more carefully.
    
    # Actually, I think I've been making an error. The matrix multiplication tensor is NOT the 
    # all-ones tensor. Here's why:
    
    # The tensor T is defined such that for any matrices A, B, C:
    # <T, A ⊗ B ⊗ C> = tr(C^T * (A*B)) - tr(C^T * C) = 0 when C = A*B
    
    # No wait, that's not the right formulation either.
    
    # Let me try yet another approach. The standard result is that the matrix multiplication 
    # tensor M_{n,m,p} has rank between max(n*m, m*p, n*p) and n*m*p.
    
    # For (3,5,5): lower bound is 25, upper bound is 75.
    
    # I think the correct tensor is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} (e_i ⊗ e_j) ⊗ (e_j ⊗ e_k) ⊗ (e_k ⊗ e_i)
    
    # Wait, that doesn't make sense dimensionally.
    
    # OK let me just accept that I don't fully understand the tensor structure and try a different 
    # approach. Let me try to find a decomposition by optimization.
    
    # Actually, let me reconsider. I think the correct formulation is:
    # We want to find U, V, W such that:
    # For all i in [n], j in [m], k in [p]:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is because the product A[i,j]*B[j,k] contributes with coefficient 1 to C[i,k].
    
    # Now, is this really the all-ones tensor? Let's check with a small example.
    # For (1,1,1): U is 1x r, V is 1x r, W is 1x r.
    # Condition: sum_r U[0,r]*V[0,r]*W[0,r] = 1
    # This can be satisfied with r=1: U[0,0]=V[0,0]=W[0,0]=1.
    # Rank 1 for (1,1,1) makes sense.
    
    # For (1,2,2): U is 2x r, V is 4x r, W is 2x r.
    # Conditions:
    # i=0, j=0, k=0: U[0,0]*V[0,0]*W[0,0] + ... = 1
    # i=0, j=0, k=1: U[0,0]*V[1,0]*W[1,0] + ... = 1
    # i=0, j=1, k=0: U[1,0]*V[2,0]*W[0,0] + ... = 1
    # i=0, j=1, k=1: U[1,0]*V[3,0]*W[1,0] + ... = 1
    # And similarly for other i values (but n=1 so only i=0).
    
    # This is 4 equations. Can we satisfy them with r=1?
    # U[0,0]*V[0,0]*W[0,0] = 1
    # U[0,0]*V[1,0]*W[1,0] = 1
    # U[1,0]*V[2,0]*W[0,0] = 1
    # U[1,0]*V[3,0]*W[1,0] = 1
    
    # From eq 1: U[0,0]*V[0,0]*W[0,0] = 1
    # From eq 2: U[0,0]*V[1,0]*W[1,0] = 1
    # Dividing: V[1,0]*W[1,0] / (V[0,0]*W[0,0]) = 1, so V[1,0]*W[1,0] = V[0,0]*W[0,0]
    
    # From eq 3: U[1,0]*V[2,0]*W[0,0] = 1
    # From eq 4: U[1,0]*V[3,0]*W[1,0] = 1
    # Dividing: V[3,0]*W[1,0] / (V[2,0]*W[0,0]) = 1, so V[3,0]*W[1,0] = V[2,0]*W[0,0]
    
    # We have 4 equations and 8 unknowns (U[0,0], U[1,0], V[0,0], V[1,0], V[2,0], V[3,0], W[0,0], W[1,0]).
    # This should be solvable. For example:
    # U[0,0]=1, U[1,0]=1, V[0,0]=1, V[1,0]=1, V[2,0]=1, V[3,0]=1, W[0,0]=1, W[1,0]=1
    # Then all 4 equations are satisfied.
    
    # So rank 1 works for (1,2,2)? That can't be right for matrix multiplication.
    
    # I think I'm still misunderstanding the problem. Let me re-read the task.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 5 matrix by a 5 by 5 matrix."
    
    # I think the issue is that the tensor is NOT the all-ones tensor. The correct tensor for 
    # matrix multiplication has a more complex structure.
    
    # After more thought, I believe the correct formulation is:
    # The matrix multiplication tensor T is in (R^{n*m})^* ⊗ (R^{m*p})^* ⊗ (R^{n*p})^*
    # and is given by:
    # T = sum_{j=0}^{m-1} (sum_{i=0}^{n-1} e_{i,j}^A) ⊗ (sum_{k=0}^{p-1} e_{j,k}^B) ⊗ ... 
    # No, that's not right.
    
    # I give up trying to derive the tensor structure from first principles. Let me just try 
    # to find a decomposition by random search and verify it numerically.
    
    # The verification condition should be:
    # For all i in [n], j in [m], k in [p]:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let me just try random search with this condition.
    
    def random_search(rank, num_trials, use_half_ints=False):
        """Try random search for a valid decomposition."""
        for trial in range(num_trials):
            if time.time() - start_time > time_limit - 10:
                return None
            
            # Generate random entries
            vals = []
            for _ in range(rank * (n * m + m * p + p * n)):
                if use_half_ints:
                    val = random.choice([-2, -1, -0.5, 0, 0.5, 1, 2])
                else:
                    val = random.choice([-2, -1, 0, 1, 2])
                vals.append(val)
            
            idx = 0
            U = []
            for i in range(n * m):
                row = []
                for r_idx in range(rank):
                    row.append(vals[idx]); idx += 1
                U.append(row)
            V = []
            for i in range(m * p):
                row = []
                for r_idx in range(rank):
                    row.append(vals[idx]); idx += 1
                V.append(row)
            W = []
            for i in range(p * n):
                row = []
                for r_idx in range(rank):
                    row.append(vals[idx]); idx += 1
                W.append(row)
            
            if verify_fast(U, V, W):
                return {"U": U, "V": V, "W": W}
        return None
    
    def structured_search(rank, num_trials):
        """Try structured search exploiting the tensor symmetry."""
        for trial in range(num_trials):
            if time.time() - start_time > time_limit - 10:
                return None
            
            # Try to find a decomposition with some structure
            # For example, try to make U, V, W have sparse patterns
            
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Randomly assign non-zero entries
            for r in range(rank):
                # Pick a few random positions for U, V, W
                u_idx = random.sample(range(n * m), min(3, n * m))
                v_idx = random.sample(range(m * p), min(3, m * p))
                w_idx = random.sample(range(p * n), min(3, p * n))
                
                for idx in u_idx:
                    U[idx, r] = random.choice([-1, 1])
                for idx in v_idx:
                    V[idx, r] = random.choice([-1, 1])
                for idx in w_idx:
                    W[idx, r] = random.choice([-1, 1])
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify_fast(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        return None
    
    # Try ranks from the lower bound upward
    lower_bound = max(n * m, m * p, p * n)
    
    for rank in range(lower_bound, best_rank):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Try random search
        result = random_search(rank, num_trials=200, use_half_ints=True)
        if result is not None:
            best_result = result
            best_rank = rank
            break
        
        # Try structured search
        if time.time() - start_time < time_limit - 10:
            result = structured_search(rank, num_trials=200)
            if result is not None:
                best_result = result
                best_rank = rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
