"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    sb_rank = n * m * p  # 96
    
    # We will try to find a lower rank decomposition using randomized search
    # with the structure of matrix multiplication tensors.
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 110s out of 150s limit
    
    best_rank = sb_rank
    best_U, best_V, best_W = None, None, None
    
    # Initialize best with schoolbook
    U = np.zeros((n * m, sb_rank), dtype=np.float64)
    V = np.zeros((m * p, sb_rank), dtype=np.float64)
    W = np.zeros((p * n, sb_rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower rank decompositions
    # The tensor T[i,j,k] = delta_{i,k} for (i,j) in A, (j,k) in B -> C
    # Actually, the matrix multiplication tensor has entries T[i,j,k] = 1 if the
    # multiplication involves A[i,j]*B[j,k], which is always true for valid indices.
    # So T[i,j,k] = 1 for all valid (i,j,k).
    # Wait, that's not right. Let me reconsider.
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 if and only if
    # the product C[i,k] = sum_j A[i,j]*B[j,k] involves the term A[i,j]*B[j,k].
    # Since every (i,j,k) triple with valid indices contributes, T[i,j,k] = 1 for all.
    # No wait, that's the trivial all-ones tensor, which has rank 1.
    # That can't be right. Let me think again.
    
    # Actually, the matrix multiplication tensor is defined as:
    # T(A,B) = A*B where (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has components T[i,j,k] such that (A*B)[i,k] = sum_{j} A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k). This is the all-ones tensor of shape (n,m,p).
    # The rank of the all-ones tensor is 1 (it's just the outer product of three ones vectors).
    # But that's clearly wrong for matrix multiplication.
    
    # I think I'm confusing myself. Let me look at this differently.
    # The matrix multiplication map is: M: R^{n*m} x R^{m*p} -> R^{n*p}
    # M(A,B) = A*B
    # The associated tensor T has T[a,b,c] where a indexes A (n*m), b indexes B (m*p), c indexes C (n*p)
    # T[a,b,c] = 1 if the entry c of A*B depends on entries a of A and b of B.
    # Specifically, if a = i*m+j, b = j'*p+k, c = i'*n+k', then
    # T[a,b,c] = 1 if i=i' and k=k' and j=j', else 0.
    # So T[(i,j),(j,k),(i,k)] = 1, and 0 otherwise.
    # This is the standard matrix multiplication tensor.
    
    # Let's build the target tensor and try to decompose it.
    # T is a 3-tensor of shape (n*m, m*p, n*p)
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=i*n+k for some i,j,k
    
    # Build target tensor as a set of 1-entries
    target_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = i * n + k
                target_entries.add((a, b, c))
    
    # Now try to find a low-rank decomposition
    # Rank r means we can write T = sum_{t=1}^{r} u_t ⊗ v_t ⊗ w_t
    # where u_t is in R^{n*m}, v_t in R^{m*p}, w_t in R^{n*p}
    # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    
    # We'll use a randomized approach: try different ranks and see if we can find a solution.
    
    def check_decomposition(U_mat, V_mat, W_mat, tol=1e-8):
        """Check if U⊗V⊗W equals the target tensor."""
        # Compute T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
        # and check it matches target
        # For efficiency, we can check a subset of entries
        # But for correctness, we need to check all.
        # T has shape (n*m, m*p, n*p) = (16, 24, 24)
        # Total entries: 16*24*24 = 9216
        # For each entry, sum over t=1..r
        # This is O(n*m*m*p*n*p*r) which is manageable for small r.
        
        nm = n * m
        mp = m * p
        pn = p * n
        r = U_mat.shape[1]
        
        # Build T as a 3D array
        T = np.zeros((nm, mp, pn), dtype=np.float64)
        for t in range(r):
            T += np.einsum('a,b,c->abc', U_mat[:, t], V_mat[:, t], W_mat[:, t])
        
        # Check against target
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    expected = 1.0 if (a, b, c) in target_entries else 0.0
                    if abs(T[a, b, c] - expected) > tol:
                        return False
        return True
    
    def try_rank(r, max_iters=50, time_budget=20.0):
        """Try to find a rank-r decomposition using randomized optimization."""
        start = time.time()
        nm = n * m
        mp = m * p
        pn = p * n
        
        for it in range(max_iters):
            if time.time() > start + time_budget:
                break
            if time.time() > deadline:
                return None
            
            # Random initialization
            U_mat = np.random.randn(nm, r) * 0.5
            V_mat = np.random.randn(mp, r) * 0.5
            W_mat = np.random.randn(pn, r) * 0.5
            
            # Gradient descent on the objective: sum_{(a,b,c) in target} (T[a,b,c]-1)^2 + sum_{(a,b,c) not in target} T[a,b,c]^2
            # This is equivalent to minimizing ||T - T_target||^2 where T_target is the target tensor
            
            # Precompute target tensor
            T_target = np.zeros((nm, mp, pn), dtype=np.float64)
            for (a, b, c) in target_entries:
                T_target[a, b, c] = 1.0
            
            lr = 0.01
            for step in range(100):
                if time.time() > start + time_budget or time.time() > deadline:
                    break
                
                # Compute T = sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t]
                T = np.zeros((nm, mp, pn), dtype=np.float64)
                for t in range(r):
                    T += np.einsum('a,b,c->abc', U_mat[:, t], V_mat[:, t], W_mat[:, t])
                
                # Gradient
                grad_T = T - T_target
                
                # Gradient wrt U_mat[a,t] = sum_{b,c} grad_T[a,b,c] * V_mat[b,t] * W_mat[c,t]
                # = np.einsum('abc,bc->a', grad_T, V_mat[:,t]*W_mat[:,t])... 
                # More efficiently:
                for t in range(r):
                    # dT/du_a = sum_{b,c} grad_T[a,b,c] * v_b * w_c
                    # = (grad_T * (v⊗w))[a,:,:] summed over b,c
                    vw = np.outer(V_mat[:, t], W_mat[:, t])  # (mp, pn)
                    U_mat[:, t] -= lr * np.einsum('abc,bc->a', grad_T, vw)
                    
                    # dT/dv_b = sum_{a,c} grad_T[a,b,c] * u_a * w_c
                    uw = np.outer(U_mat[:, t], W_mat[:, t])  # (nm, pn)
                    V_mat[:, t] -= lr * np.einsum('abc,ac->b', grad_T, uw)
                    
                    # dT/dw_c = sum_{a,b} grad_T[a,b,c] * u_a * v_b
                    uv = np.outer(U_mat[:, t], V_mat[:, t])  # (nm, mp)
                    W_mat[:, t] -= lr * np.einsum('abc,ab->c', grad_T, uv)
                
                # Check if converged
                if np.max(np.abs(grad_T)) < 1e-10:
                    break
            
            if check_decomposition(U_mat, V_mat, W_mat, tol=1e-6):
                return (U_mat, V_mat, W_mat)
        
        return None
    
    # Try ranks from 80 down to 96 (schoolbook)
    # We know rank 96 works (schoolbook)
    # Let's try to find if lower ranks work
    
    # Actually, let's first verify the schoolbook works
    if check_decomposition(best_U, best_V, best_W, tol=1e-8):
        pass  # Good
    
    # Try lower ranks
    for r in range(80, 96):
        if time.time() > deadline - 5:
            break
        
        result = try_rank(r, max_iters=30, time_budget=min(15.0, max(0, deadline - time.time() - 5)))
        if result is not None:
            U_mat, V_mat, W_mat = result
            best_rank = r
            best_U, best_V, best_W = U_mat.copy(), V_mat.copy(), W_mat.copy()
            break  # Found a lower rank, use it
    
    # Convert to lists of lists with integer/half-integer entries if possible
    def to_half_integer_list(mat):
        """Convert numpy matrix to list of lists, rounding to nearest half-integer if close."""
        rows = []
        for row in mat:
            new_row = []
            for val in row:
                # Round to nearest 0.5
                rounded = round(val * 2) / 2.0
                if abs(val - rounded) < 1e-6:
                    new_row.append(rounded)
                else:
                    new_row.append(round(val, 10))
            rows.append(new_row)
        return rows
    
    U_list = to_half_integer_list(best_U)
    V_list = to_half_integer_list(best_V)
    W_list = to_half_integer_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
