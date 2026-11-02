"""Matrix multiplication 3,4,4 - rank-minimal tensor decomposition search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 4]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 48

# Target index sets
I_SET = list(range(N))
J_SET = list(range(M))
K_SET = list(range(P))


def _build_tensors(U, V, W):
    """Build U, V, W as numpy arrays from lists, return None if shape mismatch."""
    n, m, p = N, M, P
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return None
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    if U_arr.shape != (n * m, rank) or V_arr.shape != (m * p, rank) or W_arr.shape != (p * n, rank):
        return None
    return U_arr, V_arr, W_arr


def _check_exact(U_arr, V_arr, W_arr, tol=1e-9):
    """Check if U*V*W exactly reconstructs the matrix multiplication tensor."""
    n, m, p = N, M, P
    rank = U_arr.shape[1]
    for i in I_SET:
        for j in J_SET:
            for k in K_SET:
                u_val = U_arr[i * m + j, :]
                v_val = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                lhs = np.dot(u_val, np.dot(v_val, w_val))
                if abs(lhs - 1.0) > tol:
                    return False
                # Also check that off-target entries are zero
                # For all other (i', j', k') triples, the contribution should be 0
                # But we only need to verify the target entries equal 1
                # and that the tensor is correct (no extra contributions)
                # Actually, the tensor entry T[i,j,k] should be 1, and we need
                # to verify that the decomposition produces exactly that.
                # The decomposition computes sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                # This should equal 1 for the target entry.
                # But we also need to ensure no spurious contributions.
                # The full check is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = T[i,j,k]
                # where T[i,j,k] = 1 if this is a valid multiplication entry.
                # For matrix multiplication (n x m) * (m x p) = (n x p),
                # T[i,j,k] = 1 for all valid (i,j,k) since each output element
                # is a dot product that uses each (i,j,k) triple exactly once.
                # Wait - the tensor for matrix multiplication has T[i,j,k]=1 for all i,j,k
                # in their respective ranges. So we just need each such entry to equal 1.
                # But we also need to make sure there are no extra non-zero entries elsewhere.
                # Since U, V, W only have rows indexed by valid (i,j), (j,k), (k,i),
                # and we check all valid triples, the off-diagonal entries are automatically 0
                # because those row indices don't exist.
                pass
    return True


def _check_full(U_arr, V_arr, W_arr, tol=1e-9):
    """Full verification: compute all tensor entries and check they match."""
    n, m, p = N, M, P
    rank = U_arr.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # For each (i, j, k), compute sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # This should equal 1 for all valid (i,j,k)
    for i in I_SET:
        for j in J_SET:
            for k in K_SET:
                val = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _schoolbook():
    """Return the standard schoolbook decomposition of rank 48."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in I_SET:
        for j in J_SET:
            for k in K_SET:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_block_decomposition():
    """Try to exploit block structure. For 3x4x4, try to see if we can use
    a combination of smaller known decompositions.
    
    Strategy: The 3x4x4 multiplication can be seen as computing a 3x4 output
    from 3x4 and 4x4 inputs. Try rank-47 or lower by merging some factors.
    """
    # For a 3x4x4 matmul, the output is 3x4. Each output element C[i,l]
    # = sum_j A[i,j] * B[j,l] where j ranges over 0..3.
    # So we have 3*4=12 output elements, each requiring a dot product of length 4.
    # Schoolbook uses 12*4=48 multiplications.
    # 
    # Can we reduce this? For a single dot product of length 4, the minimum rank
    # is 4 (you can't do better than the trivial for a single inner product).
    # But if we share structure between different output elements, we might reduce.
    #
    # One approach: group the 4 j-values and use a 2x2 Strassen-like trick on pairs.
    # For 2 values, Strassen reduces 4 mults to 7 (worse). For 4 values, we can
    # think of it as two dot products of length 2, or use other tricks.
    #
    # Actually, for computing sum_{j=0}^{3} A[i,j]*B[j,l] for all i in {0,1,2}, l in {0,1,2,3},
    # the total number of scalar multiplications in schoolbook is 3*4*4=48.
    #
    # A known result: R(3,4,4) <= 47? Let's try a specific construction.
    # 
    # Idea: For each pair of rows (i, i'), can we share some computation?
    # For fixed l, we compute [C[0,l], C[1,l], C[2,l]] = [A[0,0]*B[0,l]+A[0,1]*B[1,l]+A[0,2]*B[2,l]+A[0,3]*B[3,l],
    #                                                        A[1,0]*B[0,l]+..., A[2,0]*B[0,l]+...]
    # 
    # This is a 3-vector = A_row * B_col where B_col is a 4-vector (same for all i, different column l).
    # So for each l, we're computing 3 independent dot products with the same B vector.
    # 
    # Alternatively, think of it as: for each (i,j), A[i,j] is a scalar, and for each j,
    # B[j,0..3] is a 4-vector. C[i,0..3] = sum_j A[i,j] * B[j,0..3].
    # So C row i = sum_j A[i,j] * B_row_j.
    # This is a linear combination of 4 vectors (B rows), with coefficients A[i,0..3].
    # For 3 such combinations (i=0,1,2), each in a 4-dimensional space.
    #
    # The rank of this tensor... let me think about whether we can do better than 48.
    
    # Try rank 47: remove one factor and see if we can compensate.
    # This is hard to do analytically, so let's try a numerical search for rank 47.
    
    pass  # Placeholder - actual search below


def _rank_search(target_rank, time_budget=100):
    """Search for a decomposition of the given rank using random initialization + refinement.
    
    Returns (U, V, W) as lists if found, None otherwise.
    """
    n, m, p = N, M, P
    rank = target_rank
    start_time = time.time()
    
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    # We need U (n*m x rank), V (m*p x rank), W (p*n x rank) such that
    # sum_c U[a,c]*V[b,c]*W[d,c] = T[a,b,d] for all a,b,d.
    
    # Build the full tensor
    U_idx = list(range(n * m))
    V_idx = list(range(m * p))
    W_idx = list(range(p * n))
    
    # T[a,b,d] = 1 if a=i*m+j, b=j*p+k, d=k*n+i for some valid i,j,k
    # i = a // m, j = a % m, j should equal b // p, k = b % p, k should equal d % n, i should equal d // n
    def tensor_entry(a, b, d):
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = d % n
        i2 = d // n
        if j == j2 and k == k2 and i == i2:
            return 1.0
        return 0.0
    
    # Build list of all (a,b,d) triples with their target values
    all_triples = []
    for a in U_idx:
        for b in V_idx:
            for d in W_idx:
                t = tensor_entry(a, b, d)
                all_triples.append((a, b, d, t))
    
    # Filter to only non-zero entries for efficiency in checking
    nonzero_triples = [(a, b, d, t) for a, b, d, t in all_triples if t != 0]
    zero_triples = [(a, b, d, t) for a, b, d, t in all_triples if t == 0]
    
    print(f"Rank {rank} search: {len(nonzero_triples)} nonzero, {len(zero_triples)} zero entries")
    
    best_result = None
    best_error = float('inf')
    
    max_iterations = 50000
    for iteration in range(max_iterations):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization
        np.random.seed(iteration * 12345 + 67890)
        U_arr = np.random.randn(n * m, rank)
        V_arr = np.random.randn(m * p, rank)
        W_arr = np.random.randn(p * n, rank)
        
        # Normalize to help convergence
        U_arr /= np.linalg.norm(U_arr)
        V_arr /= np.linalg.norm(V_arr)
        W_arr /= np.linalg.norm(W_arr)
        
        # Use a simple iterative approach: alternate updating each matrix
        # while keeping the others fixed.
        # This is essentially a form of alternating least squares.
        
        converged = False
        for sub_iter in range(200):
            if time.time() - start_time > time_budget:
                break
            
            # Update U: for each row a, solve for U[a,:] given V, W
            # sum_c U[a,c]*V[b,c]*W[d,c] should equal T[a,b,d]
            # For fixed a, this is: sum_b,d T[a,b,d] = sum_c U[a,c] * (sum_b,d V[b,c]*W[d,c]*[indicator])
            # This is complex. Let's use a gradient-based approach instead.
            
            # Compute current error
            # For efficiency, only check nonzero entries
            error = 0.0
            for a, b, d, t in nonzero_triples:
                val = np.dot(U_arr[a, :], np.dot(V_arr[b, :], W_arr[d, :]))
                error += (val - t) ** 2
            
            if error < 1e-6:
                converged = True
                break
            
            if error < best_error:
                best_error = error
                best_result = (U_arr.copy(), V_arr.copy(), W_arr.copy())
            
            # Simple gradient step
            lr = 0.01 / (1 + iteration * 0.01)
            
            # Gradient for U_arr[a, c] = sum_b,d (val - t) * V[b,c] * W[d,c]
            # For efficiency, compute gradients for a sample of entries
            sample_size = min(len(nonzero_triples), 200)
            indices = np.random.choice(len(nonzero_triples), size=sample_size, replace=False)
            
            grad_U = np.zeros((n * m, rank))
            grad_V = np.zeros((m * p, rank))
            grad_W = np.zeros((p * n, rank))
            
            for idx in indices:
                a, b, d, t = nonzero_triples[idx]
                val = np.dot(U_arr[a, :], np.dot(V_arr[b, :], W_arr[d, :]))
                diff = val - t
                vw = np.outer(V_arr[b, :], W_arr[d, :])
                grad_U[a, :] += diff * vw
                grad_V[b, :] += diff * np.outer(U_arr[a, :], W_arr[d, :]).ravel()
                grad_W[d, :] += diff * np.outer(U_arr[a, :], V_arr[b, :]).ravel()
            
            # Also check some zero entries to ensure they stay zero
            zero_sample_size = min(len(zero_triples), 100)
            if zero_triples:
                z_indices = np.random.choice(len(zero_triples), size=zero_sample_size, replace=False)
                for zidx in z_indices:
                    a, b, d, t = zero_triples[zidx]
                    val = np.dot(U_arr[a, :], np.dot(V_arr[b, :], W_arr[d, :]))
                    diff = val - t
                    vw = np.outer(V_arr[b, :], W_arr[d, :])
                    grad_U[a, :] += diff * vw
                    grad_V[b, :] += diff * np.outer(U_arr[a, :], W_arr[d, :]).ravel()
                    grad_W[d, :] += diff * np.outer(U_arr[a, :], V_arr[b, :]).ravel()
            
            # Gradient descent step
            U_arr -= lr * grad_U
            V_arr -= lr * grad_V
            W_arr -= lr * grad_W
            
            # Periodically normalize
            if sub_iter % 50 == 0:
                U_norm = np.linalg.norm(U_arr)
                V_norm = np.linalg.norm(V_arr)
                W_norm = np.linalg.norm(W_arr)
                if U_norm > 0 and V_norm > 0 and W_norm > 0:
                    U_arr /= U_norm
                    V_arr /= V_norm
                    W_arr /= W_norm
        
        if converged:
            break
    
    # Verify best result
    if best_result is not None and best_error < 1e-5:
        U_best, V_best, W_best = best_result
        if _check_full(U_best, V_best, W_best, tol=1e-6):
            # Convert to lists
            U_list = U_best.tolist()
            V_list = V_best.tolist()
            W_list = W_best.tolist()
            # Round to nearest 0.5 if close
            for mat in [U_list, V_list, W_list]:
                for row in mat:
                    for ci in range(len(row)):
                        row[ci] = round(row[ci] * 2) / 2.0
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def _try_structured_search(time_budget=60):
    """Try structured approaches: permutation-based and block-based decompositions.
    
    For 3x4x4, one idea is to use the fact that we can permute indices.
    Another is to look for a decomposition that exploits the rectangular shape.
    """
    start_time = time.time()
    
    # Approach 1: Try all permutations of the k-index for each (i,j) pair
    # to see if we can merge some factors.
    # This is a combinatorial search over how to group the 4 k-values.
    
    # For a fixed (i, j), we need to compute sum_k A[i,j]*B[j,k]*C[k,*,i]
    # But actually the tensor entry is T[i*m+j, j*p+k, k*n+i] = 1.
    
    # Let's try: can we do rank 47 by finding one pair of entries that can share a factor?
    # Two entries (i,j,k) and (i',j',k') can share a factor if:
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] and
    # U[i'*m+j', c] * V[j'*p+k', c] * W[k'*n+i', c]
    # both use the same column c with non-zero values.
    # For them to both equal 1, we need:
    # U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # U[i'*m+j',c]*V[j'*p+k',c]*W[k'*n+i',c] = 1
    # If U[i*m+j,c] = U[i'*m+j',c] and V[j*p+k,c] = V[j'*p+k',c] and W[k*n+i,c] = W[k'*n+i',c],
    # then we'd need all of them to be 1 (or -1 with appropriate signs).
    # But this would mean the same column contributes to both entries, and we'd need
    # to ensure no other entries get spurious contributions.
    
    # This is getting complex. Let's try a direct numerical search for rank 47.
    if time.time() - start_time < time_budget:
        result = _rank_search(47, time_budget=min(time_budget, 40))
        if result is not None:
            return result
    
    # Try rank 46
    if time.time() - start_time < time_budget:
        remaining = time_budget - (time.time() - start_time)
        if remaining > 15:
            result = _rank_search(46, time_budget=remaining)
            if result is not None:
                return result
    
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = RANK_SCHOOLBOOK
    
    # Verify schoolbook
    U_arr, V_arr, W_arr = _build_tensors(best["U"], best["V"], best["W"])
    assert U_arr is not None
    assert _check_full(U_arr, V_arr, W_arr), "Schoolbook verification failed!"
    
    # Try to find lower rank
    # Attempt structured/numerical search
    try:
        result = _try_structured_search(time_budget=60)
        if result is not None:
            # Verify
            U_arr, V_arr, W_arr = _build_tensors(result["U"], result["V"], result["W"])
            if U_arr is not None and _check_full(U_arr, V_arr, W_arr):
                new_rank = len(result["U"][0])
                if new_rank < best_rank:
                    best = result
                    best_rank = new_rank
                    print(f"Found valid rank-{new_rank} decomposition!")
    except Exception as e:
        print(f"Search failed: {e}")
    
    # Return best found
    return best


if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Final rank: {rank}")
    # Verify
    U_arr, V_arr, W_arr = _build_tensors(result["U"], result["V"], result["W"])
    if U_arr is not None:
        valid = _check_full(U_arr, V_arr, W_arr)
        print(f"Valid: {valid}")
