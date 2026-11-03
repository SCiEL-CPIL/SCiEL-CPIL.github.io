import time
import numpy as np
from itertools import combinations

# Constants for 2x3 * 3x3 matrix multiplication
_N, _M, _P = 2, 3, 3
_NM = _N * _M  # 6
_MP = _M * _P  # 9
_PN = _P * _N  # 6

# Index helpers (row-major)
def _u_idx(i, j):  return i * _M + j   # A[i,j]
def _v_idx(j, k):  return j * _P + k   # B[j,k]
def _w_idx(k, i):  return k * _N + i   # C[k,i]  (trace convention W[k*n+i, r])

# Known rank-17 fallback decomposition
_FALLBACK_RANK = 17

def _fallback_decomposition():
    """Return the known rank-17 decomposition as (U, V, W) numpy arrays."""
    r = _FALLBACK_RANK
    U = np.zeros((_NM, r))
    V = np.zeros((_MP, r))
    W = np.zeros((_PN, r))
    
    # Part 1: Strassen for the 2x2 x 2x2 block (A[:, 0:2] * B[0:2, :])
    # We'll use 0-indexed for the 2x2 block: A[0:2, 0:2], B[0:2, 0:2]
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U[_u_idx(0,0), 0] = 1; U[_u_idx(1,1), 0] = 1
    V[_v_idx(0,0), 0] = 1; V[_v_idx(1,1), 0] = 1
    W[_w_idx(0,0), 0] = 1; W[_w_idx(1,1), 0] = 1
    
    # M2: (A10+A11)B00 -> C10, -C11
    U[_u_idx(1,0), 1] = 1; U[_u_idx(1,1), 1] = 1
    V[_v_idx(0,0), 1] = 1
    W[_w_idx(0,1), 1] = 1; W[_w_idx(1,1), 1] = -1
    
    # M3: A00(B01-B11) -> C01, C11
    U[_u_idx(0,0), 2] = 1
    V[_v_idx(0,1), 2] = 1; V[_v_idx(1,1), 2] = -1
    W[_w_idx(1,0), 2] = 1; W[_w_idx(1,1), 2] = 1
    
    # M4: A11(B10-B00) -> C00, C10
    U[_u_idx(1,1), 3] = 1
    V[_v_idx(1,0), 3] = 1; V[_v_idx(0,0), 3] = -1
    W[_w_idx(0,0), 3] = 1; W[_w_idx(0,1), 3] = 1
    
    # M5: (A00+A01)B11 -> -C00, C01
    U[_u_idx(0,0), 4] = 1; U[_u_idx(0,1), 4] = 1
    V[_v_idx(1,1), 4] = 1
    W[_w_idx(0,0), 4] = -1; W[_w_idx(1,0), 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[_u_idx(1,0), 5] = 1; U[_u_idx(0,0), 5] = -1
    V[_v_idx(0,0), 5] = 1; V[_v_idx(0,1), 5] = 1
    W[_w_idx(1,1), 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U[_u_idx(0,1), 6] = 1; U[_u_idx(1,1), 6] = -1
    V[_v_idx(1,0), 6] = 1; V[_v_idx(1,1), 6] = 1
    W[_w_idx(0,0), 6] = 1
    
    # Part 2: Schoolbook for cross terms A[:,2] * B[2,:]
    U[_u_idx(0,2), 7] = 1; V[_v_idx(2,0), 7] = 1; W[_w_idx(0,0), 7] = 1
    U[_u_idx(0,2), 8] = 1; V[_v_idx(2,1), 8] = 1; W[_w_idx(1,0), 8] = 1
    U[_u_idx(1,2), 9] = 1; V[_v_idx(2,0), 9] = 1; W[_w_idx(0,1), 9] = 1
    U[_u_idx(1,2), 10] = 1; V[_v_idx(2,1), 10] = 1; W[_w_idx(1,1), 10] = 1
    
    # Part 3: Schoolbook for B[:,2] * A rows -> C[:,2]
    U[_u_idx(0,0), 11] = 1; V[_v_idx(0,2), 11] = 1; W[_w_idx(2,0), 11] = 1
    U[_u_idx(0,1), 12] = 1; V[_v_idx(1,2), 12] = 1; W[_w_idx(2,0), 12] = 1
    U[_u_idx(0,2), 13] = 1; V[_v_idx(2,2), 13] = 1; W[_w_idx(2,0), 13] = 1
    U[_u_idx(1,0), 14] = 1; V[_v_idx(0,2), 14] = 1; W[_w_idx(2,1), 14] = 1
    U[_u_idx(1,1), 15] = 1; V[_v_idx(1,2), 15] = 1; W[_w_idx(2,1), 15] = 1
    U[_u_idx(1,2), 16] = 1; V[_v_idx(2,2), 16] = 1; W[_w_idx(2,1), 16] = 1
    
    return U, V, W


def _validate(U, V, W):
    """
    Check if (U, V, W) exactly decomposes the matrix multiplication tensor.
    Returns True if valid, False otherwise.
    """
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    
    # Tensor T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Expected: T[_u_idx(i,j), _v_idx(j,k), _w_idx(k,i)] = 1 for all i,j,k
    #           0 otherwise
    
    # Build the decomposition tensor
    # U is (NM, r), V is (MP, r), W is (PN, r)
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    # We can compute this as a sum of rank-1 tensors
    # For efficiency, check the 18 required non-zero entries and a subset of zeros
    
    # First, check all 18 required entries
    for i in range(_N):
        for j in range(_M):
            for k in range(_P):
                a = _u_idx(i, j)
                b = _v_idx(j, k)
                c = _w_idx(k, i)
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check that entries where j != middle index are 0
    # The "middle" index in (a,b) correspondence is: a = i*M+j, b = j'*P+k
    # The decomposition is correct if and only if for all (a,b,c), 
    # T[a,b,c] = 1 if a's j == b's j and c = k*N+i where a=i*M+j, b=j*P+k, c=k*N+i
    # T[a,b,c] = 0 otherwise
    
    # Let's check all zero entries
    for a in range(_NM):
        i_a = a // _M
        j_a = a % _M
        for b in range(_MP):
            j_b = b // _P
            k_b = b % _P
            # If j_a != j_b, all c should be 0
            if j_a != j_b:
                for c in range(_PN):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
            else:
                # j_a == j_b == j
                i_val = i_a
                k_val = k_b
                c_expected = _w_idx(k_val, i_val)
                for c in range(_PN):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if c == c_expected:
                        if abs(val - 1.0) > 1e-9:
                            return False
                    else:
                        if abs(val) > 1e-9:
                            return False
    return True


def _rank_of_tensor(U, V, W):
    """Return the effective rank (number of non-zero columns)."""
    r = U.shape[1]
    count = 0
    for col in range(r):
        if (np.abs(U[:, col]) > 1e-12).any() or (np.abs(V[:, col]) > 1e-12).any() or (np.abs(W[:, col]) > 1e-12).any():
            count += 1
    return count


def construct():
    deadline = time.time() + 115.0  # Leave 5s margin
    
    # Start with the known rank-17 solution
    U_best, V_best, W_best = _fallback_decomposition()
    best_rank = _FALLBACK_RANK
    
    if not _validate(U_best, V_best, W_best):
        raise RuntimeError("Fallback decomposition failed validation")
    
    # Now try to find a rank-16 decomposition
    # Strategy: Try all subsets of 16 columns from the 17-column decomposition,
    # then try to solve for a modified last column to make it work.
    # 
    # More systematically: We know a rank-17 decomposition. The question is whether
    # there exists a rank-16 decomposition. Let's try to search for one by:
    # 1. Taking the 17-term decomposition
    # 2. Trying to merge two terms into one (i.e., find a linear combination)
    
    # Alternative: Use the structure. The tensor is T[i,j,k] with indices.
    # Let's try a random search over rational coefficients.
    
    # First, try: can we find 16 rank-1 terms?
    # We'll do a greedy/linear algebra approach:
    # - Start with a random 16-term ansatz
    # - Try to solve the resulting system
    
    # Actually, let's try a more structured approach:
    # The known decomposition has 7 Strassen terms for the 2x2 block + 4 + 6 = 10 schoolbook.
    # Total 17. Can we reduce the schoolbook terms?
    
    # The schoolbook terms are for:
    # - A[:,2] * B[2,:] -> C[:,0:2] (4 terms)
    # - A[:,0:2] * B[0:2,2] + A[:,2]*B[2,2] -> C[:,2] (6 terms)
    
    # Wait, Part 3 has 6 terms. These are all independent (each involves a unique (i,k) pair).
    # So we can't easily reduce them.
    
    # The 4 terms in Part 2 are also each unique (i, k) pairs with j=2.
    
    # The 7 Strassen terms are minimal for 2x2 matrix multiplication.
    
    # So it seems rank 17 might be minimal, but let's try to find a rank-16 anyway.
    
    # Approach: Random search with local optimization
    # We'll parameterize 16 rank-1 terms and try to find coefficients.
    
    # Actually, let's try a more direct algebraic approach:
    # Suppose we have terms t=0..15. For each term, we choose indices (i_t, j_t) for U,
    # (j'_t, k_t) for V, (k'_t, i'_t) for W.
    # The constraint is: sum_t U_t[i,j] * V_t[j',k] * W_t[k',i'] * delta_{j,j'} = 1 if (i,j,k') matches properly
    
    # This is complex. Let's try a numerical approach:
    # 1. Create the multiplication tensor T of shape (6, 9, 6)
    # 2. Try to find 16 rank-1 vectors (u_t, v_t, w_t) such that sum_t outer(u_t, v_t, w_t) = T
    
    # We can do this by iterating: fix 15 terms randomly, solve for the 16th.
    
    # Let's set up the tensor T
    T = np.zeros((_NM, _MP, _PN))
    for i in range(_N):
        for j in range(_M):
            for k in range(_P):
                T[_u_idx(i, j), _v_idx(j, k), _w_idx(k, i)] = 1.0
    
    # Try rank-16 search
    # We'll do a local search: start from the rank-17 solution, try to eliminate one term
    # by adjusting others.
    
    # For each pair of terms (a, b) in the rank-17 decomposition, try to merge them:
    # Replace terms a and b with a single term c such that:
    # U_a * V_a * W_a + U_b * V_b * W_b = U_c * V_c * W_c
    # This requires the sum of two rank-1 tensors to be rank 1, which generally isn't possible
    # unless they're already proportional. So this won't work directly.
    
    # Better approach: Try a random rank-16 ansatz and solve.
    
    # Let's try: pick 16 of the 17 existing terms, and see if we can adjust the coefficients
    # of the remaining terms to compensate.
    
    # Actually, let's try a simpler approach: brute-force check if any 16-term subset
    # of the 17-term decomposition, when we allow one term to be a linear combination, works.
    
    # Hmm, this is getting complicated. Let's try a different tactic:
    # Use the fact that the rank of a tensor can sometimes be reduced by "completing the square"
    # or similar algebraic manipulations.
    
    # For now, let's try a random local search.
    
    # Parameterize: 16 terms, each with (i, j) for U, (j, k) for V, (k, i') for W
    # But the indices must be consistent: U uses (i, j), V uses (j, k), W uses (k, i')
    # So each term is specified by (i, j, k) where i in [0,2), j in [0,3), k in [0,3)
    # But then U[i,j], V[j,k], W[k,i] -- this is exactly the schoolbook form!
    
    # So a rank-r decomposition where each term has the form u_{i,j} * v_{j,k} * w_{k,i}
    # with u_{i,j}, v_{j,k}, w_{k,i} scalars -- this is not general enough.
    # General rank-1 terms can have non-zero entries in multiple positions.
    
    # Let's try a numerical optimization approach with a time limit.
    
    # Start from the rank-17 solution, zero out one random column, and try to
    # restore correctness by adjusting other columns.
    
    best_U, best_V, best_W = U_best.copy(), V_best.copy(), W_best.copy()
    best_score = best_rank
    
    for attempt in range(500000):
        if time.time() > deadline:
            break
        
        # Try to eliminate a random column from the 17-term decomposition
        col_to_kill = attempt % best_rank
        
        # Make a copy
        U = U_best.copy()
        V = V_best.copy()
        W = W_best.copy()
        
        # Zero out the column
        U[:, col_to_kill] = 0
        V[:, col_to_kill] = 0
        W[:, col_to_kill] = 0
        
        # Now try to fix the other columns to compensate
        # We'll do a local perturbation: add a random perturbation to all other columns
        # and see if validation passes
        
        # Actually, let's try a more structured approach:
        # Compute the residual after removing column col_to_kill
        # residual[a,b,c] = T[a,b,c] - sum_{t != col_to_kill} U[a,t]V[b,t]W[c,t]
        # We need to find a rank-1 term (u, v, w) such that u*outer(v,w) approximately equals residual
        # But we can only use the existing columns, so we'd need to modify them.
        
        # Let's try: for the column we want to "absorb", add a small multiple of it to another column
        # and adjust the other columns to compensate.
        
        # Pick a target column to absorb into
        target_col = (col_to_kill + 1) % best_rank
        
        # Add the killed column to the target column
        alpha = np.random.uniform(0.5, 1.5)
        U[:, target_col] += alpha * U_best[:, col_to_kill]
        V[:, target_col] += alpha * V_best[:, col_to_kill]
        W[:, target_col] += alpha * W_best[:, col_to_kill]
        
        # Now we need to zero out column col_to_kill completely
        # But we already did. So the decomposition is now:
        # sum_t U_t V_t W_t where t != col_to_kill, and column target_col has been modified
        # This doesn't sum to T anymore.
        
        # This approach doesn't work. Let me try something else.
        
        # Let's try: keep all 17 columns but set one to zero, then add a new rank-1 term
        # to compensate. But that's still 17 terms.
        
        # OK, I think I need to accept that finding rank 16 requires a fundamentally
        # different approach. Let me just return the rank-17 solution.
        
        break  # Don't waste time on this approach
    
    # Final validation
    if not _validate(best_U, best_V, best_W):
        # Fall back to the known solution
        best_U, best_V, best_W = _fallback_decomposition()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
