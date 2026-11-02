"""Rewritten program for 3x4 x 4x5 matrix multiplication tensor decomposition."""
import numpy as np
import time
from fractions import Fraction
from itertools import product, combinations
import random
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k.
    The tensor represents: C[i,k] = sum_j A[i,j] * B[j,k]
    In the bilinear form view, T is in R^{(n*m) x (m*p) x (n*p)} but we use
    the standard 3-tensor formulation where T[i,j,k] = 1."""
    T = np.ones((n, m, p), dtype=float)
    return T


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid rank-r decomposition.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    r = U.shape[1]
    # Reshape for efficient computation
    U_reshaped = U.T  # (r, n*m)
    V_reshaped = V.T  # (r, m*p)
    W_reshaped = W.T  # (r, p*n)
    
    # For each (i,j,k), compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This is equivalent to: for each r, the outer product U_r ⊗ V_r ⊗ W_r
    # We need sum_r to equal 1 for all (i,j,k)
    
    # Efficient approach: iterate over r and accumulate
    result = np.zeros((n, m, p), dtype=float)
    for r_idx in range(r):
        u = U_reshaped[r_idx].reshape(n, m)
        v = V_reshaped[r_idx].reshape(m, p)
        w = W_reshaped[r_idx].reshape(p, n)
        # Outer product: u[i,j] * v[j,k] * w[k,i]
        # This is a 3D tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    result[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    
    # Check if result is all 1s
    max_err = np.max(np.abs(result - 1.0))
    return max_err < tol, max_err


def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using matrix operations."""
    r = U.shape[1]
    result = np.zeros((n, m, p), dtype=float)
    
    for r_idx in range(r):
        u = U[:, r_idx].reshape(n, m)
        v = V[:, r_idx].reshape(m, p)
        w = W[:, r_idx].reshape(p, n)
        
        # u[i,j] * v[j,k] * w[k,i] for all i,j,k
        # Can be computed as: for each i, (u[i,:] ⊗ v) * w[:,i]
        # Or: sum over j of u[i,j]*v[j,k] gives (u@v)[i,k], then multiply by w[k,i]
        uv = u @ v  # (n, p)
        # Now result[i,k] += uv[i,k] * w[k,i]
        # w is (p, n), so w[k,i] is element (k,i) of w
        # We need: for each i,k: uv[i,k] * w[k,i]
        # This is element-wise product of uv and w.T
        result_slice = uv * w.T  # (n, p)
        # But we need to add to result[i, j, k]... wait, this doesn't have j dimension.
        # 
        # Actually, the tensor decomposition is:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] (using W indexed by k*n+i)
        # 
        # So for fixed r: contribution to T[i,j,k] is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # 
        # Let me redo this more carefully.
        # U has shape (n*m, r), indexed by (i*m+j, r)
        # V has shape (m*p, r), indexed by (j*p+k, r)  
        # W has shape (p*n, r), indexed by (k*n+i, r)
        #
        # For fixed r, define:
        # u_r[i,j] = U[i*m+j, r]
        # v_r[j,k] = V[j*p+k, r]
        # w_r[k,i] = W[k*n+i, r]
        #
        # Contribution: u_r[i,j] * v_r[j,k] * w_r[k,i]
        #
        # Sum over j: this is like (u_r * v_r * w_r) but with the j index contracted.
        # Actually no - we're not summing over j here. The tensor entry T[i,j,k] 
        # gets contribution u_r[i,j] * v_r[j,k] * w_r[k,i] for each r.
        #
        # So for fixed r, the 3D array is:
        # A_r[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
        
        # Let's compute this directly but efficiently
        # A_r[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
        # = (u_r * v_r) element-wise? No, u_r is (n,m), v_r is (m,p)
        # For fixed j: u_r[:,j] * v_r[j,:] gives (n,p) matrix, then multiply by w_r[:,j]... 
        # No, w_r[k,i] depends on both k and i.
        
        # Let's just do it with broadcasting
        # u_r: (n, m), v_r: (m, p), w_r: (p, n)
        # A_r[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
        
        # Use einsum
        ar = np.einsum('ij,jk,ki->ijk', u, v, w)
        result += ar
    
    max_err = np.max(np.abs(result - 1.0))
    return max_err < tol, max_err


def _schoolbook(n, m, p):
    """Return the schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_rank_with_random_search(r, n, m, p, num_trials=100, time_limit=5.0):
    """Try to find a rank-r decomposition using random search with verification."""
    start = time.time()
    
    for trial in range(num_trials):
        if time.time() - start > time_limit:
            break
        
        # Random sparse initialization
        U = np.zeros((n * m, r), dtype=float)
        V = np.zeros((m * p, r), dtype=float)
        W = np.zeros((p * n, r), dtype=float)
        
        for col in range(r):
            # Randomly select positions
            ui = random.randint(0, n * m - 1)
            vi = random.randint(0, m * p - 1)
            wi = random.randint(0, p * n - 1)
            
            # Random values from a small set
            U[ui, col] = random.choice([-1.0, 0.5, 1.0, 2.0])
            V[vi, col] = random.choice([-1.0, 0.5, 1.0, 2.0])
            W[wi, col] = random.choice([-1.0, 0.5, 1.0, 2.0])
        
        valid, err = _verify_decomposition_fast(U, V, W, n, m, p)
        if valid:
            return {"U": U, "V": V, "W": W}
    
    return None


def _try_divide_and_conquer(n, m, p):
    """
    Divide and conquer approach: partition the middle dimension m into two parts.
    For m=4, split into 2+2.
    Then we have two sub-problems: 3x2 x 2x5 and 3x2 x 2x5.
    The total rank would be 2 * rank(3x2 x 2x5).
    Schoolbook for 3x2 x 2x5 is 30, so total would be 60 (same as before).
    But if we can find a lower rank for 3x2 x 2x5, we get improvement.
    
    Actually, the divide and conquer for matrix multiplication works differently.
    We can write:
    C = A * B = [A1 | A2] * [[B1], [B2]] = A1*B1 + A2*B2
    where A1, A2 are 3x2 and B1, B2 are 2x5.
    
    If rank(3x2 x 2x5) = r_sub, then total rank = 2*r_sub.
    We need 2*r_sub < 60, i.e., r_sub < 30.
    
    The minimum rank of 3x2 x 2x5 is at least max(3*2, 2*5, 3*5) = max(6,10,15) = 15
    and at most 3*2*5 = 30.
    
    Known results: rank of 2x2x2 is 7 (Strassen), rank of 3x3x3 is 23 (Strassen).
    For 3x2 x 2x5, it's not a standard case. Let's try to find a decomposition.
    """
    # For now, let's try to find a rank-r decomposition for 3x2 x 2x5
    # and see if we can get total rank < 60.
    
    n_sub, m_sub, p_sub = 3, 2, 5
    
    # Try various ranks for the sub-problem
    for r_sub in range(15, 30):
        result = _try_rank_with_random_search(r_sub, n_sub, m_sub, p_sub, num_trials=50, time_limit=2.0)
        if result is not None:
            # Combine two sub-decompositions
            r_total = 2 * r_sub
            U = np.zeros((n * m, r_total), dtype=float)
            V = np.zeros((m * p, r_total), dtype=float)
            W = np.zeros((p * n, r_total), dtype=float)
            
            # First half: A1 * B1 where A1 = A[:, 0:2], B1 = B[0:2, :]
            U[:, 0:r_sub] = result["U"]
            V[:m_sub * p_sub, 0:r_sub] = result["V"]
            W[:, 0:r_sub] = result["W"]
            
            # Second half: A2 * B2 where A2 = A[:, 2:4], B2 = B[2:4, :]
            U[:, r_sub:2*r_sub] = result["U"]
            V[m_sub * p_sub:, r_sub:2*r_sub] = result["V"]
            W[:, r_sub:2*r_sub] = result["W"]
            
            # Verify the full decomposition
            valid, err = _verify_decomposition_fast(U, V, W, n, m, p)
            if valid:
                return {"U": U, "V": V, "W": W, "rank": r_total}
    
    return None


def _try_systematic_search(r, n, m, p, time_limit=10.0):
    """
    Try a more systematic approach for finding rank-r decompositions.
    Use the fact that the tensor has a specific structure.
    """
    start = time.time()
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_{r=1}^R U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # One approach: assume each rank-1 term has sparse support.
    # Try to find a set of sparse rank-1 tensors that sum to 1.
    
    # For each rank-1 term, it contributes u_r[i,j] * v_r[j,k] * w_r[k,i]
    # If we make these sparse, we can try to cover all (i,j,k) entries.
    
    # Let's try a greedy approach: start with 0 and add rank-1 terms
    # to minimize the residual.
    
    residual = np.ones((n, m, p), dtype=float)
    U_list = []
    V_list = []
    W_list = []
    
    for step in range(r):
        if time.time() - start > time_limit:
            break
        
        # Find the best rank-1 term to add
        best_err = np.inf
        best_u, best_v, best_w = None, None, None
        
        # Try all possible sparse patterns (limit search)
        # For simplicity, try terms with support on a single (i,j), (j,k), (k,i)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Term: u[i,j]=1, v[j,k]=1, w[k,i]=1
                    # This contributes 1 to T[i,j,k] and 0 elsewhere
                    # But we can also use non-unit values
                    
                    # Try coefficient c
                    for c in [0.5, 1.0, 2.0]:
                        # Compute new residual
                        new_residual = residual.copy()
                        new_residual[i, j, k] -= c
                        
                        # Count how many entries are now closer to 0
                        err = np.sum(new_residual ** 2)
                        if err < best_err:
                            best_err = err
                            best_u = (i, j, c)
                            best_v = (j, k, c)
                            best_w = (k, i, c)
        
        if best_u is None:
            break
        
        # Add this term
        u_val = np.zeros((n, m))
        v_val = np.zeros((m, p))
        w_val = np.zeros((p, n))
        
        i, j, c = best_u
        u_val[i, j] = c
        j, k, c = best_v
        v_val[j, k] = c
        k, i, c = best_w
        w_val[k, i] = c
        
        # Update residual
        residual -= np.einsum('ij,jk,ki->ijk', u_val, v_val, w_val)
        
        U_list.append(u_val)
        V_list.append(v_val)
        W_list.append(w_val)
        
        # Check if we've achieved a perfect decomposition
        if np.max(np.abs(residual)) < 1e-9:
            break
    
    if np.max(np.abs(residual)) < 1e-9:
        # Build the result
        r_actual = len(U_list)
        U = np.zeros((n * m, r_actual), dtype=float)
        V = np.zeros((m * p, r_actual), dtype=float)
        W = np.zeros((p * n, r_actual), dtype=float)
        
        for idx in range(r_actual):
            U_list[idx].reshape(n * m)
            V_list[idx].reshape(m * p)
            W_list[idx].reshape(p * n)
            U[:, idx] = U_list[idx].reshape(n * m)
            V[:, idx] = V_list[idx].reshape(m * p)
            W[:, idx] = W_list[idx].reshape(p * n)
        
        return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # First, get the schoolbook solution as fallback
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try divide and conquer approach
    if time.time() < deadline - 20:
        dac_result = _try_divide_and_conquer(n, m, p)
        if dac_result is not None:
            r = dac_result["rank"]
            if r < best_rank:
                best_rank = r
                best_result = {
                    "U": dac_result["U"],
                    "V": dac_result["V"],
                    "W": dac_result["W"]
                }
    
    # Try systematic search for various ranks
    if time.time() < deadline - 10:
        for r in range(55, 35, -1):
            if time.time() > deadline - 5:
                break
            if r >= best_rank:
                continue
            
            result = _try_systematic_search(r, n, m, p, time_limit=3.0)
            if result is not None:
                valid, err = _verify_decomposition_fast(result["U"], result["V"], result["W"], n, m, p)
                if valid:
                    best_rank = r
                    best_result = result
                    break
    
    # Try random search for lower ranks
    if time.time() < deadline - 5:
        for r in range(best_rank - 1, 30, -1):
            if time.time() > deadline - 2:
                break
            
            result = _try_rank_with_random_search(r, n, m, p, num_trials=20, time_limit=2.0)
            if result is not None:
                valid, err = _verify_decomposition_fast(result["U"], result["V"], result["W"], n, m, p)
                if valid:
                    best_rank = r
                    best_result = result
                    break
    
    return {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}

# EVOLVE-BLOCK-END
