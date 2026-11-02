import time
import sys
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit
    
    # Schoolbook decomposition as guaranteed fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    def build_full_tensor(n, m, p):
        """Build the full matrix multiplication tensor."""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    T_full = build_full_tensor(n, m, p)
    
    def verify_decomp_full(U, V, W, n, m, p, T, tol=1e-9):
        """Check if decomposition is exact for ALL tensor entries."""
        nm, m_p, p_n = n*m, m*p, p*n
        r = len(U[0]) if U else 0
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        # Reconstruct all entries and compare
        for a in range(nm):
            for b in range(m_p):
                val = 0.0
                for r_idx in range(r):
                    for c in range(p_n):
                        val += U_mat[a, r_idx] * V_mat[b, r_idx] * W_mat[c, r_idx]
                # Actually this is wrong - let me fix the reconstruction
                # For each (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r]
                # Let me redo this properly
                pass
        
        # Proper verification: check all (a,b,c) triples
        for a in range(nm):
            for b in range(m_p):
                for c in range(p_n):
                    val = 0.0
                    for r_idx in range(r):
                        val += U_mat[a, r_idx] * V_mat[b, r_idx] * W_mat[c, r_idx]
                    if abs(val - T[a, b, c]) > tol:
                        return False
        return True
    
    def verify_decomp_fast(U, V, W, n, m, p, tol=1e-9):
        """Faster verification using matrix operations."""
        nm, m_p, p_n = n*m, m*p, p*n
        r = len(U[0]) if U else 0
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        # For each rank component, compute the outer product contribution
        # and check against the tensor
        # More efficient: for each (a,b), compute the vector over c
        for a in range(nm):
            for b in range(m_p):
                # Compute sum_r U[a,r]*V[b,r]*W[:,r] for all c
                contributions = np.zeros(p_n)
                for r_idx in range(r):
                    contributions += U_mat[a, r_idx] * V_mat[b, r_idx] * W_mat[:, r_idx]
                
                # Check against expected values
                # Determine which (i,j,k) this (a,b) corresponds to
                i = a // m
                j = a % m
                jp = b // p
                k = b % p
                
                # The only non-zero entry should be at c = k*n + i if j == jp
                if j == jp:
                    c_expected = k * n + i
                    if abs(contributions[c_expected] - 1.0) > tol:
                        return False
                    # All other c should be 0
                    for c in range(p_n):
                        if c != c_expected and abs(contributions[c]) > tol:
                            return False
                else:
                    # All entries should be 0
                    if np.max(np.abs(contributions)) > tol:
                        return False
        return True
    
    def verify_decomp(U, V, W, n, m, p, tol=1e-9):
        """Use the fast verification."""
        return verify_decomp_fast(U, V, W, n, m, p, tol)
    
    # Try to find lower rank decomposition
    # Strategy: Start with schoolbook and try to eliminate columns
    # by finding linear dependencies or combining terms
    
    np.random.seed(42)
    
    # First, try systematic column elimination from schoolbook
    U_arr = np.array(U_school, dtype=float)
    V_arr = np.array(V_school, dtype=float)
    W_arr = np.array(W_school, dtype=float)
    current_rank = rank_schoolbook
    
    # Try removing columns one by one (greedy)
    for col_to_remove in range(current_rank - 1, -1, -1):
        if time.time() > deadline - 30:
            break
        
        U_test = np.delete(U_arr, col_to_remove, axis=1)
        V_test = np.delete(V_arr, col_to_remove, axis=1)
        W_test = np.delete(W_arr, col_to_remove, axis=1)
        
        if verify_decomp(U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p, tol=1e-8):
            U_arr = U_test
            V_arr = V_test
            W_arr = W_test
            current_rank -= 1
    
    if current_rank < best_rank:
        best_rank = current_rank
        best_U = U_arr.tolist()
        best_V = V_arr.tolist()
        best_W = W_arr.tolist()
    
    # Try structured combinations: look for pairs of schoolbook terms that can be merged
    # using algebraic identities like (a+b)(c+d) = ac + ad + bc + bd
    # For matrix multiplication, we can try to combine terms that share indices
    
    # Try random perturbations of schoolbook and optimize
    for target_rank in range(min(best_rank, 60) - 1, 45, -1):
        if time.time() > deadline - 20:
            break
        
        for attempt in range(10):
            if time.time() > deadline - 15:
                break
            
            # Initialize near schoolbook structure
            U_mat = np.zeros((n*m, target_rank))
            V_mat = np.zeros((m*p, target_rank))
            W_mat = np.zeros((p*n, target_rank))
            
            # Fill with a subset of schoolbook columns
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= target_rank:
                            break
                        U_mat[i * m + j, col] = 1.0
                        V_mat[j * p + k, col] = 1.0
                        W_mat[k * n + i, col] = 1.0
                        col += 1
                    if col >= target_rank:
                        break
                if col >= target_rank:
                    break
            
            # Add small random perturbations to remaining columns
            for r_idx in range(col, target_rank):
                # Random sparse column
                num_nonzero = np.random.randint(2, 6)
                u_idx = np.random.choice(n*m, size=num_nonzero, replace=False)
                v_idx = np.random.choice(m*p, size=num_nonzero, replace=False)
                w_idx = np.random.choice(p*n, size=num_nonzero, replace=False)
                
                for idx in u_idx:
                    U_mat[idx, r_idx] = float(np.random.choice([-1, 1, 0.5, -0.5]))
                for idx in v_idx:
                    V_mat[idx, r_idx] = float(np.random.choice([-1, 1, 0.5, -0.5]))
                for idx in w_idx:
                    W_mat[idx, r_idx] = float(np.random.choice([-1, 1, 0.5, -0.5]))
            
            # Try to fix up by adjusting existing columns
            # Use a simple iterative improvement: for each constraint violation, adjust
            for iteration in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Check if valid
                if verify_decomp(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), n, m, p, tol=1e-7):
                    best_rank = target_rank
                    best_U = U_mat.tolist()
                    best_V = V_mat.tolist()
                    best_W = W_mat.tolist()
                    break
                
                # Find a violated constraint and try to fix it
                # For simplicity, just break and try next attempt
                break
        
        if best_rank <= target_rank:
            break
    
    # Final refinement: try to eliminate more columns from the best found
    if best_rank < rank_schoolbook and time.time() < deadline - 10:
        U_arr = np.array(best_U, dtype=float)
        V_arr = np.array(best_V, dtype=float)
        W_arr = np.array(best_W, dtype=float)
        r = U_arr.shape[1]
        
        for col_to_remove in range(r - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            
            U_test = np.delete(U_arr, col_to_remove, axis=1)
            V_test = np.delete(V_arr, col_to_remove, axis=1)
            W_test = np.delete(W_arr, col_to_remove, axis=1)
            
            if verify_decomp(U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p, tol=1e-8):
                U_arr = U_test
                V_arr = V_test
                W_arr = W_test
                best_U = U_test.tolist()
                best_V = V_test.tolist()
                best_W = W_test.tolist()
                best_rank -= 1
                r = U_arr.shape[1]
    
    return {"U": best_U, "V": best_V, "W": best_W}
