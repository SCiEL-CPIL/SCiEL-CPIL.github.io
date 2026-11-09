"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition as fallback
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
    
    def verify(U, V, W):
        """Verify that the decomposition correctly computes matrix multiplication."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        # Test with random matrices
        rng = np.random.RandomState(42)
        for trial in range(10):
            A = rng.rand(n, m)
            B = rng.rand(m, p)
            C_expected = A @ B
            
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_val = np.dot(U_np[:, r], A.flatten())
                v_val = np.dot(V_np[:, r], B.flatten())
                # W index is k*n + i, so we need to map back to C[i, k]
                w_vec = W_np[:, r]
                for idx in range(p * n):
                    k = idx // n
                    i = idx % n
                    C_computed[i, k] += u_val * v_val * w_vec[idx]
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True
    
    def try_milp_rank(target_rank, max_time=60):
        """
        Try to find a decomposition of the given rank using MILP.
        We restrict entries to {-2, -1, 0, 1, 2} to keep the problem tractable.
        """
        if time.time() - start_time > deadline:
            return None
        
        # For matmul tensor: T[i,j,k] = 1 for all valid (i,j,k)
        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is a nonlinear constraint. Linearize by fixing U and V to small integers
        # and solving for W, or use a different approach.
        
        # Alternative: Use the fact that for rank-r decomposition, we can write
        # C = A @ B = sum_r (u_r^T A) * (v_r^T B) * w_r
        # where u_r is in R^(n*m), v_r in R^(m*p), w_r in R^(p*n)
        
        # Let's try a structured approach: assume U and V have entries in {-1, 0, 1}
        # and W has entries in {-1, 0, 1}
        
        # For small rank, we can try all combinations of U and V with entries in {-1,0,1}
        # but that's exponential. Instead, use a greedy approach.
        
        # Greedy: Start with schoolbook and try to merge terms
        # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if there exists a 
        # linear combination that preserves the tensor identity.
        
        # Simpler: Try to find a decomposition where some U, V, W vectors are shared
        # across multiple (i,j,k) positions.
        
        # For 3x3x5, the middle dimension m=3 allows for some sharing.
        # Specifically, for fixed i and k, we need sum_j U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # if we think of it as a sum over j, but actually the tensor identity is:
        # For each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        # Let's try a direct search for small rank with integer entries in {-1, 0, 1}
        
        if target_rank > 40:
            return None  # Too large for exhaustive search
        
        # This approach is getting complex. Let's use a simpler heuristic:
        # Try to find a decomposition by solving a system of equations.
        
        # For a given rank r, we have r unknown vectors U_r, V_r, W_r.
        # The constraint is: for each (i,j,k) in [0,n)x[0,m)x[0,p),
        #   sum_{r'=0}^{r-1} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1
        
        # This is a system of n*m*p equations in 3*r*(n*m + m*p + p*n) unknowns.
        # For n=3, m=3, p=5: 45 equations in 3*r*(9+15+15) = 3*r*39 = 117r unknowns.
        # For r=45: 45 equations in 5265 unknowns - very underdetermined.
        
        # The issue is that this is a nonlinear system (products of unknowns).
        # Linearize by fixing some variables.
        
        # Fix U and V to be standard basis vectors (schoolbook-like) and solve for W.
        # Or fix U to be standard basis and solve for V and W.
        
        # Let's try: fix U[i*m+j, r] = delta_{i,i_r} * delta_{j,j_r} * delta_{r, r(i,j)}
        # i.e., each (i,j) pair is assigned to a specific rank index.
        
        # This is getting too complex. Let's use a simpler approach:
        # Try to find a decomposition by random search with integer entries.
        
        rng = np.random.RandomState(42)
        best_result = None
        
        for attempt in range(1000):
            if time.time() - start_time > deadline:
                break
            
            # Randomly initialize U, V, W with entries in {-1, 0, 1}
            U = rng.randint(-1, 2, size=(n*m, target_rank))
            V = rng.randint(-1, 2, size=(m*p, target_rank))
            W = rng.randint(-1, 2, size=(p*n, target_rank))
            
            # Check if this satisfies the tensor identity
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for r in range(target_rank):
                            total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if total != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        return best_result
    
    # Try to find a lower-rank decomposition
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try ranks from low to high
    for target_rank in range(30, best_rank):
        if time.time() - start_time > deadline:
            break
        
        result = try_milp_rank(target_rank)
        if result is not None and verify(result["U"], result["V"], result["W"]):
            best_result = result
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
