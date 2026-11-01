"""Matrix multiplication 2x3x4 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from fractions import Fraction

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    # Build the target tensor entries
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    
    num_entries = len(tensor_entries)  # 24
    
    # Schoolbook fallback (rank 24)
    best_rank = n * m * p
    U_sb = [[0] * best_rank for _ in range(nm)]
    V_sb = [[0] * best_rank for _ in range(mp)]
    W_sb = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Strategy: Enumerate candidate rank-1 terms and greedily select minimal subset
    # A rank-1 term is (u_vec, v_vec, w_vec) where:
    #   u_vec in {0, ±1, ±0.5}^nm
    #   v_vec in {0, ±1, ±0.5}^mp  
    #   w_vec in {0, ±1, ±0.5}^pn
    # such that for all tensor entries (u,v,w): u[u]*v[v]*w[w] = 1 (or 0 for non-entries)
    # Actually, the condition is: sum_r U[u,r]*V[v,r]*W[w,r] = T[u,v,w] for all u,v,w
    # For tensor entries: sum = 1, for non-entries: sum = 0
    
    # Better approach: Directly search for structured decompositions
    # The 2x3x4 tensor has special structure. Let's exploit it.
    
    # Key insight: The tensor can be written as:
    # T = T_i=0 + T_i=1
    # where T_i=0 has entries (j, j*4+k, k) for j=0..2, k=0..3
    # and T_i=1 has entries (3+j, j*4+k, 4+k) for j=0..2, k=0..3
    
    # For each i-slice, we have a 3x4 "matrix" tensor
    # The 3x4 matrix multiplication (as a 1x3x4 tensor) has rank at most 12
    # But we can do better using known results or structured search
    
    # Let's try a direct algebraic approach:
    # For the full 2x3x4 tensor, try to find a rank-r decomposition
    # by solving a system of polynomial equations
    
    # Alternative: Use the fact that we can share structure between i=0 and i=1
    # If we use the same V factors for both slices, we save rank
    
    # Structured search: 
    # Let U[i,j,r] = a[i,r] * b[j,r] (rank-1 in i,j for each r)
    # Let W[k,i,r] = c[k,r] * d[i,r] (rank-1 in k,i for each r)
    # Then T[i,j,k] = sum_r a[i,r]*b[j,r]*V[j,k,r]*c[k,r]*d[i,r]
    # For this to work: sum_r (a[i,r]*d[i,r]) * b[j,r] * V[j,k,r] * c[k,r] = 1 for all i,j,k
    # This requires a[i,r]*d[i,r] to be independent of i, i.e., = 1 for all i
    # So d[i,r] = 1/a[i,r]
    
    # This gives us a structured family. Let's search within it.
    
    # Simpler structured approach:
    # Try U[i,j,r] = 1 for all (i,j) where r corresponds to (j,k) pairs
    # This is essentially the schoolbook but with potential sharing
    
    # Let's try a more systematic search using linear algebra:
    # For a given rank r, we need to find U (6xr), V (12xr), W (8xr) such that
    # for all (u,v,w) in tensor_entries: sum_r U[u,r]*V[v,r]*W[w,r] = 1
    # for all other (u,v,w): sum_r U[u,r]*V[v,r]*W[w,r] = 0
    
    # This is a system of 6*12*8 = 576 equations (but only 24 are =1, rest =0)
    # Total unknowns: 6r + 12r + 8r = 26r
    # For r=12: 312 unknowns, 576 equations (overdetermined but structured)
    
    # Use iterative refinement with smart initialization:
    
    def verify_solution(U, V, W, rank):
        """Check if the decomposition is exact."""
        for (u, v, w) in tensor_entries:
            val = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
            if abs(val - 1.0) > 1e-10:
                return False
        # Check non-entries (sample for speed)
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    if (u, v, w) not in tensor_entries:
                        val = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
                        if abs(val) > 1e-10:
                            return False
        return True
    
    # Try structured initializations for various ranks
    for target_rank in range(12, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        found = False
        
        # Approach: Initialize with structured patterns, then refine
        # Pattern 1: Based on (j,k) pairs - schoolbook-like but with potential compression
        # Pattern 2: Random with half-integer rounding
        
        np.random.seed(42)
        
        num_attempts = 50
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize
            if attempt < 20:
                # Structured: U depends on j only, W depends on k only
                U_init = np.zeros((nm, target_rank))
                V_init = np.zeros((mp, target_rank))
                W_init = np.zeros((pn, target_rank))
                
                # For each (j,k) pair, assign a rank index
                for r in range(min(target_rank, m * p)):
                    j = r // p
                    k = r % p
                    # U[i,j,r] = 1 for all i
                    for i in range(n):
                        U_init[i * m + j, r] = 1
                    # V[j,k,r] = 1
                    V_init[j * p + k, r] = 1
                    # W[k,i,r] = 1 for all i
                    for i in range(n):
                        W_init[k * n + i, r] = 1
                # Add random perturbation for remaining columns
                for r in range(m * p, target_rank):
                    U_init[:, r] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=nm)
                    V_init[:, r] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=mp)
                    W_init[:, r] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=pn)
            else:
                # Random initialization
                U_init = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], size=(nm, target_rank))
                V_init = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], size=(mp, target_rank))
                W_init = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], size=(pn, target_rank))
            
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()
            
            # Iterative refinement: alternate solving for each matrix
            for iteration in range(200):
                if time.time() > deadline - 3:
                    break
                
                # Solve for W given U, V
                # For each tensor entry (u,v,w): sum_r U[u,r]*V[v,r]*W[w,r] = 1
                # This is linear in W[w,r] for fixed U,V
                A_W = np.zeros((num_entries, pn * target_rank))
                b_W = np.ones(num_entries)
                
                for e_idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
                    for r in range(target_rank):
                        col = w_idx * target_rank + r
                        A_W[e_idx, col] = U_curr[u_idx, r] * V_curr[v_idx, r]
                
                W_flat, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
                W_curr = W_flat.reshape(pn, target_rank)
                W_curr = np.round(W_curr * 2) / 2  # Round to half-integers
                
                # Solve for U given V, W
                A_U = np.zeros((num_entries, nm * target_rank))
                for e_idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
                    for r in range(target_rank):
                        col = u_idx * target_rank + r
                        A_U[e_idx, col] = V_curr[v_idx, r] * W_curr[w_idx, r]
                
                U_flat, _, _, _ = np.linalg.lstsq(A_U, b_W, rcond=None)
                U_curr = U_flat.reshape(nm, target_rank)
                U_curr = np.round(U_curr * 2) / 2
                
                # Solve for V given U, W
                A_V = np.zeros((num_entries, mp * target_rank))
                for e_idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
                    for r in range(target_rank):
                        col = v_idx * target_rank + r
                        A_V[e_idx, col] = U_curr[u_idx, r] * W_curr[w_idx, r]
                
                V_flat, _, _, _ = np.linalg.lstsq(A_V, b_W, rcond=None)
                V_curr = V_flat.reshape(mp, target_rank)
                V_curr = np.round(V_curr * 2) / 2
                
                # Check convergence
                error = 0.0
                for (u, v, w) in tensor_entries:
                    val = sum(U_curr[u, r] * V_curr[v, r] * W_curr[w, r] for r in range(target_rank))
                    error += abs(val - 1.0)
                
                if error < 1e-10:
                    # Verify full solution
                    if verify_solution(
                        U_curr.tolist(), V_curr.tolist(), W_curr.tolist(), target_rank
                    ):
                        best_rank = target_rank
                        best_result = {
                            "U": U_curr.tolist(),
                            "V": V_curr.tolist(),
                            "W": W_curr.tolist()
                        }
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break
    
    # Final verification
    if best_rank < n * m * p:
        U, V, W = best_result["U"], best_result["V"], best_result["W"]
        if verify_solution(U, V, W, best_rank):
            return best_result
    
    # Return schoolbook if nothing better found
    return best_result
# EVOLVE-BLOCK-END
