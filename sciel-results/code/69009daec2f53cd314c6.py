"""Matrix multiplication 2x3x4 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    # Build the target tensor: T[u,v,w] = 1 if (i,j,k) is a valid triple
    # u = i*m+j, v = j*p+k, w = k*n+i
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
    
    # Fast verification using numpy
    def verify_solution(U, V, W, rank):
        """Check if the decomposition is exact using vectorized operations."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # For each tensor entry, check sum_r U[u,r]*V[v,r]*W[w,r] = 1
        for (u, v, w) in tensor_entries:
            val = np.dot(U[u], V[v] * W[w])
            if abs(val - 1.0) > 1e-8:
                return False
        
        # Check non-entries: all other (u,v,w) should give 0
        # We can check by computing the full tensor and verifying
        # Full tensor: T[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
        # This is a 3-tensor of shape (nm, mp, pn)
        # Compute efficiently: for each r, outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        
        # Check all entries
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    if (u, v, w) in tensor_entries:
                        continue
                    val = np.dot(U[u], V[v] * W[w])
                    if abs(val) > 1e-8:
                        return False
        return True
    
    # Try to find a lower-rank decomposition using structured patterns
    # The key insight: for matrix multiplication, we can sometimes share
    # factors between different (i,j,k) triples
    
    # Strategy: Try ranks from 12 to 23, using multiple initialization strategies
    for target_rank in range(12, best_rank):
        if time.time() > deadline - 15:
            break
        
        found = False
        
        # Multiple initialization strategies
        np.random.seed(42)
        
        num_attempts = 30
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
            
            if attempt < 10:
                # Structured: based on (j,k) pairs with potential sharing
                U_init = np.zeros((nm, target_rank))
                V_init = np.zeros((mp, target_rank))
                W_init = np.zeros((pn, target_rank))
                
                # First m*p columns: schoolbook-like
                for r in range(min(target_rank, m * p)):
                    j = r // p
                    k = r % p
                    for i in range(n):
                        U_init[i * m + j, r] = 1
                    V_init[j * p + k, r] = 1
                    for i in range(n):
                        W_init[k * n + i, r] = 1
                
                # Remaining columns: random half-integers
                for r in range(m * p, target_rank):
                    U_init[:, r] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=nm)
                    V_init[:, r] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=mp)
                    W_init[:, r] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=pn)
            elif attempt < 20:
                # Random with structured bias
                U_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(nm, target_rank))
                V_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(mp, target_rank))
                W_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(pn, target_rank))
            else:
                # Pure random
                U_init = np.random.randn(nm, target_rank) * 0.5
                V_init = np.random.randn(mp, target_rank) * 0.5
                W_init = np.random.randn(pn, target_rank) * 0.5
            
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()
            
            # Iterative refinement with alternating least squares
            for iteration in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Solve for W given U, V
                # For each tensor entry (u,v,w): sum_r U[u,r]*V[v,r]*W[w,r] = 1
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
                    val = np.dot(U_curr[u], V_curr[v] * W_curr[w])
                    error += abs(val - 1.0)
                
                if error < 1e-8:
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
