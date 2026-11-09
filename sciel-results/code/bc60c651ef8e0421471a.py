"""Find rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 3, 8]

def construct():
    """Find rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Index helpers
    def idx_u(i, j):
        return i * m + j
    
    def idx_v(j, k):
        return j * p + k
    
    def idx_w(k, i):
        return k * n + i
    
    # Schoolbook decomposition as baseline
    schoolbook_rank = n * m * p  # 72
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[idx_u(i, j), col] = 1.0
                V_sb[idx_v(j, k), col] = 1.0
                W_sb[idx_w(k, i), col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Strategy: use the fact that we can combine terms
    # For each rank candidate, try to find a valid decomposition
    
    # The tensor entries are all 1, so we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    # Try ranks from lower bound up
    lower_bound = max(n * m, m * p, p * n)  # 24
    
    # For small ranks, we can try to solve the system
    # Fix U and V to have simple structure, then solve for W
    
    # Approach: try U and V with entries in {-1, 0, 1, 0.5}
    # and solve for W
    
    # First, let's try a structured approach
    # For rank r, we have r "terms" each contributing to the sum
    
    # Try to find decompositions by searching over small entry sets
    entry_set = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # For efficiency, try a few candidate ranks
    # The theoretical lower bound is 24, but practical lower bound may be higher
    
    # Try ranks: 24, 25, 26, ..., up to 72
    # But this is computationally expensive, so limit the search
    
    elapsed = time.time() - start_time
    if elapsed < 100:
        # Try a few candidate ranks near the lower bound
        for candidate_rank in range(lower_bound, min(lower_bound + 20, schoolbook_rank + 1)):
            if time.time() - start_time > deadline - 30:
                break
            
            # For this rank, try to find U, V, W with entries in entry_set
            # such that the tensor equation is satisfied
            
            # This is a constraint satisfaction problem
            # Use a greedy approach: build up the decomposition term by term
            
            U = np.zeros((n * m, candidate_rank))
            V = np.zeros((m * p, candidate_rank))
            W = np.zeros((p * n, candidate_rank))
            
            # Residual tensor
            residual = np.ones((n, m, p))
            
            # Greedy: at each step, find a rank-1 component that
            # maximally reduces the residual
            
            for step in range(candidate_rank):
                if time.time() - start_time > deadline - 10:
                    break
                
                # Try to find a good rank-1 component
                # A rank-1 component is u ⊗ v ⊗ w where:
                # u[i*m+j] * v[j*p+k] * w[k*n+i] should match residual[i,j,k]
                
                # For simplicity, try binary or small integer patterns
                best_score = -1
                best_u, best_v, best_w = None, None, None
                
                # Try a few random candidates
                np.random.seed(42 + step * 1000 + candidate_rank)
                for trial in range(50):
                    if time.time() - start_time > deadline - 5:
                        break
                    
                    # Random initialization with entries from entry_set
                    u = np.random.choice(entry_set, size=(n * m,))
                    v = np.random.choice(entry_set, size=(m * p,))
                    w = np.random.choice(entry_set, size=(p * n,))
                    
                    # Compute how well this matches the residual
                    score = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = u[idx_u(i, j)] * v[idx_v(j, k)] * w[idx_w(k, i)]
                                if abs(val - residual[i, j, k]) < 1e-9:
                                    score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
                
                if best_u is None:
                    break
                
                # Add this component
                U[:, step] = best_u
                V[:, step] = best_v
                W[:, step] = best_w
                
                # Update residual
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            residual[i, j, k] -= best_u[idx_u(i, j)] * best_v[idx_v(j, k)] * best_w[idx_w(k, i)]
                
                # Check if we've satisfied all terms
                if np.allclose(residual, 0, atol=1e-9):
                    if step + 1 < best_rank:
                        best_rank = step + 1
                        best_U = U[:, :step + 1].copy()
                        best_V = V[:, :step + 1].copy()
                        best_W = W[:, :step + 1].copy()
                    break
    
    # Verify the best decomposition found
    if best_rank < schoolbook_rank:
        verified = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(best_U[idx_u(i, j), :] * best_V[idx_v(j, k), :] * best_W[idx_w(k, i), :])
                    if abs(val - 1.0) > 1e-9:
                        verified = False
                        break
                if not verified:
                    break
        
        if verified:
            # Convert to lists with integer/half-integer entries
            U_list = [[round(x, 1) for x in row] for row in best_U.tolist()]
            V_list = [[round(x, 1) for x in row] for row in best_V.tolist()]
            W_list = [[round(x, 1) for x in row] for row in best_W.tolist()]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return schoolbook decomposition
    U_list = U_sb.tolist()
    V_list = V_sb.tolist()
    W_list = W_sb.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
