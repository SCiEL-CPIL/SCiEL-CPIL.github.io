import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def make_rank28():
        """Construct the known rank-28 decomposition using Strassen for 2x2 blocks."""
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
        for i in range(n):
            for k in range(p):
                U[i * m + 2][r] = 1
                V[2 * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
        # Part 2: A1 B13 (j=0,1, k=4) - 4 terms
        for i in range(n):
            for j in range(2):
                U[i * m + j][r] = 1
                V[j * p + 4][r] = 1
                W[4 * n + i][r] = 1
                r += 1
        # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen - 7 terms
        # M1: (A00+A11)(B00+B11)
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 0][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = 1; W[1 * n + 1][r] = 1; r += 1
        # M2: (A10+A11)B00
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 0][r] = 1; W[0 * n + 1][r] = 1; W[1 * n + 1][r] = -1; r += 1
        # M3: A00(B10-B00)
        U[0 * m + 0][r] = 1; V[0 * p + 1][r] = 1; V[1 * p + 1][r] = -1; W[1 * n + 0][r] = 1; W[1 * n + 1][r] = 1; r += 1
        # M4: (A00-A10)B11
        U[1 * m + 1][r] = 1; V[1 * p + 0][r] = 1; V[0 * p + 0][r] = -1; W[0 * n + 0][r] = 1; W[0 * n + 1][r] = 1; r += 1
        # M5: (A00+A01)B11
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = -1; W[1 * n + 0][r] = 1; r += 1
        # M6: (A10-A00)(B00+B01)
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1; V[0 * p + 0][r] = 1; V[0 * p + 1][r] = 1; W[1 * n + 1][r] = 1; r += 1
        # M7: (A01-A11)(B10+B11)
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1; V[1 * p + 0][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = 1; r += 1
        # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen - 7 terms
        # M1: (A00+A11)(B02+B13)
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 2][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = 1; W[3 * n + 1][r] = 1; r += 1
        # M2: (A10+A11)B02
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 2][r] = 1; W[2 * n + 1][r] = 1; W[3 * n + 1][r] = -1; r += 1
        # M3: A00(B13-B03)
        U[0 * m + 0][r] = 1; V[0 * p + 3][r] = 1; V[1 * p + 3][r] = -1; W[3 * n + 0][r] = 1; W[3 * n + 1][r] = 1; r += 1
        # M4: (A00-A10)B13
        U[1 * m + 1][r] = 1; V[1 * p + 2][r] = 1; V[0 * p + 2][r] = -1; W[2 * n + 0][r] = 1; W[2 * n + 1][r] = 1; r += 1
        # M5: (A00+A01)B13
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = -1; W[3 * n + 0][r] = 1; r += 1
        # M6: (A10-A00)(B02+B03)
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1; V[0 * p + 2][r] = 1; V[0 * p + 3][r] = 1; W[3 * n + 1][r] = 1; r += 1
        # M7: (A01-A11)(B12+B13)
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1; V[1 * p + 2][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = 1; r += 1
        
        return {"U": U, "V": V, "W": W}
    
    def validate(U, V, W, rank):
        """Validate that the decomposition exactly represents matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for rr in range(rank):
                        s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_structural_reduction(base_U, base_V, base_W, base_rank):
        """
        Try to reduce rank by finding alternative decompositions of the 2x2x2 sub-tensors.
        The key insight: each 2x2x2 block can potentially be decomposed with rank < 7
        if we allow the rank-1 terms to span across blocks.
        """
        U0 = np.array(base_U, dtype=float)
        V0 = np.array(base_V, dtype=float)
        W0 = np.array(base_W, dtype=float)
        
        best_rank = base_rank
        best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
        
        # Strategy: Try to merge some rank-1 terms across blocks
        # For each pair of blocks, try to find a combined decomposition
        
        # The 2x2x2 tensor has known minimal rank 7 (Strassen is optimal)
        # But when blocks share indices, we might save terms
        
        # Try: Can we combine the k=4 column with adjacent blocks?
        # The k=4 column uses 4 terms (A1 * B_col4)
        # If we could express B_col4 in terms of other columns, we might save
        
        # Alternative: Try rank-26 by finding 2 terms that can be eliminated
        # through a more clever combination
        
        # Let's try a systematic search over small modifications
        # Focus on the Strassen terms - can we modify them to cover more?
        
        # Try: Replace some Strassen terms with terms that span multiple k values
        # For example, a term that contributes to both k=0 and k=1
        
        # Generate candidate rank-1 terms that are "multi-block"
        # A rank-1 term (u, v, w) contributes to tensor[i,j,k] = u[i,j]*v[j,k]*w[k,i]
        
        # Try all possible "structured" rank-1 terms with small integer/half-integer values
        vals = [-1, -0.5, 0, 0.5, 1]
        
        # Try to find a rank-26 decomposition by modifying the base
        # Remove 2 terms and add 0 new terms (pure reduction)
        # Or remove 3 terms and add 1 new term (net -2)
        
        # For now, try a focused search: modify the Strassen blocks
        # to see if we can save 1-2 terms
        
        # Try replacing the 7 Strassen terms for B11 with a different 6-term decomposition
        # (if one exists for the specific structure)
        
        # Actually, let's try a different approach: use the fact that n=2 is small
        # The output has only 10 elements (2x5), so we have 10 equations
        # Each rank-1 term gives 10 constraints (one per output element)
        
        # Try to find a rank-26 by solving a system
        # This is complex, so let's try a more targeted search
        
        # Try: Can we save 1 term by combining Part 2 (k=4) with Part 1 (j=2)?
        # Part 1 has terms for j=2, all k. Part 2 has terms for j=0,1, k=4.
        # These are orthogonal in j, so no direct saving.
        
        # Try: Can we save terms in the Strassen blocks by using the specific values?
        # For 2x2x2, rank 7 is minimal, so no saving within a block.
        
        # The only hope is cross-block savings. Let's try a random search
        # but with a better structure: only modify entries that are non-zero in the base
        
        for target_rank in range(base_rank - 1, max(20, base_rank - 5), -1):
            if time.time() > deadline - 10:
                break
            
            # Try to reduce to target_rank by removing (base_rank - target_rank) terms
            num_remove = base_rank - target_rank
            
            for attempt in range(30):
                if time.time() > deadline - 5:
                    break
                
                # Randomly select terms to remove
                remove_indices = np.random.choice(base_rank, size=num_remove, replace=False)
                
                U_rem = np.delete(U0, remove_indices, axis=1)
                V_rem = np.delete(V0, remove_indices, axis=1)
                W_rem = np.delete(W0, remove_indices, axis=1)
                
                # The error tensor is the sum of removed terms
                error = np.zeros((n, m, p))
                for idx in remove_indices:
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                error[i, j, k] += U0[i*m+j, idx] * V0[j*p+k, idx] * W0[k*n+i, idx]
                
                # Try to find rank-1 terms that approximate the error
                # We need to find (u, v, w) such that u[i,j]*v[j,k]*w[k,i] ≈ error[i,j,k]
                
                # Try multiple random rank-1 approximations
                found = False
                for trial in range(50):
                    # Random direction for u
                    u_dir = np.random.choice(vals, size=n*m)
                    if np.all(u_dir == 0):
                        continue
                    
                    # For each (j,k), solve for v[j,k] and w[k,i]
                    # error[i,j,k] ≈ u[i,j] * v[j,k] * w[k,i]
                    
                    v_dir = np.zeros(m*p)
                    w_dir = np.zeros(p*n)
                    
                    # For fixed u, we have: error[i,j,k] / u[i,j] ≈ v[j,k] * w[k,i]
                    # This is a matrix factorization problem for each j
                    # Let E_j be the matrix with E_j[k,i] = error[i,j,k] / u[i,j] (where u[i,j] != 0)
                    # We want E_j ≈ v[:,j] * w_j where v[:,j] is column j of V and w_j is row j of W
                    
                    # Use SVD to find rank-1 approximation
                    for j in range(m):
                        E_j = np.zeros((p, n))
                        for k in range(p):
                            for i in range(n):
                                if abs(u_dir[i*m+j]) > 1e-9:
                                    E_j[k, i] = error[i, j, k] / u_dir[i*m+j]
                        # Rank-1 approximation via SVD
                        try:
                            U_svd, s, Vt = np.linalg.svd(E_j, full_matrices=False)
                            if len(s) > 0 and s[0] > 1e-9:
                                v_dir[j*p:(j+1)*p] = U_svd[:, 0] * np.sqrt(s[0])
                                w_dir[j*n:(j+1)*n] = np.sqrt(s[0]) * Vt[0, :]
                        except:
                            pass
                    
                    # Check if this rank-1 term helps
                    U_try = np.column_stack([U_rem, u_dir])
                    V_try = np.column_stack([V_rem, v_dir])
                    W_try = np.column_stack([W_rem, w_dir])
                    
                    # Validate
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s_val = 0.0
                                for rr in range(target_rank):
                                    s_val += U_try[i*m+j, rr] * V_try[j*p+k, rr] * W_try[k*n+i, rr]
                                if abs(s_val - 1.0) > 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        best_U, best_V, best_W = U_try, V_try, W_try
                        best_rank = target_rank
                        found = True
                        break
                
                if found:
                    break
        
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist(), "rank": best_rank}
    
    def focused_search(base_U, base_V, base_W, base_rank):
        """
        More focused search: try specific structural modifications.
        """
        U0 = np.array(base_U, dtype=float)
        V0 = np.array(base_V, dtype=float)
        W0 = np.array(base_W, dtype=float)
        
        best_rank = base_rank
        best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
        
        # Try to find rank-26 by a more targeted approach
        # The base has 28 terms. Let's see if we can save 2.
        
        # Idea: The Part 2 terms (k=4, j=0,1) are 4 terms.
        # Can we express these using a combination of other terms?
        
        # For C[i,4] = A[i,0]*B[0,4] + A[i,1]*B[1,4] + A[i,2]*B[2,4]
        # The A[i,2]*B[2,4] part is in Part 1 (term for j=2, k=4)
        # The A[i,0]*B[0,4] + A[i,1]*B[1,4] part is in Part 2 (4 terms)
        
        # What if we modify the Strassen terms to also contribute to k=4?
        # This would require changing the W matrix to have non-zero entries for k=4
        
        # Try: Modify some Strassen terms to have W entries for k=4
        # and adjust U,V accordingly
        
        # This is complex. Let's try a different approach:
        # Use the fact that we only need integer/half-integer entries
        
        # Try all combinations of small modifications to the base decomposition
        # Focus on the 7 Strassen terms for each 2x2x2 block
        
        # For the B11 block (k=0,1), try alternative 7-term decompositions
        # that might be more "compatible" with other blocks
        
        # Actually, let's try a simpler approach: random search with better initialization
        # Start from the base and make small random changes
        
        for target_rank in range(base_rank - 1, max(22, base_rank - 4), -1):
            if time.time() > deadline - 10:
                break
            
            num_remove = base_rank - target_rank
            
            for attempt in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Start from base, remove num_remove random terms
                remove_indices = np.random.choice(base_rank, size=num_remove, replace=False)
                
                U_rem = np.delete(U0, remove_indices, axis=1)
                V_rem = np.delete(V0, remove_indices, axis=1)
                W_rem = np.delete(W0, remove_indices, axis=1)
                
                # Perturb the remaining terms slightly
                U_try = U_rem.copy()
                V_try = V_rem.copy()
                W_try = W_rem.copy()
                
                # Make small integer/half-integer perturbations
                for _ in range(num_remove * 3):
                    idx_u = np.random.randint(0, n*m)
                    idx_v = np.random.randint(0, m*p)
                    idx_w = np.random.randint(0, p*n)
                    col_u = np.random.randint(0, target_rank)
                    col_v = np.random.randint(0, target_rank)
                    col_w = np.random.randint(0, target_rank)
                    
                    U_try[idx_u, col_u] += np.random.choice([-1, -0.5, 0.5, 1])
                    V_try[idx_v, col_v] += np.random.choice([-1, -0.5, 0.5, 1])
                    W_try[idx_w, col_w] += np.random.choice([-1, -0.5, 0.5, 1])
                
                # Validate
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s_val = 0.0
                            for rr in range(target_rank):
                                s_val += U_try[i*m+j, rr] * V_try[j*p+k, rr] * W_try[k*n+i, rr]
                            if abs(s_val - 1.0) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_U, best_V, best_W = U_try, V_try, W_try
                    best_rank = target_rank
                    break
        
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist(), "rank": best_rank}
    
    # Start with rank-28
    best = make_rank28()
    best_rank = 28
    
    # Try structural reduction
    if time.time() < deadline - 30:
        result = try_structural_reduction(best["U"], best["V"], best["W"], best_rank)
        if result["rank"] < best_rank:
            best = {"U": result["U"], "V": result["V"], "W": result["W"]}
            best_rank = result["rank"]
    
    # Try focused search
    if time.time() < deadline - 10:
        result = focused_search(best["U"], best["V"], best["W"], best_rank)
        if result["rank"] < best_rank:
            best = {"U": result["U"], "V": result["V"], "W": result["W"]}
            best_rank = result["rank"]
    
    return best
# EVOLVE-BLOCK-END
