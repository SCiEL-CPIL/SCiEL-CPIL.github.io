import time
import numpy as np
from collections import defaultdict

DIMENSIONS = [2, 5, 6]
DEADLINE = 115.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # 1. Define the target tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if we are multiplying A[i,j] * B[j,k]
    # The decomposition requires: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Note: The interface defines U as (n*m, rank), V as (m*p, rank), W as (p*n, rank).
    # Indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Target tensor T is non-zero only when the indices are consistent with matrix mult.
    # Actually, the tensor for matrix multiplication C = A * B is defined by:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # So the tensor T[i, j, k] = 1 for all valid i, j, k.
    
    # 2. Generate a pool of candidate rank-1 tensors.
    # A rank-1 tensor is defined by three sets of indices:
    # U_support: set of (i, j) pairs
    # V_support: set of (j, k) pairs
    # W_support: set of (k, i) pairs
    # And coefficients (usually 1, -1, or 1/2 etc, but let's stick to integers for now to avoid floating point issues in exactness checks, 
    # though half-integers are allowed. Strassen uses integers).
    
    # To find a minimal rank, we need to cover the "1"s in the tensor T.
    # However, the tensor T is the standard matrix multiplication tensor.
    # T[i, j, k] = 1.
    # We want to represent T as a sum of rank-1 tensors.
    
    # Candidate generation strategy:
    # We will generate candidates based on "blocks".
    # A candidate can be a "standard" term (single i, single j, single k) -> Rank 1 contribution to T[i,j,k].
    # Or a "Strassen-like" term involving sums/differences.
    
    # Let's define a candidate by its coefficients in U, V, W.
    # Since the dimension is small (2x5x6), we can enumerate potential "structures".
    
    # Structure types:
    # 1. Single (i, j, k): U has 1 at (i,j), V has 1 at (j,k), W has 1 at (k,i).
    #    This contributes +1 to T[i,j,k].
    # 2. Strassen 2x2x2 blocks.
    #    We can try to identify all possible 2x2x2 sub-tensors within the 2x5x6 space.
    #    For a 2x2x2 block at offsets (io, jo, ko), we have 7 specific rank-1 tensors (M1..M7).
    
    # Let's build a dictionary of all possible candidate rank-1 tensors.
    # Key: tuple of (U_vec, V_vec, W_vec) flattened? No, too big.
    # Better: Represent each candidate by its effect on the target tensor T.
    # Effect is a vector of length n*m*p.
    # T[i, j, k] is the target value.
    # A candidate rank-1 tensor contributes:
    # contrib[i, j, k] = U[i, j] * V[j, k] * W[k, i]
    
    # We want to find a set of candidates {c1, c2, ..., cr} such that:
    # sum_{r} contrib_r[i, j, k] = 1 for all i, j, k.
    
    # This is a linear system. But we want minimal r.
    # This is equivalent to finding a sparse solution to A x = b, where A is the matrix of candidate contributions.
    # Since n*m*p = 60, the vector length is 60.
    
    # Let's generate candidates:
    candidates = []
    
    # Type 1: Standard unit tensors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i, j] = 1
                # V[j, k] = 1
                # W[k, i] = 1
                # Contribution: 1 at (i, j, k), 0 elsewhere.
                # We represent this as a vector of length 60.
                vec = np.zeros(n * m * p)
                vec[i * m * p + j * p + k] = 1.0
                # Store the coefficients for reconstruction later
                # U_coeff: dict (i,j) -> val
                # V_coeff: dict (j,k) -> val
                # W_coeff: dict (k,i) -> val
                u_c = {(i, j): 1}
                v_c = {(j, k): 1}
                w_c = {(k, i): 1}
                candidates.append((vec, u_c, v_c, w_c))
                
    # Type 2: Strassen 2x2x2 blocks
    # We need to iterate over all possible 2x2x2 blocks.
    # i offsets: 0..n-2 (0..0 for n=2)
    # j offsets: 0..m-2 (0..3 for m=5)
    # k offsets: 0..p-2 (0..4 for p=6)
    
    # Strassen 7 terms for 2x2x2:
    # Let local indices be 0,1.
    # M1: U(0,0), U(1,1); V(0,0), V(1,1); W(0,0), W(1,1)
    # M2: U(1,0), U(1,1); V(0,0); W(1,0), W(1,1) with signs
    # ... (Standard Strassen definitions)
    
    def get_strassen_terms(io, jo, ko):
        terms = []
        # Local indices:
        # i: io, io+1
        # j: jo, jo+1
        # k: ko, ko+1
        
        # M1
        u = {(io, jo): 1, (io+1, jo+1): 1}
        v = {(jo, ko): 1, (jo+1, ko+1): 1}
        w = {(ko, io): 1, (ko+1, io+1): 1}
        terms.append((u, v, w))
        
        # M2
        u = {(io+1, jo): 1, (io+1, jo+1): 1}
        v = {(jo, ko): 1}
        w = {(ko, io+1): 1, (ko+1, io+1): -1}
        terms.append((u, v, w))
        
        # M3
        u = {(io, jo): 1}
        v = {(jo, ko+1): 1, (jo+1, ko+1): -1}
        w = {(ko+1, io): 1, (ko+1, io+1): 1}
        terms.append((u, v, w))
        
        # M4
        u = {(io+1, jo+1): 1}
        v = {(jo+1, ko): 1, (jo, ko): -1}
        w = {(ko, io): 1, (ko, io+1): 1}
        terms.append((u, v, w))
        
        # M5
        u = {(io, jo): 1, (io, jo+1): 1}
        v = {(jo+1, ko+1): 1}
        w = {(ko, io): -1, (ko+1, io): 1}
        terms.append((u, v, w))
        
        # M6
        u = {(io+1, jo): 1, (io, jo): -1}
        v = {(jo, ko): 1, (jo, ko+1): 1}
        w = {(ko+1, io+1): 1}
        terms.append((u, v, w))
        
        # M7
        u = {(io, jo+1): 1, (io+1, jo+1): -1}
        v = {(jo+1, ko): 1, (jo+1, ko+1): 1}
        w = {(ko, io): 1}
        terms.append((u, v, w))
        
        return terms

    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 1):
                for u_c, v_c, w_c in get_strassen_terms(io, jo, ko):
                    # Compute contribution vector
                    vec = np.zeros(n * m * p)
                    for (i, j), uv in u_c.items():
                        for (j2, k), vv in v_c.items():
                            if j == j2: # Must match j index
                                for (k2, i2), wv in w_c.items():
                                    if k == k2 and i == i2: # Must match k and i
                                        vec[i * m * p + j * p + k] += uv * vv * wv
                    candidates.append((vec, u_c, v_c, w_c))
    
    # Type 3: Mixed Strassen/Standard? 
    # The greedy approach should be able to combine any of these.
    # However, Strassen terms have "negative" or "multiple" entries.
    # The target is all 1s.
    
    # 3. Greedy Set Cover / Linear Combination
    # We want to approximate T (all 1s) with a sum of candidates.
    # Let X be the matrix of candidates (rows = candidates, cols = 60 tensor entries).
    # We want to find a sparse vector y such that X^T y = T.
    # Wait, candidates are rows. So X is (num_candidates, 60).
    # We want sum_r y_r * candidate_r = T.
    
    # Since the system is likely underdetermined or just determined for small ranks,
    # we can use a greedy approach:
    # 1. Initialize residual R = T.
    # 2. Find candidate c that maximizes dot(c, R) / norm(c)^2 (or similar metric for reduction).
    #    Actually, since we want exact match, we should look for candidates that align well with the residual.
    #    However, Strassen terms have negative entries.
    #    
    # Alternative: Since the dimension is small, we can try to solve for small ranks directly.
    # But "Greedy" is faster.
    
    # Let's refine the metric:
    # We want to reduce the L2 error ||R||.
    # If we add y*c, new residual is R - y*c.
    # Optimal y = <R, c> / <c, c>.
    # Reduction in error squared = <R,c>^2 / <c,c>.
    # We pick the candidate that maximizes this reduction.
    
    # However, we need the final result to be EXACT.
    # Greedy might get stuck.
    # Given the small size (60 dimensions), we can use a hybrid:
    # 1. Greedy to get a good starting point (e.g., rank ~50).
    # 2. Then, use local search or integer programming if time permits?
    #    No, 120s is tight for complex IP.
    #    
    # Let's stick to a robust Greedy that handles the "exactness" by checking at the end.
    # If Greedy fails to be exact, we fallback to the known 54-rank solution.
    
    # Actually, there is a better way for this specific problem.
    # The tensor is sparse in a structural sense? No, it's dense (all 60 entries are 1).
    # 
    # Let's implement the Greedy.
    
    T_target = np.ones(n * m * p)
    residual = T_target.copy()
    
    selected_indices = []
    selected_coeffs = []
    
    max_iter = 100
    for _ in range(max_iter):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        best_idx = -1
        best_reduction = 0
        best_coeff = 0
        
        # Iterate through all candidates
        # Optimization: Precompute norms?
        # There are ~60 standard + 1*4*5*7 = 20+140 = 160 candidates.
        # 160 * 60 ops is tiny.
        
        for idx, (vec, u_c, v_c, w_c) in enumerate(candidates):
            # Skip if already selected? No, we can use same candidate multiple times?
            # No, rank-1 decomposition usually implies distinct terms, but coefficients can scale.
            # Actually, if we scale a candidate, it's still the same "structure".
            # So we just pick the structure and determine the coefficient.
            
            dot_rc = np.dot(residual, vec)
            dot_cc = np.dot(vec, vec)
            
            if dot_cc == 0:
                continue
                
            coeff = dot_rc / dot_cc
            reduction = (dot_rc ** 2) / dot_cc
            
            # We only want to add if it reduces error significantly
            if reduction > best_reduction:
                best_reduction = reduction
                best_idx = idx
                best_coeff = coeff
                
        if best_idx == -1 or best_reduction < 1e-9:
            break
            
        # Add to selection
        selected_indices.append(best_idx)
        selected_coeffs.append(best_coeff)
        
        # Update residual
        residual -= best_coeff * candidates[best_idx][0]
        
        # Check for convergence
        if np.linalg.norm(residual) < 1e-9:
            break
            
    # Check if exact
    if np.linalg.norm(residual) < 1e-9:
        rank_found = len(selected_indices)
        
        # Construct U, V, W
        U = np.zeros((n * m, rank_found))
        V = np.zeros((m * p, rank_found))
        W = np.zeros((p * n, rank_found))
        
        for r, idx in enumerate(selected_indices):
            coeff = selected_coeffs[r]
            vec, u_c, v_c, w_c = candidates[idx]
            
            for (i, j), val in u_c.items():
                U[i * m + j, r] = val * coeff
            for (j, k), val in v_c.items():
                V[j * p + k, r] = val
            for (k, i), val in w_c.items():
                W[k * n + i, r] = val
                
        # Verify exactness with random test
        A_test = np.random.randint(-10, 10, (n, m))
        B_test = np.random.randint(-10, 10, (m, p))
        C_test = A_test @ B_test
        
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0
                for j in range(m):
                    for r in range(rank_found):
                        val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                C_calc[i, k] = val
                
        if np.allclose(C_test, C_calc, atol=1e-5):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback: Return the known 54-rank solution from the seed
    # Reconstructing the seed logic quickly to ensure we return something valid
    
    n, m, p = DIMENSIONS
    rank_new = 54
    U_new = np.zeros((n * m, rank_new))
    V_new = np.zeros((m * p, rank_new))
    W_new = np.zeros((p * n, rank_new))
    
    current_rank = 0
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal current_rank
        # r=0: M1
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=1: M2
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = -1
        current_rank += 1
        # r=2: M3
        U_new[(i_off)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = -1
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=3: M4
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = -1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=4: M5
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = -1
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        current_rank += 1
        # r=5: M6
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off), current_rank] = -1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=6: M7
        U_new[(i_off)*m + (j_off+1), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = -1
        V_new[(j_off+1)*p + (k_off), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        current_rank += 1

    def add_standard_block(i_off, j_off, k_off):
        nonlocal current_rank
        U_new[(i_off)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        current_rank += 1
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        current_rank += 1

    # Part 1: A[:, 0:2] * B[0:2, :]
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # Part 2a: A[:, 2:3] * B[2:3, :]
    for k in range(6):
        add_standard_block(0, 2, k)
        
    # Part 2b: A[:, 3:5] * B[3:5, :]
    add_strassen_block(0, 3, 0)
    add_strassen_block(0, 3, 2)
    add_strassen_block(0, 3, 4)
    
    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
