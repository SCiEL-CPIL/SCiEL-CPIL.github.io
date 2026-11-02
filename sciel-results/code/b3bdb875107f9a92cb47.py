"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    # Build the matrix multiplication tensor T[i,j,k] = delta[i*k + j, j*p + k]
    # Actually, the multiplication tensor for M_{n,m,p}:
    # (A * B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # The tensor is T[i,j,k] which is 1 if (i,j,k) corresponds to a valid term, 0 otherwise
    # More precisely, the structure tensor has T[i,j,k] = 1 for all (i,j,k) since
    # each product A_{i,j} * B_{j,k} contributes to C_{i,k}
    #
    # Wait, let me reconsider. The matrix multiplication tensor T^{n,m,p} is defined as:
    # T = sum_{i,j,k} e_i^A \otimes e_j^B \otimes e_k^C * delta_{i,k}... 
    # No. The correct definition: T(i,j,k) = 1 if the (i,j) entry of A and (j,k) entry of B
    # multiply to give the (i,k) entry of C. So T(i,j,k) = 1 for all valid (i,j,k) tuples.
    #
    # A rank-r decomposition gives: T(i,j,k) = sum_{r=1}^{rank} U[i,j,r] * V[j,k,r] * W[i,k,r]
    # where U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    #
    # The constraint is: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other combinations... wait, there are no other combinations since
    # every (i,j,k) with i in [0,n), j in [0,m), k in [0,p) is a valid term.
    #
    # Actually the tensor is exactly: T[i,j,k] = 1 for all i,j,k in their respective ranges.
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    # The current best known is rank 21 (3 independent Strassen blocks of rank 7 each).
    # The theoretical lower bound is ceil(n*m*p / something)... 
    # For M_{2,2,6}, the trivial rank is n*m*p = 24.
    # Strassen gives 7 per 2x2x2 block, and we have 3 such blocks (since p=6=3*2),
    # giving rank 21.
    #
    # Can we do better? The question is whether there's a rank-20 decomposition.
    # Let's try a systematic search using the structure.
    
    # First, let's establish the rank-21 solution (3 independent Strassen blocks)
    def strassen_3blocks():
        rank = 21
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        # Strassen factors for 2x2 matrix multiplication
        # For a 2x2 x 2x2 multiplication:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        U_coeffs = [
            {0: 1, 3: 1},       # M1: A00 + A11
            {2: 1, 3: 1},       # M2: A10 + A11
            {0: 1},             # M3: A00
            {3: 1},             # M4: A11
            {0: 1, 1: 1},       # M5: A00 + A01
            {2: 1, 0: -1},      # M6: A10 - A00
            {1: 1, 3: -1}       # M7: A01 - A11
        ]
        
        V_coeffs = [
            [(0, 0, 1), (1, 1, 1)],      # M1: B00 + B11
            [(0, 0, 1)],                 # M2: B00
            [(0, 1, 1), (1, 1, -1)],     # M3: B01 - B11
            [(1, 0, 1), (0, 0, -1)],     # M4: B10 - B00
            [(1, 1, 1)],                 # M5: B11
            [(0, 0, 1), (0, 1, 1)],      # M6: B00 + B01
            [(1, 0, 1), (1, 1, 1)]       # M7: B10 + B11
        ]
        
        W_map = [
            [(0, 0, 1), (1, 1, 1)],   # C00 += M1, C11 += M1
            [(1, 0, 1), (1, 1, -1)],  # C10 += M2, C11 -= M2
            [(0, 1, 1), (1, 1, 1)],   # C01 += M3, C11 += M3
            [(0, 0, 1), (1, 0, 1)],   # C00 += M4, C10 += M4
            [(0, 0, -1), (0, 1, 1)],  # C00 -= M5, C01 += M5
            [(1, 1, 1)],              # C11 += M6
            [(0, 0, 1)]               # C00 += M7
        ]
        
        for r in range(3):
            for k in range(7):
                col = r * 7 + k
                for idx, val in U_coeffs[k].items():
                    U[idx, col] = val
                for j_local, k_local, val in V_coeffs[k]:
                    global_k = 2 * r + k_local
                    v_idx = j_local * p + global_k
                    V[v_idx, col] = val
                for i, j_local, val in W_map[k]:
                    global_k = 2 * r + j_local
                    w_idx = global_k * n + i
                    W[w_idx, col] = val
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U, V, W):
        """Verify the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    def verify_strict(U, V, W):
        """Verify with exact integer/half-integer arithmetic."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1.0:
                        return False
        return True
    
    # Start with the known rank-21 solution
    result = strassen_3blocks()
    
    if not verify(result["U"], result["V"], result["W"]):
        # Fallback: schoolbook decomposition (rank 24)
        U = [[0]*24 for _ in range(4)]
        V = [[0]*24 for _ in range(12)]
        W = [[0]*24 for _ in range(12)]
        for i in range(2):
            for j in range(2):
                for k in range(6):
                    r = i*12 + j*6 + k
                    U[i*2+j][r] = 1
                    V[j*6+k][r] = 1
                    W[k*2+i][r] = 1
        result = {"U": U, "V": V, "W": W}
        return result
    
    # Now try to find a rank-20 decomposition
    # Strategy: Start from the rank-21 solution and try to eliminate one term
    # by finding a linear combination that cancels it out.
    #
    # Alternative: Direct search for rank-20 using the fact that the tensor
    # has special structure.
    #
    # Let's try a different approach: use the fact that for M_{2,2,6},
    # we can think of it as 3 copies of M_{2,2,2} but potentially with
    # shared structure between them.
    #
    # Key insight: In the 3-block Strassen construction, the U matrix is
    # block-diagonal in a sense (each block of 7 columns only touches
    # specific rows of U). Specifically:
    # - Block 0 (cols 0-6): U rows 0,1,2,3 all used
    # - Block 1 (cols 7-13): U rows 0,1,2,3 all used  
    # - Block 2 (cols 14-20): U rows 0,1,2,3 all used
    #
    # But V and W have different structures per block:
    # - Block r uses V rows corresponding to B_{j, 2r} and B_{j, 2r+1}
    # - Block r uses W rows corresponding to C_{2r, i} and C_{2r+1, i}
    #
    # To get rank 20, we need to find a way to share one term across blocks.
    
    # Try rank-20 search: modify the rank-21 solution by introducing
    # a shared term and adjusting the others.
    
    # Let's try a more direct approach: parameterize a rank-20 solution
    # and solve the resulting system of equations.
    
    # For rank 20, we have:
    # U: 4 x 20, V: 12 x 20, W: 12 x 20
    # Total unknowns: 4*20 + 12*20 + 12*20 = 80 + 240 + 240 = 560
    # Constraints: n*m*p = 2*2*6 = 24 equations
    # This is heavily underdetermined, so solutions should exist if rank 20 is achievable.
    
    # The problem is that the equations are cubic (products of 3 variables).
    # Let's try a randomized approach: fix some variables and solve for others.
    
    # Approach: Fix U and V randomly (or with structure), then solve for W.
    # Given U and V, the constraint becomes:
    # For each (i,j,k): sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    # This is linear in W! So for fixed U and V, we can solve for W linearly.
    
    # But we need the system to be consistent. With 24 equations and 240 unknowns (W entries),
    # we have many degrees of freedom. The question is whether we can find U and V such
    # that the resulting linear system for W has a solution.
    
    # Let's try: fix U to have a specific structure (like the Strassen pattern but with
    # 20 columns), and see if we can find V and W.
    
    # Actually, let's try a cleaner approach. The rank-21 solution has a very specific
    # structure. Let's see if we can perturb it to get rank 20.
    
    # Another idea: use the "decomposition sharing" approach. If two of the three
    # Strassen blocks share one multiplication, we save one rank.
    
    # Let's try: can we make block 0 and block 1 share one of the 7 Strassen terms?
    # This would require that for some k in {0,...,6}, the U, V, W contributions
    # of term k in block 0 and block 1 can be combined.
    
    # For sharing to work, we need:
    # U_block0[k] = U_block1[k] (same U pattern)
    # V_block0[k] and V_block1[k] can be different (since they act on different B entries)
    # W_block0[k] and W_block1[k] can be different (since they act on different C entries)
    
    # But wait - if U is the same, then the "multiplier" is the same linear combination
    # of A entries. The V and W just determine which B and C entries are involved.
    #
    # In the standard 3-block construction:
    # Block 0: U uses A entries for all of A (since n=2, m=2, the same A is used)
    # Block 1: U uses the same A entries
    # Block 2: U uses the same A entries
    #
    # So U is actually the SAME for all three blocks! The difference is only in V and W.
    # This means we already have a "shared" structure in U.
    
    # Let me re-examine: in the 3-block construction, U is 4x21 where:
    # Columns 0-6: Strassen U pattern for block 0
    # Columns 7-13: Strassen U pattern for block 1 (same as block 0!)
    # Columns 14-20: Strassen U pattern for block 2 (same as blocks 0 and 1!)
    
    # So U[:, 0:7] = U[:, 7:14] = U[:, 14:21]
    
    # This means we could potentially merge some columns if the V and W patterns
    # allow it. But since V and W are different for each block (acting on different
    # B and C entries), we can't simply merge columns.
    
    # However, we might be able to find a rank-20 solution where one column
    # serves dual purpose across two blocks.
    
    # Let's try a direct numerical search for rank 20.
    # Strategy: 
    # 1. Initialize with rank-21 solution
    # 2. Try to eliminate one column by adjusting the others
    
    # For rank 20, let's try the following approach:
    # Take the rank-21 solution and try to express one column as a linear combination
    # of the others (in a generalized sense that preserves the tensor identity).
    
    # This is complex. Let's try a different tactic:
    # Use the fact that the problem is equivalent to finding a rank-20 decomposition
    # of a specific tensor, and use known results from algebraic complexity theory.
    
    # For M_{2,2,6}, the best known upper bound is 21 (from 3 independent Strassen).
    # The lower bound is at least 20 (from various methods).
    # It's an open question whether rank 20 is achievable.
    
    # Given the uncertainty, let's stick with rank 21 unless we can find rank 20
    # within the time budget.
    
    # Try a quick randomized search for rank 20:
    best_result = result
    best_rank = 21
    
    # Try a few random rank-20 attempts
    np.random.seed(42)
    
    for attempt in range(10):
        if time.time() > deadline - 30:
            break
        
        # Try to find rank-20 by modifying the rank-21 solution
        # Remove one column and try to adjust the remaining 20 columns
        
        # This is a complex nonlinear problem. Let's try a simpler heuristic:
        # Check if any column of the rank-21 solution is "redundant" in the sense
        # that removing it and scaling the others can still satisfy the constraints.
        
        # For each column r to remove, check if the remaining 20 columns can be
        # scaled to satisfy all 24 constraints.
        
        U_arr = np.array(result["U"], dtype=np.float64)
        V_arr = np.array(result["V"], dtype=np.float64)
        W_arr = np.array(result["W"], dtype=np.float64)
        
        for remove_col in range(21):
            if time.time() > deadline - 30:
                break
            
            # Remove column remove_col
            mask = [i for i in range(21) if i != remove_col]
            U_sub = U_arr[:, mask]
            V_sub = V_arr[:, mask]
            W_sub = W_arr[:, mask]
            
            # Now we have 20 columns. Can we find scaling factors s_r such that
            # for all (i,j,k): sum_{r in mask} s_r * U_sub[i,j,r] * V_sub[j,k,r] * W_sub[k,i,r] = 1?
            
            # This is still nonlinear in s_r. But let's try: set all s_r = 1 first
            # and check how close we are.
            
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_sub[i*m+j, :] * V_sub[j*p+k, :] * W_sub[k*n+i, :])
                        err = abs(val - 1.0)
                        max_err = max(max_err, err)
            
            if max_err < 1e-10:
                # We found a rank-20 solution!
                best_result = {
                    "U": U_sub.tolist(),
                    "V": V_sub.tolist(),
                    "W": W_sub.tolist()
                }
                best_rank = 20
                break
        
        if best_rank == 20:
            break
    
    # Try another approach: direct construction of rank-20
    # Use the insight that M_{2,2,6} might have a rank-20 decomposition
    # based on more clever sharing.
    
    if best_rank > 20 and time.time() < deadline - 20:
        # Try to construct rank-20 using a modified Strassen approach
        # where one of the 21 terms is replaced by a combination.
        
        # Let's try: use 6 terms for block 0, 6 for block 1, 6 for block 2, 
        # and 2 shared terms = 20 total.
        # This requires that 2 of the Strassen terms can be shared between blocks.
        
        # This is speculative. Let's try a numerical optimization approach.
        
        # Parameterize rank-20 solution and minimize the error.
        # Use the rank-21 solution as initialization.
        
        U_init = np.array(result["U"], dtype=np.float64)
        V_init = np.array(result["V"], dtype=np.float64)
        W_init = np.array(result["W"], dtype=np.float64)
        
        # Take first 20 columns as initial guess
        U20 = U_init[:, :20].copy()
        V20 = V_init[:, :20].copy()
        W20 = W_init[:, :20].copy()
        
        # Compute initial error
        def compute_error(U20, V20, W20):
            errs = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U20[i*m+j, :] * V20[j*p+k, :] * W20[k*n+i, :])
                        errs.append((val - 1.0) ** 2)
            return np.sum(errs)
        
        # Try a few random perturbations
        for pert in range(5):
            if time.time() > deadline - 10:
                break
            
            U_trial = U20 + np.random.randn(4, 20) * 0.01
            V_trial = V20 + np.random.randn(12, 20) * 0.01
            W_trial = W20 + np.random.randn(12, 20) * 0.01
            
            err = compute_error(U_trial, V_trial, W_trial)
            
            if err < 1e-10:
                best_result = {
                    "U": U_trial.tolist(),
                    "V": V_trial.tolist(),
                    "W": W_trial.tolist()
                }
                best_rank = 20
                break
    
    # Final verification
    if best_rank <= 20:
        if verify(best_result["U"], best_result["V"], best_result["W"]):
            return best_result
    
    # Return the rank-21 solution
    return result
# EVOLVE-BLOCK-END
