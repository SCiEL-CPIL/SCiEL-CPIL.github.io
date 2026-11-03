import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # Try to find a lower rank solution
    # We search for ranks from 35 down to 20
    for r in range(35, 19, -1):
        if time.time() > deadline:
            break
            
        # Setup linear system
        # Variables: U (9x r), V (12x r), W (12x r)
        # Total vars: 33 * r
        num_vars = 33 * r
        num_constraints = n * m * m * p * p * n  # 1296
        
        # We can't easily build the full 1296x33r matrix in a simple way without heavy memory.
        # Instead, we can use a randomized search with verification.
        
        # Random search with local search
        # Initialize with random integers/half-integers
        found = False
        attempts = 0
        max_attempts = 5000  # Limit per rank
        
        while not found and attempts < max_attempts:
            if time.time() > deadline:
                break
            attempts += 1
            
            # Random initialization
            vals = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(33, r))
            U = vals[:9, :].tolist()
            V = vals[9:21, :].tolist()
            W = vals[21:33, :].tolist()
            
            # Verify
            valid = True
            # Check non-zero constraints
            for i in range(n):
                for j in range(m):
                    for l in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m+j][c] * V[j*p+l][c] * W[l*n+i][c]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                continue
                
            # Check zero constraints
            # We only need to check a subset if we are lucky, but we must check all.
            # To speed up, we can check randomly sampled zero constraints first.
            # But for correctness, we must check all.
            # 1260 zero constraints.
            # Each check is O(r).
            # Total O(1260 * r).
            # For r=30, 37800 ops. 5000 attempts -> 190M ops. Might be too slow in Python.
            
            # Optimization: Use numpy for verification
            U_np = np.array(U)
            V_np = np.array(V)
            W_np = np.array(W)
            
            # Reshape to tensors
            # U: (n, m, r)
            # V: (m, p, r)
            # W: (p, n, r)
            U_t = U_np.reshape(n, m, r)
            V_t = V_np.reshape(m, p, r)
            W_t = W_np.reshape(p, n, r)
            
            # Compute T_reconstructed
            # T[i, j, j', l, l', i'] = sum_c U[i,j,c] V[j',l,c] W[l',i',c]
            # We want T[i, j, j, l, l, i] = 1 and others 0.
            
            # This is equivalent to:
            # For each i, j, l:
            # sum_c U[i,j,c] V[j,l,c] W[l,i,c] == 1
            # For all other (i,j,j',l,l',i'):
            # sum_c U[i,j,c] V[j',l,c] W[l',i',c] == 0
            
            # Let's check the non-zero ones first
            ok = True
            for i in range(n):
                for j in range(m):
                    for l in range(p):
                        s = np.sum(U_t[i, j, :] * V_t[j, l, :] * W_t[l, i, :])
                        if abs(s - 1.0) > 1e-9:
                            ok = False
                            break
                if not ok:
                    break
            if not ok:
                continue
                
            # Check zero ones
            # We can construct the full tensor and check sparsity
            # T_full = np.einsum('ijc,jlc,lic->ijjlil', U_t, V_t, W_t)
            # This is too large (3x3x3x4x4x3 = 1296 elements)
            # But we can check it.
            
            # Actually, we can just compute the full tensor and compare to the target.
            # Target tensor T_target has 1 at (i, j, j, l, l, i)
            
            # Let's use einsum to compute the sum for all indices
            # S[i, j, j', l, l', i'] = sum_c U[i, j, c] V[j', l, c] W[l', i', c]
            # Note: V is indexed by (j', l), W by (l', i')
            # U[i, j, c] V[j', l, c] W[l', i', c]
            
            # We can do this in steps to save memory
            # Temp1[i, j, j', l, c] = U[i, j, c] V[j', l, c]
            # Temp2[i, j, j', l, l', i'] = sum_c Temp1[i, j, j', l, c] W[l', i', c]
            
            # This is 3x3x3x4x4x3 = 1296 elements.
            # It's small enough.
            
            Temp1 = np.einsum('ijc,jlc->ijjl c', U_t, V_t) # Wait, indices
            # U_t: i, j, c
            # V_t: j', l, c
            # We want to sum over c.
            # Result: i, j, j', l
            Temp1 = np.einsum('ijc,jlc->ijjl', U_t, V_t)
            
            # Now multiply by W_t[l', i', c] and sum over c? No, c is already summed.
            # Wait, the sum is over c.
            # S[i, j, j', l, l', i'] = sum_c (U[i, j, c] V[j', l, c]) W[l', i', c]
            # This is not separable like that.
            # It is sum_c U[i, j, c] V[j', l, c] W[l', i', c]
            
            # Let's compute it directly for the specific indices we care about.
            # Or use a loop over c.
            S = np.zeros((n, m, m, p, p, n))
            for c in range(r):
                # u_c: (n, m)
                # v_c: (m, p)
                # w_c: (p, n)
                u_c = U_t[:, :, c]
                v_c = V_t[:, :, c]
                w_c = W_t[:, :, c]
                
                # Outer product u_c (i,j) and v_c (j',l) and w_c (l',i')
                # This creates a 6D tensor.
                # S += u_c[:,:,None,None,None,None] * v_c[None,None,:,None,None,None] * w_c[None,None,None,:,None,None] * ...
                # This is expensive.
                
                # Alternative:
                # S[i, j, j', l, l', i'] = sum_c U[i,j,c] V[j',l,c] W[l',i',c]
                # We can compute this by iterating over i, j, j', l, l', i'
                # But that's 1296 * r ops.
                
                # Let's just check the non-zero constraints and a random sample of zero constraints.
                # If it passes, we verify fully.
                
            # Given the complexity, let's just use the full check for the few candidates that pass the non-zero check.
            # In the loop above, we already checked non-zero.
            # Now check zero.
            
            zero_ok = True
            # Iterate over all indices
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for l in range(p):
                            for lp in range(p):
                                for ip in range(n):
                                    if (j == jp and l == lp and ip == i):
                                        continue
                                    s = 0.0
                                    for c in range(r):
                                        s += U[i*m+j][c] * V[jp*p+l][c] * W[lp*n+ip][c]
                                    if abs(s) > 1e-9:
                                        zero_ok = False
                                        break
                                if not zero_ok:
                                    break
                            if not zero_ok:
                                break
                        if not zero_ok:
                            break
                    if not zero_ok:
                        break
                if not zero_ok:
                    break
            if not zero_ok:
                continue
                
            if valid and zero_ok:
                found = True
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
                break
                
    return {"U": best_U, "V": best_V, "W": best_W}
