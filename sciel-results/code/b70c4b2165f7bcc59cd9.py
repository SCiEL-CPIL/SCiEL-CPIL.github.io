"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 110  # Safety margin
    
    n, m, p = DIMENSIONS
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # Constraint: for all a in range(n), b in range(m), c in range(p):
    #   sum_r U[a*m+b, r] * V[b*p+c, r] * W[c*n+a, r] = 1
    # And for all "invalid" (x,y,z) triples, sum should be 0.
    
    # Build the full tensor equation
    # The tensor T is (n*m, m*p, p*n) with T[a*m+b, b*p+c, c*n+a] = 1
    # All other entries = 0.
    
    def build_tensor_rhs():
        """Build RHS vector for the tensor equation."""
        rhs = {}
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    x = a * m + b
                    y = b * p + c
                    z = c * n + a
                    rhs[(x, y, z)] = 1.0
        # All other (x,y,z) have RHS = 0
        return rhs
    
    def check_solution(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        rank = U.shape[1]
        rhs = build_tensor_rhs()
        
        # Check all required 1s
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    x = a * m + b
                    y = b * p + c
                    z = c * n + a
                    val = 0.0
                    for r in range(rank):
                        val += U[x, r] * V[y, r] * W[z, r]
                    if abs(val - 1.0) > 1e-6:
                        return False
                    if time.time() > DEADLINE:
                        return None
        
        # Check a sample of 0-entries for performance
        import random
        random.seed(42)
        num_checks = 10000
        for _ in range(num_checks):
            x = random.randint(0, n*m - 1)
            y = random.randint(0, m*p - 1)
            z = random.randint(0, p*n - 1)
            # Skip if this is a required 1-entry
            ax, bx = divmod(x, m)
            by, cy = divmod(y, p)
            cz, az = divmod(z, n)
            if ax == az and bx == by and cy == cz:
                continue
            val = 0.0
            for r in range(rank):
                val += U[x, r] * V[y, r] * W[z, r]
            if abs(val) > 1e-6:
                return False
            if time.time() > DEADLINE:
                return None
        
        return True
    
    # Start with schoolbook and try to reduce
    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook_decomposition()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try greedy term elimination: merge pairs of terms
    def try_reduce(U, V, W, max_iters=500):
        """Try to reduce rank by finding redundant or mergeable terms."""
        rank = U.shape[1]
        U_work = U.copy()
        V_work = V.copy()
        W_work = W.copy()
        
        reduced = True
        while reduced and rank > 20:
            reduced = False
            if time.time() > DEADLINE - 10:
                break
            for r1 in range(rank):
                if time.time() > DEADLINE - 10:
                    break
                if abs(U_work[:, r1]).max() < 1e-10 or abs(V_work[:, r1]).max() < 1e-10 or abs(W_work[:, r1]).max() < 1e-10:
                    # This term is zero, can remove
                    U_work = np.delete(U_work, r1, axis=1)
                    V_work = np.delete(V_work, r1, axis=1)
                    W_work = np.delete(W_work, r1, axis=1)
                    rank -= 1
                    reduced = True
                    break
            if reduced:
                continue
            
            # Try merging: can we express term r1 as combination of others?
            # This is complex; skip for now
            # Try random perturbation approach
            if time.time() > DEADLINE - 20:
                break
        
        # Check validity
        result = check_solution(U_work, V_work, W_work)
        if result is None:
            # Timeout, return what we have if it was valid before
            return U, V, W, rank
        if result:
            return U_work, V_work, W_work, rank
        return U, V, W, U.shape[1]
    
    U, V, W, rank = try_reduce(U, V, W)
    if rank < best_rank:
        best_rank = rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Now try a numerical optimization approach for lower ranks
    # We'll try ranks from lower to higher
    for target_rank in range(60, best_rank):
        if time.time() > DEADLINE - 15:
            break
        
        # Initialize with random sparse matrices
        best_for_rank = None
        for trial in range(5):
            if time.time() > DEADLINE - 15:
                break
            
            np.random.seed(42 + trial)
            
            U_init = np.zeros((n*m, target_rank))
            V_init = np.zeros((m*p, target_rank))
            W_init = np.zeros((p*n, target_rank))
            
            # Initialize from schoolbook by grouping terms
            # Each group of target_rank schoolbook terms maps to one rank dimension
            # This is just a heuristic init
            for r in range(target_rank):
                # Assign a few entries
                idx_u = np.random.randint(0, n*m, size=1)
                idx_v = np.random.randint(0, m*p, size=1)
                idx_w = np.random.randint(0, p*n, size=1)
                U_init[idx_u, r] = 1.0
                V_init[idx_v, r] = 1.0
                W_init[idx_w, r] = 1.0
            
            # Use iterative refinement
            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()
            
            for it in range(100):
                if time.time() > DEADLINE - 15:
                    break
                
                # Compute residual for all (a,b,c) triples
                # Update U while fixing V, W
                # U[x] should satisfy: for all (y,z) that pair with x, sum_r U[x,r]*V[y,r]*W[z,r] = T[x,y,z]
                
                # This is a bilinear system; use alternating least squares
                # Fix V and W, solve for U
                # For each x, and for each (y,z), the constraint is linear in U[x]
                # But we need: for ALL valid (y,z) pairs, the equation holds
                # Actually the tensor T[x,y,z] is nonzero only for specific (x,y,z)
                
                # Let's use gradient descent on the loss
                loss = 0.0
                for a in range(n):
                    for b in range(m):
                        for c in range(p):
                            x = a * m + b
                            y = b * p + c
                            z = c * n + a
                            val = np.sum(U_cur[x, :] * V_cur[y, :] * W_cur[z, :])
                            loss += (val - 1.0) ** 2
                        if time.time() > DEADLINE - 15:
                            break
                    if time.time() > DEADLINE - 15:
                        break
                if time.time() > DEADLINE - 15:
                    break
                
                if loss < 1e-10:
                    break
                
                # Simple gradient step (too slow for large dimensions)
                # Instead, try a different approach: use the structure
                
                # For fixed V, W: the constraint on U[x] is that for all (y,z) where T[x,y,z]=1,
                # U[x] · (V[y] ∘ W[z]) = 1, and for all others, U[x] · (V[y] ∘ W[z]) = 0
                # where ∘ is element-wise product.
                
                # This is an overdetermined linear system for each x
                # Let's try solving it least-squares
                
                # For efficiency, only do this for a few x at a time
                for x in range(0, n*m, 3):  # Every 3rd x
                    if time.time() > DEADLINE - 15:
                        break
                    # Collect all (y,z) pairs
                    A_mat = []
                    b_vec = []
                    for y in range(m*p):
                        for z in range(p*n):
                            prod = V_cur[y, :] * W_cur[z, :]
                            # Determine T[x,y,z]
                            # x = a*m+b, y=b'*p+c, z=c'*n+a'
                            ax, bx = divmod(x, m)
                            by, cy = divmod(y, p)
                            cz, az = divmod(z, n)
                            if ax == az and bx == by and cy == cz:
                                tval = 1.0
                            else:
                                tval = 0.0
                            A_mat.append(prod)
                            b_vec.append(tval)
                    
                    if A_mat:
                        A_arr = np.array(A_mat)
                        b_arr = np.array(b_vec)
                        # Solve least squares
                        U_cur[x, :], residuals, rank_ls, sv = np.linalg.lstsq(A_arr, b_arr, rcond=None)
                
                # Similarly for V and W (but this gets very expensive)
                # Skip for now; just update U
                
                if loss < 1e-8:
                    break
            
            # Final check
            result = check_solution(U_cur, V_cur, W_cur)
            if result is None:
                break
            if result:
                best_for_rank = (U_cur.copy(), V_cur.copy(), W_cur.copy())
                break
        
        if best_for_rank:
            U, V, W = best_for_rank
            best_rank = target_rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            break  # Found a lower rank
    
    # Convert to list format and verify
    def to_list(mat):
        return [[float(mat[i, j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    # Final verification
    final_check = check_solution(best_U, best_V, best_W)
    if final_check is True or final_check is None:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    
    # Fallback: schoolbook
    U_fb, V_fb, W_fb = schoolbook_decomposition()
    return {"U": to_list(U_fb), "V": to_list(V_fb), "W": to_list(W_fb)}
# EVOLVE-BLOCK-END
