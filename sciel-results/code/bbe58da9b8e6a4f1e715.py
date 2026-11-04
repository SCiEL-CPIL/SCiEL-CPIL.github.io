"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n=3, m=4, p=8):
    """Check if the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor from U, V, W
    # T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # The true multiplication tensor has T[i*m+j, j*p+k, k*n+i] = 1 if i,j,k valid
    # Actually the tensor is: T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a, b, c] = 1 if the indices are consistent: j = j' and i = i' and k = k'
    
    # Let's check by computing: for each term c, U[:,c] * V[:,c] * W[:,c]
    # U[:,c] is (n*m), V[:,c] is (m*p), W[:,c] is (p*n)
    # The product U[:,c] outer V[:,c] outer W[:,c] gives a (n*m, m*p, p*n) tensor
    # Sum over c should equal the multiplication tensor
    
    # The multiplication tensor M[a,b,c]: 
    # a = i*m+j, b = j'*p+k, c = k'*n+i'
    # M[a,b,c] = 1 if j == j' and i == i' and k == k', else 0
    # Wait, that's not right. The multiplication is C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor T[i,j,k] = A[i,j]*B[j,k]*C[i,k] ... no.
    # The matrix multiplication tensor is defined as:
    # T[i*m+j, j*p+k, k*n+i] = 1 for valid i,j,k
    # All other entries are 0.
    
    # Let me just check: compute the sum of outer products and see if it matches.
    
    # Actually, let me use a simpler check: 
    # For the decomposition to work, we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) are valid indices
    # and 0 otherwise. But since i in [0,n), j in [0,m), k in [0,p), all combinations
    # (i,j,k) are valid, so we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # for all i,j,k, and for inconsistent indices (where the j in U doesn't match j in V,
    # or i in U doesn't match i in W, or k in V doesn't match k in W), it should be 0.
    
    # Wait, I need to be more careful. The tensor T has indices (a,b,c) where
    # a = i*m+j (i in [0,n), j in [0,m))
    # b = j'*p+k (j' in [0,m), k in [0,p))  
    # c = k'*n+i' (k' in [0,p), i' in [0,n))
    
    # The multiplication tensor: T[a,b,c] = 1 iff j = j' AND i = i' AND k = k'
    # Otherwise T[a,b,c] = 0.
    
    # Let's verify by checking a few key properties:
    # For the diagonal case (a,b,c all consistent), the value should be 1.
    # For inconsistent cases, it should be 0.
    
    # Efficient check: 
    # For each c, compute the outer product contribution.
    # But that's expensive for large rank. Let's spot check.
    
    # Check 1: For all i,j,k in range, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-6:
                    return False
    
    # Check 2: For inconsistent indices, should be 0
    # E.g., U[i*m+j, c] with j, V[j'*p+k, c] with j' != j should contribute 0
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j == j2:
                    continue
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j2*p+k, c] * W[k*n+i, c]
                    if abs(s) > 1e-6:
                        return False
    
    for j in range(m):
        for k in range(p):
            for k2 in range(p):
                if k == k2:
                    continue
                for i in range(n):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k2*n+i, c]
                    if abs(s) > 1e-6:
                        return False
    
    for i in range(n):
        for i2 in range(n):
            if i == i2:
                continue
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i2, c]
                    if abs(s) > 1e-6:
                        return False
    
    return True

def construct():
    n, m, p = 3, 4, 8
    
    # Start with schoolbook decomposition (rank 96)
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0), dtype=float)
    V0 = np.zeros((m * p, rank0), dtype=float)
    W0 = np.zeros((p * n, rank0), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1
    
    best_U = U0.copy()
    best_V = V0.copy()
    best_W = W0.copy()
    best_rank = rank0
    
    # Try to find lower rank decompositions using random initialization + iterative refinement
    # Strategy: For a given target rank r, initialize U, V, W randomly and try to minimize
    # the error between the computed tensor and the target tensor.
    
    # Target tensor (sparse): T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # We'll use a gradient-free approach: for each target rank, try multiple random inits.
    
    # Time budget
    t_start = time.time()
    time_limit = 100  # Leave margin before 150s kill
    
    # Try ranks from 90 down to 96 (96 is schoolbook)
    # Actually, let's try to go lower. Theoretical lower bound is at least max(n*m, m*p, p*n) = max(12,32,24) = 32
    # But that's very optimistic. Let's try ranks 80-96 first.
    
    target_ranks = list(range(90, 97)) + list(range(80, 90)) + list(range(70, 80)) + list(range(60, 70))
    
    rng = np.random.RandomState(42)
    
    for target_rank in target_ranks:
        if time.time() - t_start > time_limit:
            break
        
        # If we already found a rank <= target_rank, skip
        if best_rank <= target_rank:
            continue
        
        # Try multiple random initializations for this rank
        n_attempts = 20 if target_rank >= 85 else 10
        if time.time() - t_start > time_limit - 30:
            n_attempts = max(1, n_attempts // 2)
        
        for attempt in range(n_attempts):
            if time.time() - t_start > time_limit:
                break
            
            # Random initialization with small values
            scale = 1.0 / np.sqrt(target_rank)
            U = rng.randn(n * m, target_rank) * scale
            V = rng.randn(m * p, target_rank) * scale
            W = rng.randn(p * n, target_rank) * scale
            
            # Iteratively update each factor to best fit the target tensor
            # For fixed V, W: U[i*m+j, :] should be such that
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i*m+j, j*p+k, k*n+i]
            # This is a linear system for each (i,j) pair.
            
            # For a fixed (i,j), the equation for k is:
            # sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
            # This gives p equations for U[i*m+j, :] (p unknowns... no, target_rank unknowns)
            # We have p equations, target_rank unknowns. Underdetermined if target_rank > p.
            
            # Better approach: for each (i,j,k) triple, we need:
            # U[i*m+j,:] . (V[j*p+k,:] * W[k*n+i,:]) = 1
            
            # And for inconsistent triples (different j, k, or i), we need:
            # U[i*m+j,:] . (V[j'*p+k,:]*W[k*n+i,:]) = 0 when j' != j
            # etc.
            
            # This is complex. Let's use a simpler iterative approach:
            # Alternate between updating U, V, W using least squares.
            
            for it in range(30):
                if time.time() - t_start > time_limit:
                    break
                
                # Update U: for each (i,j), find U[i*m+j,:] that best satisfies
                # For all k: U[i*m+j,:] . (V[j*p+k,:] * W[k*n+i,:]) = 1
                # This is: A_{ij} @ U[i*m+j,:] = 1_p
                # where A_{ij}[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # Solve least squares.
                
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Also add constraints from inconsistent indices
                        # For j' != j: U[i*m+j,:] . (V[j'*p+k,:]*W[k*n+i,:]) = 0
                        # For i' != i: U[i*m+j,:] . (V[j*p+k,:]*W[k*n+i',:]) = 0
                        
                        # Build full constraint matrix
                        all_rows = []
                        all_rhs = []
                        
                        # Consistent: for each k, want value 1
                        for k in range(p):
                            all_rows.append(V[j * p + k, :] * W[k * n + i, :])
                            all_rhs.append(1.0)
                        
                        # Inconsistent j: for each j' != j, k, want value 0
                        for j2 in range(m):
                            if j2 == j:
                                continue
                            for k in range(p):
                                all_rows.append(V[j2 * p + k, :] * W[k * n + i, :])
                                all_rhs.append(0.0)
                        
                        # Inconsistent i: for each i' != i, k, want value 0
                        for i2 in range(n):
                            if i2 == i:
                                continue
                            for k in range(p):
                                all_rows.append(V[j * p + k, :] * W[k * n + i2, :])
                                all_rhs.append(0.0)
                        
                        A_full = np.array(all_rows)
                        b_full = np.array(all_rhs)
                        
                        # Solve least squares
                        try:
                            U[i * m + j, :], residuals, rank_sol, sv = np.linalg.lstsq(A_full, b_full, rcond=None)
                        except:
                            pass
                
                # Update V: for each (j,k), find V[j*p+k,:] that best satisfies
                # For all i: V[j*p+k,:] . (U[i*m+j,:] * W[k*n+i,:]) = 1
                for j in range(m):
                    for k in range(p):
                        all_rows = []
                        all_rhs = []
                        
                        # Consistent: for each i, want value 1
                        for i in range(n):
                            all_rows.append(U[i * m + j, :] * W[k * n + i, :])
                            all_rhs.append(1.0)
                        
                        # Inconsistent j: for each j' != j, i, want value 0
                        for j2 in range(m):
                            if j2 == j:
                                continue
                            for i in range(n):
                                all_rows.append(U[i * m + j2, :] * W[k * n + i, :])
                                all_rhs.append(0.0)
                        
                        # Inconsistent k: for each k' != k, i, want value 0
                        for k2 in range(p):
                            if k2 == k:
                                continue
                            for i in range(n):
                                all_rows.append(U[i * m + j, :] * W[k2 * n + i, :])
                                all_rhs.append(0.0)
                        
                        A_full = np.array(all_rows)
                        b_full = np.array(all_rhs)
                        
                        try:
                            V[j * p + k, :], residuals, rank_sol, sv = np.linalg.lstsq(A_full, b_full, rcond=None)
                        except:
                            pass
                
                # Update W: for each (k,i), find W[k*n+i,:] that best satisfies
                # For all j: W[k*n+i,:] . (U[i*m+j,:] * V[j*p+k,:]) = 1
                for k in range(p):
                    for i in range(n):
                        all_rows = []
                        all_rhs = []
                        
                        # Consistent: for each j, want value 1
                        for j in range(m):
                            all_rows.append(U[i * m + j, :] * V[j * p + k, :])
                            all_rhs.append(1.0)
                        
                        # Inconsistent k: for each k' != k, j, want value 0
                        for k2 in range(p):
                            if k2 == k:
                                continue
                            for j in range(m):
                                all_rows.append(U[i * m + j, :] * V[j * p + k2, :])
                                all_rhs.append(0.0)
                        
                        # Inconsistent i: for each i' != i, j, want value 0
                        for i2 in range(n):
                            if i2 == i:
                                continue
                            for j in range(m):
                                all_rows.append(U[i2 * m + j, :] * V[j * p + k, :])
                                all_rhs.append(0.0)
                        
                        A_full = np.array(all_rows)
                        b_full = np.array(all_rhs)
                        
                        try:
                            W[k * n + i, :], residuals, rank_sol, sv = np.linalg.lstsq(A_full, b_full, rcond=None)
                        except:
                            pass
                
                # Check convergence
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err += (s - 1.0) ** 2
                
                if err < 1e-4:
                    break
            
            # Check if this solution is valid
            if _verify(U, V, W, n, m, p):
                # Round to nearest half-integer or integer for cleanliness
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
                    U, V, W = U_rounded, V_rounded, W_rounded
                else:
                    # Keep original if rounding breaks it
                    pass
                
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    print(f"Found valid decomposition with rank {target_rank}")
    
    # Convert to list of lists for return
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    # Convert to proper Python types (int where possible)
    U_out = [[(int(v) if abs(v - round(v)) < 1e-9 else (2*int(round(v*2)))//2 if abs(v*2 - round(v*2)) < 1e-9 else v) for v in row] for row in U_out]
    V_out = [[(int(v) if abs(v - round(v)) < 1e-9 else (2*int(round(v*2)))//2 if abs(v*2 - round(v*2)) < 1e-9 else v) for v in row] for row in V_out]
    W_out = [[(int(v) if abs(v - round(v)) < 1e-9 else (2*int(round(v*2)))//2 if abs(v*2 - round(v*2)) < 1e-9 else v) for v in row] for row in W_out]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
