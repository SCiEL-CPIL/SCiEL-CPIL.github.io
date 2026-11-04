"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 27
    dim_w = p * n  # 18
    
    # Build the target tensor: for each (i,j,k), the product entry is 1
    # The decomposition requires: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
    # And 0 for all other combinations (but in matrix mult, only (i,j,k) with valid indices appear)
    
    # Actually, the tensor T is defined such that T[(i,j), (j,k), (k,i)] = 1
    # The decomposition is: T = sum_c U_c ⊗ V_c ⊗ W_c where U_c is a vector in R^{n*m}, etc.
    
    # Let's set up the equations. For each term c, we have:
    # U_c is a vector of length n*m, V_c of length m*p, W_c of length p*n
    # The product U_c ⊗ V_c ⊗ W_c gives a tensor, and we sum over c.
    
    # The constraint is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for all other triples (a,b,c) where the pattern doesn't match, the sum should be 0.
    # But actually, in matrix multiplication, only the triples (i,j), (j,k), (k,i) are connected.
    
    # Let me think differently. The matrix multiplication tensor M has:
    # M[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # All other entries are 0.
    
    # We want: sum_c U[:,c] * V[:,c] * W[:,c] = M (as a tensor)
    
    # This means for each triple (a,b,c):
    # sum_d U[a,d]*V[b,d]*W[c,d] = M[a,b,c]
    
    # So we have dim_u * dim_v * dim_w equations, but many are trivially satisfied (M=0).
    
    # Let's collect the non-trivial equations: those where M[a,b,c] = 1.
    # a = i*m+j, b = j*p+k, c = k*n+i
    
    # Number of non-trivial equations: n*m*p = 54
    
    # For rank r, we have r*(dim_u + dim_v + dim_w) = r*(6+27+18) = 51r unknowns.
    # We have 54 non-trivial equations (plus the constraint that all other entries are 0, which gives more equations).
    
    # Actually, we need ALL entries to match, so we have dim_u*dim_v*dim_w = 6*27*18 = 2916 equations.
    # But most have M=0, so they're homogeneous.
    
    # This is a hard problem. Let me try a different approach.
    
    # Start with schoolbook and try to reduce rank.
    start_time = time.time()
    deadline = start_time + 140  # leave 10s margin
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((dim_u, schoolbook_rank))
    V_sb = np.zeros((dim_v, schoolbook_rank))
    W_sb = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # For each rank r, try random initializations and optimize
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        # Compute the tensor: sum_c U[:,c] * V[:,c] * W[:,c]
        # This is an einsum: 'ac,bc,cc->abc'
        # But we only need to check the non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    val = np.sum(U[a,:] * V[b,:] * W[c,:])
                    if abs(val - 1) > tol:
                        return False
        # Also check that some zero entries are actually zero
        # Check a few random zero entries
        rng = np.random.RandomState(42)
        for _ in range(100):
            a = rng.randint(dim_u)
            b = rng.randint(dim_v)
            c = rng.randint(dim_w)
            # Check if this is a valid (i,j,k) triple
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if not (j == j2 and k == k2 and i == i2):
                # This should be zero
                val = np.sum(U[a,:] * V[b,:] * W[c,:])
                if abs(val) > tol:
                    return False
        return True
    
    def round_to_half_int(X):
        """Round to nearest half-integer."""
        return np.round(X * 2) / 2
    
    # Try ranks from schoolbook_rank down to schoolbook_rank - 5
    for rank in range(schoolbook_rank, max(schoolbook_rank - 10, 0), -1):
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Initialize
            if rank == schoolbook_rank:
                U = U_sb[:, :rank].copy()
                V = V_sb[:, :rank].copy()
                W = W_sb[:, :rank].copy()
            else:
                # Random initialization with small values
                rng = np.random.RandomState(trial)
                U = rng.randn(dim_u, rank) * 0.5
                V = rng.randn(dim_v, rank) * 0.5
                W = rng.randn(dim_w, rank) * 0.5
            
            # Optimize using gradient descent on the loss
            # Loss = sum over all (a,b,c) of (sum_d U[a,d]*V[b,d]*W[c,d] - M[a,b,c])^2
            # But this is expensive. Let's use a simpler approach.
            
            # Use the fact that we can solve for one variable given the others.
            # For fixed V and W, the equations are linear in U.
            
            # Build the linear system for U:
            # For each (i,j,k): sum_d U[i*m+j, d] * V[j*p+k, d] * W[k*n+i, d] = 1
            # This is a linear system in U.
            
            # Let's set up the system for U given V and W.
            # Number of equations: n*m*p = 54
            # Number of unknowns: dim_u * rank = 6 * rank
            
            # For rank < 54, this is underdetermined, so we can find a solution.
            
            # Build A matrix where A[row, col] and we solve A @ U_flat = b
            rows = []
            b_vec = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m+j
                        b_idx = j*p+k
                        c_idx = k*n+i
                        # The equation: sum_d U[a,d] * V[b_idx,d] * W[c_idx,d] = 1
                        # This is linear in U[a,:]
                        # Coefficients for U[a,d] are V[b_idx,d] * W[c_idx,d]
                        row = np.zeros(dim_u * rank)
                        for d in range(rank):
                            row[a*rank + d] = V[b_idx, d] * W[c_idx, d]
                        rows.append(row)
                        b_vec.append(1.0)
            
            A = np.array(rows)
            b = np.array(b_vec)
            
            # Solve least squares: A @ U_flat = b
            try:
                U_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                U = U_flat.reshape(dim_u, rank)
            except:
                continue
            
            # Now verify if this works
            # Check the main equations
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m+j
                        b_idx = j*p+k
                        c_idx = k*n+i
                        val = np.sum(U[a,:] * V[b_idx,:] * W[c_idx,:])
                        if abs(val - 1) > 1e-8:
                            ok = False
                            break
                if not ok:
                    break
            
            if not ok:
                continue
            
            # Check some zero entries
            rng = np.random.RandomState(123)
            for _ in range(50):
                a = rng.randint(dim_u)
                b_idx = rng.randint(dim_v)
                c_idx = rng.randint(dim_w)
                i = a // m
                j = a % m
                j2 = b_idx // p
                k = b_idx % p
                k2 = c_idx // n
                i2 = c_idx % n
                if not (j == j2 and k == k2 and i == i2):
                    val = np.sum(U[a,:] * V[b_idx,:] * W[c_idx,:])
                    if abs(val) > 1e-8:
                        ok = False
                        break
            
            if not ok:
                continue
            
            # Round to half-integers
            U_r = round_to_half_int(U)
            V_r = round_to_half_int(V)
            W_r = round_to_half_int(W)
            
            # Verify the rounded version
            if verify_decomposition(U_r, V_r, W_r, tol=1e-10):
                # Found a valid decomposition!
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                best_rank = rank
                found = True
                break
        
        if found:
            # Found a lower rank, continue trying to go even lower
            continue
        else:
            # Didn't find for this rank, break (assume lower ranks won't work either)
            break
    
    return best_result
# EVOLVE-BLOCK-END
