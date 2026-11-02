"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = 2, 2, 11

    # Build the rank-39 Strassen-based construction
    rank = 39  # 5 Strassen blocks (7 each) + 1 schoolbook column (4)

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    r = 0

    # 5 Strassen blocks for columns (0,1), (2,3), (4,5), (6,7), (8,9)
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1

        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1

        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1

        # m1 = (a+d)(e+h)
        U[0, r] += 1; U[3, r] += 1
        V[v_e, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1; W[w_c11, r] += 1
        r += 1

        # m2 = (c+d)e
        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c10, r] += 1; W[w_c11, r] += -1
        r += 1

        # m3 = a(f-h)
        U[0, r] += 1
        V[v_f, r] += 1; V[v_h, r] += -1
        W[w_c01, r] += 1; W[w_c11, r] += 1
        r += 1

        # m4 = d(g-e)
        U[3, r] += 1
        V[v_g, r] += 1; V[v_e, r] += -1
        W[w_c00, r] += 1; W[w_c10, r] += 1
        r += 1

        # m5 = (a+b)h
        U[0, r] += 1; U[1, r] += 1
        V[v_h, r] += 1
        W[w_c00, r] += -1; W[w_c01, r] += 1
        r += 1

        # m6 = (c-a)(e+f)
        U[2, r] += 1; U[0, r] += -1
        V[v_e, r] += 1; V[v_f, r] += 1
        W[w_c11, r] += 1
        r += 1

        # m7 = (b-d)(g+h)
        U[1, r] += 1; U[3, r] += -1
        V[v_g, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1
        r += 1

        if time.time() > deadline:
            break

    # Last column k=10: schoolbook
    if r < rank:
        k2 = 10
        v_e2 = k2
        v_g2 = 11 + k2
        w_c02 = 2 * k2
        w_c12 = 2 * k2 + 1

        U[0, r] += 1
        V[v_e2, r] += 1
        W[w_c02, r] += 1
        r += 1

        U[1, r] += 1
        V[v_g2, r] += 1
        W[w_c02, r] += 1
        r += 1

        U[2, r] += 1
        V[v_e2, r] += 1
        W[w_c12, r] += 1
        r += 1

        U[3, r] += 1
        V[v_g2, r] += 1
        W[w_c12, r] += 1
        r += 1

    # Verify decomposition
    U_f = U.astype(np.float64)
    V_f = V.astype(np.float64)
    W_f = W.astype(np.float64)
    
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                target[alpha, beta, gamma] = 1.0
    
    current = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r_idx in range(rank):
        current += np.einsum('a,b,c->abc', U_f[:, r_idx], V_f[:, r_idx], W_f[:, r_idx])
    
    if np.max(np.abs(current - target)) > 1e-10:
        # Decomposition is wrong, return as is
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt rank reduction: find sparse integer solutions to A@c = target
    # where A is the 1936x39 matrix of flattened rank-1 terms
    
    def flatten_term(U_col, V_col, W_col):
        tensor = np.einsum('a,b,c->abc', U_col, V_col, W_col)
        return tensor.flatten()
    
    terms = [flatten_term(U_f[:, r_idx], V_f[:, r_idx], W_f[:, r_idx]) for r_idx in range(rank)]
    target_flat = target.flatten()
    
    # Build A matrix (1936 x 39)
    A = np.column_stack(terms)
    
    # Find the null space of A to identify dependencies
    try:
        U_svd, s, Vt_svd = np.linalg.svd(A, full_matrices=True)
        tol = 1e-10 * max(s[0] if len(s) > 0 else 1, 1.0)
        nullity = rank - np.sum(s > tol)
        
        if nullity > 0 and time.time() < deadline - 20:
            # Null space vectors: Vt_svd[rank - nullity:]
            null_basis = Vt_svd[rank - nullity:].T  # shape (39, nullity)
            
            # Find a particular solution
            c_particular, _, _, _ = np.linalg.lstsq(A, target_flat, rcond=None)
            c_particular_rounded = np.round(c_particular)
            
            # Check if particular solution is integer
            residual = A @ c_particular_rounded - target_flat
            is_integer_sol = np.max(np.abs(residual)) < 1e-8
            
            if is_integer_sol:
                # Try to use null space to eliminate entries
                # For each null vector, try to add multiples to eliminate one component
                current_c = c_particular_rounded.copy()
                
                for _ in range(50):  # Max 50 elimination attempts
                    if time.time() > deadline - 10:
                        break
                    
                    # Find the smallest non-zero entry
                    nonzero = np.abs(current_c) > 0.5
                    if not np.any(nonzero):
                        break
                    
                    # Try to eliminate the first non-zero entry
                    idx = np.where(nonzero)[0][0]
                    val = current_c[idx]
                    
                    # For each null vector, see if we can reduce |val|
                    eliminated = False
                    for nv in range(nullity):
                        nv_vec = null_basis[:, nv]
                        if abs(nv_vec[idx]) < 1e-10:
                            continue
                        
                        # We want to find integer k such that val + k * nv_vec[idx] is closer to 0
                        # k = -round(val / nv_vec[idx])
                        k_candidate = -np.round(val / nv_vec[idx])
                        k_int = int(k_candidate)
                        
                        new_c = current_c + k_int * nv_vec
                        new_c = np.round(new_c)
                        
                        # Verify it's still a valid solution
                        new_residual = A @ new_c - target_flat
                        if np.max(np.abs(new_residual)) < 1e-8:
                            # Count non-zeros
                            old_nz = np.sum(np.abs(current_c) > 0.5)
                            new_nz = np.sum(np.abs(new_c) > 0.5)
                            if new_nz < old_nz:
                                current_c = new_c
                                eliminated = True
                                break
                    
                    if not eliminated:
                        break
                
                # Extract indices where current_c != 0
                active_indices = np.where(np.abs(current_c) > 0.5)[0]
                
                if len(active_indices) < rank:
                    # Verify the solution
                    U_new = U[:, active_indices]
                    V_new = V[:, active_indices]
                    W_new = W[:, active_indices]
                    
                    # The coefficients are current_c[active_indices]
                    # But we need the decomposition to use coefficients of 1
                    # So we need to absorb the coefficients into U, V, or W
                    
                    # If coefficient is an integer, we can multiply it into U
                    U_new = U_new * current_c[active_indices]
                    
                    # Verify
                    U_f2 = U_new.astype(np.float64)
                    V_f2 = V_new.astype(np.float64)
                    W_f2 = W_new.astype(np.float64)
                    current2 = np.zeros((n*m, m*p, p*n), dtype=np.float64)
                    for r_idx in range(len(active_indices)):
                        current2 += np.einsum('a,b,c->abc', U_f2[:, r_idx], V_f2[:, r_idx], W_f2[:, r_idx])
                    
                    if np.max(np.abs(current2 - target)) < 1e-8:
                        U = U_new
                        V = V_new
                        W = W_new
                        rank = len(active_indices)
    except:
        pass
    
    # Final verification
    U_f = U.astype(np.float64)
    V_f = V.astype(np.float64)
    W_f = W.astype(np.float64)
    current = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r_idx in range(rank):
        current += np.einsum('a,b,c->abc', U_f[:, r_idx], V_f[:, r_idx], W_f[:, r_idx])
    
    if np.max(np.abs(current - target)) > 1e-8:
        # Fallback - rebuild original
        U_orig = np.zeros((n * m, 39), dtype=np.int64)
        V_orig = np.zeros((m * p, 39), dtype=np.int64)
        W_orig = np.zeros((p * n, 39), dtype=np.int64)
        r = 0
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            v_e, v_f, v_g, v_h = k0, k1, 11+k0, 11+k1
            w_c00, w_c10, w_c01, w_c11 = 2*k0, 2*k0+1, 2*k1, 2*k1+1
            
            U_orig[0,r]+=1; U_orig[3,r]+=1; V_orig[v_e,r]+=1; V_orig[v_h,r]+=1; W_orig[w_c00,r]+=1; W_orig[w_c11,r]+=1; r+=1
            U_orig[2,r]+=1; U_orig[3,r]+=1; V_orig[v_e,r]+=1; W_orig[w_c10,r]+=1; W_orig[w_c11,r]+=-1; r+=1
            U_orig[0,r]+=1; V_orig[v_f,r]+=1; V_orig[v_h,r]+=-1; W_orig[w_c01,r]+=1; W_orig[w_c11,r]+=1; r+=1
            U_orig[3,r]+=1; V_orig[v_g,r]+=1; V_orig[v_e,r]+=-1; W_orig[w_c00,r]+=1; W_orig[w_c10,r]+=1; r+=1
            U_orig[0,r]+=1; U_orig[1,r]+=1; V_orig[v_h,r]+=1; W_orig[w_c00,r]+=-1; W_orig[w_c01,r]+=1; r+=1
            U_orig[2,r]+=1; U_orig[0,r]+=-1; V_orig[v_e,r]+=1; V_orig[v_f,r]+=1; W_orig[w_c11,r]+=1; r+=1
            U_orig[1,r]+=1; U_orig[3,r]+=-1; V_orig[v_g,r]+=1; V_orig[v_h,r]+=1; W_orig[w_c00,r]+=1; r+=1
        
        k2=10; v_e2=k2; v_g2=11+k2; w_c02=2*k2; w_c12=2*k2+1
        U_orig[0,r]+=1; V_orig[v_e2,r]+=1; W_orig[w_c02,r]+=1; r+=1
        U_orig[1,r]+=1; V_orig[v_g2,r]+=1; W_orig[w_c02,r]+=1; r+=1
        U_orig[2,r]+=1; V_orig[v_e2,r]+=1; W_orig[w_c12,r]+=1; r+=1
        U_orig[3,r]+=1; V_orig[v_g2,r]+=1; W_orig[w_c12,r]+=1; r+=1
        
        return {"U": U_orig.tolist(), "V": V_orig.tolist(), "W": W_orig.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
