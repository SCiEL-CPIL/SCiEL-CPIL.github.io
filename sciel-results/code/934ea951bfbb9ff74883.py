"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Vectorized check:
    # T[i,l,k] = sum_r U[i,l,r] * V[l,k,r] * W[k,i,r]
    # We want T[i,l,k] == 1 for all i,l,k.
    # Let's construct the tensor T.
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Indices: a = i*m+l, b = l*p+k, c = k*n+i
    
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U_arr[:, r]
        v = V_arr[:, r]
        w = W_arr[:, r]
        
        # u is vector of length n*m
        # v is vector of length m*p
        # w is vector of length p*n
        
        # Reshape u to (n, m)
        u_mat = u.reshape(n, m)
        # Reshape v to (m, p)
        v_mat = v.reshape(m, p)
        # Reshape w to (p, n)
        w_mat = w.reshape(p, n)
        
        # Contribution to T[i,l,k] is u[i,l] * v[l,k] * w[k,i]
        # This is not a simple matrix product.
        # T[i,l,k] = sum_r u_r[i,l] v_r[l,k] w_r[k,i]
        
        # We can loop over l since m is small (5)
        for l in range(m):
            u_l = u_mat[:, l] # length n
            v_l = v_mat[l, :] # length p
            # w_mat is (p, n), we need w[k,i]
            # term[i,k] = u_l[i] * v_l[k] * w_mat[k,i]
            # outer product of u_l and v_l is (n, p)
            prod = np.outer(u_l, v_l) # prod[i,k] = u_l[i] * v_l[k]
            # Multiply element-wise with w_mat.T (which is n, p)
            # w_mat is (p, n), so w_mat.T is (n, p)
            # w_mat.T[i,k] = w_mat[k,i]
            T[:, l, :] += prod * w_mat.T
            
    # Check if T is all ones
    return np.allclose(T, 1.0, atol=1e-9)

def _build_strassen_decomp(n, m, p, j_pairs, k_pairs):
    """Build a Strassen-based decomposition with given pairings."""
    rank = 7 * len(j_pairs) * len(k_pairs) + 2 * (m - 2*len(j_pairs)) * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1
            
            # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            
            # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1
            
            # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1
    
    # Remaining j indices: standard products
    used_j = set()
    for j0, j1 in j_pairs:
        used_j.add(j0)
        used_j.add(j1)
    
    for j in range(m):
        if j in used_j:
            continue
        for k in range(p):
            U[0*m + j][col] += 1
            V[j*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            
            U[1*m + j][col] += 1
            V[j*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1
    
    return U, V, W

def _try_eliminate_term_fast(U, V, W, n, m, p, rank, term_idx):
    """
    Try to eliminate term term_idx.
    Returns (success, new_U, new_V, new_W, new_rank)
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check if decomposition without term_idx is valid
    U_new = U_arr.copy()
    V_new = V_arr.copy()
    W_new = W_arr.copy()
    
    U_new[:, term_idx] = 0
    V_new[:, term_idx] = 0
    W_new[:, term_idx] = 0
    
    if _verify_decomp(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p, rank):
        # Remove column
        U_final = []
        for row in U_new:
            U_final.append([row[j] for j in range(rank) if j != term_idx])
        V_final = []
        for row in V_new:
            V_final.append([row[j] for j in range(rank) if j != term_idx])
        W_final = []
        for row in W_new:
            W_final.append([row[j] for j in range(rank) if j != term_idx])
        return True, U_final, V_final, W_final, rank - 1
    
    return False, None, None, None, rank

def _snap_to_half_integer(arr, tol=1e-6):
    """Snap values close to integers or half-integers."""
    result = []
    for row in arr:
        new_row = []
        for val in row:
            if abs(val - round(val)) < tol:
                new_row.append(float(round(val)))
            elif abs(val - (round(val*2)/2.0)) < tol:
                new_row.append(round(val*2)/2.0)
            else:
                new_row.append(float(val))
        result.append(new_row)
    return result

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 140  # 10s safety margin
    
    def _time_left():
        return _deadline - time.time()
    
    best_rank = 60
    best_result = None
    
    # 1. Landsberg Lower Bound Check (Flattening Rank)
    # Flatten tensor T[i,l,k] along i. Matrix M1 of size (n) x (m*p).
    # M1[i, l*p+k] = T[i,l,k]
    # Rank of M1 is a lower bound on tensor rank.
    # For matrix mult tensor, T[i,l,k] = 1 if i,l,k are "connected".
    # Actually, T[i,l,k] is 1 for all i,l,k in the definition of the tensor product?
    # No, the tensor for matrix multiplication $M_{n,m,p}$ has entries $T_{i,l,k} = \delta_{i,i}\delta_{l,l}\delta_{k,k}$?
    # The tensor is defined by $C_{i,k} = \sum_l A_{i,l} B_{l,k}$.
    # The structure tensor has entries $T_{i,l,k} = 1$ for all $i,l,k$?
    # No, the entries are $T_{i,l,k}$ such that $\langle T, A \otimes B \rangle = \langle C, A B \rangle$.
    # $T_{i,l,k}$ is 1 if the indices match the flow.
    # In the standard basis, $T_{i,l,k} = 1$ for all valid indices?
    # Let's check: $A_{i,l} B_{l,k} C_{k,i}^*$.
    # Yes, the tensor is the identity map on the space of matrices?
    # For $2 \times 2 \times 2$, the tensor has 8 ones.
    # Flattening over i (size 2):
    # Row i=0: T[0,l,k] for l=0,1 k=0,1 -> 1,1,1,1
    # Row i=1: T[1,l,k] for l=0,1 k=0,1 -> 1,1,1,1
    # Rank is 1.
    # So flattening rank is 1. This is a very weak bound.
    
    # We proceed with Strassen search.
    
    configurations = [
        ([(0,1),(2,3)], [(0,1),(2,3),(4,5)], "standard"),
        ([(0,1),(2,3)], [(0,2),(1,4),(3,5)], "alt_k1"),
        ([(0,1),(2,3)], [(0,3),(1,4),(2,5)], "alt_k2"),
        ([(0,2),(1,3)], [(0,1),(2,3),(4,5)], "alt_j1"),
        ([(0,3),(1,2)], [(0,1),(2,3),(4,5)], "alt_j2"),
        ([(0,1),(2,4)], [(0,1),(2,3),(4,5)], "alt_j3"), # j=3 left
        ([(0,4),(1,2)], [(0,1),(2,3),(4,5)], "alt_j4"), # j=3 left
        ([(0,1),(2,3)], [(0,4),(1,5),(2,3)], "alt_k3"),
    ]
    
    for j_pairs, k_pairs, desc in configurations:
        if _time_left() < 5:
            break
        
        U, V, W = _build_strassen_decomp(n, m, p, j_pairs, k_pairs)
        rank = len(U[0])
        
        if not _verify_decomp(U, V, W, n, m, p, rank):
            continue
        
        if rank < best_rank:
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}
        
        # Try to eliminate redundant terms
        current_U, current_V, current_W, current_rank = U, V, W, rank
        # Sort terms by norm to try eliminating small ones first
        norms = []
        for r in range(current_rank):
            norm_sq = sum(current_U[i][r]**2 for i in range(n*m))
            norm_sq += sum(current_V[l][r]**2 for l in range(m*p))
            norm_sq += sum(current_W[k][r]**2 for k in range(p*n))
            norms.append(norm_sq)
        
        order = np.argsort(norms)
        
        eliminated = set()
        for idx in order:
            if _time_left() < 2:
                break
            if idx in eliminated:
                continue
            
            success, new_U, new_V, new_W, new_rank = _try_eliminate_term_fast(
                current_U, current_V, current_W, n, m, p, current_rank, idx
            )
            
            if success:
                current_U, current_V, current_W, current_rank = new_U, new_V, new_W, new_rank
                eliminated.add(idx)
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_result = {"U": current_U, "V": current_W, "W": current_W} # Typo fix in next line
                    best_result = {"U": current_U, "V": current_V, "W": current_W}
            else:
                # If elimination fails, we might still be able to eliminate it later
                # if we eliminate other terms first? 
                # Usually elimination is monotonic in this context? 
                # If term r is not redundant, adding more terms doesn't make it redundant.
                # Removing other terms might make it redundant? No, if it's not covered by others,
                # removing others makes it MORE necessary.
                # So if it fails now, it will fail later.
                pass
    
    if best_result is not None:
        best_result["U"] = _snap_to_half_integer(best_result["U"])
        best_result["V"] = _snap_to_half_integer(best_result["V"])
        best_result["W"] = _snap_to_half_integer(best_result["W"])
        
        if not _verify_decomp(best_result["U"], best_result["V"], best_result["W"], n, m, p, best_rank):
            U, V, W = _build_strassen_decomp(n, m, p, [(0,1),(2,3)], [(0,1),(2,3),(4,5)])
            best_result = {"U": U, "V": V, "W": W}
            best_rank = len(U[0])
    
    return best_result
# EVOLVE-BLOCK-END
