"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product, combinations_with_replacement, permutations

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # We need T[i*m+l, l*p+k, k*n+i] = 1 for all i,l,k
    # This means: sum_r U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r] == 1 for all i,l,k
    for i in range(n):
        for l in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+l, :], V_arr[l*p+k, :] * W_arr[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_standard_decomp(n, m, p):
    """Build the standard (trivial) matrix multiplication decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for l in range(m):
            for k in range(p):
                U[i*m+l][col] = 1
                V[l*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _build_strassen_decomp(n, m, p, j_pairs, k_pairs):
    """Build a Strassen-based decomposition with given pairings.
    
    For each pair of j indices (j0, j1) and pair of k indices (k0, k1),
    we use 7 Strassen terms to compute all 4 output entries (i=0,1) x (k=k0,k1).
    For remaining j indices, we use the standard 1 term per (j,k) pair.
    """
    U = [[0] * (7 * len(j_pairs) * len(k_pairs) + 2 * (m - 2*len(j_pairs)) * p) for _ in range(n * m)]
    V = [[0] * (7 * len(j_pairs) * len(k_pairs) + 2 * (m - 2*len(j_pairs)) * p) for _ in range(m * p)]
    W = [[0] * (7 * len(j_pairs) * len(k_pairs) + 2 * (m - 2*len(j_pairs)) * p) for _ in range(p * n)]
    
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
    
    # Trim to actual rank used
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return U, V, W

def _try_eliminate_term(U, V, W, n, m, p, rank, term_idx):
    """Try to eliminate term term_idx by checking if it's redundant."""
    # Check if the decomposition still works without this term
    U_new = [row[:] for row in U]
    V_new = [row[:] for row in V]
    W_new = [row[:] for row in W]
    
    for i in range(n*m):
        U_new[i][term_idx] = 0
    for l in range(m*p):
        V_new[l][term_idx] = 0
    for k in range(p*n):
        W_new[k][term_idx] = 0
    
    if _verify_decomp(U_new, V_new, W_new, n, m, p, rank):
        # Remove the column
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

def _build_grouped_decomp(n, m, p, j_pairs, k_pairs):
    """Build decomposition where leftover j's use grouped k-pairs (rank 2 per group)."""
    used_j = set()
    for j0, j1 in j_pairs:
        used_j.add(j0)
        used_j.add(j1)
    leftover_js = [j for j in range(m) if j not in used_j]
    
    # Strassen part: 7 terms per (j_pair, k_pair)
    # Leftover part: for each leftover j, for each k_pair, use 2 terms
    #   Term A: U[i,j] = 1 for both i, V[j,k0]=1, W[k0,i]=1 for both i -> covers i=0 and i=1 for k0
    #   Wait, that's not quite right. Let me think again.
    #   For a single j with 2 k's (k0, k1):
    #   C[0,k0] = A[0,j]*B[j,k0]
    #   C[1,k0] = A[1,j]*B[j,k0]  
    #   C[0,k1] = A[0,j]*B[j,k1]
    #   C[1,k1] = A[1,j]*B[j,k1]
    #   We can do this in 2 terms:
    #   r1: U[0,j,r1]=1, U[1,j,r1]=0, V[j,k0,r1]=1, V[j,k1,r1]=0, W[k0,0,r1]=1, W[k0,1,r1]=0, W[k1,0,r1]=0, W[k1,1,r1]=0
    #   That only covers C[0,k0]. 
    #   Actually for R(2,1,2): we need 4 output entries.
    #   r1: U[0,j]=1, U[1,j]=0, V[j,k0]=1, V[j,k1]=0, W[k0,0]=1, W[k0,1]=0, W[k1,0]=0, W[k1,1]=0
    #   r2: U[0,j]=0, U[1,j]=1, V[j,k0]=1, V[j,k1]=0, W[k0,0]=0, W[k0,1]=1, W[k1,0]=0, W[k1,1]=0
    #   r3: U[0,j]=1, U[1,j]=0, V[j,k0]=0, V[j,k1]=1, W[k0,0]=0, W[k0,1]=0, W[k1,0]=1, W[k1,1]=0
    #   r4: U[0,j]=0, U[1,j]=1, V[j,k0]=0, V[j,k1]=1, W[k0,0]=0, W[k0,1]=0, W[k1,0]=0, W[k1,1]=1
    #   That's 4 terms, same as standard.
    #   
    #   But wait! We can do better:
    #   r1: U[0,j]=1, U[1,j]=1, V[j,k0]=1, V[j,k1]=0, W[k0,0]=1, W[k0,1]=1, W[k1,0]=0, W[k1,1]=0
    #   This gives C[0,k0] += 1*1*1 = 1, C[1,k0] += 1*1*1 = 1. Good for k0.
    #   r2: U[0,j]=1, U[1,j]=1, V[j,k0]=0, V[j,k1]=1, W[k0,0]=0, W[k0,1]=0, W[k1,0]=1, W[k1,1]=1
    #   This gives C[0,k1] += 1*1*1 = 1, C[1,k1] += 1*1*1 = 1. Good for k1.
    #   So 2 terms cover all 4 entries! That's a saving.
    
    strassen_terms = 7 * len(j_pairs) * len(k_pairs)
    leftover_terms = 2 * len(leftover_js) * len(k_pairs)
    rank = strassen_terms + leftover_terms
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen blocks
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
    
    # Leftover j's with grouped k-pairs (2 terms per group)
    for j in leftover_js:
        for k0, k1 in k_pairs:
            # Term for k0: both i values
            U[0*m + j][col] += 1
            U[1*m + j][col] += 1
            V[j*p + k0][col] += 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            
            # Term for k1: both i values
            U[0*m + j][col] += 1
            U[1*m + j][col] += 1
            V[j*p + k1][col] += 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 130  # 20s safety margin
    
    def _time_left():
        return _deadline - time.time()
    
    best_rank = n * m * p  # standard rank = 60
    best_result = None
    
    # Generate all possible j-pairings and k-pairings
    all_js = list(range(m))
    all_ks = list(range(p))
    
    def _get_all_pairings(indices):
        """Generate all ways to partition indices into pairs (and possibly one singleton)."""
        if len(indices) == 0:
            yield []
            return
        if len(indices) == 1:
            yield []  # no pairs possible
            return
        if len(indices) == 2:
            yield [(indices[0], indices[1])]
            return
        # For 3+ indices, try all pairings
        first = indices[0]
        for other_idx in range(1, len(indices)):
            other = indices[other_idx]
            remaining = [x for x in indices[1:] if x != other]
            for sub_pairing in _get_all_pairings(remaining):
                yield [(first, other)] + sub_pairing
        # Also yield pairings that don't include the first element in a pair
        # (i.e., first is a singleton)
        for sub_pairing in _get_all_pairings(indices[1:]):
            yield sub_pairing
    
    # Try various configurations
    configurations = []
    
    # Standard Strassen: pair up j's and k's
    j_pairings = list(_get_all_pairings(all_js))
    k_pairings = list(_get_all_pairings(all_ks))
    
    # Limit to reasonable number of pairings to stay within time
    j_pairings = j_pairings[:50]
    k_pairings = k_pairings[:50]
    
    for j_pairs in j_pairings:
        if _time_left() < 10:
            break
        for k_pairs in k_pairings:
            if _time_left() < 5:
                break
            
            # Try standard Strassen
            try:
                U, V, W = _build_strassen_decomp(n, m, p, j_pairs, k_pairs)
                rank = len(U[0])
                if _verify_decomp(U, V, W, n, m, p, rank):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U, "V": V, "W": W}
            except Exception:
                pass
            
            # Try grouped Strassen (better handling of leftover j's)
            try:
                U, V, W = _build_grouped_decomp(n, m, p, j_pairs, k_pairs)
                rank = len(U[0])
                if _verify_decomp(U, V, W, n, m, p, rank):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U, "V": V, "W": W}
            except Exception:
                pass
    
    # Try term elimination on best result
    if best_result is not None:
        current_U = best_result["U"]
        current_V = best_result["V"]
        current_W = best_result["W"]
        current_rank = best_rank
        
        for attempt in range(current_rank):
            if _time_left() < 3:
                break
            
            # Try to eliminate each term, starting with those having smallest norm
            norms = []
            for r in range(current_rank):
                norm_sq = sum(current_U[i][r]**2 for i in range(n*m))
                norm_sq += sum(current_V[l][r]**2 for l in range(m*p))
                norm_sq += sum(current_W[k][r]**2 for k in range(p*n))
                norms.append(norm_sq)
            
            sorted_indices = sorted(range(current_rank), key=lambda r: norms[r])
            
            eliminated = False
            for r in sorted_indices:
                success, new_U, new_V, new_W, new_rank = _try_eliminate_term(
                    current_U, current_V, current_W, n, m, p, current_rank, r
                )
                if success:
                    current_U, current_V, current_W = new_U, new_V, new_W
                    current_rank = new_rank
                    if current_rank < best_rank:
                        best_rank = current_rank
                        best_result = {"U": current_U, "V": current_V, "W": current_W}
                    eliminated = True
                    break
            
            if not eliminated:
                break
    
    # Snap to half-integers for cleaner output
    if best_result is not None:
        best_result["U"] = _snap_to_half_integer(best_result["U"])
        best_result["V"] = _snap_to_half_integer(best_result["V"])
        best_result["W"] = _snap_to_half_integer(best_result["W"])
        
        # Final verification
        if not _verify_decomp(best_result["U"], best_result["V"], best_result["W"], n, m, p, best_rank):
            # Fallback to standard
            U, V, W = _build_standard_decomp(n, m, p)
            best_result = {"U": U, "V": V, "W": W}
            best_rank = len(U[0])
    
    return best_result
# EVOLVE-BLOCK-END
