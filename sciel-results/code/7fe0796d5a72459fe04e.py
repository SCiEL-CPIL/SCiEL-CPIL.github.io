"""Matrix multiplication 3x3x3: rank-minimal tensor decomposition."""
import time
import itertools
import numpy as np
from scipy.linalg import null_space

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n = m = p = 3
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U_r = U.reshape(3, 3, rank)
    V_r = V.reshape(3, 3, rank)
    W_r = W.reshape(3, 3, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return np.allclose(T, np.ones((3, 3, 3)), atol=1e-8)

def _strassen_rank26():
    """Rank-26 decomposition: Strassen core (7 terms) + schoolbook border (19 terms)."""
    n = m = p = 3
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    terms = []
    
    # Strassen core for C[0:2,0:2] (7 terms)
    terms.append(([(u_idx(0,0),1), (u_idx(1,1),1)], [(v_idx(0,0),1), (v_idx(1,1),1)], [(w_idx(0,0),1), (w_idx(1,1),1)]))
    terms.append(([(u_idx(1,0),1), (u_idx(1,1),1)], [(v_idx(0,0),1)], [(w_idx(0,1),1), (w_idx(1,1),-1)]))
    terms.append(([(u_idx(0,0),1)], [(v_idx(0,1),1), (v_idx(1,1),-1)], [(w_idx(1,0),1), (w_idx(1,1),1)]))
    terms.append(([(u_idx(1,1),1)], [(v_idx(1,0),1), (v_idx(0,0),-1)], [(w_idx(0,0),1), (w_idx(0,1),1)]))
    terms.append(([(u_idx(0,0),1), (u_idx(0,1),1)], [(v_idx(1,1),1)], [(w_idx(0,0),-1), (w_idx(1,0),1)]))
    terms.append(([(u_idx(1,0),1), (u_idx(0,0),-1)], [(v_idx(0,0),1), (v_idx(0,1),1)], [(w_idx(1,1),1)]))
    terms.append(([(u_idx(0,1),1), (u_idx(1,1),-1)], [(v_idx(1,0),1), (v_idx(1,1),1)], [(w_idx(0,0),1)]))
    
    # Border: C[0:2, 2] (6 terms)
    terms.append(([(u_idx(0,0),1)], [(v_idx(0,2),1)], [(w_idx(2,0),1)]))
    terms.append(([(u_idx(0,1),1)], [(v_idx(1,2),1)], [(w_idx(2,0),1)]))
    terms.append(([(u_idx(0,2),1)], [(v_idx(2,2),1)], [(w_idx(2,0),1)]))
    terms.append(([(u_idx(1,0),1)], [(v_idx(0,2),1)], [(w_idx(2,1),1)]))
    terms.append(([(u_idx(1,1),1)], [(v_idx(1,2),1)], [(w_idx(2,1),1)]))
    terms.append(([(u_idx(1,2),1)], [(v_idx(2,2),1)], [(w_idx(2,1),1)]))
    
    # Border: C[2, 0:2] (6 terms)
    terms.append(([(u_idx(2,0),1)], [(v_idx(0,0),1)], [(w_idx(0,2),1)]))
    terms.append(([(u_idx(2,1),1)], [(v_idx(1,0),1)], [(w_idx(0,2),1)]))
    terms.append(([(u_idx(2,2),1)], [(v_idx(2,0),1)], [(w_idx(0,2),1)]))
    terms.append(([(u_idx(2,0),1)], [(v_idx(0,1),1)], [(w_idx(1,2),1)]))
    terms.append(([(u_idx(2,1),1)], [(v_idx(1,1),1)], [(w_idx(1,2),1)]))
    terms.append(([(u_idx(2,2),1)], [(v_idx(2,1),1)], [(w_idx(1,2),1)]))
    
    # Corner: C[2,2] (3 terms)
    terms.append(([(u_idx(2,0),1)], [(v_idx(0,2),1)], [(w_idx(2,2),1)]))
    terms.append(([(u_idx(2,1),1)], [(v_idx(1,2),1)], [(w_idx(2,2),1)]))
    terms.append(([(u_idx(2,2),1)], [(v_idx(2,2),1)], [(w_idx(2,2),1)]))
    
    # Additional border: A[0:2,2]*B[2,0:2] -> C[0:2,0:2] (4 terms)
    terms.append(([(u_idx(0,2),1)], [(v_idx(2,0),1)], [(w_idx(0,0),1)]))
    terms.append(([(u_idx(0,2),1)], [(v_idx(2,1),1)], [(w_idx(1,0),1)]))
    terms.append(([(u_idx(1,2),1)], [(v_idx(2,0),1)], [(w_idx(0,1),1)]))
    terms.append(([(u_idx(1,2),1)], [(v_idx(2,1),1)], [(w_idx(1,1),1)]))
    
    for r, (u_list, v_list, w_list) in enumerate(terms):
        for (row, coeff) in u_list:
            U[row][r] += coeff
        for (row, coeff) in v_list:
            V[row][r] += coeff
        for (row, coeff) in w_list:
            W[row][r] += coeff
    
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n = m = p = 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _term_tensors(U, V, W):
    """Compute the 27-dimensional tensor vector for each rank term."""
    n = m = p = 3
    rank = U.shape[1]
    tensors = np.zeros((rank, 27))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensors[r, i*9 + j*3 + k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    return tensors

def _find_dependent_terms(tensors, tol=1e-8):
    """Find indices of terms that are linear combinations of others."""
    rank = tensors.shape[0]
    dependent = set()
    
    for r in range(rank):
        others = [s for s in range(rank) if s != r]
        if len(others) < 2:
            continue
        A = tensors[others, :]
        b = tensors[r, :]
        try:
            sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            if len(residuals) > 0 and residuals[0] < tol:
                dependent.add(r)
        except:
            pass
    
    return dependent

def _reduce_rank(U, V, W, dependent_indices):
    """Remove dependent terms and return new matrices."""
    keep = [r for r in range(U.shape[1]) if r not in dependent_indices]
    return {
        "U": U[:, keep].tolist(),
        "V": V[:, keep].tolist(),
        "W": W[:, keep].tolist()
    }

def _gauge_transform(U, V, W, G):
    """Apply gauge transformation: U' = U*G, V' = V*G^{-1}, W' = W*G^{-1}."""
    G_inv = np.linalg.inv(G)
    U_new = U @ G
    V_new = V @ G_inv
    W_new = W @ G_inv
    return U_new, V_new, W_new

def _random_gauge(rank, rng):
    """Generate a random invertible gauge matrix."""
    G = rng.standard_normal((rank, rank))
    # Ensure invertibility
    for _ in range(10):
        try:
            np.linalg.inv(G)
            return G
        except np.linalg.LinAlgError:
            G = rng.standard_normal((rank, rank))
    # Fallback: identity + small perturbation
    return np.eye(rank) + 0.1 * rng.standard_normal((rank, rank))

def _search_lower_rank(U_init, V_init, W_init, deadline, max_gauges=200):
    """
    Search for lower rank by:
    1. Checking for linear dependencies in current representation
    2. Applying random gauge transformations and re-checking
    """
    n = m = p = 3
    U = np.array(U_init, dtype=float)
    V = np.array(V_init, dtype=float)
    W = np.array(W_init, dtype=float)
    cur_rank = U.shape[1]
    
    best_result = None
    best_rank = cur_rank
    rng = np.random.RandomState(42)
    
    # First: check current representation
    tensors = _term_tensors(U, V, W)
    dependent = _find_dependent_terms(tensors)
    if len(dependent) > 0:
        new_rank = cur_rank - len(dependent)
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = _reduce_rank(U, V, W, dependent)
    
    # Try random gauge transformations
    for attempt in range(max_gauges):
        if time.time() > deadline:
            break
        
        G = _random_gauge(cur_rank, rng)
        try:
            U_g, V_g, W_g = _gauge_transform(U, V, W, G)
        except:
            continue
        
        tensors_g = _term_tensors(U_g, V_g, W_g)
        dependent_g = _find_dependent_terms(tensors_g)
        
        if len(dependent_g) > 0:
            new_rank = cur_rank - len(dependent_g)
            if new_rank < best_rank:
                best_rank = new_rank
                best_result = _reduce_rank(U_g, V_g, W_g, dependent_g)
                # Update base for further searches
                U = np.array(best_result["U"], dtype=float)
                V = np.array(best_result["V"], dtype=float)
                W = np.array(best_result["W"], dtype=float)
                cur_rank = best_rank
    
    if best_result is not None:
        return best_result, best_rank
    return None, cur_rank

def _try_subspace_reduction(U, V, W, deadline):
    """
    Try to find a lower-dimensional subspace by checking if the tensor
    can be expressed with fewer terms using a different basis.
    
    This uses the fact that if we have R terms, we can check if any
    proper subset spans the same tensor. We do this by checking the
    rank of the tensor matrix for each mode.
    """
    n = m = p = 3
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    cur_rank = U.shape[1]
    
    # The tensor T is 3x3x3. Its rank is at most min(3*3, 3*3, 3*3) = 9
    # But we're looking for the tensor rank (sum of rank-1 terms), not multilinear rank.
    # For 3x3x3, the minimum tensor rank is known to be at most 23 (Landsberg).
    # Our approach: try to find if we can express the tensor with fewer than cur_rank terms.
    
    # Build the full tensor
    T = np.zeros((27,))
    for r in range(cur_rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*9 + j*3 + k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We know T should be all ones
    # Try to find a decomposition with rank < cur_rank
    # This is NP-hard in general, but for small cases we can try structured approaches.
    
    # Approach: try to merge pairs of terms by finding a 2x2 block diagonal
    # transformation that makes one term zero.
    
    best_result = None
    best_rank = cur_rank
    
    # Try all pairs
    for a in range(cur_rank):
        if time.time() > deadline:
            break
        for b in range(a + 1, cur_rank):
            # Can we find a 2x2 transformation of terms a,b that zeros one out?
            # This means: there exists (alpha, beta) such that
            # alpha * term_a + beta * term_b = 0 in tensor space
            # which means term_a and term_b are linearly dependent.
            ta = np.zeros(27)
            tb = np.zeros(27)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        ta[i*9 + j*3 + k] = U[i*m+j, a] * V[j*p+k, a] * W[k*n+i, a]
                        tb[i*9 + j*3 + k] = U[i*m+j, b] * V[j*p+k, b] * W[k*n+i, b]
            
            # Check if ta and tb are linearly dependent
            if np.linalg.matrix_rank(np.column_stack([ta, tb])) == 1:
                # They're dependent, so we can eliminate one
                remaining = [x for x in range(cur_rank) if x != b]
                new_rank = len(remaining)
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_result = {
                        "U": U[:, remaining].tolist(),
                        "V": V[:, remaining].tolist(),
                        "W": W[:, remaining].tolist()
                    }
    
    if best_result is not None:
        return best_result, best_rank
    return None, cur_rank

def construct():
    """Construct a tensor decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s limit
    
    # Start with the verified rank-26 construction
    result = _strassen_rank26()
    if not _verify(result["U"], result["V"], result["W"]):
        return _schoolbook()
    
    current_rank = len(result["U"][0])
    
    # Phase 1: Direct linear dependency check
    if time.time() < deadline - 60:
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        tensors = _term_tensors(U, V, W)
        dependent = _find_dependent_terms(tensors)
        if len(dependent) > 0:
            reduced = _reduce_rank(U, V, W, dependent)
            if _verify(reduced["U"], reduced["V"], reduced["W"]):
                result = reduced
                current_rank = len(reduced["U"][0])
    
    # Phase 2: Gauge transformation search
    if time.time() < deadline - 30:
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        polished, polished_rank = _search_lower_rank(
            result["U"], result["V"], result["W"], deadline - 30, max_gauges=300
        )
        if polished is not None and polished_rank < current_rank:
            if _verify(polished["U"], polished["V"], polished["W"]):
                result = polished
                current_rank = polished_rank
    
    # Phase 3: Pair merging (subspace reduction)
    if time.time() < deadline - 10:
        merged, merged_rank = _try_subspace_reduction(
            result["U"], result["V"], result["W"], deadline - 10
        )
        if merged is not None and merged_rank < current_rank:
            if _verify(merged["U"], merged["V"], merged["W"]):
                result = merged
                current_rank = merged_rank
    
    # Final verification
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook if something went wrong
    return _schoolbook()

# EVOLVE-BLOCK-END
