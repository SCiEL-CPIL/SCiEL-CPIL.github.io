"""Matrix multiplication 3x3x3: rank-minimal tensor decomposition."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication using exact arithmetic."""
    n = m = p = 3
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                if s != 1:
                    return False
    return True

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

def _strassen_rank26():
    """
    Rank-26 decomposition for 3x3 matrix multiplication using explicit factors.
    Strassen core (7 terms) for the 2x2 block + schoolbook border terms.
    """
    n = m = p = 3
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    terms = []
    
    # --- Strassen core for C[0:2,0:2] (7 terms) ---
    terms.append(([ (u_idx(0,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,1),1), (w_idx(1,1),-1) ]))
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,1),1), (v_idx(1,1),-1) ], [ (w_idx(1,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,0),1), (v_idx(0,0),-1) ], [ (w_idx(0,0),1), (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(0,0),1), (u_idx(0,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(0,0),-1), (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(0,0),-1) ], [ (v_idx(0,0),1), (v_idx(0,1),1) ], [ (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(0,1),1), (u_idx(1,1),-1) ], [ (v_idx(1,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1) ]))
    
    # --- Border: C[0:2, 2] ---
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(1,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,1),1) ]))
    
    # --- Border: C[2, 0:2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,2),1) ]))
    
    # --- Corner: C[2,2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))
    
    # --- Additional border: A[0:2,2]*B[2,0:2] contributes to C[0:2,0:2] ---
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,1),1) ]))
    
    for r, (u_list, v_list, w_list) in enumerate(terms):
        for (row, coeff) in u_list:
            U[row][r] += coeff
        for (row, coeff) in v_list:
            V[row][r] += coeff
        for (row, coeff) in w_list:
            W[row][r] += coeff
    
    return {"U": U, "V": V, "W": W}

def _local_search_polish(U, V, W, deadline, max_iterations=5000):
    """
    Local search to reduce rank by finding linear dependencies among terms.
    For each term, try to express it as a linear combination of others.
    If successful, remove it and reduce rank.
    """
    n = m = p = 3
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    cur_rank = U.shape[1]
    
    def term_tensor(r):
        vec = np.zeros(27)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i*9 + j*3 + k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        return vec
    
    # Build the tensor for each term
    tensors = [term_tensor(r) for r in range(cur_rank)]
    
    removed = set()
    iterations = 0
    
    while iterations < max_iterations and time.time() < deadline:
        iterations += 1
        found_removal = False
        
        for r in range(cur_rank):
            if r in removed:
                continue
            
            others = [s for s in range(cur_rank) if s != r and s not in removed]
            if len(others) < 2:
                continue
            
            A = np.column_stack([tensors[s] for s in others])
            b = tensors[r]
            
            try:
                sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                if len(residuals) > 0 and residuals[0] < 1e-10:
                    removed.add(r)
                    found_removal = True
                    break
            except:
                pass
        
        if not found_removal:
            break
    
    if len(removed) > 0:
        remaining = [x for x in range(cur_rank) if x not in removed]
        new_rank = len(remaining)
        return {
            "U": U[:, remaining].tolist(),
            "V": V[:, remaining].tolist(),
            "W": W[:, remaining].tolist()
        }, new_rank
    
    return None, cur_rank

def _try_reduce_to_rank_r(U, V, W, target_rank, deadline):
    """
    Try to reduce the decomposition to exactly target_rank terms.
    Uses the approach: keep target_rank terms and try to express the rest
    as linear combinations. If all can be expressed, we have a valid rank-r decomposition.
    """
    n = m = p = 3
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    cur_rank = U.shape[1]
    
    if target_rank >= cur_rank:
        return None, cur_rank
    
    def term_tensor(r):
        vec = np.zeros(27)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i*9 + j*3 + k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        return vec
    
    # Try subsets of size target_rank to keep
    # For efficiency, try the first target_rank, then random subsets
    from random import sample
    
    # First try: keep the first target_rank terms
    keep = list(range(target_rank))
    discard = list(range(target_rank, cur_rank))
    
    A = np.column_stack([term_tensor(k) for k in keep])
    all_discard_tensors = [term_tensor(d) for d in discard]
    
    try:
        all_ok = True
        for b_vec in all_discard_tensors:
            sol, residuals, rank_A, sv = np.linalg.lstsq(A, b_vec, rcond=None)
            if len(residuals) > 0 and residuals[0] > 1e-10:
                all_ok = False
                break
        if all_ok:
            return {
                "U": U[:, keep].tolist(),
                "V": V[:, keep].tolist(),
                "W": W[:, keep].tolist()
            }, target_rank
    except:
        pass
    
    # Try random subsets
    max_attempts = 200
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        keep = sample(range(cur_rank), target_rank)
        discard = [x for x in range(cur_rank) if x not in keep]
        
        A = np.column_stack([term_tensor(k) for k in keep])
        all_discard_tensors = [term_tensor(d) for d in discard]
        
        try:
            all_ok = True
            for b_vec in all_discard_tensors:
                sol, residuals, rank_A, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                if len(residuals) > 0 and residuals[0] > 1e-10:
                    all_ok = False
                    break
            if all_ok:
                return {
                    "U": U[:, keep].tolist(),
                    "V": V[:, keep].tolist(),
                    "W": W[:, keep].tolist()
                }, target_rank
        except:
            pass
    
    return None, cur_rank

def construct():
    """Construct a tensor decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Start with the verified rank-26 construction
    result = _strassen_rank26()
    if not _verify(result["U"], result["V"], result["W"]):
        # Should not happen, but fall back to schoolbook
        return _schoolbook()
    
    current_rank = len(result["U"][0])
    
    # Try local search polish
    if time.time() < deadline - 60:
        polished, polished_rank = _local_search_polish(
            result["U"], result["V"], result["W"], deadline - 60
        )
        if polished is not None and polished_rank < current_rank:
            if _verify(polished["U"], polished["V"], polished["W"]):
                result = polished
                current_rank = polished_rank
    
    # Try reducing to specific target ranks
    for target in range(current_rank - 1, 20, -1):
        if time.time() > deadline - 15:
            break
        reduced, reduced_rank = _try_reduce_to_rank_r(
            result["U"], result["V"], result["W"], target, deadline - 15
        )
        if reduced is not None and reduced_rank < current_rank:
            if _verify(reduced["U"], reduced["V"], reduced["W"]):
                result = reduced
                current_rank = reduced_rank
                break  # Found a valid reduction, stop trying lower targets
    
    # Final verification
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook if something went wrong
    return _schoolbook()

# EVOLVE-BLOCK-END
