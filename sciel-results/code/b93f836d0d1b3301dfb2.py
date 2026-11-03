"""Matrix multiplication 3x3x3: rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n = m = p = 3
    U = [[float(x) for x in row] for row in U]
    V = [[float(x) for x in row] for row in V]
    W = [[float(x) for x in row] for row in W]
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
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

def _try_reduce_rank(base_result, target_rank, deadline):
    """
    Attempt to reduce the rank of a valid decomposition by merging terms.
    Uses the base decomposition as a starting point and tries linear
    combinations of terms that preserve the identity property.
    """
    n = m = p = 3
    U = np.array(base_result["U"], dtype=float)
    V = np.array(base_result["V"], dtype=float)
    W = np.array(base_result["W"], dtype=float)
    cur_rank = U.shape[1]
    
    if cur_rank <= target_rank:
        return base_result
    
    # Build the 27 constraint equations.
    # For each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # We'll try to find a rank-`target_rank` sub-decomposition.
    
    best = None
    best_rank = cur_rank
    
    # Try to remove one term at a time (greedy)
    removed = set()
    for _ in range(cur_rank - target_rank):
        if time.time() > deadline:
            break
        found = False
        for a in range(cur_rank):
            if a in removed:
                continue
            # Try removing term a: check if the 27 equations
            # can be satisfied using only the other terms.
            remaining = [x for x in range(cur_rank) if x != a and x not in removed]
            if len(remaining) < target_rank:
                continue
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in remaining:
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                removed.add(a)
                found = True
                break
        if not found:
            break
    
    if len(removed) > 0:
        remaining = [x for x in range(cur_rank) if x not in removed]
        new_rank = len(remaining)
        if new_rank < best_rank:
            best_rank = new_rank
            best = {
                "U": U[:, remaining].tolist(),
                "V": V[:, remaining].tolist(),
                "W": W[:, remaining].tolist()
            }
    
    return best if best is not None else base_result

def construct():
    """Construct a tensor decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Try rank-26 construction (Strassen core + schoolbook border)
    result = _strassen_rank26()
    if _verify(result["U"], result["V"], result["W"]):
        # Try to reduce rank further
        if time.time() < deadline:
            reduced = _try_reduce_rank(result, 25, deadline)
            if reduced is not None and _verify(reduced["U"], reduced["V"], reduced["W"]):
                result = reduced
        
        if time.time() < deadline:
            return result
    
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
