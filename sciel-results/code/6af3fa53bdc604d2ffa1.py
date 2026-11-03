"""Rewritten program: structured sparse linear algebra / binary IP for exact tensor decomposition."""
DIMENSIONS = [3, 3, 8]
import random
random.seed(1)


# EVOLVE-BLOCK-START
import numpy as np

def _verify(U, V, W, n, m, p):
    """Check that sum_r U[a,r]*V[b,r]*W[c,r] equals 1 iff (a,b,c)=(i*m+j, j*p+k, k*n+i)."""
    nm, mp, pn = n*m, m*p, p*n
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)
    # Compute tensor via einsum
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    # Check all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                if T[a,b,c] != 1:
                    return False
    # Check all non-target entries are 0
    # Build a set of target indices
    targets = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                targets.add((i*m+j, j*p+k, k*n+i))
    # Iterate over all (a,b,c) not in targets
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a,b,c) not in targets:
                    if T[a,b,c] != 0:
                        return False
    return True

def _generate_candidates(n, m, p):
    """
    Generate a dictionary of candidate rank-1 terms (u, v, w) with integer entries in {-1,0,1}.
    We include:
      1. Schoolbook terms: u has 1 at (i,j), v has 1 at (j,k), w has 1 at (k,i).
      2. Strassen-like terms for 3x3x3 blocks: use known rank-23 patterns.
      3. Additional sparse patterns to allow mixing.
    Each candidate is a tuple (u_list, v_list, w_list) where each is a list of length nm, mp, pn respectively.
    """
    nm, mp, pn = n*m, m*p, p*n
    candidates = []
    
    # 1. Schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0]*nm; v = [0]*mp; w = [0]*pn
                u[i*m+j] = 1
                v[j*p+k] = 1
                w[k*n+i] = 1
                candidates.append((u, v, w))
    
    # 2. Strassen-like patterns for 3x3x3
    # We'll generate a few known algebraic identities for 3x3 matrix multiplication.
    # For 3x3x3, the tensor has 27 target entries. Strassen gives rank 23.
    # We'll add some non-trivial rank-1 terms that help reduce rank.
    # Here, we add terms that are sums of schoolbook terms with signs.
    # For example, for a 2x2 block, Strassen uses (a11+a22), (a12+a21), etc.
    # We'll generate terms for all 2x2 subblocks of the 3x3 matrix.
    
    # Generate Strassen-like terms for all 2x2 subblocks
    for i1 in range(n-1):
        for j1 in range(m-1):
            # 2x2 block starting at (i1, j1)
            i2 = i1+1
            j2 = j1+1
            for k in range(p):
                # For each column k, add Strassen-like terms for the 2x2 block
                # The 2x2 Strassen decomposition has 7 terms.
                # We'll add these as candidate rank-1 terms.
                
                # Term 1: (a_i1_j1 + a_i2_j2) * (b_j1_k + b_j2_k) -> contributes to (i1,k) and (i2,k)
                u = [0]*nm; v = [0]*mp; w = [0]*pn
                u[i1*m+j1] = 1
                u[i2*m+j2] = 1
                v[j1*p+k] = 1
                v[j2*p+k] = 1
                w[k*n+i1] = 1
                w[k*n+i2] = 1
                candidates.append((u, v, w))
                
                # Term 2: (a_i1_j1 + a_i2_j2) * (b_j1_k - b_j2_k) -> contributes to (i1,k) and (i2,k) with signs
                u = [0]*nm; v = [0]*mp; w = [0]*pn
                u[i1*m+j1] = 1
                u[i2*m+j2] = 1
                v[j1*p+k] = 1
                v[j2*p+k] = -1
                w[k*n+i1] = 1
                w[k*n+i2] = 1
                candidates.append((u, v, w))
                
                # Term 3: (a_i1_j1 - a_i2_j2) * (b_j1_k + b_j2_k)
                u = [0]*nm; v = [0]*mp; w = [0]*pn
                u[i1*m+j1] = 1
                u[i2*m+j2] = -1
                v[j1*p+k] = 1
                v[j2*p+k] = 1
                w[k*n+i1] = 1
                w[k*n+i2] = 1
                candidates.append((u, v, w))
                
                # Term 4: (a_i1_j1 - a_i2_j2) * (b_j1_k - b_j2_k)
                u = [0]*nm; v = [0]*mp; w = [0]*pn
                u[i1*m+j1] = 1
                u[i2*m+j2] = -1
                v[j1*p+k] = 1
                v[j2*p+k] = -1
                w[k*n+i1] = 1
                w[k*n+i2] = 1
                candidates.append((u, v, w))
                
                # Term 5: a_i1_j1 * b_j1_k
                u = [0]*nm; v = [0]*mp; w = [0]*pn
                u[i1*m+j1] = 1
                v[j1*p+k] = 1
                w[k*n+i1] = 1
                candidates.append((u, v, w))
                
                # Term 6: a_i2_j1 * b_j1_k
                u = [0]*nm; v = [0]*mp; w = [0]*pn
                u[i2*m+j1] = 1
                v[j1*p+k] = 1
                w[k*n+i2] = 1
                candidates.append((u, v, w))
                
                # Term 7: a_i1_j2 * b_j2_k
                u = [0]*nm; v = [0]*mp; w = [0]*pn
                u[i1*m+j2] = 1
                v[j2*p+k] = 1
                w[k*n+i1] = 1
                candidates.append((u, v, w))
                
                # Term 8: a_i2_j2 * b_j2_k
                u = [0]*nm; v = [0]*mp; w = [0]*pn
                u[i2*m+j2] = 1
                v[j2*p+k] = 1
                w[k*n+i2] = 1
                candidates.append((u, v, w))
    
    # 3. Additional sparse patterns: try all combinations of 2 non-zero entries in u, v, w
    # This is too many, so we'll add a few random ones
    np.random.seed(42)
    for _ in range(500):
        u = [0]*nm; v = [0]*mp; w = [0]*pn
        # Pick 1-2 non-zero entries in each
        for _ in range(np.random.randint(1,3)):
            idx = np.random.randint(0, nm)
            u[idx] = np.random.choice([-1,1])
        for _ in range(np.random.randint(1,3)):
            idx = np.random.randint(0, mp)
            v[idx] = np.random.choice([-1,1])
        for _ in range(np.random.randint(1,3)):
            idx = np.random.randint(0, pn)
            w[idx] = np.random.choice([-1,1])
        candidates.append((u, v, w))
    
    return candidates

def _greedy_select(candidates, n, m, p, max_rank=60):
    """
    Greedy selection of a minimal subset of candidates that exactly satisfies the tensor equations.
    We start with an empty set and iteratively add the candidate that best reduces the residual.
    """
    nm, mp, pn = n*m, m*p, p*n
    
    # Build target tensor
    T_target = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1
    
    # Initialize residual
    residual = T_target.copy()
    
    selected = []
    used = set()
    
    for _ in range(max_rank):
        best_idx = -1
        best_score = float('inf')
        best_residual = None
        
        for idx, (u, v, w) in enumerate(candidates):
            if idx in used:
                continue
            u_arr = np.array(u, dtype=np.int64)
            v_arr = np.array(v, dtype=np.int64)
            w_arr = np.array(w, dtype=np.int64)
            term = np.einsum('a,b,c->abc', u_arr, v_arr, w_arr)
            new_residual = residual - term
            
            # Score: number of non-zero entries in new_residual
            score = np.count_nonzero(new_residual)
            
            if score < best_score:
                best_score = score
                best_idx = idx
                best_residual = new_residual
        
        if best_idx == -1:
            break
        
        selected.append(best_idx)
        used.add(best_idx)
        residual = best_residual
        
        if best_score == 0:
            break
    
    if best_score != 0:
        return None
    
    # Build U, V, W
    rank = len(selected)
    U = [[0]*rank for _ in range(nm)]
    V = [[0]*rank for _ in range(mp)]
    W = [[0]*rank for _ in range(pn)]
    
    for col, idx in enumerate(selected):
        u, v, w = candidates[idx]
        for a in range(nm):
            U[a][col] = u[a]
        for b in range(mp):
            V[b][col] = v[b]
        for c in range(pn):
            W[c][col] = w[c]
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = 3, 3, 8
    nm, mp, pn = n*m, m*p, p*n
    
    # Generate candidates
    candidates = _generate_candidates(n, m, p)
    
    # Try greedy selection with various max_rank values
    for max_rank in [46, 48, 50, 52, 54, 56, 58, 60]:
        result = _greedy_select(candidates, n, m, p, max_rank=max_rank)
        if result is not None:
            # Verify
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return result
    
    # Fallback: schoolbook rank 72
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
