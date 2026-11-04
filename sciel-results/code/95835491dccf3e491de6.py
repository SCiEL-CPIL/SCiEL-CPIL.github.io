import numpy as np
import time
import itertools

def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms as (u_coeffs, v_coeffs, w_coeffs) for 2x2x2."""
    # Each term is a list of (index_tuple, value) for U, V, W
    # U is indexed by (i, j) where i is row of A, j is col of A
    # V is indexed by (j, k) where j is row of B, k is col of B  
    # W is indexed by (k, i) where k is col of B (= row of C), i is row of C
    # Trace convention: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = C[i,k]
    # Wait, the task says W[k*n+i, r] so W is indexed by (k, i) where k is column of C and i is row of C
    # For 3x5x5: n=3, m=5, p=5
    # U shape: (n*m, rank) = (15, rank)
    # V shape: (m*p, rank) = (25, rank)
    # W shape: (p*n, rank) = (15, rank)
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For 2x2x2 sub-block at offset (i0, j0, k0):
    # U[i0+a, j0+b, r] for a in 0,1 and b in 0,1
    # V[j0+b, k0+c, r] for b in 0,1 and c in 0,1
    # W[k0+c, i0+a, r] for c in 0,1 and a in 0,1
    
    # Strassen for 2x2x2:
    # C = A * B where A is 2x2, B is 2x2
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    
    # Standard Strassen:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each term r, we need:
    # U coefficients: for each (a,b) in {0,1}^2, the coefficient of A[a,b]
    # V coefficients: for each (b,c) in {0,1}^2, the coefficient of B[b,c]  
    # W coefficients: for each (c,a) in {0,1}^2, the coefficient for C[a,c]
    
    terms = []
    
    # Term 1: M1 = (A11 + A22)(B11 + B22)
    # U: A11 coeff 1, A22 coeff 1 -> (0,0):1, (1,1):1
    # V: B11 coeff 1, B22 coeff 1 -> (0,0):1, (1,1):1
    # W: C11 coeff 1, C22 coeff 1 -> (0,0):1, (1,1):1
    terms.append(([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]))
    
    # Term 2: M2 = (A21 + A22)B11
    # U: A21 coeff 1, A22 coeff 1 -> (1,0):1, (1,1):1
    # V: B11 coeff 1 -> (0,0):1
    # W: C21 coeff 1, C22 coeff -1 -> (0,1):1, (1,1):-1
    terms.append(([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)]))
    
    # Term 3: M3 = A11(B12 - B22)
    # U: A11 coeff 1 -> (0,0):1
    # V: B12 coeff 1, B22 coeff -1 -> (0,1):1, (1,1):-1
    # W: C11 coeff 1, C12 coeff 1 -> (0,0):1, (1,0):1
    terms.append(([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,0,1), (1,0,1)]))
    
    # Term 4: M4 = A22(B21 - B11)
    # U: A22 coeff 1 -> (1,1):1
    # V: B21 coeff 1, B11 coeff -1 -> (1,0):1, (0,0):-1
    # W: C11 coeff -1, C21 coeff 1 -> (0,0):-1, (0,1):1
    terms.append(([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,-1), (0,1,1)]))
    
    # Term 5: M5 = (A11 + A12)B22
    # U: A11 coeff 1, A12 coeff 1 -> (0,0):1, (0,1):1
    # V: B22 coeff 1 -> (1,1):1
    # W: C11 coeff -1, C12 coeff 1 -> (0,0):-1, (1,0):1
    terms.append(([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (1,0,1)]))
    
    # Term 6: M6 = (A21 - A11)(B11 + B12)
    # U: A21 coeff 1, A11 coeff -1 -> (1,0):1, (0,0):-1
    # V: B11 coeff 1, B12 coeff 1 -> (0,0):1, (0,1):1
    # W: C22 coeff 1 -> (1,1):1
    terms.append(([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]))
    
    # Term 7: M7 = (A12 - A22)(B21 + B22)
    # U: A12 coeff 1, A22 coeff -1 -> (0,1):1, (1,1):-1
    # V: B21 coeff 1, B22 coeff 1 -> (1,0):1, (1,1):1
    # W: C11 coeff 1 -> (0,0):1
    terms.append(([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]))
    
    return terms

def _get_splits(d):
    """Generate all compositions of d into parts of size 1 or 2."""
    results = []
    def _gen(remaining, current):
        if remaining == 0:
            results.append(list(current))
            return
        if remaining >= 1:
            current.append(1)
            _gen(remaining - 1, current)
            current.pop()
        if remaining >= 2:
            current.append(2)
            _gen(remaining - 2, current)
            current.pop()
    _gen(d, [])
    return results

def _compute_rank(n_splits, m_splits, p_splits):
    """Compute the rank for given splits."""
    rank = 0
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                if nb == 2 and mb == 2 and pb == 2:
                    rank += 7
                else:
                    rank += nb * mb * pb
    return rank

def _build_decomposition(n_splits, m_splits, p_splits, n, m, p):
    """Build the full decomposition for given splits."""
    n_off = [0]
    for s in n_splits[:-1]:
        n_off.append(n_off[-1] + s)
    m_off = [0]
    for s in m_splits[:-1]:
        m_off.append(m_off[-1] + s)
    p_off = [0]
    for s in p_splits[:-1]:
        p_off.append(p_off[-1] + s)
    
    rank = _compute_rank(n_splits, m_splits, p_splits)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = _strassen_2x2x2_terms()
    
    col = 0
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                i_off = n_off[i_b]
                j_off = m_off[j_b]
                k_off = p_off[k_b]
                
                if nb == 2 and mb == 2 and pb == 2:
                    for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                        for a, b, val in u_coeffs:
                            gi = i_off + a
                            gj = j_off + b
                            U[gi * m + gj][col] = val
                        for b, c, val in v_coeffs:
                            gj = j_off + b
                            gk = k_off + c
                            V[gj * p + gk][col] = val
                        for c, a, val in w_coeffs:
                            gk = k_off + c
                            gi = i_off + a
                            W[gk * n + gi][col] = val
                        col += 1
                else:
                    for li in range(nb):
                        for lj in range(mb):
                            for lk in range(pb):
                                gi = i_off + li
                                gj = j_off + lj
                                gk = k_off + lk
                                U[gi * m + gj][col] = 1
                                V[gj * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the tensor identity for the decomposition."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U_np.shape[1]):
                    s += U_np[i*m + j, r] * V_np[j*p + k, r] * W_np[k*n + i, r]
                if not np.isclose(s, 1.0, atol=1e-10):
                    return False
    return True

def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
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
    return U, V, W

def construct():
    n, m, p = 3, 5, 5
    
    deadline = time.time() + 100
    
    # Generate all possible splits
    n_splits_list = _get_splits(n)
    m_splits_list = _get_splits(m)
    p_splits_list = _get_splits(p)
    
    best_rank = n * m * p
    best_U = best_V = best_W = None
    
    # Try all combinations of splits, sorted by expected rank
    candidates = []
    for n_splits in n_splits_list:
        for m_splits in m_splits_list:
            for p_splits in p_splits_list:
                rank = _compute_rank(n_splits, m_splits, p_splits)
                candidates.append((rank, n_splits, m_splits, p_splits))
    
    candidates.sort(key=lambda x: x[0])
    
    for rank, n_splits, m_splits, p_splits in candidates:
        if time.time() > deadline - 10:
            break
        
        if rank >= best_rank:
            continue
        
        U, V, W = _build_decomposition(n_splits, m_splits, p_splits, n, m, p)
        
        if _verify_decomposition(U, V, W, n, m, p):
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break
    
    if best_U is not None:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    U, V, W = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}
