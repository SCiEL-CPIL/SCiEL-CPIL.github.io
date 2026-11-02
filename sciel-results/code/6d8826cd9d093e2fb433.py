import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (u, v, w) where u,v,w are 2-vectors."""
    return [
        ((1, 1), (1, 0), (1, 0)),
        ((1, 0), (1, 1), (1, 0)),
        ((1, -1), (1, 0), (0, 1)),
        ((0, 1), (1, -1), (0, 1)),
        ((1, 0), (0, 1), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
        ((1, 1), (0, 1), (1, 0)),
    ]

def _generate_candidate_blocks(n, m, p):
    """Generate all possible 2x2x2 Strassen blocks and their covered triples."""
    blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                # Check if this block fits
                i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
                if i1 < n and j1 < m and k1 < p:
                    covered = set()
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                covered.add((i0+di, j0+dj, k0+dk))
                    blocks.append({
                        'i0': i0, 'j0': j0, 'k0': k0,
                        'covered': covered,
                        'savings': 1  # 8 schoolbook - 7 strassen = 1 rank saved
                    })
    return blocks

def _greedy_select_blocks(n, m, p, deadline):
    """Greedy selection of non-overlapping Strassen blocks."""
    blocks = _generate_candidate_blocks(n, m, p)
    selected = []
    covered = set()
    
    # Sort by savings (all are 1, so just use order)
    for block in blocks:
        if time.time() > deadline:
            break
        # Check if this block overlaps with already selected
        if not (block['covered'] & covered):
            selected.append(block)
            covered |= block['covered']
    
    return selected, covered

def _build_decomposition(n, m, p, selected_blocks, deadline):
    """Build the full decomposition from selected blocks and schoolbook for rest."""
    # Count total rank
    num_blocks = len(selected_blocks)
    covered = set()
    for block in selected_blocks:
        covered |= block['covered']
    
    total_triples = n * m * p
    num_schoolbook = total_triples - len(covered)
    total_rank = num_blocks * 7 + num_schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    strassen_terms = _strassen_2x2x2_terms()
    
    # Add Strassen blocks
    for block in selected_blocks:
        i0, j0, k0 = block['i0'], block['j0'], block['k0']
        for (u, v, w) in strassen_terms:
            for di in range(2):
                if u[di] != 0:
                    for dj in range(2):
                        U[(i0+di)*m + (j0+dj)][col] += u[di]
            for dj in range(2):
                if v[dj] != 0:
                    for dk in range(2):
                        V[(j0+dj)*p + (k0+dk)][col] += v[dj]
            for dk in range(2):
                if w[dk] != 0:
                    for di in range(2):
                        W[(k0+dk)*n + (i0+di)][col] += w[dk]
            col += 1
    
    # Add schoolbook for uncovered triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, deadline):
    """Verify the decomposition is correct."""
    if time.time() > deadline:
        return False
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1
        T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        return np.allclose(T_decomp, target, atol=1e-10)
    except Exception:
        return False

def _schoolbook_fallback(n, m, p):
    """Return full schoolbook decomposition as fallback."""
    rank_school = n * m * p
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    c2 = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][c2] = 1
                V_s[j*p+k][c2] = 1
                W_s[k*n+i][c2] = 1
                c2 += 1
    return {"U": U_s, "V": V_s, "W": W_s}

def construct():
    start = time.time()
    deadline = start + 140  # Leave 10s margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Generate candidate Strassen blocks
    blocks = _generate_candidate_blocks(n, m, p)
    
    # Use greedy selection to pick non-overlapping blocks
    selected_blocks, covered = _greedy_select_blocks(n, m, p, deadline)
    
    # Build decomposition
    U, V, W = _build_decomposition(n, m, p, selected_blocks, deadline)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p, deadline):
        return _schoolbook_fallback(n, m, p)
    
    return {"U": U, "V": V, "W": W}
