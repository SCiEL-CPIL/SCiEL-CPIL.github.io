import time
import sys
from fractions import Fraction
import random

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm, mp, pn = n * m, m * p, p * n

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build the rank-44 solution first
    rank = 44
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    col = 0

    # Part 1: A1 (2x2) * B1 (2x8) using Strassen
    for block_idx in range(4):
        k0 = block_idx * 2
        k1 = k0 + 1
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        u_idx = li * m + la
                        U[u_idx][col] = u_val

            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] = v_val

            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] = w_val

            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A2 (2x1) * B2 (1x8)
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}

    if not _validate(candidate):
        return _schoolbook()

    # Try to find a rank-43 solution by exploiting linear dependencies
    # Approach: For each pair (Strassen term, schoolbook a=2 term), check if we can
    # absorb the schoolbook term into the Strassen term with minimal perturbation
    if _elapsed_ok():
        best_rank = 44
        best_candidate = candidate

        # Try rank 43: remove one a=2 term and modify one Strassen term
        # The a=2 terms are in columns 28..43 (16 terms)
        # The Strassen terms are in columns 0..27 (28 terms)
        
        # For each a=2 term to remove and each Strassen term to modify:
        # We need the modified Strassen term to still satisfy the 2x2x2 block equations
        # AND also contribute to the a=2 equation for the removed term.
        
        # Let's try a direct approach: for each Strassen term s and a=2 term t,
        # try to find a new rank-1 term that replaces both.
        
        # A rank-1 term is (u, v, w) where:
        # u is in R^6, v is in R^24, w is in R^16
        # The contribution to tensor entry (i,a,k) is u[i*m+a] * v[a*p+k] * w[k*n+i]
        
        # For the 2x2x2 Strassen block (block b, covering k in {2b, 2b+1}):
        # The Strassen term r contributes to entries (i, a, k) for i in {0,1}, a in {0,1}, k in {2b, 2b+1}
        # with value S_U[r][i*2+a] * S_V[r][a*2+(k-2b)] * S_W[r][(k-2b)*2+i]
        
        # For the a=2 term (i0, k0):
        # It contributes 1 to entry (i0, 2, k0) and 0 to all other entries.
        
        # To merge: we want a new rank-1 term (u', v', w') such that:
        # 1. For all (i,a,k) in the Strassen block: u'[i*m+a]*v'[a*p+k]*w'[k*n+i] = S_U[r][i*2+a]*S_V[r][a*2+(k-2b)]*S_W[r][(k-2b)*2+i]
        # 2. For entry (i0, 2, k0): u'[i0*m+2]*v'[2*p+k0]*w'[k0*n+i0] = 1
        # 3. For all other entries: u'[i*m+a]*v'[a*p+k]*w'[k*n+i] = 0
        
        # This is a very constrained system. Let's try specific cases.
        
        # Case: merge a=2 term (i0, k0) with Strassen term (block b, r) where k0 is in block b
        # i.e., k0 = 2b or k0 = 2b+1
        
        for i0 in range(2):
            for k0 in range(8):
                if not _elapsed_ok():
                    break
                b = k0 // 2
                for r in range(7):
                    if not _elapsed_ok():
                        break
                    
                    # Try to construct a merged rank-1 term
                    # The Strassen term has support on:
                    # U indices: (i, a) for i in {0,1}, a in {0,1} where S_U[r][i*2+a] != 0
                    # V indices: (a, k) for a in {0,1}, k in {0,1} (local) where S_V[r][a*2+k] != 0
                    # W indices: (k, i) for k in {0,1} (local), i in {0,1} where S_W[r][k*2+i] != 0
                    
                    # The a=2 term has support on:
                    # U index: (i0, 2)
                    # V index: (2, k0)
                    # W index: (k0, i0)
                    
                    # For the merged term to work, we need:
                    # u'[i0*m+2] * v'[2*p+k0] * w'[k0*n+i0] = 1
                    # AND for the Strassen block entries, the product must match.
                    
                    # Let's try: set u'[i0*m+2] = alpha, v'[2*p+k0] = beta, w'[k0*n+i0] = gamma
                    # with alpha * beta * gamma = 1
                    
                    # For the Strassen part, we need the products to match S_U, S_V, S_W values.
                    # This requires that for each (i,a,k_local) in the block:
                    # u'[i*m+a] * v'[a*p+(2b+k_local)] * w'[(2b+k_local)*n+i] = S_U[r][i*2+a] * S_V[r][a*2+k_local] * S_W[r][k_local*2+i]
                    
                    # And for entries outside the block and outside the a=2 position, the product must be 0.
                    
                    # This is a system of 8 equations (for the 2x2x2 block) plus 1 equation (for a=2)
                    # with 6+24+16 = 46 unknowns, but many are constrained to be 0.
                    
                    # Let's try a specific simple case: 
                    # Set u' to have non-zero only at positions that are in the Strassen support plus (i0, 2)
                    # Set v' to have non-zero only at positions in the Strassen support plus (2, k0)
                    # Set w' to have non-zero only at positions in the Strassen support plus (k0, i0)
                    
                    # For this to work, we need the Strassen support to be "compatible" with the a=2 position.
                    
                    # Let's just try to solve numerically for a few cases.
                    
                    # Build the system:
                    # Unknowns: u'[idx] for idx in U_support, v'[idx] for idx in V_support, w'[idx] for idx in W_support
                    # where U_support = {(i,a) : S_U[r][i*2+a] != 0} ∪ {(i0, 2)}
                    # V_support = {(a, 2b+k) : S_V[r][a*2+k] != 0} ∪ {(2, k0)}
                    # W_support = {(2b+k, i) : S_W[r][k*2+i] != 0} ∪ {(k0, i0)}
                    
                    u_support = set()
                    for i in range(2):
                        for a in range(2):
                            if S_U[r][i*2+a] != 0:
                                u_support.add(i*m+a)
                    u_support.add(i0*m+2)
                    
                    v_support = set()
                    for a in range(2):
                        for k_local in range(2):
                            if S_V[r][a*2+k_local] != 0:
                                v_support.add(a*p + 2*b + k_local)
                    v_support.add(2*p + k0)
                    
                    w_support = set()
                    for k_local in range(2):
                        for i in range(2):
                            if S_W[r][k_local*2+i] != 0:
                                w_support.add((2*b+k_local)*n + i)
                    w_support.add(k0*n + i0)
                    
                    # Number of unknowns
                    n_u = len(u_support)
                    n_v = len(v_support)
                    n_w = len(w_support)
                    n_unknowns = n_u + n_v + n_w
                    
                    # Equations: one for each (i,a,k) in the 2x2x2 block, plus one for (i0,2,k0)
                    # But we also need equations to ensure other entries are 0.
                    
                    # This is getting complex. Let's try a different approach:
                    # Use the fact that the tensor has a specific structure and try to find
                    # a rank-43 decomposition by solving a linear system.
                    
                    # Actually, let's try a simpler heuristic: check if there's a rank-43
                    # solution by trying to find a linear dependency in the 44 rank-1 terms.
                    
                    # The 44 rank-1 terms span a 44-dimensional space. If there's a linear
                    # dependency, we might be able to reduce the rank.
                    
                    # For now, let's just return the rank-44 solution.
                    pass

        return candidate

    return candidate

# EVOLVE-BLOCK-END
