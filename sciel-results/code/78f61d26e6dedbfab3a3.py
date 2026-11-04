"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _decompose_strassen(n, m, p):
    """Build the known rank-17 decomposition (Strassen + direct)."""
    n_a, n_b, n_c = n*m, m*p, p*n
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    def set_term(r, u_dict, v_dict, w_dict):
        for a, val in u_dict.items():
            U[a, r] = val
        for b, val in v_dict.items():
            V[b, r] = val
        for c, val in w_dict.items():
            W[c, r] = val
    
    # Strassen for A1(2x2) x B1_left(2x2)
    set_term(0, {0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1})
    set_term(1, {3:1, 4:1}, {0:1}, {1:1, 3:-1})
    set_term(2, {0:1}, {1:1, 4:-1}, {2:1, 3:1})
    set_term(3, {4:1}, {3:1, 0:-1}, {0:1, 1:1})
    set_term(4, {0:1, 1:1}, {4:1}, {0:-1, 2:1})
    set_term(5, {3:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_term(6, {1:1, 4:-1}, {3:1, 4:1}, {0:1})
    
    # Direct for A1(2x2) x B1_right(2x1)
    set_term(7, {0:1}, {2:1}, {4:1})
    set_term(8, {1:1}, {5:1}, {4:1})
    set_term(9, {3:1}, {2:1}, {5:1})
    set_term(10, {4:1}, {5:1}, {5:1})
    
    # Direct for A2(2x1) x B2(1x3)
    set_term(11, {2:1}, {6:1}, {0:1})
    set_term(12, {2:1}, {7:1}, {2:1})
    set_term(13, {2:1}, {8:1}, {4:1})
    set_term(14, {5:1}, {6:1}, {1:1})
    set_term(15, {5:1}, {7:1}, {3:1})
    set_term(16, {5:1}, {8:1}, {5:1})
    
    return U, V, W

def _try_rank16_systematic(deadline):
    """
    Systematic search for rank-16 decomposition.
    
    Strategy: The residual after Strassen has 10 non-zero entries.
    We need 9 rank-1 terms to produce these 10 entries.
    One term must cover 2 entries, the rest cover 1 each.
    
    We enumerate which pair of entries is covered by a single term,
    then solve for the remaining 8 terms (which must be direct).
    """
    n_a, n_b, n_c = 6, 9, 6
    
    # The 10 target positions (from A1*B1_right and A2*B2)
    targets = [
        (0, 2, 4),  # P1: A[0,0]*B[0,2]->C[0,2]
        (1, 5, 4),  # P2: A[0,1]*B[1,2]->C[0,2]
        (3, 2, 5),  # P3: A[1,0]*B[0,2]->C[1,2]
        (4, 5, 5),  # P4: A[1,1]*B[1,2]->C[1,2]
        (2, 6, 0),  # P5: A[0,2]*B[2,0]->C[0,0]
        (2, 7, 2),  # P6: A[0,2]*B[2,1]->C[0,1]
        (2, 8, 4),  # P7: A[0,2]*B[2,2]->C[0,2]
        (5, 6, 1),  # P8: A[1,2]*B[2,0]->C[1,0]
        (5, 7, 3),  # P9: A[1,2]*B[2,1]->C[1,1]
        (5, 8, 5),  # P10: A[1,2]*B[2,2]->C[1,2]
    ]
    
    target_set = set(targets)
    
    # Enumerate all pairs of targets and check if a rank-1 term
    # can cover exactly those two (and no others from the target set).
    valid_pairs = []
    for i in range(len(targets)):
        for j in range(i+1, len(targets)):
            a1, b1, c1 = targets[i]
            a2, b2, c2 = targets[j]
            
            u_set = {a1, a2}
            v_set = {b1, b2}
            w_set = {c1, c2}
            
            produced = set()
            for a in u_set:
                for b in v_set:
                    for c in w_set:
                        produced.add((a, b, c))
            
            if produced & target_set == {(a1,b1,c1), (a2,b2,c2)}:
                valid_pairs.append((i, j))
    
    # Try all sign combinations for the pair term
    for i, j in valid_pairs:
        if time.time() > deadline:
            return None
        
        a1, b1, c1 = targets[i]
        a2, b2, c2 = targets[j]
        
        # Try different sign patterns for the pair term
        for su1 in [1, -1]:
            for su2 in [1, -1]:
                for sv1 in [1, -1]:
                    for sv2 in [1, -1]:
                        for sc1 in [1, -1]:
                            for sc2 in [1, -1]:
                                # Check that the product gives +1 at both targets
                                if su1 * sv1 * sc1 != 1:
                                    continue
                                if su2 * sv2 * sc2 != 1:
                                    continue
                                
                                # Check that no other target is in the cross product
                                u_set = {a1, a2}
                                v_set = {b1, b2}
                                w_set = {c1, c2}
                                produced = set()
                                for a in u_set:
                                    for b in v_set:
                                        for c in w_set:
                                            produced.add((a, b, c))
                                
                                if produced & target_set != {(a1,b1,c1), (a2,b2,c2)}:
                                    continue
                                
                                # Construct the full decomposition
                                U = np.zeros((n_a, 16))
                                V = np.zeros((n_b, 16))
                                W = np.zeros((n_c, 16))
                                
                                # Strassen
                                strassen_U, strassen_V, strassen_W = _decompose_strassen(2, 3, 3)
                                U[:, :7] = strassen_U[:, :7]
                                V[:, :7] = strassen_V[:, :7]
                                W[:, :7] = strassen_W[:, :7]
                                
                                # Term 7: pair term
                                U[a1, 7] = su1
                                U[a2, 7] = su2
                                V[b1, 7] = sv1
                                V[b2, 7] = sv2
                                W[c1, 7] = sc1
                                W[c2, 7] = sc2
                                
                                # Terms 8-15: direct for remaining targets
                                remaining = [t for idx, t in enumerate(targets) if idx != i and idx != j]
                                if len(remaining) != 8:
                                    continue
                                
                                for k, (a, b, c) in enumerate(remaining):
                                    col = 8 + k
                                    U[a, col] = 1
                                    V[b, col] = 1
                                    W[c, col] = 1
                                
                                # Verify
                                T = _build_tensor(2, 3, 3)
                                if _verify_decomposition(U, V, W, T):
                                    return U, V, W
    
    return None

def _try_rank16_with_shared_terms(deadline):
    """
    Try to find a rank-16 decomposition where more than one term shares indices,
    allowing for a more complex structure than just one pair term.
    
    This explores whether we can use 2 terms that each cover 2 targets,
    or 1 term covering 3 targets, etc.
    """
    n_a, n_b, n_c = 6, 9, 6
    
    targets = [
        (0, 2, 4),
        (1, 5, 4),
        (3, 2, 5),
        (4, 5, 5),
        (2, 6, 0),
        (2, 7, 2),
        (2, 8, 4),
        (5, 6, 1),
        (5, 7, 3),
        (5, 8, 5),
    ]
    
    target_set = set(targets)
    
    # Try: one term covers 3 targets
    # For a rank-1 term to cover 3 targets (a1,b1,c1), (a2,b2,c2), (a3,b3,c3):
    # U support: {a1,a2,a3}, V support: {b1,b2,b3}, W support: {c1,c2,c3}
    # The term produces entries at all 27 combinations in the cross product
    # We need exactly the 3 targets to be in target_set
    
    for i in range(len(targets)):
        for j in range(i+1, len(targets)):
            for k in range(j+1, len(targets)):
                if time.time() > deadline:
                    return None
                
                a1, b1, c1 = targets[i]
                a2, b2, c2 = targets[j]
                a3, b3, c3 = targets[k]
                
                u_set = {a1, a2, a3}
                v_set = {b1, b2, b3}
                w_set = {c1, c2, c3}
                
                produced = set()
                for a in u_set:
                    for b in v_set:
                        for c in w_set:
                            produced.add((a, b, c))
                
                # Check that exactly the 3 targets are in produced
                if produced & target_set == {(a1,b1,c1), (a2,b2,c2), (a3,b3,c3)}:
                    # This is a valid 3-target term
                    # Now we need 7 more terms for the remaining 7 targets
                    remaining = [t for idx, t in enumerate(targets) 
                                if idx not in (i, j, k)]
                    if len(remaining) != 7:
                        continue
                    
                    # Try sign patterns
                    for su1 in [1, -1]:
                        for su2 in [1, -1]:
                            for su3 in [1, -1]:
                                for sv1 in [1, -1]:
                                    for sv2 in [1, -1]:
                                        for sv3 in [1, -1]:
                                            for sc1 in [1, -1]:
                                                for sc2 in [1, -1]:
                                                    for sc3 in [1, -1]:
                                                        if su1*sv1*sc1 != 1:
                                                            continue
                                                        if su2*sv2*sc2 != 1:
                                                            continue
                                                        if su3*sv3*sc3 != 1:
                                                            continue
                                                        
                                                        U = np.zeros((n_a, 16))
                                                        V = np.zeros((n_b, 16))
                                                        W = np.zeros((n_c, 16))
                                                        
                                                        strassen_U, strassen_V, strassen_W = _decompose_strassen(2, 3, 3)
                                                        U[:, :7] = strassen_U[:, :7]
                                                        V[:, :7] = strassen_V[:, :7]
                                                        W[:, :7] = strassen_W[:, :7]
                                                        
                                                        U[a1, 7] = su1
                                                        U[a2, 7] = su2
                                                        U[a3, 7] = su3
                                                        V[b1, 7] = sv1
                                                        V[b2, 7] = sv2
                                                        V[b3, 7] = sv3
                                                        W[c1, 7] = sc1
                                                        W[c2, 7] = sc2
                                                        W[c3, 7] = sc3
                                                        
                                                        for idx, (a, b, c) in enumerate(remaining):
                                                            col = 8 + idx
                                                            U[a, col] = 1
                                                            V[b, col] = 1
                                                            W[c, col] = 1
                                                        
                                                        T = _build_tensor(2, 3, 3)
                                                        if _verify_decomposition(U, V, W, T):
                                                            return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # First, try the systematic rank-16 search with pair terms
    result = _try_rank16_systematic(deadline)
    if result is not None:
        U, V, W = result
        T = _build_tensor(n, m, p)
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank-16 with 3-target terms
    result = _try_rank16_with_shared_terms(deadline)
    if result is not None:
        U, V, W = result
        T = _build_tensor(n, m, p)
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank-17
    U, V, W = _decompose_strassen(n, m, p)
    T = _build_tensor(n, m, p)
    if not _verify_decomposition(U, V, W, T):
        # Schoolbook fallback
        rank_fb = n * m * p
        U_fb = np.zeros((n*m, rank_fb))
        V_fb = np.zeros((m*p, rank_fb))
        W_fb = np.zeros((p*n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
