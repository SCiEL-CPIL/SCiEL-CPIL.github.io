"""Matrix multiplication 2x3 x 3x10 - rank-minimal decomposition."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Check that the decomposition exactly reproduces C = A*B using NumPy."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Check required entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(Ua[a, :rank] * Va[b, :rank], Wa[c, :rank])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check spurious entries
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i = a // m
                j = a % m
                j_prime = b // p
                k = b % p
                i_prime = c % n
                k_prime = c // n
                
                if i == i_prime and j == j_prime and k == k_prime:
                    continue
                
                val = np.dot(Ua[a, :rank] * Va[b, :rank], Wa[c, :rank])
                if abs(val) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _rank55(n, m, p):
    """Block-based rank-55 decomposition for 2x3 x 3x10."""
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1: (A00 + A11) * (B0,k0 + B1,k1)
        U[0 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M2: (A10 + A11) * B0,k0
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1
        
        # M3: A00 * (B0,k1 - B1,k1)
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M4: A11 * (B1,k0 - B0,k0)
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # M5: (A00 + A01) * B1,k1
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # M6: (A10 - A00) * (B0,k0 + B0,k1)
        U[1 * 3 + 0][col] += 1
        U[0 * 3 + 0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M7: (A01 - A11) * (B1,k0 + B1,k1)
        U[0 * 3 + 1][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # Direct terms for row 2 (A02, A12) with B2
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

def _try_rank54(n, m, p, deadline):
    """
    Try to find rank-54 by systematically searching for a valid decomposition.
    Uses tensor slicing ideas: treat the 2x3x3x10 tensor and look for
    a decomposition with 54 terms by merging some of the 4 direct terms
    in each block with the Strassen terms.
    
    Key insight: For each 2x3 x 3x2 block, the 4 direct terms are:
      A02*B2k0, A02*B2k1, A12*B2k0, A12*B2k1
    
    We can try to express 3 of these 4 products using only 3 bilinear terms
    by combining them with the Strassen terms. This is possible if we allow
    the Strassen terms to also contribute to the "direct" outputs.
    
    Specifically, we can modify the Strassen decomposition so that some of
    its terms also produce outputs in the C0k0, C0k1, C1k0, C1k1 positions
    from the A02, A12 rows.
    """
    import random
    random.seed(42)
    
    rank = 54
    n, m, p = 2, 3, 10
    
    # We need to find 54 terms. The natural block structure gives 55.
    # To get 54, we need to save 1 term across all 5 blocks.
    # This means one block uses 10 terms and the others use 11.
    
    # For a 2x3 x 3x2 block with 10 terms, we need a specific structure.
    # Let's try to construct one by modifying the 11-term structure.
    
    # The 11 terms are:
    # 7 Strassen terms for A[0:2,0:2] * B[0:2,0:2]
    # 4 direct terms for A[0:2,2] * B[2,0:2]
    
    # To save 1, we need to merge one direct term with a Strassen term.
    # This requires the Strassen term to also produce a "direct" output.
    
    # Let's try a systematic approach: for each block, try all possible
    # ways to merge one direct term with a Strassen term.
    
    # Actually, let's try a different approach: use a global decomposition
    # that doesn't strictly follow the block structure.
    
    # For now, let's try to find a valid rank-54 by random search
    # with a structured initialization.
    
    for attempt in range(5000):
        if time.time() > deadline - 10:
            break
        
        U = np.zeros((6, rank))
        V = np.zeros((30, rank))
        W = np.zeros((20, rank))
        
        # Start with the rank-55 structure and try to zero out one column
        # by redistributing its contributions to other columns.
        
        col = 0
        for t in range(5):
            k0 = 2 * t
            k1 = 2 * t + 1
            
            if t == 0 and attempt % 5 == 0:
                # Try a 10-term block for block 0
                # Use a modified Strassen + 3 direct terms
                terms = _get_10_term_block_v2(k0, k1, attempt)
                for term in terms:
                    for (ui, s) in term[0]:
                        U[ui, col] += s
                    for (vi, s) in term[1]:
                        V[vi, col] += s
                    for (wi, s) in term[2]:
                        W[wi, col] += s
                    col += 1
            else:
                # Standard 11-term block
                terms = _get_11_term_block(k0, k1)
                for term in terms:
                    for (ui, s) in term[0]:
                        U[ui, col] += s
                    for (vi, s) in term[1]:
                        V[vi, col] += s
                    for (wi, s) in term[2]:
                        W[wi, col] += s
                    col += 1
        
        if col != rank:
            continue
        
        # Verify
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _verify(U_list, V_list, W_list, n, m, p, rank):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _get_11_term_block(k0, k1):
    """Return the standard 11-term decomposition for a 2x3 x 3x2 block."""
    terms = []
    
    # M1: (A00 + A11) * (B0,k0 + B1,k1)
    terms.append([
        [(0, 1), (4, 1)],
        [(k0, 1), (10+k1, 1)],
        [(2*k0, 1), (2*k1+1, 1)]
    ])
    
    # M2: (A10 + A11) * B0,k0
    terms.append([
        [(3, 1), (4, 1)],
        [(k0, 1)],
        [(2*k0+1, 1), (2*k1+1, -1)]
    ])
    
    # M3: A00 * (B0,k1 - B1,k1)
    terms.append([
        [(0, 1)],
        [(k1, 1), (10+k1, -1)],
        [(2*k1, 1), (2*k1+1, 1)]
    ])
    
    # M4: A11 * (B1,k0 - B0,k0)
    terms.append([
        [(4, 1)],
        [(10+k0, 1), (k0, -1)],
        [(2*k0, 1), (2*k0+1, 1)]
    ])
    
    # M5: (A00 + A01) * B1,k1
    terms.append([
        [(0, 1), (1, 1)],
        [(10+k1, 1)],
        [(2*k0, -1), (2*k1, 1)]
    ])
    
    # M6: (A10 - A00) * (B0,k0 + B0,k1)
    terms.append([
        [(3, 1), (0, -1)],
        [(k0, 1), (k1, 1)],
        [(2*k1+1, 1)]
    ])
    
    # M7: (A01 - A11) * (B1,k0 + B1,k1)
    terms.append([
        [(1, 1), (4, -1)],
        [(10+k0, 1), (10+k1, 1)],
        [(2*k0, 1)]
    ])
    
    # Direct terms
    terms.append([
        [(2, 1)],
        [(20+k0, 1)],
        [(2*k0, 1)]
    ])
    
    terms.append([
        [(2, 1)],
        [(20+k1, 1)],
        [(2*k1, 1)]
    ])
    
    terms.append([
        [(5, 1)],
        [(20+k0, 1)],
        [(2*k0+1, 1)]
    ])
    
    terms.append([
        [(5, 1)],
        [(20+k1, 1)],
        [(2*k1+1, 1)]
    ])
    
    return terms

def _get_10_term_block_v2(k0, k1, seed):
    """
    Try to find a 10-term decomposition for a 2x3 x 3x2 block.
    
    Strategy: Modify the 7 Strassen terms so that one of them also
    contributes to the direct outputs. This allows us to eliminate
    one of the 4 direct terms.
    
    For example, if we modify M1 to also produce C0k0 from A02*B2k0,
    we can eliminate the direct term A02*B2k0.
    """
    import random
    rng = random.Random(seed)
    
    # Start with the 7 Strassen terms
    terms = []
    
    # M1: (A00 + A11) * (B0,k0 + B1,k1) + alpha * A02 * B2k0
    # To make this work, we need:
    # (A00 + A11) * (B0,k0 + B1,k1) contributes to C0k0, C0k1, C1k0, C1k1
    # and alpha * A02 * B2k0 contributes to C0k0
    # But we can't add A02 to the U factor without affecting other terms.
    
    # Alternative: use a different combination.
    # Let's try:
    # T1 = (A00 + A11 + A02) * (B0,k0 + B1,k1)
    # This would contribute to C0k0 from A02*B2k0? No, A02 is in row 2,
    # and B2k0 is in column 2, so A02*B2k0 goes to C0k0 (since i=0, k=k0).
    # Wait, A02 is A[0,2], B2k0 is B[2,k0], so the product is C[0,k0].
    # But in T1, A02 is multiplied by (B0,k0 + B1,k1), not B2k0.
    
    # So this doesn't work directly. Let me think differently.
    
    # The issue is that the Strassen terms use A[0:2,0:2] and B[0:2,0:2],
    # while the direct terms use A[0:2,2] and B[2,0:2].
    # These are disjoint sets of indices, so we can't easily merge them.
    
    # Unless we use a non-block decomposition. Let's try a fully
    # general 10-term decomposition.
    
    # For now, return a placeholder that will fail verification.
    # This needs to be a valid 10-term decomposition.
    
    # Let's try a different approach: use 6 terms for the 2x2x2 part
    # and 4 terms for the 2x1x2 part, but with some overlap.
    
    # Actually, the rank of the 2x2x2 tensor is 7 (Strassen), so we
    # can't do better than 7 for that part.
    
    # The rank of the 2x1x2 tensor is 4 (direct), so we can't do
    # better than 4 for that part.
    
    # Total: 7 + 4 = 11. To get 10, we need a non-block decomposition.
    
    # Let's try to find one by solving a system of equations.
    
    # For a 10-term decomposition, we have 10 unknown terms.
    # Each term has 3 factors: U (6 entries), V (30 entries), W (20 entries).
    # But we can assume the factors are sparse with small integer entries.
    
    # Let's try a random search with structured initialization.
    
    # For now, return the 11-term structure truncated (will fail).
    terms_11 = _get_11_term_block(k0, k1)
    return terms_11[:10]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # First, try the known rank-55 decomposition
    try:
        res55 = _rank55(n, m, p)
        if _verify(res55["U"], res55["V"], res55["W"], n, m, p, 55):
            # Found valid rank-55, now try to improve
            if time.time() < deadline - 30:
                res54 = _try_rank54(n, m, p, deadline)
                if res54 is not None:
                    return res54
            return res55
    except Exception:
        pass
    
    # Fallback: schoolbook rank-60
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
