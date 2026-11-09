import numpy as np
import time
import itertools

DIMENSIONS = [2, 5, 5]

def _schoolbook(n, m, p):
    """Return the trivial rank-nmp decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that U, V, W exactly decompose the matmul tensor."""
    rank = U.shape[1]
    T_decomp = np.zeros((n, m, p))
    for r in range(rank):
        U_r = U[:, r].reshape(n, m)
        V_r = V[:, r].reshape(m, p)
        W_r = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_decomp[i, j, k] += U_r[i, j] * V_r[j, k] * W_r[k, i]
    T_true = np.ones((n, m, p))
    return np.allclose(T_decomp, T_true, atol=tol)

def _strassen_block(j_pair, k_pair):
    """
    Return the 7 (U, V, W) contributions for a Strassen block
    covering j in j_pair and k in k_pair.
    
    j_pair: tuple of 2 j indices
    k_pair: tuple of 2 k indices
    
    Returns list of 7 tuples, each (U_entry, V_entry, W_entry)
    where each entry is a dict mapping (index) -> value
    """
    j1, j2 = j_pair
    k1, k2 = k_pair
    
    # Strassen's 7 multiplications for 2x2x2:
    # M1 = (A11 - A22) * (B22 + B12)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # In our notation:
    # A[i,j] corresponds to U entry at (i*m + j)
    # B[j,k] corresponds to V entry at (j*p + k)
    # C[i,k] corresponds to W entry at (k*n + i)
    
    # For the 2x2x2 block with i in {0,1}, j in {j1,j2}, k in {k1,k2}:
    # Let's map: i0=0, i1=1; j0=j1, j1=j2; k0=k1, k1=k2
    
    # A00 = A[0,j1], A01 = A[0,j2], A10 = A[1,j1], A11 = A[1,j2]
    # B00 = B[j1,k1], B01 = B[j1,k2], B10 = B[j2,k1], B11 = B[j2,k2]
    # C00 = C[0,k1], C01 = C[0,k2], C10 = C[1,k1], C11 = C[1,k2]
    
    # Strassen:
    # M1 = (A00 - A11) * (B11 + B01) = (A[0,j1] - A[1,j2]) * (B[j2,k2] + B[j1,k2])
    # M2 = (A10 + A11) * B00 = (A[1,j1] + A[1,j2]) * B[j1,k1]
    # M3 = A00 * (B01 - B11) = A[0,j1] * (B[j1,k2] - B[j2,k2])
    # M4 = A11 * (B10 - B00) = A[1,j2] * (B[j2,k1] - B[j1,k1])
    # M5 = (A00 + A01) * B11 = (A[0,j1] + A[0,j2]) * B[j2,k2]
    # M6 = (A10 - A00) * (B00 + B01) = (A[1,j1] - A[0,j1]) * (B[j1,k1] + B[j1,k2])
    # M7 = (A01 - A11) * (B10 + B11) = (A[0,j2] - A[1,j2]) * (B[j2,k1] + B[j2,k2])
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a product of a linear combination of A's and a linear combination of B's
    # So for each M, we have one U vector (for A), one V vector (for B), and the W vector
    # is determined by how M contributes to each C entry.
    
    contributions = []
    
    # M1: U has A[0,j1]=1, A[1,j2]=-1; V has B[j2,k2]=1, B[j1,k2]=1
    # W: C[0,k1]+=1, C[0,k2]-=1 (from -M5? no, let me re-derive)
    # Actually, let me just define U_r, V_r, W_r for each of the 7 terms.
    
    # Term 1 (M1):
    U1 = {0 * 5 + j1: 1, 1 * 5 + j2: -1}  # A[0,j1]=1, A[1,j2]=-1
    V1 = {j1 * 5 + k2: 1, j2 * 5 + k2: 1}  # B[j1,k2]=1, B[j2,k2]=1
    W1 = {k1 * 2 + 0: 1, k1 * 2 + 1: 1,  # C[0,k1] += M1, C[1,k1] += M1
          k2 * 2 + 0: 0, k2 * 2 + 1: 0}   # C[0,k2], C[1,k2] don't get M1 directly
    # Wait, I need to be more careful. Let me re-derive.
    
    # C[0,k1] = M1 + M4 - M5 + M7
    # C[0,k2] = M3 + M5
    # C[1,k1] = M2 + M4
    # C[1,k2] = M1 - M2 + M3 + M6
    
    # So:
    # M1 contributes to: C[0,k1] (+1), C[1,k2] (+1)
    # M2 contributes to: C[1,k1] (+1), C[1,k2] (-1)
    # M3 contributes to: C[0,k2] (+1), C[1,k2] (+1)
    # M4 contributes to: C[0,k1] (+1), C[1,k1] (+1)
    # M5 contributes to: C[0,k1] (-1), C[0,k2] (+1)
    # M6 contributes to: C[1,k2] (+1)
    # M7 contributes to: C[0,k1] (+1)
    
    # Now define each term:
    
    # M1: U: A[0,j1]=1, A[1,j2]=-1; V: B[j1,k2]=1, B[j2,k2]=1
    # W: C[0,k1]=1, C[1,k2]=1
    contributions.append({
        'U': {0 * 5 + j1: 1, 1 * 5 + j2: -1},
        'V': {j1 * 5 + k2: 1, j2 * 5 + k2: 1},
        'W': {k1 * 2 + 0: 1, k2 * 2 + 1: 1}
    })
    
    # M2: U: A[1,j1]=1, A[1,j2]=1; V: B[j1,k1]=1
    # W: C[1,k1]=1, C[1,k2]=-1
    contributions.append({
        'U': {1 * 5 + j1: 1, 1 * 5 + j2: 1},
        'V': {j1 * 5 + k1: 1},
        'W': {k1 * 2 + 1: 1, k2 * 2 + 1: -1}
    })
    
    # M3: U: A[0,j1]=1; V: B[j1,k2]=1, B[j2,k2]=-1
    # W: C[0,k2]=1, C[1,k2]=1
    contributions.append({
        'U': {0 * 5 + j1: 1},
        'V': {j1 * 5 + k2: 1, j2 * 5 + k2: -1},
        'W': {k2 * 2 + 0: 1, k2 * 2 + 1: 1}
    })
    
    # M4: U: A[1,j2]=1; V: B[j2,k1]=1, B[j1,k1]=-1
    # W: C[0,k1]=1, C[1,k1]=1
    contributions.append({
        'U': {1 * 5 + j2: 1},
        'V': {j2 * 5 + k1: 1, j1 * 5 + k1: -1},
        'W': {k1 * 2 + 0: 1, k1 * 2 + 1: 1}
    })
    
    # M5: U: A[0,j1]=1, A[0,j2]=1; V: B[j2,k2]=1
    # W: C[0,k1]=-1, C[0,k2]=1
    contributions.append({
        'U': {0 * 5 + j1: 1, 0 * 5 + j2: 1},
        'V': {j2 * 5 + k2: 1},
        'W': {k1 * 2 + 0: -1, k2 * 2 + 0: 1}
    })
    
    # M6: U: A[1,j1]=1, A[0,j1]=-1; V: B[j1,k1]=1, B[j1,k2]=1
    # W: C[1,k2]=1
    contributions.append({
        'U': {1 * 5 + j1: 1, 0 * 5 + j1: -1},
        'V': {j1 * 5 + k1: 1, j1 * 5 + k2: 1},
        'W': {k2 * 2 + 1: 1}
    })
    
    # M7: U: A[0,j2]=1, A[1,j2]=-1; V: B[j2,k1]=1, B[j2,k2]=1
    # W: C[0,k1]=1
    contributions.append({
        'U': {0 * 5 + j2: 1, 1 * 5 + j2: -1},
        'V': {j2 * 5 + k1: 1, j2 * 5 + k2: 1},
        'W': {k1 * 2 + 0: 1}
    })
    
    return contributions

def _build_with_strassen_blocks(n, m, p, strassen_jk_pairs, deadline):
    """
    Build a decomposition where the specified (j,k) pairs use Strassen,
    and all other (j,k) pairs use schoolbook.
    
    strassen_jk_pairs: list of (j_pair, k_pair) tuples
    """
    start_time = time.time()
    
    # First, determine which (j,k) pairs are covered by Strassen blocks
    strassen_jk = set()
    for j_pair, k_pair in strassen_jk_pairs:
        for j in j_pair:
            for k in k_pair:
                strassen_jk.add((j, k))
    
    # Build U, V, W
    # Start with empty matrices
    # We'll collect all contributions and then assemble
    
    # For schoolbook terms (not in strassen_jk):
    schoolbook_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (j, k) not in strassen_jk:
                    schoolbook_terms.append((i, j, k))
    
    # For Strassen terms:
    strassen_terms = []
    for j_pair, k_pair in strassen_jk_pairs:
        contribs = _strassen_block(j_pair, k_pair)
        for c in contribs:
            strassen_terms.append(c)
    
    total_rank = len(schoolbook_terms) + len(strassen_terms)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    # Add schoolbook terms
    for i, j, k in schoolbook_terms:
        U[i * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + i, col] = 1
        col += 1
    
    # Add Strassen terms
    for c in strassen_terms:
        for idx, val in c['U'].items():
            U[idx, col] = val
        for idx, val in c['V'].items():
            V[idx, col] = val
        for idx, val in c['W'].items():
            W[idx, col] = val
        col += 1
    
    # Validate
    if _validate(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = best["U"].shape[1]
    
    # Try different combinations of Strassen blocks
    # A Strassen block covers 2 j-values and 2 k-values, saving 1 rank
    # We need to find compatible sets of blocks
    
    # All possible j-pairs: C(5,2) = 10
    # All possible k-pairs: C(5,2) = 10
    # Total possible blocks: 100
    
    # For each block, it covers 4 (j,k) pairs
    # Two blocks are compatible if they don't share any (j,k) pair
    
    # Let's try to find a maximum set of compatible blocks
    # This is a set packing problem
    
    j_pairs = list(itertools.combinations(range(m), 2))
    k_pairs = list(itertools.combinations(range(p), 2))
    
    all_blocks = [(jp, kp) for jp in j_pairs for kp in k_pairs]
    
    # Check compatibility: two blocks are compatible if their (j,k) sets are disjoint
    def blocks_compatible(block1, block2):
        jp1, kp1 = block1
        jp2, kp2 = block2
        set1 = set((j, k) for j in jp1 for k in kp1)
        set2 = set((j, k) for j in jp2 for k in kp2)
        return len(set1 & set2) == 0
    
    # Greedy: try to add as many blocks as possible
    # Start with all possible blocks, try different orderings
    
    best_solution = None
    best_rank = best["U"].shape[1]
    
    # Try greedy with different orderings
    for seed_order in range(20):
        if time.time() > deadline:
            break
        
        import random
        random.seed(seed_order)
        blocks_shuffled = all_blocks[:]
        random.shuffle(blocks_shuffled)
        
        selected = []
        for block in blocks_shuffled:
            if all(blocks_compatible(block, sel) for sel in selected):
                selected.append(block)
        
        # Try this selection
        if len(selected) > 0:
            result = _build_with_strassen_blocks(n, m, p, selected, deadline)
            if result is not None:
                rank = result["U"].shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_solution = result
    
    if best_solution is not None:
        return {"U": best_solution["U"].tolist(), "V": best_solution["V"].tolist(), "W": best_solution["W"].tolist()}
    
    # Return schoolbook
    return {"U": best["U"].tolist(), "V": best["V"].tolist(), "W": best["W"].tolist()}
