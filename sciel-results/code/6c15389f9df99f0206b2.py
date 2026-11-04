"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # Strassen patterns for 2x2x2 (7 rank-1 terms)
    strassen_patterns = [
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)]
        },
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)]
        },
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)]
        },
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)]
        },
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        }
    ]
    
    def get_splits(d):
        """Generate all compositions of d into parts of size 1 or 2."""
        if d == 0:
            yield []
        elif d == 1:
            yield [1]
        elif d == 2:
            yield [2]
            yield [1, 1]
        else:
            for rest in get_splits(d - 1):
                yield [1] + rest
            for rest in get_splits(d - 2):
                yield [2] + rest
    
    def compute_rank(n_splits, m_splits, p_splits):
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
    
    def build_decomposition(n_splits, m_splits, p_splits):
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
        
        U_entries = []
        V_entries = []
        W_entries = []
        rank = 0
        
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                    i_off = n_off[i_b]
                    j_off = m_off[j_b]
                    k_off = p_off[k_b]
                    
                    if nb == 2 and mb == 2 and pb == 2:
                        for pattern in strassen_patterns:
                            col = rank
                            for li, lj, val in pattern['U']:
                                U_entries.append(((i_off + li) * m + (j_off + lj), col, val))
                            for lj, lk, val in pattern['V']:
                                V_entries.append(((j_off + lj) * p + (k_off + lk), col, val))
                            for lk, li, val in pattern['W']:
                                W_entries.append(((k_off + lk) * n + (i_off + li), col, val))
                            rank += 1
                    else:
                        for li in range(nb):
                            for lj in range(mb):
                                for lk in range(pb):
                                    col = rank
                                    U_entries.append(((i_off + li) * m + (j_off + lj), col, 1))
                                    V_entries.append(((j_off + lj) * p + (k_off + lk), col, 1))
                                    W_entries.append(((k_off + lk) * n + (i_off + li), col, 1))
                                    rank += 1
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for idx, col, val in U_entries:
            U[idx][col] = val
        for idx, col, val in V_entries:
            V[idx][col] = val
        for idx, col, val in W_entries:
            W[idx][col] = val
        
        return {"U": U, "V": V, "W": W}
    
    # Generate all possible split combinations
    n_split_options = list(get_splits(n))
    m_split_options = list(get_splits(m))
    p_split_options = list(get_splits(p))
    
    # Find the split combination with minimum rank
    best_rank = float('inf')
    best_splits = None
    
    for ns in n_split_options:
        for ms in m_split_options:
            for ps in p_split_options:
                r = compute_rank(ns, ms, ps)
                if r < best_rank:
                    best_rank = r
                    best_splits = (ns, ms, ps)
    
    # Build the decomposition with the best splits
    result = build_decomposition(*best_splits)
    
    # Post-processing: try to reduce rank by eliminating linearly dependent terms
    U_np = np.array(result['U'], dtype=float)
    V_np = np.array(result['V'], dtype=float)
    W_np = np.array(result['W'], dtype=float)
    rank = U_np.shape[1]
    
    # Verify the decomposition is correct
    tensor_reconstructed = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor_reconstructed[i,j,k] += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
    target = np.ones((n, m, p), dtype=float)
    
    if not np.allclose(tensor_reconstructed, target, atol=1e-9):
        return result  # Return as-is if verification fails
    
    # Try to find and remove redundant terms
    # A term r is redundant if its contribution can be expressed as a linear combination
    # of other terms. We check this by seeing if the "slice" defined by term r lies
    # in the span of slices from other terms.
    
    # For each term, compute its contribution vector in the tensor space
    # Term r contributes: outer(U[:,r], V[:,r], W[:,r]) reshaped to (n*m*p,)
    # If this vector is in the span of the other terms' contributions, it's redundant.
    
    # Build the contribution matrix: each column is a term's flattened contribution
    # Shape: (n*m*p, rank)
    nmp = n * m * p
    contrib = np.zeros((nmp, rank), dtype=float)
    for r in range(rank):
        # Contribution of term r to each (i,j,k) entry
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contrib[i*m*p + j*p + k, r] = U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
    
    # Now check: is any column of contrib a linear combination of the others?
    # If so, that term is redundant and can be removed.
    # We do this iteratively: try removing each term and check if the remaining
    # terms still span the target tensor.
    
    # More efficient: find the rank of the contrib matrix. If rank < number of columns,
    # there are linear dependencies.
    # But we need to be careful: we need the remaining terms to still reconstruct the target.
    
    # Strategy: greedily remove terms that are linear combinations of others
    # while maintaining correctness.
    
    # Note: The QR decomposition line below was removed because standard NumPy qr 
    # does not return pivots in this manner, causing an unpacking error. 
    # The subsequent greedy algorithm using lstsq is used for reduction instead.
    
    # Actually, let's use a simpler approach: iteratively try to remove each term
    # and verify the remaining decomposition still works.
    
    # This is O(rank^2 * nmp) which for rank=48, nmp=75 is manageable.
    
    # Better approach: find a minimal subset of terms that still spans the target.
    # Use a greedy algorithm: start with all terms, try to remove each one.
    
    remaining = list(range(rank))
    improved = True
    while improved:
        improved = False
        for idx in range(len(remaining)):
            if time.time() > deadline - 5:
                break
            # Try removing term at position idx
            candidate = remaining[:idx] + remaining[idx+1:]
            if len(candidate) == 0:
                break
            
            # Check if the remaining terms still reconstruct the target
            # Build new contribution matrix
            new_contrib = contrib[:, candidate]
            # Check if target is in the column space of new_contrib
            # i.e., solve new_contrib @ x = target_flat for x
            target_flat = target.flatten()
            try:
                # Use least squares to check if target is in the span
                sol, residuals, rank_new, s = np.linalg.lstsq(new_contrib, target_flat, rcond=None)
                # Check if the reconstruction is accurate
                recon = new_contrib @ sol
                if np.allclose(recon, target_flat, atol=1e-8):
                    # We can remove this term!
                    remaining = candidate
                    improved = True
                    break
            except:
                pass
        if time.time() > deadline - 5:
            break
    
    if len(remaining) < rank:
        # Rebuild the decomposition with only the remaining terms
        new_rank = len(remaining)
        new_U = U_np[:, remaining]
        new_V = V_np[:, remaining]
        new_W = W_np[:, remaining]
        
        # Verify once more
        tensor_check = np.zeros((n, m, p), dtype=float)
        for r in range(new_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor_check[i,j,k] += new_U[i*m+j, r] * new_V[j*p+k, r] * new_W[k*n+i, r]
        
        if np.allclose(tensor_check, target, atol=1e-8):
            # Convert back to list format
            result['U'] = new_U.tolist()
            result['V'] = new_V.tolist()
            result['W'] = new_W.tolist()
    
    return result
# EVOLVE-BLOCK-END
