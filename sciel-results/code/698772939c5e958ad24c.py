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
    
    # Now try to reduce rank further by finding linear dependencies
    # Convert to numpy for efficient linear algebra
    U_np = np.array(result['U'], dtype=float)
    V_np = np.array(result['V'], dtype=float)
    W_np = np.array(result['W'], dtype=float)
    
    num_rows_u = U_np.shape[0]  # n*m = 15
    num_rows_v = V_np.shape[0]  # m*p = 25
    num_rows_w = W_np.shape[0]  # p*n = 15
    rank = U_np.shape[1]
    
    # The tensor T has entries T[(i,j), j', (k,i')] where the multiplication
    # tensor is: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Actually, the matrix multiplication tensor T_{ijk} = 1 for all i,j,k
    # The decomposition says: T_{ijk} = sum_r U_{ij,r} * V_{jk,r} * W_{ki,r}
    
    # Verify current decomposition
    # Build the tensor from U, V, W
    tensor_reconstructed = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor_reconstructed[i,j,k] += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
    
    # The target tensor is all ones
    target = np.ones((n, m, p), dtype=float)
    
    # Try to find a lower rank decomposition by searching for rank-1 terms
    # that can replace multiple existing terms
    
    # Strategy: try to find a set of rank-1 terms (sparse patterns) that sum to the target tensor
    # Use a greedy approach: start with the current decomposition, try to eliminate terms
    
    # Alternative approach: use the fact that for small matrices, we can try
    # specific known decompositions
    
    # For 3x5x5, let's try a different structural approach:
    # Split the 5x5 part into 2+2+1 or similar, and use Strassen where possible
    
    # Try all split combinations more carefully, including mixed approaches
    
    # Let's try a completely different approach: enumerate possible rank-1
    # term patterns and use a covering/set-cover style search
    
    # A rank-1 term in the matrix multiplication tensor corresponds to:
    # U_factor: a matrix of shape (n*m, 1) = vector u of length n*m
    # V_factor: a matrix of shape (m*p, 1) = vector v of length m*p  
    # W_factor: a matrix of shape (p*n, 1) = vector w of length p*n
    # Such that u⊗v⊗w contributes to the tensor
    
    # For the standard basis, each rank-1 term picks specific (i,j) for U,
    # (j,k) for V, and (k,i) for W, but they must be consistent.
    
    # Let's try a smarter approach: use the current best decomposition and
    # try to find linear combinations of columns that can be zeroed out
    
    # Check if any column can be expressed as a combination of others
    # This is expensive, so let's try a simpler heuristic
    
    # Try to find pairs of columns that are linearly dependent
    # and remove one
    
    U_cols = U_np.T  # rank x (n*m)
    V_cols = V_np.T  # rank x (m*p)
    W_cols = W_np.T  # rank x (p*n)
    
    # For each pair of columns, check if they are linearly dependent
    # by checking if the outer product structure allows cancellation
    
    # This is complex. Let's instead try a different structural decomposition.
    
    # New approach: use 3x3 Strassen-like decompositions if available
    # or try to find a better split
    
    # Actually, let's try to use the fact that we can have non-integer
    # coefficients and search more broadly
    
    # For now, return the best split-based decomposition
    return result
# EVOLVE-BLOCK-END
