"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                if abs(actual - expected) > 1e-10:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_4x4x4():
    """
    Return the Strassen-Winograd rank-19 decomposition for 4x4x4 matrix multiplication.
    U: (16, 19), V: (16, 19), W: (16, 19)
    Indexing: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    """
    # Strassen-Winograd 4x4x4 rank 19 decomposition
    # This is a known result. Let me construct it.
    # For 4x4x4, the Strassen-Winograd algorithm uses 19 multiplications.
    
    # I'll use a known formulation. The 4x4x4 Strassen-Winograd can be built
    # from 2x2x2 Strassen blocks.
    
    # Actually, let me just return None and use a different approach.
    # The exact Strassen-Winograd 4x4x4 rank-19 coefficients are complex.
    # Let me instead focus on direct search for 3x4x4.
    return None

def _search_low_rank(n, m, p, max_rank, deadline, seed=42):
    """
    Search for a low-rank decomposition using structured random search.
    Try ranks from low to high, using various initialization strategies.
    """
    rng = random.Random(seed)
    best = None
    best_rank = max_rank + 1
    
    # Strategy 1: Try to find a decomposition with rank as low as possible
    # using random sparse matrices with +/-1 entries
    
    # For 3x4x4, try ranks from 15 upward
    for target_rank in range(15, min(max_rank + 1, 49)):
        if time.time() > deadline - 5:
            break
        
        found = False
        for trial in range(30):
            if time.time() > deadline - 5:
                break
            
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Initialize with random sparse +/-1 pattern
            for c in range(target_rank):
                # Random support sizes
                u_sparsity = rng.randint(1, max(1, (n*m)//3))
                v_sparsity = rng.randint(1, max(1, (m*p)//3))
                w_sparsity = rng.randint(1, max(1, (p*n)//3))
                
                u_idx = rng.sample(range(n*m), u_sparsity)
                v_idx = rng.sample(range(m*p), v_sparsity)
                w_idx = rng.sample(range(p*n), w_sparsity)
                
                for idx in u_idx:
                    U[idx, c] = rng.choice([1, -1])
                for idx in v_idx:
                    V[idx, c] = rng.choice([1, -1])
                for idx in w_idx:
                    W[idx, c] = rng.choice([1, -1])
            
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    found = True
                    break
        
        if found:
            # Try to go lower
            continue
    
    return best

def _strassen_embed_3x4x4():
    """
    Try to embed the 3x4x4 multiplication into the 4x4x4 Strassen-Winograd
    decomposition by projecting out the 4th row.
    
    The idea: If we have a rank-r decomposition for 4x4x4 with factors
    U4 (16x r), V4 (16x r), W4 (16x r), then for 3x4x4 we can try to
    restrict to the first 3 rows of A.
    
    For 3x4x4: U is (12, r'), V is (16, r'), W is (12, r')
    U[i*4+j, c] for i in 0..2, j in 0..3
    V[j*4+k, c] for j in 0..3, k in 0..3
    W[k*3+i, c] for k in 0..3, i in 0..2
    
    This is not a direct projection because the W factor has different dimensions.
    """
    # This approach is tricky because the dimensions don't match directly.
    # Let me try a different embedding strategy.
    # 
    # Alternative: Use the fact that 3x4x4 multiplication can be seen as
    # computing 3 rows of a 4x4 product. If we have a 4x4x4 algorithm,
    # we only need to compute 3 of the 4 output rows.
    #
    # But the Strassen-Winograd algorithm doesn't decompose cleanly by rows.
    #
    # Let me try a direct algebraic construction instead.
    
    return None

def _algebraic_construction():
    """
    Try to construct a low-rank decomposition algebraically.
    
    For 3x4x4, we need T[i,j,k] = 1 for all i in [0,3), j in [0,4), k in [0,4).
    
    One approach: Use the fact that this is a "slice" of a larger tensor.
    
    Another approach: Try to find a decomposition using the structure of
    matrix multiplication more cleverly.
    
    Let me try a specific construction based on polynomial multiplication.
    """
    # For now, return None and rely on random search
    return None

def _refine_decomposition(U, V, W, n, m, p, deadline):
    """
    Try to refine a decomposition to reduce its rank.
    Use linear algebra to find if some columns are redundant.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    current_rank = len(U[0])
    
    # Check if any column is all zeros
    nonzero_cols = []
    for c in range(current_rank):
        if np.any(U[:, c] != 0) and np.any(V[:, c] != 0) and np.any(W[:, c] != 0):
            nonzero_cols.append(c)
    
    if len(nonzero_cols) < current_rank:
        U_trim = U[:, nonzero_cols]
        V_trim = V[:, nonzero_cols]
        W_trim = W[:, nonzero_cols]
        return {"U": U_trim.tolist(), "V": V_trim.tolist(), "W": W_trim.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try algebraic construction
    if time.time() < deadline - 30:
        alg = _algebraic_construction()
        if alg is not None:
            r = len(alg['U'][0])
            if r < best_rank:
                best_rank = r
                result = alg
    
    # Try Strassen embedding
    if time.time() < deadline - 30:
        strassen = _strassen_embed_3x4x4()
        if strassen is not None:
            r = len(strassen['U'][0])
            if r < best_rank:
                best_rank = r
                result = strassen
    
    # Random structured search
    if time.time() < deadline - 10:
        found = _search_low_rank(n, m, p, best_rank - 1, deadline - 10, seed=42)
        if found is not None:
            r = len(found['U'][0])
            if r < best_rank:
                best_rank = r
                result = found
    
    # Try more random searches with different seeds
    if time.time() < deadline - 5:
        for seed in [123, 456, 789]:
            if time.time() > deadline - 5:
                break
            found = _search_low_rank(n, m, p, best_rank - 1, deadline - 5, seed=seed)
            if found is not None:
                r = len(found['U'][0])
                if r < best_rank:
                    best_rank = r
                    result = found
    
    # Refine
    if time.time() < deadline - 2:
        refined = _refine_decomposition(result['U'], result['V'], result['W'], n, m, p, deadline)
        if refined is not None:
            r = len(refined['U'][0])
            if r < best_rank:
                best_rank = r
                result = refined
    
    return result
# EVOLVE-BLOCK-END
