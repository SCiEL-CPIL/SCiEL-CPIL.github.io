"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]
N, M, P = DIMENSIONS

# EVOLVE-BLOCK-START
def _verify(U, V, W, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    # U: (N*M, rank), V: (M*P, rank), W: (P*N, rank)
    n, m, p = N, M, P
    rank = U.shape[1]
    # For each (i, j, k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    # Also check off-diagonal terms: for any (i, j, k) not matching the (i', j', k') pattern
    # Actually, the tensor entry T[i,j,k] = 1 iff the indices match the multiplication pattern
    # Let me re-check the tensor definition.
    # The bilinear form: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # T_{i,j,k} = 1 for all (i,j,k) since C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all (i,j,k)
    # And T should be 0 elsewhere. But the decomposition only needs to match the 1s.
    # The schoolbook gives exactly that. For a lower-rank decomposition, we need
    # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all (i,j,k).
    # There are no other constraints - the tensor only has 1s at those positions.
    # Wait, that's not right. The tensor is defined as T[a,b,c] where a indexes A, b indexes B, c indexes C.
    # T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k) for some valid i,j,k, else 0.
    # So for the decomposition: sum_c U[a,c] V[b,c] W[c_c, c] = T[a, b, c_c]
    # This must hold for ALL a, b, c_c, not just the ones where T=1.
    # So we need to check ALL (a, b, c_c) triples.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_idx = k * n + i
                u_val = U[a]
                v_val = V[b]
                w_val = W[c_idx]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    # Check that off-diagonal entries are zero
    for a in range(n * m):
        for b in range(m * p):
            for c_idx in range(p * n):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c_idx // n
                i2 = c_idx % n
                if not (j == j2 and i == i2 and k == k2):
                    u_val = U[a]
                    v_val = V[b]
                    w_val = W[c_idx]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s) > tol:
                        return False
    return True

def _try_rank(rank, max_iter=500, time_limit=140.0):
    """Try to find a rank decomposition of the given rank using random search."""
    start_time = time.time()
    n, m, p = N, M, P
    nm, mp, pn = n * m, m * p, p * n
    
    # Initialize with random integer values
    best_score = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization with small integers
        U = np.random.randint(-2, 3, size=(nm, rank)).astype(float)
        V = np.random.randint(-2, 3, size=(mp, rank)).astype(float)
        W = np.random.randint(-2, 3, size=(pn, rank)).astype(float)
        
        # Try a few local improvement steps
        for _ in range(5):
            if time.time() - start_time > time_limit:
                break
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c_idx = k * n + i
                        s = np.dot(U[a], V[b] * W[c_idx])
                        err += (s - 1.0) ** 2
            # Off-diagonal errors
            for a in range(nm):
                for b in range(mp):
                    for c_idx in range(pn):
                        i = a // m
                        j = a % m
                        j2 = b // p
                        k = b % p
                        k2 = c_idx // n
                        i2 = c_idx % n
                        if not (j == j2 and i == i2 and k == k2):
                            s = np.dot(U[a], V[b] * W[c_idx])
                            err += s ** 2
            
            if err < best_score:
                best_score = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                if err < 1e-6:
                    return U.copy(), V.copy(), W.copy()
            
            # Local perturbation
            if np.random.random() < 0.5:
                idx = np.random.randint(0, nm)
                col = np.random.randint(0, rank)
                U[idx, col] += np.random.randint(-1, 2)
            else:
                idx = np.random.randint(0, mp)
                col = np.random.randint(0, rank)
                V[idx, col] += np.random.randint(-1, 2)
            if np.random.random() < 0.5:
                idx = np.random.randint(0, pn)
                col = np.random.randint(0, rank)
                W[idx, col] += np.random.randint(-1, 2)
            else:
                # Try adjusting a random entry in U or V
                mat = np.random.choice([U, V])
                idx = np.random.randint(0, mat.shape[0])
                col = np.random.randint(0, mat.shape[1])
                mat[idx, col] += np.random.randint(-1, 2)
    
    if best_U is not None and best_score < 1e-6:
        return best_U, best_V, best_W
    return None

def _build_block_decomp(n, m, p, rank_nmp):
    """Build a rank-nmp schoolbook decomposition for M_{n,m,p}."""
    U = np.zeros((n * m, rank_nmp))
    V = np.zeros((m * p, rank_nmp))
    W = np.zeros((p * n, rank_nmp))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_from_blocks(n, m, p_blocks, block_ranks, block_decomps):
    """Build M_{n,m,p} decomposition from block structure M_{n,m,p} = direct sum over blocks.
    
    p = sum of block sizes. Each block is an m x p_b matrix.
    The decomposition is the block-diagonal combination.
    """
    total_rank = sum(block_ranks)
    nm = n * m
    mp = m * p_blocks
    pn = p_blocks * n
    
    U_full = np.zeros((nm, total_rank))
    V_full = np.zeros((mp, total_rank))
    W_full = np.zeros((pn, total_rank))
    
    col_offset = 0
    for bi, (pb, rb, (Ub, Vb, Wb)) in enumerate(zip([b for b in [p_blocks]] if False else [], [], [])):
        pass
    
    # Simpler: just do it directly
    # The blocks partition the p dimension.
    # For M_{n,m,p}, A is n x m, B is m x p.
    # B can be split into m x p_1, m x p_2, ..., m x p_t blocks.
    # C = A * B = A * [B_1 | B_2 | ... | B_t] = [A*B_1 | A*B_2 | ... | A*B_t]
    # Each A*B_b is n x p_b.
    # The tensor for M_{n,m,p} is the concatenation of tensors for M_{n,m,p_b}.
    # So the decomposition is: U is the same for all blocks (since A is the same),
    # V is block-concatenated, W is block-concatenated.
    
    # Actually for the tensor decomposition:
    # T_{n,m,p}[a,b,c] = sum_c_idx U[a, c_idx] * V[b, c_idx] * W[c, c_idx]
    # where a indexes (i,j), b indexes (j,k), c indexes (i,k).
    # When we split p into blocks, b = (j, k) where k ranges over 1..p.
    # For block b_i with p_i columns, the k values are offset.
    # U is the same across all blocks (n*m rows, total_rank cols).
    # V has m*p rows: for block b_i, rows (j, k_i) where k_i is within block i.
    # W has p*n rows: for block b_i, rows (k_i, i_row).
    
    # Let's just construct it properly.
    U_full = np.zeros((nm, total_rank))
    V_full = np.zeros((m * p_blocks, total_rank))
    W_full = np.zeros((p_blocks * n, total_rank))
    
    col_offset = 0
    k_offset = 0
    for bi in range(len(block_ranks)):
        pb = p_blocks  # This doesn't work for multiple blocks of different sizes
        pass
    
    return None  # This approach is getting complicated; use a different strategy

def construct():
    start_time = time.time()
    n, m, p = N, M, P
    
    # Strategy 1: Try to find a low-rank decomposition via random search
    # Start from schoolbook rank n*m*p = 54
    # Try ranks from low to high
    
    # First, build the schoolbook solution as fallback
    schoolbook_rank = n * m * p
    U_sb, V_sb, W_sb = _build_block_decomp(n, m, p, schoolbook_rank)
    
    # Try random search for lower ranks
    # Theoretical lower bound: rank >= n*m*p / (something)
    # For M_{n,m,p}, rank >= max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Also rank >= n*m*p / min(n,m,p) = 54/2 = 27
    
    # Try a few candidate ranks
    candidate_ranks = [27, 28, 29, 30, 32, 35, 38, 40, 45, 50, 54]
    
    for rank in candidate_ranks:
        if time.time() - start_time > 130:
            break
        result = _try_rank(rank, max_iter=200, time_limit=120 - (time.time() - start_time))
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, tol=1e-6):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END

