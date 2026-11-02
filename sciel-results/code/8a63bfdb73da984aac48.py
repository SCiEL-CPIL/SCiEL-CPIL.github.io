"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook(n, m, p):
    """Build the standard n*m*p rank decomposition for (n,m) x (m,p) -> (p,n) output."""
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

def _verify(decomp, n, m, p):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta_{j,k} (i.e., identity in m)."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                expected = 1 if j == k else 0
                if abs(s - expected) > 1e-9:
                    return False
    return True

def _find_small_decomp(n, m, p, target_rank, max_attempts=100):
    """Try to find a rank decomposition for n x m x p using random search."""
    for _ in range(max_attempts):
        U = np.random.randint(-2, 3, size=(n * m, target_rank)) * 0.5
        V = np.random.randint(-2, 3, size=(m * p, target_rank)) * 0.5
        W = np.random.randint(-2, 3, size=(p * n, target_rank)) * 0.5
        # Ensure some non-zeros
        if U.size > 0:
            U[np.random.choice(U.size, size=min(target_rank, U.size), replace=False)] = 1
        if V.size > 0:
            V[np.random.choice(V.size, size=min(target_rank, V.size), replace=False)] = 1
        if W.size > 0:
            W[np.random.choice(W.size, size=min(target_rank, W.size), replace=False)] = 1
        
        decomp = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        if _verify(decomp, n, m, p):
            return decomp
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0
    
    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Strategy: Exploit p=9 as 3 blocks of 3.
    # Decompose 2x3x3, then replicate for the 3 blocks.
    # Total rank = 3 * rank_small.
    # Lower bound for 2x3x3 is 9.
    
    n_small = 2
    m_small = 3
    p_small = 3
    
    # Try ranks from lower bound (9) up to 18 (schoolbook for small)
    for r_small in range(9, 19):
        if time.time() > deadline - 20:
            break
        
        small_decomp = _find_small_decomp(n_small, m_small, p_small, r_small, max_attempts=50)
        if small_decomp:
            # Construct full decomp by replication
            rank = 3 * r_small
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # U is same for all blocks (A is shared)
            # U[i*m+j, block*q + c] = small_U[i*m+j, c]
            for i in range(n):
                for j in range(m):
                    for q in range(3):
                        for c in range(r_small):
                            U[i * m + j][q * r_small + c] = small_decomp["U"][i * m + j][c]
            
            # V is block specific
            # V[j*p+k, block*q + c] = small_V[j*p_small+k, c] where k is local col
            for j in range(m):
                for q in range(3):
                    for k_local in range(p_small):
                        k_global = q * p_small + k_local
                        for c in range(r_small):
                            V[j * p + k_global][q * r_small + c] = small_decomp["V"][j * p_small + k_local][c]
            
            # W is block specific
            # W[k*n+i, block*q + c] = small_W[k_local*n+i, c] where k is local row
            for i in range(n):
                for q in range(3):
                    for k_local in range(p_small):
                        k_global = q * p_small + k_local
                        for c in range(r_small):
                            W[k_global * n + i][q * r_small + c] = small_decomp["W"][k_local * n + i][c]
            
            decomp = {"U": U, "V": V, "W": W}
            if _verify(decomp, n, m, p):
                if rank < best_rank:
                    best = decomp
                    best_rank = rank
                return best

    # Fallback to schoolbook
    return best
# EVOLVE-BLOCK-END
