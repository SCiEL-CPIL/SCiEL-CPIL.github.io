"""Matrix multiplication 2,5,6 tensor decomposition using block Strassen with optimal partition search."""
import time

DIMENSIONS = [2, 5, 6]

def _strassen_terms():
    """Return the 7 Strassen terms for 2x2x2 multiplication."""
    return [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [0,0,1,-1], [-1,1,0,0]),
        ([1,1,0,0], [0,0,0,1], [1,0,1,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,-1]),
        ([0,1,0,-1], [0,0,1,1], [-1,0,0,1]),
    ]

def _verify_strassen(terms):
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0
                for U, V, W in terms:
                    total += U[i] * V[j] * W[k]
                if total != 1:
                    return False
    return True

def _generate_partitions(n):
    """Generate all partitions of range(n) into blocks of size 1 or 2."""
    if n == 0:
        yield []
        return
    # Try block of size 1
    for rest in _generate_partitions(n - 1):
        yield [[0]] + [[x + 1 for x in block] for block in rest]
    # Try block of size 2
    if n >= 2:
        for rest in _generate_partitions(n - 2):
            yield [[0, 1]] + [[x + 2 for x in block] for block in rest]

def _block_cost(j_size, k_size, n):
    """Cost (rank contribution) for a sub-block of size j_size x k_size."""
    if j_size == 2 and k_size == 2:
        return 7
    elif j_size == 2 and k_size == 1:
        return 2
    elif j_size == 1 and k_size == 2:
        return 2
    elif j_size == 1 and k_size == 1:
        return n
    else:
        return j_size * k_size * n

def _find_best_partition(n, m, p, deadline):
    """Find the partition of j and k dimensions that minimizes total rank."""
    best_rank = n * m * p
    best_j_part = None
    best_k_part = None
    
    j_partitions = list(_generate_partitions(m))
    k_partitions = list(_generate_partitions(p))
    
    for jp in j_partitions:
        for kp in k_partitions:
            rank = 0
            for j_block in jp:
                for k_block in kp:
                    rank += _block_cost(len(j_block), len(k_block), n)
            if rank < best_rank:
                best_rank = rank
                best_j_part = jp
                best_k_part = kp
            if time.time() > deadline:
                return best_rank, best_j_part, best_k_part
    
    return best_rank, best_j_part, best_k_part

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    strassen = _strassen_terms()
    if not _verify_strassen(strassen):
        # Fallback to schoolbook
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
    
    # Find best partition
    best_rank, best_j_part, best_k_part = _find_best_partition(n, m, p, DEADLINE)
    
    # If no valid partition found, use schoolbook
    if best_j_part is None or best_k_part is None:
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
    
    # Build decomposition using best partition
    all_terms = []
    
    for j_block in best_j_part:
        for k_block in best_k_part:
            j_size = len(j_block)
            k_size = len(k_block)
            
            if j_size == 2 and k_size == 2:
                j0, j1 = j_block[0], j_block[1]
                k0, k1 = k_block[0], k_block[1]
                for U_flat, V_flat, W_flat in strassen:
                    U_sparse = {}
                    V_sparse = {}
                    W_sparse = {}
                    for idx, val in enumerate(U_flat):
                        if val != 0:
                            i = idx // 2
                            j_local = idx % 2
                            j = j0 if j_local == 0 else j1
                            U_sparse[i * m + j] = val
                    for idx, val in enumerate(V_flat):
                        if val != 0:
                            j_local = idx // 2
                            k_local = idx % 2
                            j = j0 if j_local == 0 else j1
                            k = k0 if k_local == 0 else k1
                            V_sparse[j * p + k] = val
                    for idx, val in enumerate(W_flat):
                        if val != 0:
                            k_local = idx // 2
                            i = idx % 2
                            k = k0 if k_local == 0 else k1
                            W_sparse[k * n + i] = val
                    all_terms.append((U_sparse, V_sparse, W_sparse))
            elif j_size == 2 and k_size == 1:
                j0, j1 = j_block[0], j_block[1]
                k0 = k_block[0]
                for j in [j0, j1]:
                    U_sparse = {i * m + j: 1 for i in range(n)}
                    V_sparse = {j * p + k0: 1}
                    W_sparse = {k0 * n + i: 1 for i in range(n)}
                    all_terms.append((U_sparse, V_sparse, W_sparse))
            elif j_size == 1 and k_size == 2:
                j0 = j_block[0]
                k0, k1 = k_block[0], k_block[1]
                for k in [k0, k1]:
                    U_sparse = {i * m + j0: 1 for i in range(n)}
                    V_sparse = {j0 * p + k: 1}
                    W_sparse = {k * n + i: 1 for i in range(n)}
                    all_terms.append((U_sparse, V_sparse, W_sparse))
            else:
                j0 = j_block[0]
                k0 = k_block[0]
                U_sparse = {i * m + j0: 1 for i in range(n)}
                V_sparse = {j0 * p + k0: 1}
                W_sparse = {k0 * n + i: 1 for i in range(n)}
                all_terms.append((U_sparse, V_sparse, W_sparse))
    
    rank = len(all_terms)
    
    # Check time
    if time.time() - start_time > DEADLINE:
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
    
    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (U_sparse, V_sparse, W_sparse) in enumerate(all_terms):
        for idx, val in U_sparse.items():
            U[idx][col] = val
        for idx, val in V_sparse.items():
            V[idx][col] = val
        for idx, val in W_sparse.items():
            W[idx][col] = val
    
    return {"U": U, "V": V, "W": W}
