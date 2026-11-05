"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _schoolbook(n=3, m=5, p=6):
    """Standard rank n*m*p decomposition."""
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


def _strassen_2x2x2():
    """Rank-7 Strassen decomposition for 2x2 matrix multiplication."""
    M1_U = [1, 0, 0, 1]; M1_V = [1, 0, 0, 1]; M1_W = [1, 1, 1, 1]
    M2_U = [0, 0, 1, 1]; M2_V = [1, 0, 0, 0]; M2_W = [0, 0, 1, 0]
    M3_U = [1, 0, 0, 0]; M3_V = [0, 1, 0, -1]; M3_W = [0, 1, 0, 0]
    M4_U = [0, 0, 0, 1]; M4_V = [1, 0, 1, 0]; M4_W = [0, 0, 1, 0]
    M5_U = [1, 1, 0, 0]; M5_V = [0, 0, 0, 1]; M5_W = [0, 0, 0, 1]
    M6_U = [1, 0, 1, 0]; M6_V = [1, 1, 0, 0]; M6_W = [1, 0, 0, 0]
    M7_U = [0, 1, 0, 1]; M7_V = [0, 0, 1, 1]; M7_W = [0, 0, 0, 1]

    U = [[M1_U[t], M2_U[t], M3_U[t], M4_U[t], M5_U[t], M6_U[t], M7_U[t]] for t in range(4)]
    V = [[M1_V[t], M2_V[t], M3_V[t], M4_V[t], M5_V[t], M6_V[t], M7_V[t]] for t in range(4)]
    W = [[M1_W[t], M2_W[t], M3_W[t], M4_W[t], M5_W[t], M6_W[t], M7_W[t]] for t in range(4)]
    return U, V, W


def _strassen_block_decomposition(n=3, m=5, p=6):
    """Block Strassen: rank 84 for 3x5x6."""
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    sU, sV, sW = _strassen_2x2x2()
    col = 0

    # 6 Strassen 2x2x2 blocks for i in {0,1}, j in {0,1}/{2,3}, k in {0,1}/{2,3}/{4,5}
    for i_off in [0, 0, 0, 0, 0, 0]:
        for j_off in [0, 0, 0, 2, 2, 2]:
            for k_off in [0, 2, 4, 0, 2, 4]:
                for t in range(7):
                    for li in range(2):
                        for lj in range(2):
                            for lk in range(2):
                                gi = li
                                gj = j_off + lj
                                gk = k_off + lk
                                U[gi * m + gj][col] += sU[li * 2 + lj][t]
                                V[gj * p + gk][col] += sV[lj * 2 + lk][t]
                                W[gk * n + gi][col] += sW[lk * 2 + li][t]
                    col += 1

    # Schoolbook for remaining parts
    # i in {0,1}, j=4, all k
    for i in range(2):
        for k in range(p):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # i=2, all j, all k
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n=3, m=5, p=6):
    """Check exact decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for t in range(rank):
                    val += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                if val != 1:
                    return False
    return True


def _compute_residual(U, V, W, n=3, m=5, p=6):
    """Compute sum of squared residuals (should be 0 for valid)."""
    rank = len(U[0])
    total = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for t in range(rank):
                    val += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                total += (val - 1) ** 2
    return total


def _random_int(low, high):
    return random.randint(low, high)


def _two_column_search(U, V, W, n, m, p, deadline, max_iters=20000):
    """
    Local search: try to replace pairs of columns with new integer columns
    that reduce the residual. This is the 'error_repair' mechanism.
    """
    rank = len(U[0])
    best_res = _compute_residual(U, V, W, n, m, p)
    
    if best_res == 0:
        return U, V, W, best_res

    nU = n * m
    nV = m * p
    nW = p * n
    
    # Precompute for speed: for each column t, store the vectors
    # Actually, let's just do direct computation but with early termination
    
    iters = 0
    while iters < max_iters:
        if time.time() > deadline:
            break
        iters += 1
        
        # Pick two random columns to replace
        t1 = random.randint(0, rank - 1)
        t2 = random.randint(0, rank - 1)
        if t1 == t2:
            continue
        
        # Compute contribution of columns t1 and t2
        # For each (i,j,k), the contribution is:
        # U[i*m+j][t1]*V[j*p+k][t1]*W[k*n+i][t1] + U[i*m+j][t2]*V[j*p+k][t2]*W[k*n+i][t2]
        # We want to find new columns u', v', w' (integers) such that
        # u'[i*m+j]*v'[j*p+k]*w'[k*n+i] approximates the target residual contribution
        
        # Strategy: try random small integer vectors for u', v', w'
        # and check if they reduce the total residual.
        
        # To make this tractable, we'll try a few random candidates per iteration.
        # For each candidate, compute the new residual only for affected (i,j,k)
        # where the old columns had non-zero entries.
        
        # Collect affected indices
        affected_U = set()
        for idx in range(nU):
            if U[idx][t1] != 0 or U[idx][t2] != 0:
                affected_U.add(idx)
        affected_V = set()
        for idx in range(nV):
            if V[idx][t1] != 0 or V[idx][t2] != 0:
                affected_V.add(idx)
        affected_W = set()
        for idx in range(nW):
            if W[idx][t1] != 0 or W[idx][t2] != 0:
                affected_W.add(idx)
        
        if not affected_U or not affected_V or not affected_W:
            continue
        
        # Try a few random replacements
        found_improvement = False
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Generate random integer vectors
            new_u = [_random_int(-2, 2) for _ in range(nU)]
            new_v = [_random_int(-2, 2) for _ in range(nV)]
            new_w = [_random_int(-2, 2) for _ in range(nW)]
            
            # Compute old contribution for affected (i,j,k)
            old_contrib = {}
            for idx_u in affected_U:
                i = idx_u // m
                j = idx_u % m
                for idx_v in affected_V:
                    if idx_v // p != j:
                        continue
                    k = idx_v % p
                    idx_w = k * n + i
                    if idx_w not in affected_W:
                        continue
                    old_val = (U[idx_u][t1] * V[idx_v][t1] * W[idx_w][t1] +
                              U[idx_u][t2] * V[idx_v][t2] * W[idx_w][t2])
                    new_val = new_u[idx_u] * new_v[idx_v] * new_w[idx_w]
                    old_contrib[(i, j, k)] = old_val
                    new_contrib_val = new_val
            
            # This is getting complex. Let's simplify: just compute full residual
            # for a small number of candidates.
            pass
        
        # Simplified approach: just try to zero out one specific residual
        # by adjusting one column.
        # Find the (i,j,k) with largest residual
        max_res = 0
        max_ijk = None
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for t in range(rank):
                        val += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    res = abs(val - 1)
                    if res > max_res:
                        max_res = res
                        max_ijk = (i, j, k)
        
        if max_res == 0:
            best_res = 0
            return U, V, W, best_res
        
        i, j, k = max_ijk
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        
        # Try to fix this by modifying one column
        # We need U[idx_u][t] * V[idx_v][t] * W[idx_w][t] to contribute
        # to make the sum equal 1.
        
        # Find a column where we can make a change
        for t in random.sample(range(rank), min(10, rank)):
            if time.time() > deadline:
                break
            # Try setting U[idx_u][t], V[idx_v][t], W[idx_w][t] to small integers
            # such that the product helps fix the residual.
            current_val = 0
            for tt in range(rank):
                current_val += U[idx_u][tt] * V[idx_v][tt] * W[idx_w][tt]
            needed = 1 - current_val
            
            if needed == 0:
                continue
            
            # Try to set one of the three to make the product equal needed
            # If we set U[idx_u][t] = needed, V[idx_v][t] = 1, W[idx_w][t] = 1
            # But this might break other (i',j',k') terms.
            # For now, just try it and check if it improves.
            
            old_u = U[idx_u][t]
            old_v = V[idx_v][t]
            old_w = W[idx_w][t]
            
            # Try a few options
            for nu in [needed, needed // 2 if needed % 2 == 0 else needed]:
                for nv in [1, -1]:
                    for nw in [needed // (nu * nv) if nu * nv != 0 and needed % (nu * nv) == 0 else 1]:
                        if nu == 0 or nv == 0 or nw == 0:
                            continue
                        if nu * nv * nw != needed:
                            continue
                        # Check if this improves the residual
                        U[idx_u][t] = nu
                        V[idx_v][t] = nv
                        W[idx_w][t] = nw
                        new_res = _compute_residual(U, V, W, n, m, p)
                        if new_res < best_res:
                            best_res = new_res
                            found_improvement = True
                        else:
                            U[idx_u][t] = old_u
                            V[idx_v][t] = old_v
                            W[idx_w][t] = old_w
        
        if not found_improvement:
            # Try a different strategy: replace two columns with new random ones
            t1 = random.randint(0, rank - 1)
            t2 = random.randint(0, rank - 1)
            if t1 != t2:
                old_u = [U[idx][t1] for idx in range(nU)] + [U[idx][t2] for idx in range(nU)]
                old_v = [V[idx][t1] for idx in range(nV)] + [V[idx][t2] for idx in range(nV)]
                old_w = [W[idx][t1] for idx in range(nW)] + [W[idx][t2] for idx in range(nW)]
                
                # Try random replacements
                for attempt in range(5):
                    if time.time() > deadline:
                        break
                    new_u = [_random_int(-2, 2) for _ in range(nU)]
                    new_v = [_random_int(-2, 2) for _ in range(nV)]
                    new_w = [_random_int(-2, 2) for _ in range(nW)]
                    
                    for idx in range(nU):
                        U[idx][t1] = new_u[idx]
                    for idx in range(nV):
                        V[idx][t1] = new_v[idx]
                    for idx in range(nW):
                        W[idx][t1] = new_w[idx]
                    
                    new_res = _compute_residual(U, V, W, n, m, p)
                    if new_res < best_res:
                        best_res = new_res
                        # Keep the change, now try to fix t2
                        for idx in range(nU):
                            U[idx][t2] = _random_int(-2, 2)
                        for idx in range(nV):
                            V[idx][t2] = _random_int(-2, 2)
                        for idx in range(nW):
                            W[idx][t2] = _random_int(-2, 2)
                        new_res = _compute_residual(U, V, W, n, m, p)
                        if new_res < best_res:
                            best_res = new_res
                    else:
                        # Revert
                        for idx in range(nU):
                            U[idx][t1] = old_u[idx]
                        for idx in range(nV):
                            V[idx][t1] = old_v[idx]
                        for idx in range(nW):
                            W[idx][t1] = old_w[idx]
    
    return U, V, W, best_res


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    # Start with block Strassen (rank 84)
    decomp = _strassen_block_decomposition(n, m, p)
    best_rank = len(decomp["U"][0])
    
    if _verify(decomp["U"], decomp["V"], decomp["W"], n, m, p):
        # Try to improve via local search
        U = [row[:] for row in decomp["U"]]
        V = [row[:] for row in decomp["V"]]
        W = [row[:] for row in decomp["W"]]
        
        U, V, W, res = _two_column_search(U, V, W, n, m, p, deadline, max_iters=5000)
        
        if res == 0:
            # Check if we can reduce rank by removing zero columns
            new_U = []
            new_V = []
            new_W = []
            for t in range(len(U[0])):
                if any(U[i][t] != 0 for i in range(len(U))):
                    new_U.append([U[i][t] for i in range(len(U))])
                    new_V.append([V[i][t] for i in range(len(V))])
                    new_W.append([W[i][t] for i in range(len(W))])
            
            if new_U:
                U = [list(row) for row in zip(*new_U)]
                V = [list(row) for row in zip(*new_V)]
                W = [list(row) for row in zip(*new_W)]
                
                if _verify(U, V, W, n, m, p):
                    best_rank = len(U[0])
                    return {"U": U, "V": V, "W": W}
        
        # Return the Strassen result
        return decomp
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
