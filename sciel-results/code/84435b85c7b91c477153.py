import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

def _target_tensor(n, m, p):
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                key = (i*m+j, j*p+k, k*n+i)
                T[key] = T.get(key, 0) + 1
    return T

def _from_columns(cols, n, m, p, rank):
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for r, (Ac, Bc, Cc) in enumerate(cols):
        for (i, j), v in Ac.items():
            U[i*m+j][r] += v
        for (j, k), v in Bc.items():
            V[j*p+k][r] += v
        for (i, k), v in Cc.items():
            W[k*n+i][r] += v
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    target = _target_tensor(n, m, p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j; b = j*p+k; c = k*n+i
                s = 0
                for r in range(rank):
                    s += U[a][r]*V[b][r]*W[c][r]
                if abs(s - target.get((a, b, c), 0)) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W

def _strassen_2x2(ai, aj, bj, bk, ci, ck):
    """Return list of (Ac, Bc, Cc) for 2x2 Strassen."""
    def A(i, j): return (ai+i, aj+j)
    def B(j, k): return (bj+j, bk+k)
    def C(i, k): return (ci+i, ck+k)
    M = [
        ([(A(0,0),1),(A(1,1),1)], [(B(0,0),1),(B(1,1),1)], [(C(0,0),1),(C(1,1),1)]),
        ([(A(1,0),1),(A(1,1),1)], [(B(0,0),1)], [(C(1,0),1),(C(1,1),-1)]),
        ([(A(0,0),1)], [(B(1,0),1),(B(1,1),-1)], [(C(0,1),1),(C(1,1),1)]),
        ([(A(1,1),1)], [(B(0,1),1),(B(1,0),-1)], [(C(0,0),1),(C(1,0),1)]),
        ([(A(0,0),1),(A(0,1),1)], [(B(1,0),1)], [(C(0,0),-1),(C(0,1),1)]),
        ([(A(0,1),1),(A(0,0),-1)], [(B(0,0),1),(B(1,0),1)], [(C(1,1),1)]),
        ([(A(1,0),1),(A(1,1),-1)], [(B(1,1),1),(B(0,1),1)], [(C(0,0),1),(C(0,1),-1)]),
    ]
    return [(dict(Ac), dict(Bc), dict(Cc)) for Ac, Bc, Cc in M]

def _build_low_rank(n, m, p):
    """
    Block decomposition:
    Split A (3x3) into rows [0:2, 2:3] and B (3x4) into cols [0:2, 2:4].
    4 blocks:
      A[0:2, 0:2] (2x2) * B[0:2, 0:2] (2x2) -> C[0:2, 0:2]: Strassen rank 7
      A[0:2, 0:2] (2x2) * B[0:2, 2:4] (2x2) -> C[0:2, 2:4]: Strassen rank 7
      A[0:2, 2:3] (2x1) * B[2:3, 0:2] (1x2) -> C[0:2, 0:2]: schoolbook rank 4
      A[0:2, 2:3] (2x1) * B[2:3, 2:4] (1x2) -> C[0:2, 2:4]: schoolbook rank 4
      A[2:3, 0:2] (1x2) * B[0:2, 0:2] (2x2) -> C[2:3, 0:2]: schoolbook rank 4
      A[2:3, 0:2] (1x2) * B[0:2, 2:4] (2x2) -> C[2:3, 2:4]: schoolbook rank 4
      A[2:3, 2:3] (1x1) * B[2:3, 0:2] (1x2) -> C[2:3, 0:2]: schoolbook rank 2
      A[2:3, 2:3] (1x1) * B[2:3, 2:4] (1x2) -> C[2:3, 2:4]: schoolbook rank 2
    Total rank: 7+7+4+4+4+4+2+2 = 34
    """
    cols = []
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    cols.extend(_strassen_2x2(0, 0, 0, 0, 0, 0))
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    cols.extend(_strassen_2x2(0, 0, 0, 2, 0, 2))
    
    # Block 3: A[0:2, 2:3] (2x1) * B[2:3, 0:2] (1x2) -> C[0:2, 0:2]
    for i in range(2):
        for k in range(2):
            cols.append(( {(i, 2): 1}, {(2, k): 1}, {(i, k): 1} ))
    
    # Block 4: A[0:2, 2:3] (2x1) * B[2:3, 2:4] (1x2) -> C[0:2, 2:4]
    for i in range(2):
        for k in range(2):
            cols.append(( {(i, 2): 1}, {(2, 2+k): 1}, {(i, 2+k): 1} ))
    
    # Block 5: A[2:3, 0:2] (1x2) * B[0:2, 0:2] (2x2) -> C[2:3, 0:2]
    for j in range(2):
        for k in range(2):
            cols.append(( {(2, j): 1}, {(j, k): 1}, {(2, k): 1} ))
    
    # Block 6: A[2:3, 0:2] (1x2) * B[0:2, 2:4] (2x2) -> C[2:3, 2:4]
    for j in range(2):
        for k in range(2):
            cols.append(( {(2, j): 1}, {(j, 2+k): 1}, {(2, 2+k): 1} ))
    
    # Block 7: A[2:3, 2:3] (1x1) * B[2:3, 0:2] (1x2) -> C[2:3, 0:2]
    for k in range(2):
        cols.append(( {(2, 2): 1}, {(2, k): 1}, {(2, k): 1} ))
    
    # Block 8: A[2:3, 2:3] (1x1) * B[2:3, 2:4] (1x2) -> C[2:3, 2:4]
    for k in range(2):
        cols.append(( {(2, 2): 1}, {(2, 2+k): 1}, {(2, 2+k): 1} ))
    
    rank = len(cols)
    return _from_columns(cols, n, m, p, rank)

def _search_lower_rank(n, m, p, target_rank, time_limit):
    """
    Attempts to find a decomposition of the given rank using randomized search.
    """
    start_time = time.time()
    target = _target_tensor(n, m, p)
    indices = list(target.keys())
    
    # We will try to find U, V, W of shape (n*m, R), (m*p, R), (p*n, R)
    # such that sum_r U[a,r]V[b,r]W[c,r] == T[a,b,c]
    
    # Use a simple iterative solver
    # Initialize with random values
    U = np.random.randn(n*m, target_rank)
    V = np.random.randn(m*p, target_rank)
    W = np.random.randn(p*n, target_rank)
    
    best_error = float('inf')
    
    for _ in range(1000): # Max iterations
        if time.time() - start_time > time_limit:
            break
        
        # Calculate current error
        error = 0
        for (a, b, c) in indices:
            val = 0
            for r in range(target_rank):
                val += U[a, r] * V[b, r] * W[c, r]
            error += (val - target[(a, b, c)])**2
        
        if error < 1e-9:
            # Found a solution!
            cols = []
            for r in range(target_rank):
                Ac = {}
                Bc = {}
                Cc = {}
                for i in range(n):
                    for j in range(m):
                        if U[i*m+j, r] != 0:
                            Ac[(i, j)] = U[i*m+j, r]
                for j in range(m):
                    for k in range(p):
                        if V[j*p+k, r] != 0:
                            Bc[(j, k)] = V[j*p+k, r]
                for i in range(n):
                    for k in range(p):
                        if W[k*n+i, r] != 0:
                            Cc[(i, k)] = W[k*n+i, r]
                cols.append((Ac, Bc, Cc))
            return _from_columns(cols, n, m, p, target_rank)
        
        if error < best_error:
            best_error = error
            
        # Update step
        # This is a very crude update. We need a better one.
        # Let's just restart with different random seeds if we don't improve.
        if _ % 10 == 0:
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
    return None

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    DEADLINE = 115.0
    
    # Try the block decomposition with Strassen (rank 34)
    U, V, W = _build_low_rank(n, m, p)
    if _verify(U, V, W, n, m, p, len(U[0])) and (time.time() - t0) < DEADLINE:
        best_U, best_V, best_W = U, V, W
        best_rank = len(U[0])
    else:
        # Fallback to schoolbook (rank 36)
        U, V, W = _schoolbook(n, m, p)
        best_U, best_V, best_W = U, V, W
        best_rank = len(U[0])
        
    # Try to search for a lower rank
    # We try ranks from 23 to best_rank - 1
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5:
        # Allocate time to search
        search_time = remaining_time - 2
        # Try a few ranks
        for r in range(23, best_rank):
            if time.time() - t0 > DEADLINE - 2:
                break
            result = _search_lower_rank(n, m, p, r, search_time / (best_rank - 23))
            if result:
                best_U, best_V, best_W = result
                best_rank = r
                break # Found a lower rank, stop searching lower ones
                
    return {"U": best_U, "V": best_V, "W": best_W}
