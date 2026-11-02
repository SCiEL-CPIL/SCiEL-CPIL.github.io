"""Search for a rank-minimal exact tensor decomposition of 2x4 times 4x7 matrix multiply."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = time.time() + 110.0  # safety margin within 120s


def _construct_np(n, m, p):
    """Construct rank n*p decomposition."""
    rank = n * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    for i in range(n):
        for k in range(p):
            c = k * n + i  # column index in [0, n*p)
            for j in range(m):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
            W[k * n + i, c] = 1
    return U, V, W


def _verify_np(U, V, W, n, m, p, tol=1e-6):
    """Faster vectorized check. U, V, W are 2D arrays (rows, rank)."""
    for i in range(n):
        for k in range(p):
            w = W[k * n + i]
            s = 0.0
            for j in range(m):
                u = U[i * m + j]
                v = V[j * p + k]
                if np.any(u != 0) and np.any(v != 0):
                    s += np.sum(u * v * w)
            if abs(s - 1.0) > tol:
                return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros(n * m * rank, dtype=int)
    V = np.zeros(m * p * rank, dtype=int)
    W = np.zeros(p * n * rank, dtype=int)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[(i * m + j) * rank + c] = 1
                V[(j * p + k) * rank + c] = 1
                W[(k * n + i) * rank + c] = 1
                c += 1
    U = U.reshape(n * m, rank)
    V = V.reshape(m * p, rank)
    W = W.reshape(p * n, rank)
    return U, V, W


def _try_smaller_rank(target_rank, n, m, p, attempts=200):
    """
    For ranks < n*p, try to find a decomposition using integer entries.
    """
    for _ in range(attempts):
        if time.time() > DEADLINE:
            return None
        U = np.random.randint(0, 3, size=(n * m, target_rank))
        V = np.random.randint(0, 3, size=(m * p, target_rank))
        W = np.random.randint(0, 3, size=(p * n, target_rank))
        ok = True
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(s - 1.0) > 1e-6:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return {'U': U, 'V': V, 'W': W}
    return None


def _try_kronecker_structure(n, m, p):
    """
    Try to exploit Kronecker product structure.
    For m=4, we can view the middle dimension as 2x2.
    This might allow a lower rank decomposition.
    """
    import random
    for rank in range(10, 14):
        if time.time() > DEADLINE - 10:
            break
        for _ in range(100):
            if time.time() > DEADLINE:
                break
            W = np.zeros((p * n, rank), dtype=float)
            for c in range(rank):
                k = random.randint(0, p - 1)
                i = random.randint(0, n - 1)
                W[k * n + i, c] = 1
            for i in range(n):
                for k in range(p):
                    if W[k * n + i].sum() == 0:
                        break
                else:
                    continue
                break
            else:
                U = np.zeros((n * m, rank), dtype=float)
                V = np.zeros((m * p, rank), dtype=float)
                for i in range(n):
                    for k in range(p):
                        S = np.where(W[k * n + i] != 0)[0]
                        if len(S) == 0:
                            break
                        c_first = S[0]
                        for j in range(m):
                            U[i * m + j, c_first] = 1
                            V[j * p + k, c_first] = 1
                        for c in S[1:]:
                            for j in range(m):
                                U[i * m + j, c] = 0
                                V[j * p + k, c] = 0
                if _verify_np(U, V, W, n, m, p):
                    return {'U': U, 'V': V, 'W': W}
    return None


def _try_strassen_style(n, m, p):
    """
    Try a structured approach inspired by Strassen's algorithm.
    For multiplying (2x4) by (4x7), we can think of the 4 as 2x2 blocks.
    
    Let A be 2x4, B be 4x7.
    Split A into [A1 A2] where A1, A2 are 2x2.
    Split B into [B1; B2] where B1, B2 are 2x7.
    
    Then AB = A1*B1 + A2*B2.
    
    If we can compute A1*B1 and A2*B2 with low rank decompositions,
    we might get a better overall rank.
    
    For 2x2 * 2x2, the Strassen algorithm uses 7 multiplications.
    But here we have 2x2 * 2x7, which is different.
    
    Let's try a different approach: use the fact that we can combine
    multiple (i,k) pairs into a single column by using linear combinations.
    """
    import random
    
    # For each rank from 10 to 13, try to find a decomposition
    for rank in range(10, 14):
        if time.time() > DEADLINE - 15:
            break
        
        # Strategy: assign columns to (i,k) pairs, allowing some pairs
        # to share columns. For shared columns, use linear combinations
        # in U and V to satisfy the constraints.
        
        # Generate random assignments of columns to (i,k) pairs
        for attempt in range(200):
            if time.time() > DEADLINE:
                return None
            
            # Create a random assignment: each column is assigned to one or more (i,k) pairs
            # We'll use a simpler approach: randomly assign each column to a single (i,k) pair,
            # but allow some (i,k) pairs to have multiple columns.
            
            W = np.zeros((p * n, rank), dtype=float)
            assignments = []  # For each column, list of (i,k) pairs it's assigned to
            
            # Ensure each (i,k) pair has at least one column
            all_pairs = [(i, k) for i in range(n) for k in range(p)]
            random.shuffle(all_pairs)
            
            # First, assign one column to each (i,k) pair
            col_idx = 0
            for (i, k) in all_pairs:
                if col_idx >= rank:
                    break
                W[k * n + i, col_idx] = 1
                assignments.append([(i, k)])
                col_idx += 1
            
            # If we have more columns than (i,k) pairs, assign extra columns randomly
            while col_idx < rank:
                i = random.randint(0, n - 1)
                k = random.randint(0, p - 1)
                W[k * n + i, col_idx] = 1
                assignments.append([(i, k)])
                col_idx += 1
            
            # Now, for each (i,k) pair, find all columns assigned to it
            pair_to_cols = {}
            for c, pairs in enumerate(assignments):
                for (i, k) in pairs:
                    if (i, k) not in pair_to_cols:
                        pair_to_cols[(i, k)] = []
                    pair_to_cols[(i, k)].append(c)
            
            # For each (i,k) pair, we need to solve for U and V entries
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            
            valid = True
            for (i, k) in all_pairs:
                cols = pair_to_cols.get((i, k), [])
                if not cols:
                    valid = False
                    break
                
                # For each j, we need sum_{c in cols} U[i*m+j, c] * V[j*p+k, c] = 1
                # This is a system of m equations in 2*|cols| unknowns.
                # If |cols| >= 1, we can find a solution.
                
                if len(cols) == 1:
                    c = cols[0]
                    for j in range(m):
                        U[i * m + j, c] = 1
                        V[j * p + k, c] = 1
                else:
                    # Multiple columns: use the first column for all j, set others to 0
                    c_first = cols[0]
                    for j in range(m):
                        U[i * m + j, c_first] = 1
                        V[j * p + k, c_first] = 1
                    for c in cols[1:]:
                        for j in range(m):
                            U[i * m + j, c] = 0
                            V[j * p + k, c] = 0
            
            if not valid:
                continue
            
            # Verify
            if _verify_np(U, V, W, n, m, p):
                return {'U': U, 'V': V, 'W': W}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()

    # Best result found so far (start with schoolbook)
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = n * m * p

    # 1) Try rank = n*p = 14
    U14, V14, W14 = _construct_np(n, m, p)
    if _verify_np(U14, V14, W14, n, m, p):
        best_U, best_V, best_W = U14, V14, W14
        best_rank = n * p
        print(f"Found rank {n*p} decomposition")
    else:
        print(f"Rank {n*p} construction failed verification")

    # 2) Try Strassen-style structured approach
    if best_rank > 10:
        result = _try_strassen_style(n, m, p)
        if result is not None:
            best_U, best_V, best_W = result['U'], result['V'], result['W']
            best_rank = result['U'].shape[1]
            print(f"Found rank {best_rank} decomposition via Strassen-style")

    # 3) Try Kronecker structure
    if best_rank > 10:
        result = _try_kronecker_structure(n, m, p)
        if result is not None:
            best_U, best_V, best_W = result['U'], result['V'], result['W']
            best_rank = result['U'].shape[1]
            print(f"Found rank {best_rank} decomposition via Kronecker")

    # 4) Try ranks from n*p - 1 down to 1
    for rank in range(n * p - 1, 0, -1):
        if time.time() > DEADLINE - 10:
            break
        result = _try_smaller_rank(rank, n, m, p, attempts=max(10, int(60 / max(1, n * p - rank))))
        if result is not None:
            best_U, best_V, best_W = result['U'], result['V'], result['W']
            best_rank = rank
            print(f"Found rank {rank} decomposition")
            break

    # Convert to list-of-lists with integer/half-integer entries
    def to_int_list(mat):
        m2 = mat * 2
        if np.allclose(m2, np.round(m2), atol=1e-9):
            m2 = np.round(m2).astype(int)
            if np.all(m2 % 2 == 0):
                return (m2 // 2).tolist()
            return m2.tolist()
        return mat.tolist()

    return {
        'U': to_int_list(best_U),
        'V': to_int_list(best_V),
        'W': to_int_list(best_W)
    }
