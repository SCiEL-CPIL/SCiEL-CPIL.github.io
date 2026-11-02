"""Search for a rank-minimal exact tensor decomposition of 2x4 times 4x7 matrix multiply."""
import time
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
            # U[i*m+j, c] = 1 for all j
            for j in range(m):
                U[i * m + j, c] = 1
            # V[j*p+k, c] = 1 for all j
            for j in range(m):
                V[j * p + k, c] = 1
            # W[k*n+i, c] = 1
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
    import random
    for _ in range(attempts):
        if time.time() > DEADLINE:
            return None
        U = np.random.randint(0, 3, size=(n * m, target_rank))
        V = np.random.randint(0, 3, size=(m * p, target_rank))
        W = np.random.randint(0, 3, size=(p * n, target_rank))
        # Verify
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
    # The idea: if we can factor the multiplication tensor using Kronecker products
    # of smaller tensors, we might get a lower rank.
    # For 2x4 * 4x7, if we split the 4 into 2x2:
    # A is 2x4 = 2x(2x2), B is 4x7 = (2x2)x7
    # This doesn't directly give a Kronecker product structure for the full multiplication.
    
    # Instead, let's try a different approach: use the fact that the tensor
    # T[i,j,k] = 1 for all (i,j,k) can be written as a sum of rank-1 tensors.
    # The minimal rank is the tensor rank of the all-ones tensor of shape (2,4,7).
    
    # For an all-ones tensor of shape (n,m,p), the tensor rank is min(n*m, m*p, n*p)
    # in general, but there are better bounds.
    
    # Actually, for the all-ones tensor, the rank is exactly min over partitions.
    # Let's try to find a better decomposition by solving a linear system.
    
    # For rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is a system of n*m*p equations in r^3 unknowns (with structure).
    # It's nonlinear in the entries, but we can fix some entries and solve for others.
    
    # Strategy: fix W to be a 0-1 matrix with exactly one 1 per row (i.e., W is a
    # permutation-like matrix), then the constraint becomes linear in U and V.
    # For each column c, W[k*n+i, c] = 1 for exactly one (k,i) pair.
    # Then for a given (i,j,k), only columns c with W[k*n+i,c]=1 contribute.
    # Let S_{i,k} = {c : W[k*n+i,c] = 1}. Then:
    #   sum_{c in S_{i,k}} U[i*m+j,c] * V[j*p+k,c] = 1 for all j.
    # This is a system of m equations in |S_{i,k}| * 2 unknowns (for each c in S_{i,k},
    # we have U[i*m+j,c] and V[j*p+k,c]).
    
    # Let's try with |S_{i,k}| = 1 for all (i,k), which gives the n*p construction.
    # Can we do better by having some (i,k) share columns?
    
    # If two (i,k) pairs share a column c, then for that column, W[k*n+i,c]=1 and
    # W[k'*n+i',c]=1, which means the column has two 1s in W. This is allowed.
    # Let's try rank = n*p - 1 and see if we can find a solution.
    
    # Actually, let's try a more systematic approach: use integer linear programming
    # or a greedy algorithm to assign columns to (i,k) pairs.
    
    # For now, let's try random search with small ranks.
    import random
    for rank in range(10, 14):  # try ranks 10 to 13
        if time.time() > DEADLINE - 10:
            break
        for _ in range(100):
            if time.time() > DEADLINE:
                break
            # Random 0-1 matrix for W with exactly rank columns
            W = np.zeros((p * n, rank), dtype=float)
            # Assign each column to a random (k,i) pair
            for c in range(rank):
                k = random.randint(0, p - 1)
                i = random.randint(0, n - 1)
                W[k * n + i, c] = 1
            # Now solve for U and V
            # For each (i,k), let S_{i,k} = {c : W[k*n+i,c] = 1}
            # We need sum_{c in S_{i,k}} U[i*m+j,c] * V[j*p+k,c] = 1 for all j
            # This is a system of m equations. If |S_{i,k}| = 1, it's easy.
            # If |S_{i,k}| > 1, we have more unknowns than equations, so we can find a solution.
            
            # Let's check if every (i,k) has at least one column
            for i in range(n):
                for k in range(p):
                    if W[k * n + i].sum() == 0:
                        break
                else:
                    continue
                break
            else:
                # Every (i,k) has at least one column
                # Try to solve for U and V
                U = np.zeros((n * m, rank), dtype=float)
                V = np.zeros((m * p, rank), dtype=float)
                for i in range(n):
                    for k in range(p):
                        S = np.where(W[k * n + i] != 0)[0]
                        if len(S) == 0:
                            break
                        # For each j, we need sum_{c in S} U[i*m+j,c] * V[j*p+k,c] = 1
                        # Let's set U[i*m+j,c] = 1 for the first c in S, and 0 for others
                        # Then V[j*p+k,c_first] = 1 for all j
                        c_first = S[0]
                        for j in range(m):
                            U[i * m + j, c_first] = 1
                            V[j * p + k, c_first] = 1
                        # Set other columns to 0 for this (i,k)
                        for c in S[1:]:
                            for j in range(m):
                                U[i * m + j, c] = 0
                                V[j * p + k, c] = 0
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

    # 2) Try Kronecker structure
    if best_rank > 10:
        result = _try_kronecker_structure(n, m, p)
        if result is not None:
            best_U, best_V, best_W = result['U'], result['V'], result['W']
            best_rank = result['U'].shape[1]
            print(f"Found rank {best_rank} decomposition via Kronecker")

    # 3) Try ranks from n*p - 1 down to 1
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
        # Multiply by 2 to handle half-integers, check if all even, then divide
        m2 = mat * 2
        if np.allclose(m2, np.round(m2), atol=1e-9):
            m2 = np.round(m2).astype(int)
            if np.all(m2 % 2 == 0):
                return (m2 // 2).tolist()
            return (m2.tolist())
        return mat.tolist()

    return {
        'U': to_int_list(best_U),
        'V': to_int_list(best_V),
        'W': to_int_list(best_W)
    }
