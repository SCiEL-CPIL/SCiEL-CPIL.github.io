"""Search for a rank-minimal exact tensor decomposition of 2x4 times 4x7 matrix multiply."""
import time
import random
import numpy as np
import itertools

DIMENSIONS = [2, 4, 7]
DEADLINE = time.time() + 110.0  # safety margin within 120s


def _build_U(i, j, idx, n, m):
    U = np.zeros(n*m, dtype=int)
    U[i*m + j] = idx
    return U


def _build_V(j, k, idx, m, p):
    V = np.zeros(m*p, dtype=int)
    V[j*p + k] = idx
    return V


def _build_W(i, k, idx, n, p):
    W = np.zeros(p*n, dtype=int)
    W[k*n + i] = idx
    return W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check whether the decomposition is exact."""
    for i in range(n):
        for k in range(p):
            target = 1.0
            s = 0.0
            for j in range(m):
                u = U[i*m + j]
                v = V[j*p + k]
                if np.any(u != 0) and np.any(v != 0):
                    w = W[k*n + i]
                    prod = np.dot(u, v) * w
                    if np.any(prod != 0):
                        s += np.sum(prod)
            if abs(s - target) > tol:
                return False
    return True


def _verify_np(U, V, W, n, m, p, tol=1e-6):
    """Faster vectorized check.  U, V, W are 2D arrays (rows, rank)."""
    for i in range(n):
        for k in range(p):
            w = W[k*n + i]
            s = 0.0
            for j in range(m):
                u = U[i*m + j]
                v = V[j*p + k]
                if np.any(u != 0) and np.any(v != 0):
                    s += np.sum(u * v * w)
            if abs(s - 1.0) > tol:
                return False
    return True


def _rank_of(U, V, W):
    """Return the number of non-zero columns across all three matrices."""
    r = max(U.shape[1], V.shape[1], W.shape[1]) if U is not None else 0
    count = 0
    for c in range(r):
        if np.any(U[:, c]) or np.any(V[:, c]) or np.any(W[:, c]):
            count += 1
    return count


def _construct_from_assignment(assign, n, m, p):
    """
    assign: list of (i, j, k, coeff) where coefficient is in {0, 1}
    We build U, V, W with integer entries.
    """
    rank = len(assign)
    U = np.zeros(n*m*rank, dtype=float)
    V = np.zeros(m*p*rank, dtype=float)
    W = np.zeros(p*n*rank, dtype=float)
    for c, (i, j, k, coeff) in enumerate(assign):
        U[(i*m + j) * rank + c] = coeff
        V[(j*p + k) * rank + c] = coeff
        W[(k*n + i) * rank + c] = coeff
    U = U.reshape(n*m, rank)
    V = V.reshape(m*p, rank)
    W = W.reshape(p*n, rank)
    return U, V, W


def _to_dict(U, V, W):
    rank = U.shape[1]
    return {
        'U': U[:, :rank].tolist(),
        'V': V[:, :rank].tolist(),
        'W': W[:, :rank].tolist()
    }


def _schoolbook(n, m, p):
    rank = n*m*p
    U = np.zeros(n*m*rank, dtype=int)
    V = np.zeros(m*p*rank, dtype=int)
    W = np.zeros(p*n*rank, dtype=int)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[(i*m + j)*rank + c] = 1
                V[(j*p + k)*rank + c] = 1
                W[(k*n + i)*rank + c] = 1
                c += 1
    U = U.reshape(n*m, rank)
    V = V.reshape(m*p, rank)
    W = W.reshape(p*n, rank)
    return U, V, W


def _try_random_rank(target_rank, n, m, p, attempts=500):
    """
    Try to find a decomposition of the given rank by assigning random
    (i,j,k) triples to columns and then solving a linear system to see
    if the product constraint can be satisfied with {0,1} coefficients.
    """
    for _ in range(attempts):
        if time.time() > DEADLINE:
            return None
        # Generate target_rank random (i,j,k) triples, with replacement allowed
        triples = [(random.randint(0, n-1), random.randint(0, m-1), random.randint(0, p-1))
                   for _ in range(target_rank)]
        # We want to assign coefficients in {0, 1} to each triple.
        # Build the linear system: for each (i,k), sum_{c: matches(i,k)} coeff_c * prod(U,V,W at c) = 1
        # With our assignment, for column c, U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1,
        # so the contribution to (i',k') is 1 if (i'=i, k'=k) else 0 for that column.
        # So the constraint reduces to: for each (i,k), the number of columns with (i,k) = 1.
        # That means we need exactly one column for each (i,k)? But we need the j-sum to also work.
        # Wait, actually the constraint is sum_j sum_{c} U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all (i,k).
        # In our model, U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1 only when the column corresponds
        # to the specific (i,j,k). So the product U[i*m+j',c]*V[j'*p+k,c]*W[k*n+i,c] is 1
        # only if the column's (i_col, j_col, k_col) has i_col=i, j_col=j', k_col=k.
        # So for a given (i,k), the sum is the count of columns with (i_col, k_col)=(i,k),
        # each with j_col = some value. But the j' index must match j_col.
        # Actually, the triple product for column c contributes to (i,k) only if
        # i_col=i AND k_col=k, regardless of j_col, because:
        # U[i*m+j_col, c] * V[j_col*p+k, c] * W[k*n+i, c]
        # For a fixed output (i,k), this term is 1 if i_col=i and k_col=k, else 0.
        # So the constraint is: for each (i,k), number of columns with (i_col, k_col)=(i,k) = 1.
        # But that can't be right because different columns can have same (i,k) but different j,
        # and they all contribute 1 to the same (i,k).
        # So we need for each (i,k): count of columns with i_col=i and k_col=k must equal 1.
        # But that means the total number of columns is at least n*p, and each (i,k) is hit exactly once.
        # However, we also need the j index to be consistent... wait no, the j is already in the triple.
        # So if we assign exactly one column per (i,k) pair, that gives n*p columns.
        # For n=2,p=7 that's 14 columns. Each column has an i,k pair and some j.
        # But do we also need the j to be consistent? No, because the triple product only
        # constrains i and k.
        # Actually wait, let me re-derive. The tensor is T[i,j,k] = 1 for all (i,j,k).
        # A rank-r decomposition is sum_{c=0}^{r-1} U_c[i] * V_c[j] * W_c[k].
        # We need this sum to equal 1 for all (i,j,k).
        # In the schoolbook, U_c[i]=delta(i,i_c), V_c[j]=delta(j,j_c), W_c[k]=delta(k,k_c),
        # and sum over c of delta(i,i_c)*delta(j,j_c)*delta(k,k_c) = 1 only if exactly one
        # c matches (i,j,k).
        # A more compact form: assign for each (i,k) a single column c_(i,k) with
        # U[i]=1, W[k]=1, and V[j]=some value a_{i,k,j}.
        # Then for a fixed (i,k), the sum is sum_{c: i_c=i, k_c=k} U_c[i]*V_c[j]*W_c[k]
        #   = sum_{c: i_c=i, k_c=k} a_{i,k,j}.
        # We need this to equal 1 for all j.
        # If there is exactly one column per (i,k), say c, then a_{i,k,j}=1 for all j.
        # That gives rank = n*p.
        # Let's check: U has rows indexed by (i), but we need U[i*m+j]... no wait.
        # In the problem, U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
        # The convention is W[k*n+i, c].
        # So the product is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
        # For this to equal 1 for all (i,j,k), we can set for each (i,k) one column c_{i,k}:
        #   U[i*m+j, c_{i,k}] = 1 for all j
        #   V[j*p+k, c_{i,k}] = 1 for all j
        #   W[k*n+i, c_{i,k}] = 1
        # Then for a fixed (i,j,k), only column c_{i,k} contributes:
        #   U[i*m+j, c_{i,k}] * V[j*p+k, c_{i,k}] * W[k*n+i, c_{i,k}]
        #   = 1 * 1 * 1 = 1.
        # All other columns c_{i',k'} with (i',k') != (i,k) have either
        #   U[i*m+j, c_{i',k'}] = 0 (if i' != i)
        #   or V[j*p+k, c_{i',k'}] = 0 (if k' != k)
        #   or W[k*n+i, c_{i',k'}] = 0 (if (i',k') != (i,k), since W row is k*n+i)
        # Wait, W[k*n+i, c_{i',k'}] is non-zero only if k=k' and i=i'.
        # So yes, this works! Rank = n*p = 2*7 = 14.
        if target_rank >= n*p:
            # Try the n*p construction
            U, V, W = _construct_np(n, m, p)
            if _verify_np(U, V, W, n, m, p):
                return {'U': U, 'V': V, 'W': W}
        return None


def _construct_np(n, m, p):
    """Construct rank n*p decomposition."""
    rank = n*p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    for i in range(n):
        for k in range(p):
            c = k*n + i  # arbitrary column index in [0, n*p)
            # U[i*m+j, c] = 1 for all j
            for j in range(m):
                U[i*m + j, c] = 1
            # V[j*p+k, c] = 1 for all j
            for j in range(m):
                V[j*p + k, c] = 1
            # W[k*n+i, c] = 1
            W[k*n + i, c] = 1
    return U, V, W


def _construct_rank1(n, m, p):
    """
    The matrix multiplication tensor T[i,j,k] = 1 for all (i,j,k) is the
    outer product of three all-ones vectors, so it has tensor rank 1.
    U[i*m+j, 0] = 1 for all i,j
    V[j*p+k, 0] = 1 for all j,k
    W[k*n+i, 0] = 1 for all k,i
    Then for any (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1.
    """
    U = np.ones((n*m, 1), dtype=int)
    V = np.ones((m*p, 1), dtype=int)
    W = np.ones((p*n, 1), dtype=int)
    return U, V, W


def _try_smaller_rank(target_rank, n, m, p, attempts=200):
    """
    For ranks < n*p, try to find a decomposition using integer entries.
    The key constraint is:
      For all (i,k): sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    For all (i,j,k): the same sum = 1.
    Actually the constraint is stronger: the sum must equal 1 for all (i,j,k) simultaneously.
    Let's try a randomized search with small integer entries.
    """
    for _ in range(attempts):
        if time.time() > DEADLINE:
            return None
        U = np.random.randint(0, 3, size=(n*m, target_rank))
        V = np.random.randint(0, 3, size=(m*p, target_rank))
        W = np.random.randint(0, 3, size=(p*n, target_rank))
        # Verify
        ok = True
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > 1e-6:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return {'U': U, 'V': V, 'W': W}
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()

    # Best result found so far (start with schoolbook)
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = n*m*p

    # 1) Try rank 1: the all-ones tensor is rank 1
    U1, V1, W1 = _construct_rank1(n, m, p)
    if _verify_np(U1, V1, W1, n, m, p):
        best_U, best_V, best_W = U1, V1, W1
        best_rank = 1
        print(f"Found rank 1 decomposition")
        # Rank 1 is the theoretical minimum, so we can return immediately
        return {
            'U': best_U.tolist(),
            'V': best_V.tolist(),
            'W': best_W.tolist()
        }

    # 2) Try rank = n*p
    U14, V14, W14 = _construct_np(n, m, p)
    if _verify_np(U14, V14, W14, n, m, p):
        best_U, best_V, best_W = U14, V14, W14
        best_rank = n*p
        print(f"Found rank {n*p} decomposition")

    # 3) Try ranks from n*p - 1 down to 2
    for rank in range(n*p - 1, 1, -1):
        if time.time() > DEADLINE - 10:
            break
        result = _try_smaller_rank(rank, n, m, p, attempts=max(10, int(60 / max(1, n*p-rank))))
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
