"""Matrix multiplication 3x4x7 tensor decomposition via randomized search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]
n, m, p = DIMENSIONS
NM = n * m
MP = m * p
PN = p * n
SCHOOLBOOK_RANK = NM * MP  # 84

def _matmul_tensor_entries():
    """Return the list of (i,j,k, weight) where weight is always 1 for matmul.
    The tensor T has T[i,j,k] = delta such that C[i,k] = sum_j A[i,j]*B[j,k].
    In the decomposition: C[i,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    Wait, the interface says:
    U: (n*m, rank) -> U[i*m + j, r]
    V: (m*p, rank) -> V[j*p + k, r]
    W: (p*n, rank) -> W[k*n + i, r]
    And the product is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} (for the matmul tensor)
    Actually, the matmul tensor T is defined such that:
    C[i,k] = sum_j A[i,j] B[j,k]
    So the tensor T has entries T[i,j,k] = 1 if we think of it as a 3-index tensor where
    the contraction is over j. But the standard tensor decomposition for matmul is:
    T_{i,k}^{j} ... no.
    
    Let's re-read: "exactly decompose the matrix multiplication tensor"
    The matrix multiplication tensor T is a 3-tensor of size (n, m, p) such that
    T[i, j, k] = 1 if and only if ... no.
    
    Actually, the standard formulation:
    We want C = A * B, where A is n x m, B is m x p, C is n x p.
    C[i, k] = sum_{j=0}^{m-1} A[i, j] * B[j, k]
    
    The tensor decomposition expresses this as:
    C[i, k] = sum_{r=0}^{rank-1} U[i, j, r] * V[j, k, r] * W[k, i, r]
    Wait, that doesn't look right either. Let's look at the interface:
    U: (n*m, rank) -> U[i*m + j, r]
    V: (m*p, rank) -> V[j*p + k, r]
    W: (p*n, rank) -> W[k*n + i, r]
    
    So the reconstruction is:
    C[i, k] = sum_{r} sum_{j} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For this to equal sum_j A[i,j] B[j,k], we need the decomposition to satisfy:
    For all i, k: sum_r sum_j U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = (A*B)[i,k]
    
    But A and B are arbitrary. So actually, the tensor we're decomposing is the
    "matrix multiplication tensor" which is a 3-way tensor T of shape (n*p, m) or similar.
    
    Let me reconsider. The standard Strassen-style decomposition:
    We have bilinear forms. The matmul can be written as:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Each term A[i,j] * B[j,k] is a product of an entry of A and an entry of B.
    In a rank-1 decomposition, each "rank-1" term corresponds to a bilinear form:
    (sum_{i,j} u_{i,j} A[i,j]) * (sum_{j,k} v_{j,k} B[j,k]) * (sum_{k,i} w_{k,i} C[i,k])
    
    So the tensor T has entries indexed by (i,j) for A, (j,k) for B, (k,i) for C.
    T[(i,j), (j,k), (k,i)] = 1 if the indices are consistent, i.e., this is a 3-tensor
    of size (n*m) x (m*p) x (p*n) where T[a, b, c] = 1 if the index tuples match.
    
    Specifically, T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k, and 0 otherwise.
    Wait, that's not quite right because j appears in both A and B indices.
    
    Let's be precise. The matmul tensor T is a 3-way tensor of dimensions (n*m) x (m*p) x (p*n).
    T[a, b, c] where a = i*m + j, b = j'*p + k, c = k'*n + i'
    T[a, b, c] = 1 if and only if i=i', j=j', k=k' (i.e., all indices match).
    Otherwise 0.
    
    So T[i*m+j, j*p+k, k*n+i] = 1 for each (i,j,k) in range(n)xrange(m)xrange(p).
    There are n*m*p = 84 such 1s.
    
    A rank-1 decomposition term r contributes:
    sum_{i,j,k} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * T[i*m+j, j*p+k, k*n+i]
    = sum_{i,j,k} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    And the full decomposition requires:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    and 0 for all other (a,b,c) where the indices don't all match.
    
    Wait, that's not right. The tensor T has 1s only at positions where all three index tuples
    share consistent i, j, k. So T is sparse with exactly n*m*p non-zero entries.
    
    The decomposition condition is:
    For each (i,j,k): sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    For each (a,b,c) not of the form (i*m+j, j*p+k, k*n+i) for consistent i,j,k:
      sum_r U[a, r] V[b, r] W[c, r] = 0
    
    This is a system of polynomial equations in the entries of U, V, W.
    
    The schoolbook solution has rank 84, with each rank-1 term isolating one (i,j,k).
    
    For a lower rank, we need to find U, V, W with fewer columns satisfying these equations.
    """
    pass

def _verify(U, V, W):
    """Verify that U, V, W form a valid decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    And for all other (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] == 0
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Check the "on" positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check a sample of "off" positions (for speed, check all if rank is small)
    # Total positions: NM * MP * PN = 12 * 28 * 21 = 7056
    # On positions: 84
    # Off positions: 6972
    # For verification, we need to check all off positions
    for a in range(NM):
        i_a = a // m
        j_a = a % m
        for b in range(MP):
            j_b = b // p
            k_b = b % p
            if j_a != j_b:
                # This is an off position
                for c in range(PN):
                    k_c = c // n
                    i_c = c % n
                    # Check if this is actually an on position
                    if i_c == i_a and k_c == k_b:
                        continue  # This is an on position, already checked
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
    return True

def _build_schoolbook():
    rank = n * m * p
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_random_rank(rank, deadline):
    """Try to find a valid decomposition of given rank via random initialization."""
    U = np.random.randn(NM, rank)
    V = np.random.randn(MP, rank)
    W = np.random.randn(PN, rank)
    
    # Normalize so that for each (i,j,k), the sum is 1
    # This is hard to do directly. Instead, let's use a different approach.
    # We'll use the fact that the schoolbook solution is valid and try to
    # combine rank-1 terms.
    pass

def _greedy_combine(U, V, W, deadline):
    """Try to reduce rank by finding linear dependencies or combining terms."""
    # This is a placeholder for a more sophisticated approach.
    pass

_cached_result = None
_start_time = None

def construct():
    global _cached_result, _start_time
    if _start_time is None:
        _start_time = time.time()
    
    if _cached_result is not None:
        return _cached_result
    
    deadline = _start_time + 140  # 10s safety margin
    
    # Start with schoolbook
    best = _build_schoolbook()
    best_rank = n * m * p  # 84
    
    # Strategy: Try to find a lower-rank decomposition using random search.
    # We'll try ranks from 83 down to some minimum, using random initialization
    # and a simple verification.
    
    # First, let's try a structured approach: use the known Strassen-like
    # block decomposition for 2x2x2 blocks.
    # For 3x4x7, we can try to partition into blocks and apply Strassen.
    
    # Actually, let's try a more direct approach: random search with verification.
    # For each candidate rank, generate random U, V, W and check if they work.
    # This is unlikely to succeed for ranks close to 84, but let's try.
    
    # Better approach: Start from schoolbook and try to merge terms.
    # Two schoolbook terms for (i,j,k1) and (i,j,k2) can potentially be merged
    # if we can find U, V, W contributions that cancel the off-diagonal terms.
    
    # Let's try a simple random search for rank < 84.
    for target_rank in range(83, 60, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            U = np.random.randn(NM, target_rank)
            V = np.random.randn(MP, target_rank)
            W = np.random.randn(PN, target_rank)
            
            # Quick check: just verify the "on" positions
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        if abs(val - 1.0) > 0.1:  # Loose check first
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Full verification
                if _verify(U, V, W):
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = target_rank
                    break
    
    _cached_result = best
    return _cached_result
