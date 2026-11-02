import time
import random
import numpy as np

DIMENSIONS = [3, 4, 6]
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

def _build_factor(n, m, p, entries, rank):
    """entries: list of (i,j,k) triples; build U,V,W with given rank."""
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for t, (i, j, k) in enumerate(entries):
        if t >= rank:
            break
        U[i * m + j][t] = 1.0
        V[j * p + k][t] = 1.0
        W[k * n + i][t] = 1.0
    return U, V, W

def _build_bordered(n, m, p):
    """Build bordered decomposition for n x m times m x p.
    
    Rank = (n-1)*(m-1)*(p-1) + n + m + p - 2
    
    Construction:
    - Core: (n-1)*(m-1)*(p-1) terms for i<n-1, j<m-1, k<p-1
    - Border: n+m+p-2 terms covering all remaining (i,j,k) triples
    
    The border terms work as follows:
    For each j in 0..m-1, we need to cover all (i,k) pairs where
    the triple (i,j,k) is not in the core (i.e., i=n-1 or k=p-1).
    
    We use the following strategy:
    - For j < m-1: cover (i=n-1, j, k) for all k, and (i, j, k=p-1) for all i<n-1
    - For j = m-1: cover all (i, m-1, k)
    
    More precisely, the border has (n*m*p) - (n-1)*(m-1)*(p-1) entries.
    We cover them with n+m+p-2 terms using a clever sharing pattern.
    
    Term structure for border:
    For each j in 0..m-2:
      - One term covering (n-1, j, k) for all k: U[n-1,j]=1, V[j,k]=1 for all k, W[k,n-1]=1
        But this requires separate k values... 
    
    Let me use a different approach. The standard bordered decomposition:
    
    Core terms (t = 0 to (n-1)(m-1)(p-1)-1):
      For each (i,j,k) with i<n-1, j<m-1, k<p-1:
        U[i,j,t]=1, V[j,k,t]=1, W[k,i,t]=1
    
    Border terms (total: n+m+p-2):
      Group 1: For i in 0..n-2, term covering (i, m-1, k) for all k<p-1
        U[i, m-1] = 1
        V[m-1, k] = 1 for k in 0..p-2
        W[k, i] = 1 for k in 0..p-2
        This is (n-1) terms.
      
      Group 2: For k in 0..p-2, term covering (n-1, j, k) for all j<m-1
        U[n-1, j] = 1 for j in 0..m-2
        V[j, k] = 1 for j in 0..m-2
        W[k, n-1] = 1
        This is (p-1) terms.
      
      Group 3: One term for (n-1, m-1, p-1)
        U[n-1, m-1] = 1
        V[m-1, p-1] = 1
        W[p-1, n-1] = 1
        This is 1 term.
      
      Group 4: For i in 0..n-2, term covering (i, m-1, p-1)
        U[i, m-1] = 1
        V[m-1, p-1] = 1
        W[p-1, i] = 1
        This is (n-1) terms... but wait, that overlaps with group 1.
    
    Hmm, let me reconsider. Total border entries:
    - (i, j, k) where i=n-1, j<m-1, k<p-1: (m-1)(p-1) entries
    - (i, j, k) where i<n-1, j<m-1, k=p-1: (n-1)(m-1) entries
    - (i, j, k) where i=n-1, j<m-1, k=p-1: (m-1) entries
    - (i, j, k) where i<n-1, j=m-1, k<p-1: (n-1)(p-1) entries
    - (i, j, k) where i<n-1, j=m-1, k=p-1: (n-1) entries
    - (i, j, k) where i=n-1, j=m-1, k<p-1: (p-1) entries
    - (i, j, k) where i=n-1, j=m-1, k=p-1: 1 entry
    
    Total: (m-1)(p-1) + (n-1)(m-1) + (m-1) + (n-1)(p-1) + (n-1) + (p-1) + 1
    = (m-1)(p-1) + (n-1)(m-1) + (m-1) + (n-1)(p-1) + (n-1) + (p-1) + 1
    = n*m*p - (n-1)(m-1)(p-1)
    
    For (3,4,6): 72 - 30 = 42 border entries.
    
    We need to cover these with n+m+p-2 = 11 terms.
    
    Each border term is a rank-1 tensor. A rank-1 tensor U_t ⊗ V_t ⊗ W_t
    contributes to T[i,j,k] = U_t[i,j] * V_t[j,k] * W_t[k,i].
    
    For a border term to contribute to multiple (i,j,k) triples, the U, V, W
    patterns must be consistent.
    
    Key insight: A border term with U having support on {(i0, j0)} contributes
    only to T[i0, j0, k] for various k (where V[j0,k] and W[k,i0] are nonzero).
    
    So if I set U[n-1, j0]=1 for a specific j0, then the term only affects
    T[n-1, j0, k] for k where V[j0,k]*W[k,n-1] != 0.
    
    To cover T[n-1, j, k] for all j<m-1 and k<p-1 with few terms:
    - Fix j, vary k: one term with U[n-1,j]=1, V[j,k]=1 for all k<p-1, W[k,n-1]=1 for all k<p-1
      This covers (m-1) terms * 1 = m-1 terms for the (n-1, *, *) border strip.
      
    - Fix k, vary j: one term with U[n-1,j]=1 for all j<m-1, V[j,k]=1 for all j<m-1, W[k,n-1]=1
      This covers (p-1) terms.
    
    But these two sets overlap at (n-1, j, k) for j<m-1, k<p-1!
    
    The resolution: use inclusion-exclusion or a different partition.
    
    Actually, the correct bordered decomposition uses:
    - (n-1) terms: for each i<n-1, cover (i, m-1, k) for k=0..p-1
      U[i,m-1]=1, V[m-1,k]=1 for all k, W[k,i]=1 for all k
      But V[m-1,k] for all k means this term contributes to T[i, m-1, k] = 1*1*W[k,i]
      So we need W[k,i]=1 for all k. This works! One term per i<n-1.
      Count: n-1 = 2 terms.
      Covers: (i, m-1, k) for i<n-1, all k. That's (n-1)*p = 2*6 = 12 entries.
      
    - (p-1) terms: for each k<p-1, cover (n-1, j, k) for j=0..m-2
      U[n-1,j]=1 for all j<m-1, V[j,k]=1 for all j<m-1, W[k,n-1]=1
      This term contributes T[n-1, j, k] = 1*1*1 = 1 for j<m-1.
      Count: p-1 = 5 terms.
      Covers: (n-1, j, k) for j<m-1, k<p-1. That's (m-1)*(p-1) = 3*5 = 15 entries.
      
    - (m-1) terms: for each j<m-1, cover (n-1, j, p-1)
      U[n-1,j]=1, V[j,p-1]=1, W[p-1,n-1]=1
      Count: m-1 = 3 terms.
      Covers: (n-1, j, p-1) for j<m-1. That's m-1 = 3 entries.
      
    - 1 term: cover (n-1, m-1, p-1)
      U[n-1,m-1]=1, V[m-1,p-1]=1, W[p-1,n-1]=1
      Count: 1 term.
      Covers: 1 entry.
      
    Total border terms: (n-1) + (p-1) + (m-1) + 1 = 2+5+3+1 = 11 = n+m+p-2. ✓
    
    But wait, do these terms interfere with the core? Let's check:
    - Group 1 terms (i<n-1, j=m-1, k=any): U[i,m-1]=1. In the core, j<m-1, so
      U[i,j] for j<m-1 is 0 in these terms. No interference. ✓
    - Group 2 terms (i=n-1, j<m-1, k<p-1): W[k,n-1]=1. In the core, i<n-1, so
      W[k,i] for i<n-1 is 0 in these terms. No interference. ✓
    - Group 3 terms (i=n-1, j<m-1, k=p-1): W[p-1,n-1]=1. In the core, k<p-1 and i<n-1.
      No interference. ✓
    - Group 4 term (i=n-1, j=m-1, k=p-1): No interference with core. ✓
    
    But do the border terms interfere with EACH OTHER?
    - Group 1 term for i0: contributes T[i0, m-1, k] = W[k, i0] for all k.
      Group 2 term for k0: contributes T[n-1, j, k0] for j<m-1.
      These don't overlap (different i or different j). ✓
    - Group 1 term for i0: contributes T[i0, m-1, k] for all k including k=p-1.
      Group 4 term: contributes T[n-1, m-1, p-1].
      Different i (i0<n-1 vs n-1). ✓
    - Group 2 term for k0: contributes T[n-1, j, k0] for j<m-1.
      Group 3 term for j0: contributes T[n-1, j0, p-1].
      Different k (k0<p-1 vs p-1). ✓
    - Group 3 term for j0: contributes T[n-1, j0, p-1].
      Group 4 term: contributes T[n-1, m-1, p-1].
      Different j (j0<m-1 vs m-1). ✓
    
    All good! Let me implement this.
    """
    rank = (n-1)*(m-1)*(p-1) + n + m + p - 2
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    t = 0
    
    # Core: (n-1) x (m-1) x (p-1)
    for i in range(n-1):
        for j in range(m-1):
            for k in range(p-1):
                if t >= rank:
                    return None, None, None
                U[i*m + j][t] = 1.0
                V[j*p + k][t] = 1.0
                W[k*n + i][t] = 1.0
                t += 1
    
    # Group 1: (n-1) terms for (i, m-1, k) for i<n-1, all k
    for i in range(n-1):
        if t >= rank:
            return None, None, None
        U[i*m + (m-1)][t] = 1.0
        for k in range(p):
            V[(m-1)*p + k][t] = 1.0
            W[k*n + i][t] = 1.0
        t += 1
    
    # Group 2: (p-1) terms for (n-1, j, k) for j<m-1, k<p-1
    for k in range(p-1):
        if t >= rank:
            return None, None, None
        for j in range(m-1):
            U[(n-1)*m + j][t] = 1.0
            V[j*p + k][t] = 1.0
        W[k*n + (n-1)][t] = 1.0
        t += 1
    
    # Group 3: (m-1) terms for (n-1, j, p-1) for j<m-1
    for j in range(m-1):
        if t >= rank:
            return None, None, None
        U[(n-1)*m + j][t] = 1.0
        V[j*p + (p-1)][t] = 1.0
        W[(p-1)*n + (n-1)][t] = 1.0
        t += 1
    
    # Group 4: 1 term for (n-1, m-1, p-1)
    if t >= rank:
        return None, None, None
    U[(n-1)*m + (m-1)][t] = 1.0
    V[(m-1)*p + (p-1)][t] = 1.0
    W[(p-1)*n + (n-1)][t] = 1.0
    t += 1
    
    return U, V, W

def _validate(U, V, W, n, m, p, rank):
    """Check that U,V,W exactly decompose the matrix mult tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Reconstruct tensor T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
    T = np.zeros((n, m, p), dtype=float)
    for t in range(rank):
        u_mat = U[:, t].reshape(n, m)
        v_mat = V[:, t].reshape(m, p)
        w_mat = W[:, t].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + DEADLINE - SAFETY_MARGIN
    
    # Best known: schoolbook
    rank = n * m * p
    U, V, W = _build_factor(n, m, p, [(i, j, k) for i in range(n) for j in range(m) for k in range(p)], rank)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try bordered decomposition
    bordered_rank = (n-1)*(m-1)*(p-1) + n + m + p - 2
    if bordered_rank < best_rank:
        U_b, V_b, W_b = _build_bordered(n, m, p)
        if U_b is not None:
            if _validate(U_b, V_b, W_b, n, m, p, len(U_b[0])):
                best_result = {"U": U_b, "V": V_b, "W": W_b}
                best_rank = len(U_b[0])
    
    return best_result
