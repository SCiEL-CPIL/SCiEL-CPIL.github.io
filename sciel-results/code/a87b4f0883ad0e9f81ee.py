import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = 2, 3, 7
    NM, MP, PN = n*m, m*p, p*n  # 6, 21, 14
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    # Build schoolbook as fallback
    def schoolbook():
        rank = n*m*p  # 42
        U = np.zeros((NM, rank))
        V = np.zeros((MP, rank))
        W = np.zeros((PN, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Verify exact decomposition
    def verify(U, V, W):
        rank = U.shape[1]
        # Check required entries = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check zero entries
        for a in range(NM):
            i1, j1 = divmod(a, m)
            for b in range(MP):
                j2, k1 = divmod(b, p)
                for c in range(PN):
                    k2, i2 = divmod(c, n)
                    if not (i1==i2 and j1==j2 and k1==k2):
                        s = 0.0
                        for r in range(rank):
                            s += U[a, r] * V[b, r] * W[c, r]
                        if abs(s) > 1e-9:
                            return False
        return True
    
    # Fast verify (sample)
    def verify_fast(U, V, W, seed=42):
        rank = U.shape[1]
        rng = np.random.RandomState(seed)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-8:
                        return False
        for _ in range(300):
            a = rng.randint(NM)
            b = rng.randint(MP)
            c = rng.randint(PN)
            i1, j1 = divmod(a, m)
            j2, k1 = divmod(b, p)
            k2, i2 = divmod(c, n)
            if not (i1==i2 and j1==j2 and k1==k2):
                s = 0.0
                for r in range(rank):
                    s += U[a, r] * V[b, r] * W[c, r]
                if abs(s) > 1e-8:
                    return False
        return True
    
    # Strategy: Build rank-1 terms with structured patterns
    # Pattern: for each term r, specify which (i,j), (j,k), (k,i) are nonzero
    # Then solve for values to satisfy all 42 equations + zero constraints
    
    # Key observation: if we group by j, terms with same j interact
    # For j=0: entries (i,0,k) for i in {0,1}, k in {0..6} = 14 entries
    # For j=1: entries (i,1,k) = 14 entries
    # For j=2: entries (i,2,k) = 14 entries
    # Total 42
    
    # A rank-1 term with u supported on S_u ⊆ {0..5}, v on S_v ⊆ {0..20}, w on S_w ⊆ {0..13}
    # contributes to entry (a,b,c) iff a∈S_u, b∈S_v, c∈S_w
    # For it to only hit required entries: for any a=(i1,j1), b=(j2,k1), c=(k2,i2) in supports
    # we need i1=j2=k2 and j1=k1=i2... wait, i1=i2, j1=j2, k1=k2
    
    # Let's try: terms where v is supported on a single j-slice
    # v supported on {j*p, j*p+1, ..., j*p+p-1} for some j
    # Then for the term to be valid, u must be supported on {(i,j): i∈I} and w on {(k,i): i∈I, k∈K}
    # where I ⊆ {0,1}, K ⊆ {0..6}
    # And the product must be consistent
    
    # Try rank-41: can we merge two schoolbook terms?
    # Terms (i1,j,k1) and (i2,j,k2) with same j
    # Merge into one term: u[(i1,j)] = a1, u[(i2,j)] = a2, v[(j,k1)] = b1, v[(j,k2)] = b2
    # w[(k1,i1)] = c1, w[(k2,i2)] = c2
    # Need: a1*b1*c1 = 1, a2*b2*c2 = 1 (if i1≠i2 or k1≠k2)
    # But also need: a1*b2*w[(k2,i1)] = 0 and a2*b1*w[(k1,i2)] = 0
    # So w[(k2,i1)] = 0 and w[(k1,i2)] = 0 (assuming a1,b2,a2,b1 nonzero)
    # This is satisfied if (k2,i1) ≠ (k1,i2) which means k2≠k1 or i1≠i2
    
    # So if i1≠i2, we can merge! The term has:
    # u[(i1,j)] = a1, u[(i2,j)] = a2
    # v[(j,k1)] = b1, v[(j,k2)] = b2
    # w[(k1,i1)] = c1, w[(k2,i2)] = c2
    # with a1*b1*c1 = 1, a2*b2*c2 = 1
    # And we need to check that this term doesn't create spurious nonzero entries
    
    # The term contributes to entry (a,b,c) = ((i',j'),(j',k'),(k',i''))
    # iff i'∈{i1,i2}, j'=j, k'∈{k1,k2}, i''∈{i1,i2}, and the specific combos
    # Actually: u[(i',j)] nonzero iff i'∈{i1,i2}
    # v[(j,k')] nonzero iff k'∈{k1,k2}
    # w[(k',i'')] nonzero iff (k',i'')∈{(k1,i1),(k2,i2)}
    # So entry (i',j,k',i'') gets contribution iff i'∈{i1,i2}, k'∈{k1,k2}, (k',i'')∈{(k1,i1),(k2,i2)}
    # The valid entries are (i1,j,k1,i1) and (i2,j,k2,i2)
    # Spurious: (i1,j,k2,i1) needs (k2,i1)∈{(k1,i1),(k2,i2)} → needs k2=k1 and i1=i2, or k2=k2 and i1=i2
    # If i1≠i2: (k2,i1) is in set only if i1=i2 (no) or i1=i2... 
    # (k2,i1) = (k1,i1) requires k2=k1
    # (k2,i1) = (k2,i2) requires i1=i2
    # So if i1≠i2 AND k1≠k2: no spurious entries!
    # If i1≠i2 AND k1=k2: (k2,i1)=(k1,i1) is in set, so entry (i1,j,k1,i1) gets extra from this term
    #   But that's a required entry, so it's fine (just adds to it)
    #   Wait, we need the SUM to be 1. If this term contributes to (i1,j,k1) and (i2,j,k1),
    #   and we removed the original (i1,j,k1) and (i2,j,k1) terms, we need this term to give exactly 1 to each
    
    # Let's try: merge (i1,j,k) and (i2,j,k) for same k, different i
    # Term: u[(i1,j)]=a1, u[(i2,j)]=a2, v[(j,k)]=b, w[(k,i1)]=c1, w[(k,i2)]=c2
    # Entry (i1,j,k): a1*b*c1 = 1
    # Entry (i2,j,k): a2*b*c2 = 1
    # Spurious: (i1,j,k,i2): needs w[(k,i2)]≠0 (yes) and u[(i1,j)]≠0 (yes) and v[(j,k)]≠0 (yes)
    #   So entry (i1,j,k,i2) = (a=i1, b=j*k, c=k*i2) gets a1*b*c2
    #   But (i1,j,k,i2) is NOT a required entry (i1≠i2)!
    #   So we need a1*b*c2 = 0, contradiction since all nonzero
    # So we CANNOT merge terms with same k but different i
    
    # What about same i, different k?
    # Merge (i,j,k1) and (i,j,k2): u[(i,j)]=a, v[(j,k1)]=b1, v[(j,k2)]=b2, w[(k1,i)]=c1, w[(k2,i)]=c2
    # Entry (i,j,k1): a*b1*c1 = 1
    # Entry (i,j,k2): a*b2*c2 = 1
    # Spurious: (i,j,k1,i) is required. (i,j,k2,i) is required.
    # Other entries: (i',j,k',i'') where i'≠i or k'∉{k1,k2}
    # u[(i',j)] = 0 if i'≠i, so no contribution
    # So only entries with i'=i, j'=j, k'∈{k1,k2}, i''=i are affected
    # These are exactly (i,j,k1,i) and (i,j,k2,i), both required!
    # So this works! We can merge two terms with same (i,j) but different k
    
    # So: for each (i,j), we can merge all p=7 k-values into 1 term?
    # Term: u[(i,j)]=a, v[(j,k)]=b_k for all k, w[(k,i)]=c_k for all k
    # Need: a * b_k * c_k = 1 for each k
    # Spurious entries: (i,j,k,i) for all k are required. No other entries affected.
    # Wait, entry (i,j,k,i) where k is any value: this IS required (i,j,k)
    # So yes! One term per (i,j) covers all k!
    # That gives rank = n*m = 6!
    
    # But wait: does the term affect entry (i',j,k,i) for i'≠i?
    # u[(i',j)] = 0 for i'≠i, so no.
    # What about entry (i,j,k,i'') for i''≠i?
    # w[(k,i'')] = 0 for i''≠i (we only set w[(k,i)]), so no.
    # Perfect!
    
    # So rank 6 should work! Let me verify:
    # For each (i,j) in {0,1}×{0,1,2}, create term r:
    # u[(i,j)] = 1
    # v[(j,k)] = 1 for all k
    # w[(k,i)] = 1 for all k
    # Then entry (i,j,k) = sum_r u[(i,j),r] * v[(j,k),r] * w[(k,i),r]
    # Only term r=(i,j) has u[(i,j),r]≠0, and for that term v[(j,k),r]=1, w[(k,i),r]=1
    # So entry = 1*1*1 = 1. 
    
    # Check spurious: entry (i1,j1,k,i2) with (i1,j1)≠(i2,·) or i1≠i2
    # If i1≠i2: term r=(i1,j1) has w[(k,i1),r]≠0 but we need w[(k,i2),r] which is 0 (since i2≠i1)
    #   Wait, the entry is (a,b,c) = ((i1,j1),(j1,k),(k,i2))
    #   Term r contributes if u[(i1,j1),r]≠0, v[(j1,k),r]≠0, w[(k,i2),r]≠0
    #   For term r=(i1,j1): u[(i1,j1),r]=1, v[(j1,k),r]=1, w[(k,i2),r]=? 
    #   w[(k,i2),r] is nonzero only if i2=i1 (since term r only sets w[(k,i1),r])
    #   So if i2≠i1, w[(k,i2),r]=0, no contribution
    #   For term r'≠(i1,j1): u[(i1,j1),r']=0, no contribution
    # So spurious entry = 0. 
    
    # Rank 6 works! Let me construct it.
    
    rank = n * m  # 6
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            U[i*m+j, col] = 1.0
            for k in range(p):
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
            col += 1
    
    if verify(U, V, W):
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Fallback: schoolbook
    U, V, W = schoolbook()
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
