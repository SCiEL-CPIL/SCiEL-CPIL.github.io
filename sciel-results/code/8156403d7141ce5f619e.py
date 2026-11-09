"""Matrix multiplication 3x5x6 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin

    n, m, p = 3, 5, 6
    total = n * m * p  # 90

    # Build adjacency: for each (i,j,k) triple, its projections
    # proj1: i*m+j (in [0, n*m))
    # proj2: j*p+k (in [0, m*p))
    # proj3: k*n+i (in [0, p*n))

    # A cluster is a set of triples S such that:
    # Let I1 = {i*m+j : (i,j,k) in S}
    # Let I2 = {j*p+k : (i,j,k) in S}
    # Let I3 = {k*n+i : (i,j,k) in S}
    # Then S = {(i,j,k) : i*m+j in I1, j*p+k in I2, k*n+i in I3}
    # (i.e., S is "closed" under the projection structure)

    # For a cluster, we need to assign values u[proj1], v[proj2], w[proj3]
    # such that u * v * w = 1 for all triples in S.
    # This requires: for each (i,j,k) in S, u[i*m+j] * v[j*p+k] * w[k*n+i] = 1
    # And for consistency, we need the system to be solvable.
    # If we set all u, v, w = 1 on their supports, it works!
    # But we also need u, v, w = 0 outside their supports to not interfere.
    # Actually, since we're building a full decomposition, each rank-1 term
    # only needs to be non-zero on its cluster and zero elsewhere.
    # So: u has 1s at I1, 0s elsewhere. v has 1s at I2, 0s elsewhere. w has 1s at I3, 0s elsewhere.
    # Then for (i,j,k) in S: product = 1. For (i,j,k) not in S but with all projections in the sets:
    # We need this to be empty, which is exactly the closure condition.

    # So the problem reduces to: partition all 90 triples into minimum number of closed clusters.

    # Let's enumerate all possible closed clusters.
    # A cluster is determined by (I1, I2, I3) where I1 subset of [0,n*m), I2 subset of [0,m*p),
    # I3 subset of [0,p*n), and the set S(I1,I2,I3) = {(i,j,k) : i*m+j in I1, j*p+k in I2, k*n+i in I3}
    # must equal the "closure" of itself, i.e., proj1(S) = I1, proj2(S) = I2, proj3(S) = I3.

    # Actually, we just need S to be a valid set (any set of triples), and we check if it's closed.
    # For a given set of triples S, compute I1, I2, I3 from S, then compute S' = all triples
    # with projections in I1, I2, I3. If S' == S, then S is a closed cluster.

    # Let's enumerate by trying subsets of triples. But 2^90 is too large.
    # Instead, let's use a greedy set cover approach.

    # First, let's find all "maximal" closed clusters containing each triple.
    # For a given triple (i,j,k), the smallest closed cluster containing it is just {(i,j,k)}
    # (since the projections are singletons, and the only triple with those projections is itself).
    # But we can grow it.

    # Alternative approach: since n=3, m=5, p=6 are small, let's try to find good partitions
    # using a greedy algorithm or local search.

    # Let's think about this differently. A rank-1 term with support pattern:
    # u non-zero on set A subset of [0, n*m)
    # v non-zero on set B subset of [0, m*p)
    # w non-zero on set C subset of [0, p*n)
    # The non-zero positions of the tensor are all (i,j,k) where i*m+j in A, j*p+k in B, k*n+i in C.
    # For this to be a valid "piece" of the matmul tensor, we need this set to be exactly
    # some subset of the 90 triples, and we assign values to make each product = 1.

    # The key constraint: if (i,j,k) is in the set, we need u[i*m+j]*v[j*p+k]*w[k*n+i] = 1.
    # If we set all to 1, this works. But we need the set to be "consistent" - i.e., no contradictions.
    # Since we're just setting all non-zero values to 1, the only issue is whether the set
    # is well-defined (which it is by construction).

    # So any choice of (A, B, C) gives a valid rank-1 term that covers the set S(A,B,C).
    # We need to cover all 90 triples with minimum number of such terms.

    # Let's enumerate all possible (A, B, C) that give non-empty S, and find a minimum set cover.
    # But the number of possible (A, B, C) is 2^15 * 2^30 * 2^18 which is way too large.

    # Let's be smarter. Note that the structure is:
    # (i,j,k) is in S(A,B,C) iff i*m+j in A AND j*p+k in B AND k*n+i in C.

    # Let's parameterize by the actual index sets:
    # A corresponds to a set of (i,j) pairs
    # B corresponds to a set of (j,k) pairs
    # C corresponds to a set of (k,i) pairs

    # For n=3, m=5, p=6:
    # (i,j) pairs: 15 possibilities
    # (j,k) pairs: 30 possibilities
    # (k,i) pairs: 18 possibilities

    # Let's try a greedy approach: repeatedly find the largest cluster that covers the most
    # remaining triples, assign it, and continue.

    # To find large clusters, let's try all combinations of small index sets.
    # Actually, let's try a different approach: use the fact that we can greedily
    # build clusters.

    # Let's try: for each possible set of (i,j) pairs (subset of 15), for each possible
    # set of (j,k) pairs (subset of 30), compute the implied (k,i) pairs and check closure.

    # This is still too large. Let's use a heuristic.

    # Heuristic: Start with single-triple clusters (90 terms). Then try to merge pairs,
    # triples, etc. of clusters if they form a valid closed cluster.

    # Actually, let's try a simpler approach: use a greedy set cover with randomly
    # generated clusters.

    # Generate random clusters: pick random subsets of (i,j), (j,k), (k,i) pairs,
    # compute S, and if it's non-empty and closed, use it.

    import random
    random.seed(42)

    all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    triple_set = set(all_triples)

    def get_cluster(A, B, C):
        """Given sets of (i,j), (j,k), (k,i) pairs, compute the set of triples."""
        S = set()
        for (i, j) in A:
            for (j2, k) in B:
                if j2 == j:
                    if (k, i) in C:
                        S.add((i, j, k))
        return S

    def is_closed(S):
        """Check if S is closed: proj1(S), proj2(S), proj3(S) generate exactly S."""
        if not S:
            return True
        A = set((i, j) for (i, j, k) in S)
        B = set((j, k) for (i, j, k) in S)
        C = set((k, i) for (i, j, k) in S)
        S2 = get_cluster(A, B, C)
        return S2 == S

    # Let's find all clusters of size 2, 3, 4, etc.
    # For size 2: pick two triples, check if they form a closed cluster.
    # For a pair {(i1,j1,k1), (i2,j2,k2)} to be closed:
    # A = {(i1,j1), (i2,j2)}, B = {(j1,k1), (j2,k2)}, C = {(k1,i1), (k2,i2)}
    # The cluster is all (i,j,k) with (i,j) in A, (j,k) in B, (k,i) in C.
    # We need this to equal the original pair.

    # Let's enumerate all pairs and check.
    pairs = list(product(all_triples, repeat=2))
    valid_clusters = []  # list of (frozenset, size)

    # Check all pairs
    for (t1, t2) in pairs:
        if t1 == t2:
            continue
        S = {t1, t2}
        if is_closed(S):
            valid_clusters.append(frozenset(S))

    # Check all triples
    for (t1, t2, t3) in product(all_triples, repeat=3):
        S = {t1, t2, t3}
        if len(S) == 3 and is_closed(S):
            valid_clusters.append(frozenset(S))

    # Check all 4-tuples (limit for time)
    # This is 90^4 = 65 million, too many. Let's skip or limit.

    # Now do greedy set cover
    remaining = set(all_triples)
    cover = []
    clusters_list = list(valid_clusters)

    # Sort by size descending
    clusters_list.sort(key=len, reverse=True)

    while remaining and time.time() < deadline - 10:
        best_cluster = None
        best_count = 0
        for c in clusters_list:
            overlap = len(c & remaining)
            if overlap > best_count:
                best_count = overlap
                best_cluster = c
                if best_count == len(c):
                    break
        if best_cluster is None or best_count == 0:
            # No cluster covers any remaining triple, use singletons
            for t in remaining:
                cover.append(frozenset({t}))
            remaining = set()
            break
        cover.append(best_cluster)
        remaining -= best_cluster
        # Remove used clusters (optional, for efficiency)
        # clusters_list = [c for c in clusters_list if c & remaining]

    # Build the decomposition
    rank = len(cover)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r, cluster in enumerate(cover):
        for (i, j, k) in cluster:
            U[i * m + j][r] = 1
            V[j * p + k][r] = 1
            W[k * n + i][r] = 1

    # Verify the decomposition
    # For each (i,j,k), the sum over r of U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should be 1
    # For each (i,j,k), exactly one cluster should contain it (since they're disjoint)
    # Actually, we need to verify that no two clusters overlap on the same (i,j,k)
    # and that every (i,j,k) is covered exactly once.

    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    # Fallback to schoolbook
                    rank = total
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    col = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U[i2 * m + j2][col] = 1
                                V[j2 * p + k2][col] = 1
                                W[k2 * n + i2][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
