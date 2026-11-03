import numpy as np
import itertools
import time

D = 11

def construct():
    deadline = time.time() + 120  # safety margin
    
    # Generate all 4-subsets of range(11)
    subsets = list(itertools.combinations(range(D), 4))
    n = len(subsets)  # 330
    
    # For each subset, compute its 4 triples
    subset_triples = []
    for s in subsets:
        triples = set()
        for i in range(4):
            for j in range(i+1, 4):
                for k in range(j+1, 4):
                    triples.add((s[i], s[j], s[k]))
        subset_triples.append(triples)
    
    # Build conflict graph: i and j conflict if they share exactly 3 elements
    # Two 4-sets share exactly 3 elements iff they share exactly 3 triples
    # Actually, two 4-sets share 3 elements iff |S ∩ T| = 3
    # Let's compute intersection sizes directly
    
    # For efficiency, map each triple to the list of subsets containing it
    triple_to_subsets = {}
    for idx, triples in enumerate(subset_triples):
        for t in triples:
            if t not in triple_to_subsets:
                triple_to_subsets[t] = []
            triple_to_subsets[t].append(idx)
    
    # Conflict: two subsets conflict if they share a triple (i.e., 3 elements)
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for t, idxs in triple_to_subsets.items():
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                adj[idxs[i]].add(idxs[j])
                adj[idxs[j]].add(idxs[i])
    
    # Greedy independent set: repeatedly pick vertex with min degree among remaining
    remaining = set(range(n))
    independent = []
    
    while remaining:
        # Pick vertex with minimum degree in remaining
        best = None
        best_deg = float('inf')
        for v in remaining:
            deg = len(adj[v] & remaining)
            if deg < best_deg:
                best_deg = deg
                best = v
        independent.append(best)
        remaining.discard(best)
        # Remove neighbors of best from remaining
        for u in adj[best]:
            remaining.discard(u)
    
    # Build vectors
    vectors = []
    for s in subsets:
        if s in [subsets[i] for i in independent]:  # This is inefficient
            pass
    
    # Better: use the independent set indices
    for idx in independent:
        s = subsets[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
