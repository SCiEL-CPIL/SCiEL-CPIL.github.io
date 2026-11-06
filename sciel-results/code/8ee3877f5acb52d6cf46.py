# EVOLVE-BLOCK-START
from math import gcd, log
from collections import defaultdict

def construct():
    N = 128
    # Build a maximal independent set in the hypergraph where edges are bad triples.
    # Bad triple: distinct a,b,c with lcm(a,b)==lcm(a,c)==lcm(b,c)
    
    # Precompute lcm for all pairs (1..128)
    lcms = [[0]*N for _ in range(N)]
    for i in range(1, N):
        for j in range(i+1, N):
            lcms[i-1][j-1] = (i*j)//gcd(i,j)
            lcms[j-1][i-1] = lcms[i-1][j-1]
    
    # For each triple, check if it's bad
    # Precompute for each pair (a,b) the set of c such that (a,b,c) is bad
    # A triple (a,b,c) is bad iff lcm(a,b)==lcm(a,c)==lcm(b,c)
    # Note: lcm(a,c)==lcm(a,b) implies that a,c have same lcm as a,b
    # This is equivalent to: lcm(a,b) divides lcm(a,c) and lcm(a,c) divides lcm(a,b), i.e., equal.
    # Alternatively: let L = lcm(a,b). Then lcm(a,c)==L and lcm(b,c)==L.
    # This means c's prime exponents are bounded by L's, and for each prime p where a and b
    # have different exponents, c must have the max exponent.
    
    # For efficiency: for each pair (a,b), find all c such that lcm(a,b)==lcm(a,c)==lcm(b,c)
    # We'll store conflicts as a dict: for each pair (i,j) (0-indexed), set of conflicting c indices
    # But better: for each element c, store which pairs (a,b) would conflict if c is added
    # Actually, let's use a different approach: 
    # Maintain a set S. When considering adding x, check all pairs (a,b) in S.
    # To speed up, for each x, precompute the set of (a,b) pairs that would conflict.
    
    # Precompute for each x (1..128), the list of pairs (a,b) with a<b, a!=x, b!=x
    # such that lcm(a,b)==lcm(a,x)==lcm(b,x)
    # Store as: conflicts[x] = set of frozenset({a-1, b-1}) for 0-indexed pairs
    
    conflicts = [set() for _ in range(N)]
    for a in range(1, N):
        for b in range(a+1, N):
            L = lcms[a-1][b-1]
            for c in range(1, N):
                if c == a or c == b:
                    continue
                if lcms[a-1][c-1] == L and lcms[b-1][c-1] == L:
                    # (a,b,c) is a bad triple
                    ac, bc, cc = a-1, b-1, c-1
                    conflicts[ac].add(frozenset((bc, cc)))
                    conflicts[bc].add(frozenset((ac, cc)))
                    conflicts[cc].add(frozenset((ac, bc)))
    
    # Now do greedy: sort elements by some heuristic, add if no conflict with current set
    # Heuristic: try to add elements that have fewer conflicts first? Or more?
    # For maximum independent set, greedy with min-degree is common.
    # Here, the "degree" of a vertex x is the number of edges (bad triples) it's in.
    # Let's compute degree: number of bad triples containing x
    degree = [len(conflicts[i]) for i in range(N)]
    
    # Try multiple orderings: by degree, by value, random
    # Let's try: sort by (degree, value) ascending
    order = sorted(range(N), key=lambda i: (degree[i], i+1))
    
    selected = set()
    for idx in order:
        x = idx + 1
        # Check if adding x conflicts with any pair in selected
        can_add = True
        for pair in conflicts[idx]:
            a, b = tuple(pair)
            if a in selected and b in selected:
                can_add = False
                break
        if can_add:
            selected.add(idx)
    
    result = sorted(i+1 for i in selected)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
