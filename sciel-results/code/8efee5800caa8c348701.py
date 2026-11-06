import time
from math import gcd
from itertools import combinations
import sys

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    N = 128
    N1 = N + 1

    # Precompute gcd matrix
    gcd_mat = [[0] * N1 for _ in range(N1)]
    for i in range(1, N1):
        for j in range(i + 1, N1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Find all bad triples: (a,b,c) with a<b<c such that gcd(a,b)=gcd(a,c)=gcd(b,c)
    # Group bad triples by their common gcd value
    bad_triples_by_g = {}
    for a in range(1, N1):
        for b in range(a + 1, N1):
            g_ab = gcd_mat[a][b]
            if g_ab == 1:
                continue
            for c in range(b + 1, N1):
                if gcd_mat[a][c] == g_ab and gcd_mat[b][c] == g_ab:
                    if g_ab not in bad_triples_by_g:
                        bad_triples_by_g[g_ab] = []
                    bad_triples_by_g[g_ab].append((a, b, c))

    # For each gcd value g, we have a set of numbers that participate in bad triples with gcd g.
    # But actually, a number can participate in bad triples with different gcd values.
    # The key insight: if we fix the set S, then for each g, the subset of S that forms bad triples
    # with gcd g must be independent in the hypergraph of those triples.
    
    # Alternative structural approach: 
    # Notice that if gcd(a,b) = gcd(a,c) = gcd(b,c) = g, then a/g, b/g, c/g are pairwise coprime? No.
    # Actually, let a = g*a', b = g*b', c = g*c'. Then gcd(a',b') = gcd(a',c') = gcd(b',c') = 1.
    # So a', b', c' are pairwise coprime.
    # This means the bad triples correspond to triples of numbers that are pairwise coprime after dividing by g.
    
    # Let's use a different structural decomposition:
    # For each number x, consider its "signature" in terms of which gcd classes it belongs to.
    # Actually, let's just do a global branch-and-bound but with a much better upper bound.
    
    # Precompute for each x, the set of pairs (a,b) with a<b such that {a,b,x} is a bad triple.
    completed_pairs = [set() for _ in range(N1)]
    for g, triples in bad_triples_by_g.items():
        for a, b, c in triples:
            completed_pairs[a].add((b, c))
            completed_pairs[b].add((a, c))
            completed_pairs[c].add((a, b))

    # Count how many bad triples each element is in
    triple_count = [0] * N1
    for g, triples in bad_triples_by_g.items():
        for a, b, c in triples:
            triple_count[a] += 1
            triple_count[b] += 1
            triple_count[c] += 1

    # Order elements by triple count (ascending), then by value
    order = sorted(range(1, N1), key=lambda x: (triple_count[x], x))
    n = len(order)

    # For fast checking, precompute for each index i in order, the completed_pairs of order[i]
    order_completed = [completed_pairs[order[i]] for i in range(n)]

    def can_add(S_set, x):
        """Check if x can be added to S_set without creating a bad triple."""
        cp = completed_pairs[x]
        for a in S_set:
            for b in S_set:
                if a < b and (a, b) in cp:
                    return False
        return True

    # Greedy initial solution
    def greedy():
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S

    best_set = greedy()
    best_size = len(best_set)

    # Branch and bound with a tighter upper bound using clique cover on the conflict graph
    # Conflict graph: vertices are numbers, edge between a and b if there exists c such that {a,b,c} is bad.
    # But this is not quite right for the 3-uniform hypergraph.
    
    # Better upper bound: for the remaining elements R, the maximum number we can add is at most
    # the size of a maximum independent set in the induced subhypergraph.
    # We can approximate this by: for each pair (a,b) in R that forms a bad triple with some c in R,
    # at most one of {a,b,c} can be chosen. This gives a fractional bound.
    
    # Simpler: use the fact that if we have a set of mutually conflicting pairs, we can bound.
    # Let's use a greedy coloring bound on the "conflict graph" where we connect a and b
    # if they are in a bad triple together (i.e., there exists c with {a,b,c} bad).
    
    # Precompute conflict graph adjacency
    conflict_adj = [set() for _ in range(N1)]
    for g, triples in bad_triples_by_g.items():
        for a, b, c in triples:
            conflict_adj[a].add(b)
            conflict_adj[a].add(c)
            conflict_adj[b].add(a)
            conflict_adj[b].add(c)
            conflict_adj[c].add(a)
            conflict_adj[c].add(b)

    sys.setrecursionlimit(10000)

    def branch_and_bound(idx, S_list, S_set, remaining_count):
        nonlocal best_size, best_set
        if time.time() > DEADLINE - 2:
            return

        # Upper bound: current size + remaining elements that can potentially be added
        # A better bound: for the remaining elements, compute a greedy upper bound
        # on the maximum independent set in the induced conflict graph.
        
        # Simple bound first
        if len(S_list) + remaining_count <= best_size:
            return

        if idx == n:
            if len(S_list) > best_size:
                best_size = len(S_list)
                best_set = set(S_list)
            return

        x = order[idx]

        # Option 1: exclude x
        branch_and_bound(idx + 1, S_list, S_set, remaining_count - 1)

        # Option 2: include x, if possible
        if can_add(S_set, x):
            S_list.append(x)
            S_set.add(x)
            branch_and_bound(idx + 1, S_list, S_set, remaining_count - 1)
            S_list.pop()
            S_set.remove(x)

    # This is still too slow. Let's use a different approach:
    # Use the structural decomposition by gcd values.
    
    # For each gcd value g, the numbers that can participate in bad triples with gcd g
    # are those that are multiples of g. But a number can participate in multiple g classes.
    
    # Let's try a different structural insight:
    # If we choose a set S, then for any three distinct a,b,c in S, it's not the case that
    # gcd(a,b)=gcd(a,c)=gcd(b,c).
    # Note that if gcd(a,b)=gcd(a,c)=gcd(b,c)=g, then g divides a,b,c.
    # So for each g, consider the set S_g = {x in S : g divides x}.
    # Then within S_g, no three elements can have pairwise gcd exactly g.
    # But they could have pairwise gcd > g.
    
    # This is getting complex. Let's just use a faster heuristic:
    # Simulated annealing with 1-swap and 1-add moves.

    def local_search(S, max_iters=5000):
        S = set(S)
        remaining = [x for x in range(1, N1) if x not in S]
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < DEADLINE - 5:
            improved = False
            iters += 1

            # Try to add an element
            for x in remaining[:]:
                if can_add(S, x):
                    S.add(x)
                    remaining.remove(x)
                    improved = True
                    break

            # Try 1-swap: remove one from S, add one from remaining
            if not improved and remaining:
                for a in list(S):
                    S.remove(a)
                    for b in remaining[:]:
                        if can_add(S, b):
                            S.add(b)
                            remaining.remove(b)
                            remaining.append(a)
                            improved = True
                            break
                    if not improved:
                        S.add(a)
                    if improved:
                        break

        return S

    best_set = local_search(list(best_set))
    best_size = len(best_set)

    # Try multiple random restarts with different orderings
    import random
    for seed in range(50):
        if time.time() > DEADLINE - 10:
            break
        random.seed(seed)
        
        # Random ordering
        rand_order = list(range(1, N1))
        random.shuffle(rand_order)
        
        # Greedy with random order
        S_rand = set()
        for x in rand_order:
            if can_add(S_rand, x):
                S_rand.add(x)
        
        S_rand = local_search(list(S_rand), max_iters=2000)
        if len(S_rand) > best_size:
            best_size = len(S_rand)
            best_set = S_rand

    if not best_set:
        best_set = {1}

    return sorted(best_set)


# Wrap to ensure nonempty
_orig_construct = construct
def construct():
    result = _orig_construct()
    if not result:
        return [1]
    return result
