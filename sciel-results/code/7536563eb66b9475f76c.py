"""Rewritten program: clique partition of split graphs via minimum clique cover on complement."""
# EVOLVE-BLOCK-START
def construct():
    import random
    from itertools import combinations

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def max_clique_in_complement(n, k, cross_edges_set):
        """
        Find a maximum clique in the complement graph restricted to vertices 0..n-1.
        In the original graph, a clique is a set where every pair is an edge.
        In the complement, a clique in original = independent set in complement.
        
        Actually, let's think differently. We want to cover all edges with cliques.
        A clique in the original graph is a set of vertices where all pairs are edges.
        
        For split graphs:
        - The first k vertices form a clique (all edges among them).
        - Vertices k..n-1 form an independent set (no edges among them).
        - Cross edges exist with probability p.
        
        Strategy: 
        1. The clique on the first k vertices covers all edges among them. But we can do better
           by including some independent set vertices that are connected to all vertices in the clique.
        
        Actually, let me reconsider. A clique in a split graph can include:
        - Some subset of the clique part (first k vertices)
        - Some subset of the independent set (remaining vertices) that are all connected to all vertices in the clique part subset
        
        Since the independent set has no internal edges, any clique can contain at most... wait,
        no, a clique can contain multiple independent set vertices only if they have no edges between them,
        but they do need to be connected to each other. Since there are no edges between independent set
        vertices, a clique can contain at most ONE independent set vertex, OR it can contain multiple
        independent set vertices only if... no, two independent set vertices are NOT connected, so they
        cannot both be in a clique.
        
        So each clique contains at most one vertex from the independent set (vertices k..n-1).
        
        Therefore, the problem reduces to:
        - We have the clique C = {0, ..., k-1}
        - We have independent set I = {k, ..., n-1}
        - Edges within C are all present.
        - Edges within I are all absent.
        - Cross edges are given.
        
        A clique can be:
        - A subset of C (any size >= 2)
        - A subset of C plus one vertex v from I, where v is connected to all vertices in the subset of C.
        
        We need to cover all edges exactly once with such cliques.
        
        Edges to cover:
        1. All edges within C: these must be covered by cliques that are subsets of C.
        2. All cross edges: each cross edge (u, v) with u in C, v in I must be covered by a clique 
           that contains both u and v. Since v is in I, the clique must be {u, v} or a larger set 
           containing u, v and possibly other vertices from C that are all connected to v and to each other.
        
        Key insight: Since cliques are edge-disjoint, and edges within C must be covered by cliques 
        entirely within C, let's think about how to partition the edges within C and the cross edges.
        
        Actually, a cleaner approach: Since each clique contains at most one vertex from I, we can 
        think of assigning each vertex v in I to a clique that contains v (or not using v at all, 
        but then edges incident to v must be covered by cliques not containing v, which is impossible 
        since any clique containing an edge (u,v) must contain v).
        
        So every vertex v in I that has at least one cross edge must appear in at least one clique. 
        In fact, all edges incident to v must be covered by cliques containing v.
        
        Let's use a greedy approach that is known to be good for split graphs:
        
        1. First, handle the clique part C. We need to cover all edges within C with cliques (subsets of C).
           The simplest way is one big clique C itself, but that might not be optimal if we want to 
           combine C vertices with I vertices.
        
        2. For each vertex v in I, consider the set N(v) of its neighbors in C. Any clique containing v 
           must be a subset of N(v) union {v}, and the subset of N(v) must form a clique in C (which it 
           always does since C is a clique). So any subset S of N(v) with |S| >= 1 gives a clique S ∪ {v}.
        
        3. The edges within C that are not covered by cliques that also include I vertices must be covered 
           by cliques entirely within C.
        
        This is complex. Let me try a simpler greedy:
        
        Greedy Algorithm:
        1. Start with all edges uncovered.
        2. While there are uncovered edges:
           a. Find the largest clique (in terms of number of uncovered edges it covers) that can be formed.
           b. Add it to the partition and remove its edges from the uncovered set.
        
        For efficiency, since n <= 90, we can do this.
        
        But let's think about a smarter approach specific to split graphs.
        
        Observation: In a split graph, the minimum clique cover of edges can be found by:
        - The clique part C has binomial(k,2) edges.
        - Each vertex v in I has deg(v) cross edges.
        - A clique containing v and a subset S of N(v) covers |S| cross edges and binomial(|S|,2) edges within C.
        
        Let's try: For each v in I, we can form cliques {v} ∪ S where S ⊆ N(v). 
        The edges within C covered by all such cliques must not overlap.
        
        This is still complex. Let me just implement a reasonable greedy:
        
        1. Identify all maximal cliques in the graph.
        2. Use a greedy set cover / edge cover approach.
        
        Actually, for the time limit, let's do:
        1. The first k vertices form a clique. Let's call this C.
        2. For each vertex v in I (indices k to n-1), let N(v) be its neighbors in C.
        3. We can create a clique {v} ∪ N(v) for each v, but this would over-cover edges within C.
        
        Better greedy:
        - Maintain a set of uncovered edges.
        - Repeatedly find a clique that maximizes the number of uncovered edges it covers.
        - Add it to the solution and remove those edges.
        
        To find cliques efficiently:
        - Any clique is either a subset of C, or a subset of C plus one vertex from I.
        - For subsets of C: the best is to take the largest subset of C whose edges are all uncovered. But since C is a clique, any subset is a clique. The uncovered edges within C form a graph on C; we want a clique in that graph, which is just a set of vertices where all pairs are uncovered edges. Since the original C is a complete graph, the uncovered edges within C are a complete graph on some subset minus some edges. Actually, the uncovered edges within C are a subgraph of K_k. A clique in this subgraph is a set of vertices where all pairs are still uncovered.
        
        This is getting complicated. Let me just use a simple greedy that works well in practice:
        
        1. Start with the full clique C = {0, ..., k-1}. Check how many of its edges are uncovered. If all, add it.
        2. For each v in I, form the clique {v} ∪ N(v) and check uncovered edges.
        3. Pick the clique that covers the most uncovered edges.
        4. Repeat.
        
        Given n <= 90 and the structure, this should be fast enough.
        """
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in cross_edges_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # All edges to cover
        all_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                all_edges.add((u, v))
        for u, v in cross_edges_set:
            all_edges.add((min(u, v), max(u, v)))
        
        uncovered = set(all_edges)
        cliques = []
        
        while uncovered:
            best_clique = None
            best_count = 0
            
            # Candidate 1: subsets of C
            # Try the full C
            c_clique = list(range(k))
            count = 0
            for u, v in combinations(c_clique, 2):
                if (min(u, v), max(u, v)) in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_clique = c_clique
            
            # Try adding one vertex from I to subsets of C
            for v in range(k, n):
                if not adj[v]:  # no cross edges, skip
                    continue
                # The clique is {v} ∪ S where S ⊆ N(v)
                # Try S = N(v)
                S = list(adj[v])  # neighbors in C
                if not S:
                    continue
                clique = S + [v]
                count = 0
                # Edges within S
                for u1, u2 in combinations(S, 2):
                    if (min(u1, u2), max(u1, u2)) in uncovered:
                        count += 1
                # Edges from v to S
                for u in S:
                    if (min(u, v), max(u, v)) in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_clique = clique
            
            # Also try smaller subsets if the above didn't find anything good
            if best_clique is None or best_count == 0:
                # Find any uncovered edge and make a clique of size 2
                e = next(iter(uncovered))
                best_clique = [e[0], e[1]]
                best_count = 1
            
            # Add the clique
            cliques.append(best_clique)
            # Remove covered edges
            for u, v in combinations(best_clique, 2):
                e = (min(u, v), max(u, v))
                if e in uncovered:
                    uncovered.discard(e)
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cross_edges_set = set()
        for u, v in edges:
            if u >= k or v >= k:
                cross_edges_set.add((u, v))
            else:
                cross_edges_set.add((u, v))  # This is within C, not cross
        # Actually, let me rebuild properly
        cross_edges_set = set()
        clique_edges = set()
        for u, v in edges:
            if u < k and v < k:
                clique_edges.add((u, v))
            else:
                cross_edges_set.add((u, v))
        
        cliques = max_clique_in_complement(n, k, cross_edges_set)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
