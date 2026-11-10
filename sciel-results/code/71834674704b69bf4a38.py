import random
import itertools
import time

def construct():
    random.seed(42)
    n = 20
    
    def is_K33_free(adj):
        # adj[u] is a set of neighbors
        # Check: no three vertices have three common neighbors
        # Equivalent: no K3,3 subgraph
        # For any partition of 6 vertices into two sets of 3, check if all 9 cross-edges exist
        # More efficient: for any 3 vertices S, |N(S) common| <= 2
        # Actually, K3,3 exists iff there exist disjoint sets A, B of size 3 such that
        # every vertex in A is connected to every vertex in B.
        # Let's check all pairs of 3-sets that are disjoint.
        # 20 choose 3 = 1140, pairs of disjoint 3-sets... too many.
        # Better: for each triple A, compute common neighbors, check if any 3 of them form B
        # where B is also connected to all of A. Actually if |common neighbors of A| >= 3,
        # we need to check if any 3 of those common neighbors form a set B such that B is 
        # connected to all of A. But wait, if v is a common neighbor of A, v is connected to all of A.
        # So if |common neighbors of A| >= 3, then pick any 3 of them as B, and we have K3,3!
        # Wait, that's not right. K3,3 requires A and B to be disjoint. The common neighbors 
        # of A are vertices connected to all of A. If we pick 3 common neighbors as B, 
        # then every vertex in B is connected to every vertex in A. But we also need B to be
        # a set of 3 vertices. And A and B must be disjoint. Since common neighbors are 
        # different from A (in a simple graph, no self-loops, and neighbors are distinct),
        # B is disjoint from A. So yes! If any triple A has 3 or more common neighbors, 
        # the graph contains K3,3.
        
        for A in itertools.combinations(range(n), 3):
            common = adj[A[0]] & adj[A[1]] & adj[A[2]]
            if len(common) >= 3:
                return False
        return True
    
    def try_greedy_improve(adj, seed):
        random.seed(seed)
        # Try to add edges greedily
        improved = True
        while improved:
            improved = False
            # Get all missing edges
            missing = []
            for i in range(n):
                for j in range(i+1, n):
                    if j not in adj[i]:
                        missing.append((i, j))
            random.shuffle(missing)
            for (i, j) in missing:
                # Try adding edge (i, j)
                if j in adj[i]:
                    continue
                # Add it temporarily
                adj[i].add(j)
                adj[j].add(i)
                if not is_K33_free(adj):
                    # Remove it
                    adj[i].remove(j)
                    adj[j].remove(i)
                else:
                    improved = True
                    break  # Restart with new missing list
        return adj
    
    def count_edges(adj):
        return sum(len(adj[i]) for i in range(n)) // 2
    
    # Start with random graphs and improve
    best_adj = None
    best_edges = 0
    
    # Try different densities
    for p in [0.3, 0.4, 0.5, 0.6]:
        for trial in range(50):
            adj = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i].add(j)
                        adj[j].add(i)
            
            if is_K33_free(adj):
                adj = try_greedy_improve(adj, trial * 1000 + int(p * 10))
                e = count_edges(adj)
                if e > best_edges:
                    best_edges = e
                    best_adj = [set(s) for s in adj]
            else:
                # Try to repair by removing edges
                # For each triple with >= 3 common neighbors, remove an edge
                repaired = True
                while repaired:
                    repaired = False
                    for A in itertools.combinations(range(n), 3):
                        common = adj[A[0]] & adj[A[1]] & adj[A[2]]
                        if len(common) >= 3:
                            # Remove an edge from one of the common neighbors to one of A
                            v = list(common)[0]
                            a = A[0]
                            adj[v].remove(a)
                            adj[a].remove(v)
                            repaired = True
                            break
                    # After repair, try greedy improvement
                if is_K33_free(adj):
                    adj = try_greedy_improve(adj, trial * 1000 + int(p * 10) + 100)
                    e = count_edges(adj)
                    if e > best_edges:
                        best_edges = e
                        best_adj = [set(s) for s in adj]
    
    # Also try starting from the K2,18 seed and improving
    adj = [set() for _ in range(n)]
    for a in range(2):
        for b in range(2, 20):
            adj[a].add(b)
            adj[b].add(a)
    if is_K33_free(adj):
        adj = try_greedy_improve(adj, 999999)
        e = count_edges(adj)
        if e > best_edges:
            best_edges = e
            best_adj = [set(s) for s in adj]
    
    if best_adj is None:
        # Fallback to K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    # Convert to list of edges
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    return edges
