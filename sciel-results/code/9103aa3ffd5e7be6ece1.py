"""
K4-free graph on 18 vertices minimizing the size of the largest triangle-free induced subgraph.

Construction: Take the complete tripartite graph K_{6,6,6} and add a perfect matching
within each part. This remains K4-free (a K4 in a tripartite graph requires 4 vertices
in distinct parts, but with 3 parts, a K4 needs 2 in one part; those 2 must be non-
adjacent, which they are since matchings pair up vertices and any two in the same part
are either matched or not, but even if matched, we'd need edges to 2 vertices in each
other part forming K4... let me verify this more carefully).

Actually, let me reconsider. In K_{6,6,6} with parts A, B, C:
- A K4 would need vertices from at most 3 parts (pigeonhole: 4 vertices in 3 parts means
  at least 2 in one part, say A: u, v).
- For u, v in A to be in a K4, they need edges to 2 vertices in B and 2 in C, plus
  edges between those. But K4 has 4 vertices total, so: u, v, w (in B), x (in C).
  Edges needed: u-v, u-w, u-x, v-w, v-x, w-x.
  u-v exists (matching edge). u-w, v-w exist (complete bipartite A-B).
  u-x, v-x exist (complete bipartite A-C). w-x exists (complete bipartite B-C).
  So {u, v, w, x} IS a K4!

So adding matching edges to K_{6,6,6} creates K4s. That doesn't work.

Let me think of a different approach. We need K4-free and minimize max triangle-free induced subgraph.

The Erdős-Rogers problem: f(n) = min over K4-free G on n vertices of the largest
triangle-free induced subgraph.

For n=18, K_{6,6,6} gives 12 (any two parts form a triangle-free set of size 12).

Can we do better? The known bounds: f(n) >= n/2 for K4-free graphs (Erdős-Rogers).
So for n=18, f(18) >= 9. The question is whether we can achieve 9, 10, or 11.

Let me try a different construction. Consider a graph based on a 3-coloring where each
part is not independent but has some structure.

Alternative: Use a random K4-free graph and check its properties. But we need deterministic.

Let me try: partition into 3 parts of 6, but make each part a matching (3 edges each),
and keep complete bipartite connections between parts. Then check K4-freeness.

As shown above, this creates K4s. So that fails.

What if we use a 4-partite graph? K_{5,5,4,4} (18 vertices). Any two parts give
triangle-free sets of size 10, 9, etc. The largest pair is 5+5=10. So max triangle-free
induced subgraph <= 10? But we also need to check if there are larger triangle-free sets
spanning 3 or 4 parts.

In K_{5,5,4,4}, a triangle-free set can include vertices from multiple parts. If it has
vertices from 3 parts, say 2 from part1, 2 from part2, 2 from part3 (all edges exist between
parts), then any three form a triangle. So a triangle-free set in a complete multipartite
graph can have at most one vertex per part... no wait, that's wrong.

In a complete multipartite graph, a set is triangle-free iff it doesn't contain 3 vertices
from distinct parts (since any 3 from distinct parts form a triangle). Actually no: any two
from the same part are non-adjacent, so a set with 2 from part A and 1 from part B is
triangle-free (the edge between the two A-vertices is missing). A set with 2 from A, 2 from B:
the edges within A are missing, within B are missing, but all cross edges exist. So we have
edges a1-b1, a1-b2, a2-b1, a2-b2. Does this contain a triangle? a1-b1-a2? No edge a1-a2.
a1-b1-b2? No edge b1-b2. So no triangle. Thus {a1,a2,b1,b2} is triangle-free!

So in K_{m,n,p,q}, a set with 2 from part A and 2 from part B is triangle-free (size 4).
A set with 1 from each of 4 parts: all edges present, forms K4 (triangle present).
A set with 2 from A, 1 from B, 1 from C: edges a1-b, a2-b, a1-c, a2-c, b-c. Triangle?
a1-b-c: yes! a1-b, b-c, c-a1 all exist. So this IS a triangle.

So triangle-free sets in complete multipartite: can have at most 2 from any single pair of
parts, and if using vertices from 3+ parts, must be careful.

For K_{5,5,4,4}: the set {2 from part1, 2 from part2} is triangle-free, size 4.
{5 from part1, 0 from others} is triangle-free, size 5.
{5 from part1, 5 from part2} - is this triangle-free? As shown above, yes! Size 10.

So K_{5,5,4,4} has a triangle-free induced subgraph of size 10 (two parts).

Can we do better than 10? What about K_{6,6,3,3}? Two parts: 6+6=12, 6+3=9, 3+3=6.
So max pair gives 12. Worse.

K_{4,4,4,4,2}: pairs give 8, 6. But {4 from one part, 4 from another} = 8 triangle-free.
What about {4,4,1,1}? Check: vertices a1-a4 (part1), b1-b4 (part2), c (part3), d (part4).
Triangle? a1-b1-c: a1-b1 exists, b1-c exists, c-a1 exists. Yes, triangle.
So can't mix 3+ parts freely.

It seems complete multipartite graphs have their two largest parts forming a large triangle-free set.

To beat 10, I need a non-complete-multipartite construction. Let me try a random search with
branch and bound within the time limit, or use a known good construction.

Given time constraints, let me use K_{5,5,4,4} which gives score 10, improving on 12.
"""
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 100  # safety margin
    
    # K_{5,5,4,4} construction: parts of sizes 5,5,4,4
    # Vertices 0-4: part A, 5-9: part B, 10-13: part C, 14-17: part D
    # All edges between different parts, no edges within parts
    # This is K4-free (complete multipartite graphs are K_{r+1}-free for r parts)
    # Triangle-free induced subgraph: any two parts together form a bipartite (triangle-free) subgraph
    # Largest such: 5+5 = 10
    
    # Verify it's K4-free: in a 4-partite complete graph, a K4 would need 4 vertices
    # in 4 distinct parts (one per part) - but that's only if we pick one from each part.
    # Actually a K4 needs 4 mutually adjacent vertices. In K_{5,5,4,4}, pick one from each
    # of 4 parts: they're all mutually adjacent, forming K4! 
    # So K_{5,5,4,4} is NOT K4-free!
    
    # For K4-free, we need at most 3 parts in a complete multipartite graph.
    # So we're stuck with 3-partite: K_{6,6,6} gives 12.
    
    # Need a non-complete-multipartite K4-free graph.
    # Let me try: start with K_{6,6,6} and remove some edges strategically.
    
    # Alternative: Use a known construction. The triangle-free induced subgraph size
    # for K4-free graphs on n vertices: f(n) satisfies n/2 <= f(n) <= n - n^{2/3}... 
    # For n=18, this gives roughly 9 to 12.
    
    # Let me try a random K4-free graph approach with verification.
    
    def is_k4_free(adj, n):
        """Check if graph is K4-free using bitmask for speed."""
        for a in range(n):
            for b in range(a+1, n):
                if not (adj[a] >> b) & 1:
                    continue
                for c in range(b+1, n):
                    if not ((adj[a] >> c) & 1 and (adj[b] >> c) & 1):
                        continue
                    for d in range(c+1, n):
                        if ((adj[a] >> d) & 1 and (adj[b] >> d) & 1 and (adj[c] >> d) & 1):
                            return False
        return True
    
    def max_triangle_free_size(adj, n):
        """Find largest triangle-free induced subgraph size via branch and bound."""
        # Use the fact that in a triangle-free graph, for any vertex, its neighbors
        # form an independent set. So we can use this property.
        
        best = 0
        
        # Greedy initial solution
        visited = [False] * n
        for start in range(n):
            if visited[start]:
                continue
            # Try to build a large triangle-free set containing start
            current = set()
            candidates = list(range(n))
            # Add start
            current.add(start)
            # Remove non-neighbors of start that would conflict
            # Actually, triangle-free means: for any two vertices in the set, 
            # they can't share a common neighbor that's also in the set.
            # Equivalently: the induced subgraph is triangle-free.
            
            # Simple approach: try all subsets up to size 10? Too expensive for n=18.
            # Use branch and bound with the constraint that the subgraph must be triangle-free.
            pass
        
        # Better: enumerate all subsets of size s from n, check triangle-free.
        # For n=18, C(18,10) = 43758, C(18,9) = 48620. Manageable.
        
        for s in range(n, best, -1):
            found = False
            for subset in combinations(range(n), s):
                # Check if induced subgraph is triangle-free
                triangle_free = True
                subset_set = set(subset)
                for u, v, w in combinations(subset, 3):
                    if ((adj[u] >> v) & 1 and (adj[u] >> w) & 1 and (adj[v] >> w) & 1):
                        triangle_free = False
                        break
                if triangle_free:
                    best = s
                    found = True
                    break
            if not found:
                break
        
        return best
    
    # Start with K_{6,6,6} and try to remove edges to reduce max triangle-free size
    # while maintaining K4-freeness.
    
    n = 18
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    
    # Build K_{6,6,6}
    adj = [0] * n
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    
    # K_{6,6,6} is K4-free and has max triangle-free size 12.
    # Try removing edges to reduce this.
    
    best_edges = None
    best_score = 12
    
    # Try random edge removals
    all_cross_edges = []
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    all_cross_edges.append((a, b))
    
    random.seed(42)
    attempts = 0
    while time.time() < deadline and attempts < 5000:
        attempts += 1
        # Remove a random subset of edges
        num_remove = random.randint(1, 10)
        removed = random.sample(all_cross_edges, min(num_remove, len(all_cross_edges)))
        
        adj_copy = list(adj)
        for a, b in removed:
            adj_copy[a] &= ~(1 << b)
            adj_copy[b] &= ~(1 << a)
        
        # Check K4-free
        if not is_k4_free(adj_copy, n):
            continue
        
        # Check max triangle-free size
        score = max_triangle_free_size(adj_copy, n)
        
        if score < best_score:
            best_score = score
            # Build edge list
            best_edges = []
            for a in range(n):
                for b in range(a+1, n):
                    if (adj_copy[a] >> b) & 1:
                        best_edges.append([a, b])
    
    # If we found something better, return it; otherwise return K_{6,6,6}
    if best_edges is not None:
        return best_edges
    
    # Fallback: K_{6,6,6}
    edges = []
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.append([a, b])
    return edges
