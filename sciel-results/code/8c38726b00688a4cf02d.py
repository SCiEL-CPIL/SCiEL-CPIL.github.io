"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    random.seed(42)
    
    n = 18
    best_edges = None
    best_score = 18  # worst case
    
    def build_adjacency(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_k4_free(adj):
        for a in range(n):
            for b in adj[a]:
                if b <= a:
                    continue
                for c in adj[a] & adj[b]:
                    if c <= b:
                        continue
                    if adj[a] & adj[b] & adj[c]:
                        return False
        return True
    
    def is_triangle_free_subset(adj, subset):
        """Check if induced subgraph on subset is triangle-free."""
        for a, b, c in combinations(subset, 3):
            if b in adj[a] and c in adj[a] and c in adj[b]:
                return False
        return True
    
    def max_triangle_free_size(adj):
        """Find the maximum size of a triangle-free induced subgraph via branch and bound."""
        # Try to find a triangle-free set of size k, for k from high to low
        # Use the fact that if a set is triangle-free, all its subsets are too
        
        # First, get all edges
        all_edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    all_edges.add((u, v))
        
        # For each vertex, its non-neighbors (within 0..n-1)
        # A triangle-free set can contain at most 2 neighbors of any vertex that form an edge between them...
        # Actually: in a triangle-free set S, for any v in S, the neighbors of v in S form an independent set.
        
        # Branch and bound: try to build a triangle-free set
        # State: current set (as sorted tuple), candidate vertices to add
        best_found = 0
        
        # Use a simpler approach: check all subsets of size k from largest down
        # But 18 choose k is too large for large k. Use branch and bound.
        
        def bt(current_set, candidates):
            """Branch and bound to find max triangle-free subset."""
            nonlocal best_found
            if not candidates:
                if len(current_set) > best_found:
                    best_found = len(current_set)
                return
            
            # Upper bound: len(current_set) + len(candidates)
            if len(current_set) + len(candidates) <= best_found:
                return
            
            # Try adding each candidate
            for i, v in enumerate(candidates):
                # Check if v can be added: v must not form a triangle with any two in current_set
                # i.e., for any two vertices a, b in current_set, not (a-b edge and a-v edge and b-v edge)
                # Equivalent: the neighbors of v in current_set must form an independent set
                v_neighbors_in_set = [x for x in current_set if x in adj[v]]
                # Check if v_neighbors_in_set is independent
                ok = True
                for a in range(len(v_neighbors_in_set)):
                    for b in range(a+1, len(v_neighbors_in_set)):
                        if v_neighbors_in_set[b] in adj[v_neighbors_in_set[a]]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                
                new_set = current_set + (v,)
                # New candidates: those after i in candidates that are compatible
                new_candidates = []
                for w in candidates[i+1:]:
                    # w must be compatible with v and all in new_set
                    w_neighbors_in_set = [x for x in new_set if x in adj[w]]
                    ok_w = True
                    for a in range(len(w_neighbors_in_set)):
                        for b in range(a+1, len(w_neighbors_in_set)):
                            if w_neighbors_in_set[b] in adj[w_neighbors_in_set[a]]:
                                ok_w = False
                                break
                        if not ok_w:
                            break
                    if ok_w:
                        new_candidates.append(w)
                
                bt(new_set, tuple(new_candidates))
            
            # Also branch on NOT including any more from candidates (already handled by recursion)
        
        # Start with empty set, all vertices as candidates
        bt((), tuple(range(n)))
        return best_found
    
    def edge_score(adj, u, v):
        """Score for adding edge (u,v): how many triangles it covers."""
        # Triangles formed: (u, v, w) where w is common neighbor
        return len(adj[u] & adj[v])
    
    def greedy_construct(seed):
        """Greedy: start with empty graph, add edges that maximize triangle coverage while keeping K4-free."""
        adj = [set() for _ in range(n)]
        edges = []
        
        # Generate all possible edges
        all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.seed(seed)
        random.shuffle(all_possible)
        
        for u, v in all_possible:
            if time.time() > deadline:
                break
            # Check if adding (u,v) creates a K4
            # A K4 would be formed if there exist a, b such that:
            # a,b in adj[u] & adj[v], and a-b is an edge, and a,b are both adjacent to u and v
            # More precisely: new K4 contains u, v, and two common neighbors a, b with a-b edge
            common = adj[u] & adj[v]
            creates_k4 = False
            for a in common:
                for b in common:
                    if a < b and b in adj[a]:
                        creates_k4 = True
                        break
                if creates_k4:
                    break
            if creates_k4:
                continue
            
            # Add the edge
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
        
        return edges, adj
    
    def local_search(edges, adj, max_iters=5000):
        """Flip edges to improve the objective (minimize max triangle-free size)."""
        edges_set = set()
        for e in edges:
            edges_set.add((min(e[0], e[1]), max(e[0], e[1])))
        
        current_score = max_triangle_free_size(adj)
        if current_score < best_score:
            best_score = current_score
            best_edges = [list(e) for e in edges]
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random edge to flip (add or remove)
            all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
            u, v = random.choice(all_possible)
            
            if (u, v) in edges_set:
                # Remove edge
                adj[u].discard(v)
                adj[v].discard(u)
                edges_set.discard((u, v))
            else:
                # Add edge - check K4
                common = adj[u] & adj[v]
                creates_k4 = False
                for a in common:
                    for b in common:
                        if a < b and b in adj[a]:
                            creates_k4 = True
                            break
                    if creates_k4:
                        break
                if creates_k4:
                    continue
                adj[u].add(v)
                adj[v].add(u)
                edges_set.add((u, v))
            
            # Evaluate
            new_score = max_triangle_free_size(adj)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edges = [list(e) for e in edges_set]
            else:
                # Revert
                if (u, v) in edges_set:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    edges_set.discard((u, v))
                else:
                    adj[u].discard(v)
                    adj[v].discard(u)
        
        return current_score
    
    # Phase 1: Multiple greedy constructions
    for seed in range(20):
        if time.time() > deadline:
            break
        edges, adj = greedy_construct(seed)
        score = max_triangle_free_size(adj)
        if score < best_score:
            best_score = score
            best_edges = [list(e) for e in edges]
        if best_score <= 10:
            break
    
    # Phase 2: Local search from best solution
    if best_edges is not None:
        adj = build_adjacency(best_edges)
        local_search(best_edges, adj, max_iters=20000)
    
    # Phase 3: More local search rounds with different random seeds
    for seed in range(5):
        if time.time() > deadline:
            break
        random.seed(seed + 100)
        # Perturb the best solution
        if best_edges is not None:
            adj = build_adjacency(best_edges)
            edges_set = set()
            for e in best_edges:
                edges_set.add((min(e[0], e[1]), max(e[0], e[1])))
            # Flip a few random edges
            all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
            for _ in range(5):
                u, v = random.choice(all_possible)
                if (u, v) in edges_set:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    edges_set.discard((u, v))
                else:
                    common = adj[u] & adj[v]
                    creates_k4 = False
                    for a in common:
                        for b in common:
                            if a < b and b in adj[a]:
                                creates_k4 = True
                                break
                        if creates_k4:
                            break
                    if not creates_k4:
                        adj[u].add(v)
                        adj[v].add(u)
                        edges_set.add((u, v))
            local_search([list(e) for e in edges_set], adj, max_iters=10000)
    
    # Final verification and return
    if best_edges is None:
        # Fallback: complete tripartite
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    
    # Verify K4-free
    adj = build_adjacency(best_edges)
    if not is_k4_free(adj):
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    
    return best_edges
