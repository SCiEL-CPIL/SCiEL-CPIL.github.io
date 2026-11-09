import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 15
    # Partition: A = {0,1,2,3,4,5,6}, B = {7,8,9,10,11,12,13,14}
    A = list(range(7))
    B = list(range(7, 15))
    
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def build_adj_and_deg(edge_set):
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg
    
    def check_violations(edge_set):
        """
        Returns count of violating (s,t) pairs where deg(s)==deg(t) and a 3-path exists.
        A 3-path is s-u-v-t where s-u, u-v, v-t are edges and s,t,u,v distinct.
        """
        adj, deg = build_adj_and_deg(edge_set)
        
        # Group vertices by degree
        deg_groups = {}
        for i in range(n):
            deg_groups.setdefault(deg[i], []).append(i)
        
        violations = 0
        for d, vertices in deg_groups.items():
            if len(vertices) < 2:
                continue
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    s = vertices[i]
                    t = vertices[j]
                    # Check for path s-u-v-t
                    found = False
                    for u in adj[s]:
                        if u == t:
                            continue
                        for v in adj[u]:
                            if v == s or v == t:
                                continue
                            if t in adj[v]:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        violations += 1
        return violations
    
    def find_violating_path(edge_set):
        """Return one violating (s,t) pair or None."""
        adj, deg = build_adj_and_deg(edge_set)
        deg_groups = {}
        for i in range(n):
            deg_groups.setdefault(deg[i], []).append(i)
        
        for d, vertices in deg_groups.items():
            if len(vertices) < 2:
                continue
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    s = vertices[i]
                    t = vertices[j]
                    for u in adj[s]:
                        if u == t:
                            continue
                        for v in adj[u]:
                            if v == s or v == t:
                                continue
                            if t in adj[v]:
                                return (s, t)
        return None
    
    # Start with K_{7,8} bipartite graph - this is always valid
    # In a bipartite graph, all 3-paths go from one partition to the other
    # So endpoints are in different partitions, and we need to check degrees
    # But in K_{7,8}, all A vertices have degree 8, all B vertices have degree 7
    # So no two vertices in the same partition have equal degree... wait, they DO
    # A vertices all have degree 8, B vertices all have degree 7
    # A 3-path s-u-v-t: if s in A, u in B, v in A, t in B: deg(s)=8, deg(t)=7, not equal
    # If s in A, u in B, v in A, t in B: same
    # Actually in bipartite, 3-path always goes A-B-A-B or B-A-B-A
    # So endpoints are in the same partition
    # If s,t both in A: deg(s)=8, deg(t)=8 -> equal! And there IS a path
    # So K_{7,8} is NOT valid!
    
    # Let me reconsider. We need a base that is valid.
    # A star graph is valid (only 1 vertex with degree > 0, so no pair with equal degree > 0)
    # A matching is valid (all degrees are 0 or 1, but if two vertices have degree 1, 
    #   is there a 3-path between them? No, because a 3-path needs 4 distinct vertices)
    
    # Let's think about what graphs are valid.
    # If max degree <= 1, then no 3-paths exist at all, so it's valid.
    # If we have a 3-path s-u-v-t, we need deg(s) != deg(t).
    
    # Strategy: Use a graph where all vertices have distinct degrees, or 
    # where vertices with equal degree are not endpoints of any 3-path.
    
    # A simpler approach: start with empty graph and add edges greedily.
    # But that's slow. Let's use a smarter base.
    
    # Actually, let's use a "degree-graded" graph where we try to give each vertex 
    # a different degree. With 15 vertices, max degree is 14, so we can have degrees 0-14.
    # A graph with degrees 0,1,2,...,14 would be valid because no two vertices have the same degree.
    # Such a graph exists (it's a threshold graph or similar).
    
    # Let's construct a graph with distinct degrees.
    # One way: vertex i has degree i for i=0..14.
    # This is a "degree sequence" that must be graphical.
    # The sequence 0,1,2,...,14 is graphical (by Erdős–Gallai or Havel-Hakimi).
    
    # Let's use Havel-Hakimi to construct it.
    def havel_hakimi(deg_seq):
        """Construct a simple graph with given degree sequence, or return None."""
        n = len(deg_seq)
        if sum(deg_seq) % 2 != 0:
            return None
        if any(d < 0 or d > n-1 for d in deg_seq):
            return None
        
        # Use a greedy approach
        edges = set()
        remaining = list(deg_seq)
        vertices = list(range(n))
        
        for _ in range(n):
            # Find vertex with highest remaining degree
            max_idx = max(range(n), key=lambda i: remaining[i])
            if remaining[max_idx] == 0:
                continue
            if remaining[max_idx] > 0:
                # Connect to next 'remaining[max_idx]' vertices with positive degree
                # Sort by remaining degree descending
                candidates = [i for i in range(n) if i != max_idx and remaining[i] > 0]
                candidates.sort(key=lambda i: remaining[i], reverse=True)
                
                if len(candidates) < remaining[max_idx]:
                    return None
                
                for j in range(remaining[max_idx]):
                    v = candidates[j]
                    u, v2 = min(max_idx, v), max(max_idx, v)
                    edges.add((u, v2))
                    remaining[max_idx] -= 1
                    remaining[v] -= 1
        
        if any(r != 0 for r in remaining):
            return None
        
        return edges
    
    # Try to get a graph with distinct degrees 0,1,...,14
    deg_seq = list(range(15))
    base_edges = havel_hakimi(deg_seq)
    
    if base_edges is None or check_violations(base_edges) != 0:
        # Fallback: use a star-like graph or just start empty
        # Let's use a matching plus some edges
        base_edges = set()
        for i in range(7):
            base_edges.add((i, i+7))
        # This gives degrees: 7 vertices with deg 1 (0-6), 8 vertices with deg 1 (7-14)
        # Wait, that's 14 vertices with degree 1, and 1 vertex with degree 0
        # Two vertices with degree 1: is there a 3-path? No, need 4 distinct vertices.
        # So this is valid.
        # But we can do better.
        
        # Let's try a different approach: use a bipartite graph with unequal parts
        # and make degrees within each part distinct.
        # A = 0..6, B = 7..14
        # Give A vertices degrees 1,2,3,4,5,6,7 and B vertices degrees 1,2,3,4,5,6,7,8
        # But they must be consistent (sum of degrees in A = sum in B)
        # Sum A = 1+2+3+4+5+6+7 = 28
        # Sum B = 1+2+3+4+5+6+7+8 = 36
        # Not equal. Let's adjust.
        
        # Actually, let's just use the greedy approach from empty.
        base_edges = set()
    
    # Verify base is valid
    if check_violations(base_edges) != 0:
        # Reset to empty
        base_edges = set()
    
    current_edges = set(base_edges)
    best_edges = set(current_edges)
    best_count = len(current_edges)
    
    # Greedy addition: try to add each missing edge
    for e in all_edges:
        if time.time() > deadline:
            break
        if e not in current_edges:
            candidate = current_edges | {e}
            if check_violations(candidate) == 0:
                current_edges = candidate
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = set(current_edges)
    
    # Local search: try swaps to improve
    for step in range(50000):
        if time.time() > deadline:
            break
        
        # Try adding a random missing edge
        missing = [e for e in all_edges if e not in current_edges]
        if missing and random.random() < 0.5:
            e = random.choice(missing)
            candidate = current_edges | {e}
            if check_violations(candidate) == 0:
                current_edges = candidate
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = set(current_edges)
        else:
            # Try a swap: remove one, add one
            if not current_edges:
                continue
            rem = random.choice(list(current_edges))
            candidate = current_edges - {rem}
            if not missing:
                missing = [e for e in all_edges if e not in candidate]
            if missing:
                add = random.choice(missing)
                candidate2 = candidate | {add}
                if check_violations(candidate2) == 0:
                    current_edges = candidate2
                    if len(current_edges) > best_count:
                        best_count = len(current_edges)
                        best_edges = set(current_edges)
    
    # Final greedy pass
    for e in all_edges:
        if time.time() > deadline:
            break
        if e not in best_edges:
            candidate = best_edges | {e}
            if check_violations(candidate) == 0:
                best_edges = candidate
    
    # Ensure best_edges is valid
    if check_violations(best_edges) != 0:
        # Fallback to empty
        best_edges = set()
    
    result = [[int(u), int(v)] for u, v in best_edges]
    return result
