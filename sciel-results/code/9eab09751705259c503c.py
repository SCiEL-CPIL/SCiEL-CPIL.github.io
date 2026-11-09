"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START

def _adjacency(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _cycle_lengths(adj, n):
    """Find all distinct simple cycle lengths in an undirected graph."""
    lengths = set()
    # For each pair of vertices (s, t) with s < t, find all simple paths from s to t
    # that don't use s or t as internal nodes. A path of length L from s to t
    # gives a cycle of length L+1.
    
    # Use DFS to enumerate simple paths
    # To keep it manageable, we limit path length
    max_path_len = n - 1  # at most n-1 edges in a simple path
    
    for s in range(n):
        for t in range(s + 1, n):
            if t not in adj[s]:
                continue
            # DFS from s to t, paths cannot revisit nodes, and cannot use s or t as internal
            # Actually, standard cycle finding: paths from s to t that don't use s or t internally
            # are the same as paths where s is start, t is end, no repeats.
            # But we need to ensure the path from s to t doesn't go through s or t again.
            # In a simple path from s to t, s and t appear only at endpoints.
            
            # Simple DFS enumeration
            stack = [(s, [s])]
            found_paths = []
            while stack:
                node, path = stack.pop()
                if node == t:
                    found_paths.append(len(path) - 1)  # number of edges
                    continue
                for nb in adj[node]:
                    if nb not in path:
                        # Don't go back to s (s is already in path, so this is handled)
                        # Don't go to t unless we want to end (but we check node==t above)
                        # Actually we should allow going to t, but then stop
                        if nb == t:
                            found_paths.append(len(path))  # path + one more edge to t
                        else:
                            if nb != s:  # don't revisit s
                                stack.append((nb, path + [nb]))
            for L in found_paths:
                lengths.add(L + 1)  # cycle length = path length + 1 (the edge s-t)
    
    return lengths

def _min_degree(adj, n):
    return min(len(adj[i]) for i in range(n))

def _is_triangle_free(adj, n):
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w > v and w in adj[u]:
                    return False
    return True

def _valid_graph(adj, n):
    """Check: min degree >= 3, triangle-free, has at least one cycle."""
    if _min_degree(adj, n) < 3:
        return False
    if not _is_triangle_free(adj, n):
        return False
    # Check for at least one cycle
    lengths = _cycle_lengths(adj, n)
    if not lengths:
        return False
    return True

def _score_graph(adj, n):
    """Return number of distinct cycle lengths, or None if invalid."""
    if _min_degree(adj, n) < 3:
        return None
    if not _is_triangle_free(adj, n):
        return None
    lengths = _cycle_lengths(adj, n)
    if not lengths:
        return None
    return len(lengths)

def construct():
    n = 12
    start_time = time.time()
    deadline = start_time + 70  # 70s limit to be safe within 60s wall-clock... 
    # Actually the evaluator kills at 90s, but task says 60s. Let's use 50s to be safe.
    deadline = start_time + 50
    
    best_score = None
    best_edges = None
    
    # Start with the known good solution: two K_{3,3} components
    edges_known = []
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges_known.append([u, v])
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges_known.append([u, v])
    
    adj_known = _adjacency(edges_known, n)
    score_known = _score_graph(adj_known, n)
    if score_known is not None:
        best_score = score_known
        best_edges = [list(e) for e in edges_known]
    
    # Try to find a graph with fewer distinct cycle lengths
    # Search over random edge sets
    import random
    random.seed(42)
    
    # Total possible edges
    all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Try various random graphs
    for trial in range(5000):
        if time.time() > deadline:
            break
        
        # Random bipartite-like graph
        # Partition vertices into two sets of 6
        perm = list(range(n))
        random.shuffle(perm)
        part_a = set(perm[:6])
        part_b = set(perm[6:])
        
        # Random edges between part_a and part_b
        # Each vertex needs degree >= 3
        # Try to ensure min degree by adding random edges until min degree >= 3
        edges = set()
        
        # Add random cross edges
        for _ in range(20):
            a = random.choice(list(part_a))
            b = random.choice(list(part_b))
            if a < b:
                edges.add((a, b))
            else:
                edges.add((b, a))
        
        # Check min degree
        adj = _adjacency(list(edges), n)
        if _min_degree(adj, n) < 3:
            # Try to add more edges
            for _ in range(10):
                for a in part_a:
                    for b in part_b:
                        if len(adj[a]) < 3 and len(adj[b]) < 3:
                            if a < b:
                                e = (a, b)
                            else:
                                e = (b, a)
                            if e not in edges:
                                edges.add(e)
                                adj[a].add(b)
                                adj[b].add(a)
        
        adj = _adjacency(list(edges), n)
        if _min_degree(adj, n) < 3:
            continue
        
        score = _score_graph(adj, n)
        if score is not None:
            if best_score is None or score < best_score:
                best_score = score
                best_edges = [list(e) for e in edges]
                if best_score == 1:
                    break
    
    # Also try a single connected bipartite graph
    for trial in range(2000):
        if time.time() > deadline:
            break
        
        # Random bipartite with partitions of sizes 6,6
        part_a = set(range(6))
        part_b = set(range(6, 12))
        
        edges = set()
        # Ensure each vertex has degree at least 3
        # Use a random 3-regular bipartite graph approach
        # Shuffle connections
        a_list = list(part_a)
        b_list = list(part_b)
        
        # Random permutation-based: each a connects to 3 random b's
        for a in a_list:
            random.shuffle(b_list)
            for b in b_list[:3]:
                if a < b:
                    edges.add((a, b))
                else:
                    edges.add((b, a))
        
        adj = _adjacency(list(edges), n)
        if _min_degree(adj, n) < 3:
            continue
        
        score = _score_graph(adj, n)
        if score is not None:
            if best_score is None or score < best_score:
                best_score = score
                best_edges = [list(e) for e in edges]
                if best_score == 1:
                    break
    
    if best_edges is None:
        # Fallback to known solution
        best_edges = edges_known
    
    # Ensure edges are sorted and returned as lists of ints
    result = sorted([list(sorted(e)) for e in best_edges])
    return result

# EVOLVE-BLOCK-END
