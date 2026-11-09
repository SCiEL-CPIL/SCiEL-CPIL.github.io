"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _has_c6_through_edge(adj, n, u, v):
    """
    Check if adding edge (u,v) creates a C6 that uses this edge.
    A C6 using edge (u,v) has form: u-v-a-b-c-d-u
    So we need: a in N(v), b in N(a)\{u,v}, c in N(b)\{u,v,a}, d in N(c)\{u,v,a,b} with d in N(u).
    We can iterate over a in N(v), then b in N(a) excluding u,v, then c in N(b) excluding u,v,a,
    then check if any d in N(c) excluding u,v,a,b is in N(u).
    """
    Nu = adj[u]
    Nv = adj[v]
    
    for a in Nv:
        if a == u:
            continue
        Na = adj[a]
        for b in Na:
            if b == u or b == v or b == a:
                continue
            Nb = adj[b]
            for c in Nb:
                if c == u or c == v or c == a or c == b:
                    continue
                Nc = adj[c]
                for d in Nc:
                    if d == u or d == v or d == a or d == b or d == c:
                        continue
                    if d in Nu:
                        return True
    return False

def _has_c6_full(adj, n):
    """Full C6 check - slow, used only for final verification if needed."""
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            if _has_c6_through_edge(adj, n, u, v):
                return True
    return False

def _greedy_build_fast(n, seed, time_limit=10):
    """
    Greedy build with fast local C6 check.
    Uses degree-aware edge ordering: prefer edges between low-degree vertices.
    """
    random.seed(seed)
    adj = [set() for _ in range(n)]
    edge_count = 0
    start_time = time.time()
    
    # Generate all edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Shuffle for randomization
    random.shuffle(all_edges)
    
    degrees = [0] * n
    
    for u, v in all_edges:
        if time.time() - start_time > time_limit:
            break
        
        # Try to add edge (u, v)
        adj[u].add(v)
        adj[v].add(u)
        
        if not _has_c6_through_edge(adj, n, u, v):
            edge_count += 1
            degrees[u] += 1
            degrees[v] += 1
        else:
            adj[u].discard(v)
            adj[v].discard(u)
    
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    return edges, edge_count

def _local_search_fast(edges, n, time_limit=20):
    """
    Local search: try to add edges that don't create C6, using fast check.
    Also try edge swaps.
    """
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    
    best_edges = list(edges)
    best_count = len(edges)
    start_time = time.time()
    
    all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # First pass: try to add any missing edge that doesn't create C6
    missing = [e for e in all_possible if e not in edge_set]
    random.shuffle(missing)
    
    for u, v in missing:
        if time.time() - start_time > time_limit:
            break
        adj[u].add(v)
        adj[v].add(u)
        if not _has_c6_through_edge(adj, n, u, v):
            edge_set.add((u, v))
            best_count += 1
            best_edges = [[a, b] for a, b in edge_set]
        else:
            adj[u].discard(v)
            adj[v].discard(u)
    
    # Second pass: try edge swaps
    current_edges = list(edge_set)
    missing = [e for e in all_possible if e not in edge_set]
    random.shuffle(missing)
    
    while time.time() - start_time < time_limit:
        if not current_edges or not missing:
            break
        
        # Pick a random existing edge to remove
        idx = random.randint(0, len(current_edges) - 1)
        u, v = current_edges[idx]
        
        # Remove it
        adj[u].discard(v)
        adj[v].discard(u)
        edge_set.discard((u, v))
        current_edges.pop(idx)
        
        # Try to add a random missing edge
        added = False
        for _ in range(100):
            if not missing:
                break
            a, b = random.choice(missing)
            if (a, b) in edge_set or a == b:
                continue
            adj[a].add(b)
            adj[b].add(a)
            if not _has_c6_through_edge(adj, n, a, b):
                edge_set.add((a, b))
                current_edges.append((a, b))
                missing.remove((a, b))
                added = True
                if len(edge_set) > best_count:
                    best_count = len(edge_set)
                    best_edges = [[a, b] for a, b in edge_set]
                break
            else:
                adj[a].discard(b)
                adj[b].discard(a)
        
        if not added:
            # Put the removed edge back
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
            current_edges.append((u, v))
            if (u, v) not in missing:
                missing.append((u, v))
    
    return best_edges, best_count

def _verify_c6_free(edges, n):
    """Verify the graph is C6-free using the fast per-edge check."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in edges:
        if _has_c6_through_edge(adj, n, u, v):
            return False
    return True

def construct():
    n = 40
    deadline = 140  # safety margin before 150s limit
    start_time = time.time()
    
    # Start with the seed: 8 disjoint K5s (100 edges)
    seed_edges = []
    for k in range(8):
        seed_edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
    
    best_edges = seed_edges
    best_count = len(seed_edges)
    
    # Try many greedy builds with different seeds
    seeds = list(range(100))
    random.shuffle(seeds)
    
    for seed in seeds:
        if time.time() - start_time > deadline - 30:
            break
        remaining_for_greedy = min(5, deadline - 30 - (time.time() - start_time))
        if remaining_for_greedy <= 0:
            break
        edges, count = _greedy_build_fast(n, seed=seed, time_limit=remaining_for_greedy)
        if count > best_count:
            best_count = count
            best_edges = edges
    
    # Try local search from best found
    if time.time() - start_time < deadline - 10:
        remaining = deadline - 10 - (time.time() - start_time)
        if remaining > 5:
            improved_edges, improved_count = _local_search_fast(best_edges, n, time_limit=remaining)
            if improved_count > best_count:
                best_count = improved_count
                best_edges = improved_edges
    
    # Final verification
    if not _verify_c6_free(best_edges, n):
        # Fallback to seed if verification fails
        best_edges = seed_edges
        best_count = len(seed_edges)
    
    return best_edges
# EVOLVE-BLOCK-END
