"""Spectral optimization: K20,20 init + A²-guided local search to flatten triangle distribution."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n = 40
    m = 401
    deadline = time.time() + 138  # safety margin before 150s kill

    def build_adj(edge_set):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def compute_books(adj, edge_set):
        """Return dict edge -> book size (common neighbors)."""
        books = {}
        for u, v in edge_set:
            common = adj[u] & adj[v]
            books[(u, v)] = len(common)
        return books

    def max_book(books):
        return max(books.values()) if books else 0

    def compute_a2_eigenvalues(adj):
        """Compute A² matrix and return eigenvalues for spectral analysis."""
        A = np.zeros((n, n), dtype=np.float64)
        for u, v in adj:
            A[u, v] = 1.0
            A[v, u] = 1.0
        A2 = A @ A
        eigvals = np.linalg.eigvalsh(A2)
        return eigvals

    def spectral_score(eigvals):
        """
        Spectral score: measures how 'flat' the triangle distribution is.
        Higher eigenvalue spread = more concentrated triangles.
        We want to minimize the largest eigenvalue of A² (which relates to max degree in A²).
        Also use the gap between top eigenvalues as a proxy for concentration.
        """
        # Sort eigenvalues descending
        ev = np.sort(eigvals)[::-1]
        # The largest eigenvalue of A² is related to max (A²)_{ii} = deg(i)
        # But more importantly, the spectral structure tells us about triangle concentration
        # Use: top eigenvalue minus mean of rest (measures concentration)
        if len(ev) > 1:
            return ev[0] - np.mean(ev[1:])
        return 0.0

    def k20_20_init():
        """Initialize with K_{20,20} + 1 extra edge for 401 edges."""
        edge_set = set()
        # K_{20,20}: all edges between {0..19} and {20..39}
        for i in range(20):
            for j in range(20, 40):
                edge_set.add((i, j))
        # That's 400 edges. Add one more edge within a partition.
        edge_set.add((0, 1))
        return edge_set

    # Initialize
    edge_set = k20_20_init()
    adj = build_adj(edge_set)
    books = compute_books(adj, edge_set)
    best_max = max_book(books)
    
    # Track best solution
    best_edge_set = set(edge_set)
    best_books = dict(books)
    
    # Compute initial spectral score
    eigvals = compute_a2_eigenvalues(edge_set)
    best_spectral = spectral_score(eigvals)
    
    # Local search with spectral guidance
    no_improve_rounds = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve_rounds < max_no_improve:
        # Recompute current state
        adj = build_adj(edge_set)
        books = compute_books(adj, edge_set)
        current_max = max_book(books)
        
        # Get edges sorted by book size descending
        sorted_edges = sorted(books.items(), key=lambda x: -x[1])
        
        round_improved = False
        
        # Focus on edges with book >= current_max (or >= best_max if we want to improve further)
        target_max = max(current_max, best_max)
        
        for (u, v), bk in sorted_edges:
            if time.time() >= deadline:
                break
            if bk < target_max:
                break
            
            # Generate candidate non-edges to swap in
            # Use spectral insight: prefer edges that connect vertices with
            # lower A² row sums (less triangle-prone positions)
            
            # Compute A² row sums for guidance (approximate triangle involvement per vertex)
            # A²[i][j] = number of common neighbors of i and j
            # Row sum of A² at i = sum_j A²[i][j] = sum_j |N(i) ∩ N(j)|
            # This is related to the number of triangles through i weighted by structure
            
            # For efficiency, use degree-based heuristic combined with spectral info
            # Vertices with lower degree in the current graph are less triangle-prone
            deg = [len(adj[i]) for i in range(n)]
            
            # Try a mix of strategies:
            # 1. Random non-edges (exploration)
            # 2. Non-edges involving low-degree vertices (exploitation)
            
            num_candidates = 30
            found_good = False
            
            for attempt in range(num_candidates):
                if time.time() >= deadline:
                    break
                
                if attempt < num_candidates // 2:
                    # Random non-edge
                    c = random.randrange(n)
                    d = random.randrange(n)
                else:
                    # Prefer low-degree vertices for the new edge
                    # Pick c from lower-degree vertices
                    candidates_c = sorted(range(n), key=lambda x: deg[x])
                    c = random.choice(candidates_c[:10])
                    candidates_d = sorted(range(n), key=lambda x: deg[x])
                    d = random.choice(candidates_d[:10])
                
                if c == d:
                    continue
                c, d = min(c, d), max(c, d)
                if (c, d) in edge_set:
                    continue
                if (c, d) == (u, v):
                    continue
                
                # Compute common neighbors for (c,d) after removing (u,v)
                if c == u:
                    ac = adj[u] - {v}
                elif c == v:
                    ac = adj[v] - {u}
                else:
                    ac = adj[c]
                
                if d == u:
                    ad = adj[u] - {v}
                elif d == v:
                    ad = adj[v] - {u}
                else:
                    ad = adj[d]
                
                common_cd = len(ac & ad)
                
                # Acceptance criterion:
                # 1. New edge book must be < current edge book
                # 2. Prefer swaps that reduce spectral concentration
                if common_cd < bk:
                    # Perform tentative swap
                    edge_set.discard((u, v))
                    edge_set.add((c, d))
                    adj[u].discard(v)
                    adj[v].discard(u)
                    adj[c].add(d)
                    adj[d].add(c)
                    
                    # Recompute books
                    new_books = compute_books(adj, edge_set)
                    new_max = max_book(new_books)
                    
                    if new_max <= current_max:
                        # Check if it improves over best
                        if new_max < best_max:
                            best_max = new_max
                            best_edge_set = set(edge_set)
                            best_books = dict(new_books)
                        
                        # Compute new spectral score
                        new_eigvals = compute_a2_eigenvalues(edge_set)
                        new_spectral = spectral_score(new_eigvals)
                        
                        # Accept if spectral score improved or is similar
                        # (we allow slight spectral degradation if max book improved)
                        if new_spectral <= best_spectral + 0.5:
                            best_spectral = new_spectral
                            round_improved = True
                        else:
                            # Revert
                            edge_set.discard((c, d))
                            edge_set.add((u, v))
                            adj[u].add(v)
                            adj[v].add(u)
                            adj[c].discard(d)
                            adj[d].discard(c)
                    else:
                        # Revert
                        edge_set.discard((c, d))
                        edge_set.add((u, v))
                        adj[u].add(v)
                        adj[v].add(u)
                        adj[c].discard(d)
                        adj[d].discard(c)
                    
                    found_good = True
                    break
            
            # If we found a good swap, restart the sorted edges (books changed)
            if found_good:
                # Recompute for next iteration
                break
        
        if not round_improved:
            no_improve_rounds += 1
        else:
            no_improve_rounds = 0
    
    # Return best solution found
    result_edges = best_edge_set if best_edge_set else edge_set
    return [[u, v] for u, v in result_edges]
# EVOLVE-BLOCK-END
