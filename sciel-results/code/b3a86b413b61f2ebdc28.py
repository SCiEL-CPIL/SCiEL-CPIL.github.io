import random
import time
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d
    edges = []
    edge_map = {}
    
    # Generate all edges
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                edge_map[(u, v)] = len(edges) - 1
    
    n_edges = len(edges)
    
    # Precompute for each edge: list of C4s it belongs to
    # Each C4 is a frozenset of 4 edge indices
    edge_to_c4s = [[] for _ in range(n_edges)]
    all_c4s = []
    
    for idx, (u, v) in enumerate(edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        
        for j in range(d):
            if j == i:
                continue
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            
            idx1 = edge_map[e1]
            idx2 = edge_map[e2]
            idx3 = edge_map[e3]
            
            c4 = frozenset([idx, idx1, idx2, idx3])
            all_c4s.append(c4)
            edge_to_c4s[idx].append(len(all_c4s) - 1)
    
    n_c4s = len(all_c4s)
    
    # Build: for each C4, list of its 4 edge indices
    c4_edges = [list(c4) for c4 in all_c4s]
    
    def greedy_hitting_set(deleted, remaining_c4s, rng):
        """
        Greedy hitting set: repeatedly pick the edge that covers the most uncovered C4s.
        deleted: set of edge indices already deleted
        remaining_c4s: list of C4 indices not yet hit
        Returns updated (deleted, remaining_c4s)
        """
        # For each edge not yet deleted, count how many uncovered C4s it hits
        # Only consider edges that are part of at least one uncovered C4
        
        # Build coverage: edge -> count of uncovered C4s it belongs to
        # We need to track which C4s are still uncovered
        
        # For efficiency, maintain an array
        uncovered = set(remaining_c4s)
        
        while uncovered:
            # Count for each edge how many uncovered C4s it hits
            edge_count = {}
            for c4_idx in uncovered:
                for e in c4_edges[c4_idx]:
                    if e not in deleted:
                        edge_count[e] = edge_count.get(e, 0) + 1
            
            if not edge_count:
                break
            
            # Pick edge with max count (tie-break randomly)
            max_count = max(edge_count.values())
            candidates = [e for e, c in edge_count.items() if c == max_count]
            chosen = rng.choice(candidates)
            
            deleted.add(chosen)
            
            # Remove all C4s that contain chosen
            to_remove = []
            for c4_idx in uncovered:
                if chosen in c4_edges[c4_idx]:
                    to_remove.append(c4_idx)
            for c4_idx in to_remove:
                uncovered.discard(c4_idx)
        
        return deleted, list(uncovered)
    
    def prune_deletions(deleted, rng, max_iter=50):
        """
        Try to remove edges from the hitting set if the C4s they cover are still hit by others.
        """
        # Build: for each C4, which deleted edges cover it
        c4_covered_by = {}
        for c4_idx in range(n_c4s):
            covers = []
            for e in c4_edges[c4_idx]:
                if e in deleted:
                    covers.append(e)
            c4_covered_by[c4_idx] = covers
        
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            
            # Try removing each deleted edge
            for e in list(deleted):
                # Check: is every C4 that e covers also covered by another deleted edge?
                safe = True
                for c4_idx in edge_to_c4s[e]:
                    covers = c4_covered_by[c4_idx]
                    # Count how many deleted edges cover this C4
                    count = sum(1 for ce in covers if ce in deleted)
                    if count <= 1:
                        safe = False
                        break
                
                if safe:
                    deleted.discard(e)
                    # Update c4_covered_by
                    for c4_idx in edge_to_c4s[e]:
                        if e in c4_covered_by[c4_idx]:
                            c4_covered_by[c4_idx].remove(e)
                    improved = True
        
        return deleted
    
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    best_deleted = set()
    best_size = 0  # number of edges kept = n_edges - len(deleted)
    
    # Multiple restarts with different randomizations
    n_restarts = 100
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Randomized greedy hitting set
        # Instead of pure greedy, use a randomized variant:
        # At each step, pick from top-k candidates
        
        deleted = set()
        uncovered = set(range(n_c4s))
        
        while uncovered:
            # Count coverage
            edge_count = {}
            for c4_idx in uncovered:
                for e in c4_edges[c4_idx]:
                    if e not in deleted:
                        edge_count[e] = edge_count.get(e, 0) + 1
            
            if not edge_count:
                break
            
            max_count = max(edge_count.values())
            
            # Randomized: pick from top candidates
            # With some probability, pick the best; otherwise pick randomly from top few
            candidates = [e for e, c in edge_count.items() if c == max_count]
            
            if restart % 5 == 0 and len(candidates) > 1:
                # Pure greedy for some restarts
                chosen = rng.choice(candidates)
            else:
                # Randomized: consider edges with count >= max_count - 1
                threshold = max_count - 1
                relaxed = [e for e, c in edge_count.items() if c >= threshold]
                if relaxed:
                    chosen = rng.choice(relaxed)
                else:
                    chosen = rng.choice(candidates)
            
            deleted.add(chosen)
            
            # Remove covered C4s
            to_remove = []
            for c4_idx in uncovered:
                if chosen in c4_edges[c4_idx]:
                    to_remove.append(c4_idx)
            for c4_idx in to_remove:
                uncovered.discard(c4_idx)
        
        # Prune
        deleted = prune_deletions(deleted, rng, max_iter=30)
        
        size = n_edges - len(deleted)
        if size > best_size:
            best_size = size
            best_deleted = set(deleted)
        
        # Periodic deep prune on best
        if restart % 10 == 0 and best_size > 0:
            best_deleted = prune_deletions(best_deleted, rng, max_iter=100)
            best_size = n_edges - len(best_deleted)
    
    # Final deep prune
    if time.time() - start_time < time_limit - 5:
        best_deleted = prune_deletions(best_deleted, rng, max_iter=200)
        best_size = n_edges - len(best_deleted)
    
    # Convert to list of [u, v]
    result = []
    for idx in range(n_edges):
        if idx not in best_deleted:
            result.append(list(edges[idx]))
    
    return result
