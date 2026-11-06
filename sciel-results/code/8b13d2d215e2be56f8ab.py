import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(adj, n, max_cliques):
        """
        Enumerate maximal cliques.
        adj: list of sets of neighbors.
        Returns list of cliques (sets of vertices).
        """
        cliques = []
        
        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
            # Optimization: if P is empty, no more cliques in this branch
            if not P:
                return
            
            # Pivot selection: choose u from P U X with max |P \cap N(u)|
            # To simplify, just pick first one or random? 
            # Standard BK picks max degree in P.
            pivot_candidates = P.union(X)
            if not pivot_candidates:
                return
            
            # Find pivot with max intersection with P
            pivot = -1
            max_inter = -1
            # For speed, we can just iterate P if it's small, but P grows.
            # Let's pick a random pivot or the one with max degree in P.
            # Given n <= 40, iterating is fine.
            for u in pivot_candidates:
                inter = len(P.intersection(adj[u]))
                if inter > max_inter:
                    max_inter = inter
                    pivot = u
            
            # If pivot covers all P, we can prune? 
            # Standard BK: iterate P \ N(pivot)
            P_minus_Np = P.difference(adj[pivot])
            
            for v in list(P_minus_Np):
                new_R = R.union([v])
                new_P = P.intersection(adj[v])
                new_X = X.intersection(adj[v])
                expand(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)

        expand(set(), set(range(n)), set())
        return cliques

    def solve_hitting_set(n, cliques, time_limit):
        """
        Find minimum set S of vertices hitting all cliques in 'cliques'.
        """
        if not cliques:
            return []
        
        # Precompute: for each vertex, which cliques does it hit?
        # cliques is a list of frozensets.
        # Map: vertex -> list of clique indices
        vertex_to_cliques = {i: [] for i in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        # Greedy Initialization
        # uncovered: set of clique indices
        uncovered = set(range(len(cliques)))
        cover = []
        
        while uncovered:
            best_v = -1
            best_count = -1
            for v in range(n):
                # Count how many uncovered cliques v hits
                count = 0
                for idx in vertex_to_cliques[v]:
                    if idx in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1 or best_count == 0:
                # Should not happen if problem is solvable
                break
            
            cover.append(best_v)
            # Remove all cliques hit by best_v
            for idx in vertex_to_cliques[best_v]:
                if idx in uncovered:
                    uncovered.remove(idx)
        
        best_cover = list(cover)
        best_size = len(cover)
        
        # Local Search
        # Data structure for fast "is clique covered?" check
        # For each clique, count how many vertices from current cover are in it.
        # cover_counts[c_idx] = number of vertices in cover that are in clique c_idx
        
        def update_counts(cover_set):
            counts = [0] * len(cliques)
            for v in cover_set:
                for idx in vertex_to_cliques[v]:
                    counts[idx] += 1
            return counts

        def is_valid_cover(cover_set):
            counts = update_counts(cover_set)
            for c in range(len(cliques)):
                if counts[c] == 0:
                    return False
            return True

        # Optimization: Maintain counts incrementally
        cover_set = set(best_cover)
        counts = update_counts(cover_set)
        
        improved = True
        while improved:
            improved = False
            # Try to remove
            for v in list(cover_set):
                # Check if v is the only one covering any clique
                can_remove = True
                for idx in vertex_to_cliques[v]:
                    if counts[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove v
                    cover_set.remove(v)
                    for idx in vertex_to_cliques[v]:
                        counts[idx] -= 1
                    improved = True
                    break # Restart loop as counts changed
            
            if improved:
                continue
            
            # Try swaps
            # For each v in cover, for each u not in cover:
            # Swap v->u. If valid, check if any other vertex can be removed.
            found_swap = False
            for v in list(cover_set):
                # Tentative swap state
                # We need to check if replacing v with u keeps all counts > 0
                # And if it allows removing some w.
                
                # Optimization: Only consider u that hits at least one clique 
                # that v hits? Or just any u?
                # If u doesn't hit any clique that v hits, and v is essential for some clique, 
                # then u must hit that clique.
                
                # Let's just iterate all u not in cover_set
                for u in range(n):
                    if u in cover_set:
                        continue
                    
                    # Check validity of swap
                    valid = True
                    # Cliques that v hits:
                    # For each clique idx in vertex_to_cliques[v]:
                    # If counts[idx] == 1 (v is the only one), then u MUST be in clique idx.
                    # If u is not in clique idx, swap is invalid.
                    
                    for idx in vertex_to_cliques[v]:
                        if counts[idx] == 1:
                            # v is critical for idx
                            if u not in cliques[idx]:
                                valid = False
                                break
                    
                    if not valid:
                        continue
                    
                    # If valid, simulate swap and check for removals
                    # New counts:
                    # Decrement counts for cliques containing v
                    # Increment counts for cliques containing u
                    # Note: if clique contains both v and u, count stays same.
                    
                    new_counts = counts[:]
                    for idx in vertex_to_cliques[v]:
                        new_counts[idx] -= 1
                    for idx in vertex_to_cliques[u]:
                        new_counts[idx] += 1
                    
                    # Check if any other vertex w (w != v) can be removed
                    removed_any = False
                    removal_candidates = []
                    for w in cover_set:
                        if w == v:
                            continue
                        # Check if w can be removed in new state
                        # w can be removed if for all idx in vertex_to_cliques[w], new_counts[idx] > 1
                        # (Because v is gone, but u is added. We are checking if w is redundant given S \ {v} U {u})
                        
                        # Wait, if we remove w, we further decrement new_counts.
                        # So we check: for all idx in vertex_to_cliques[w], new_counts[idx] > 1.
                        
                        can_remove_w = True
                        for idx in vertex_to_cliques[w]:
                            if new_counts[idx] <= 1:
                                can_remove_w = False
                                break
                        
                        if can_remove_w:
                            removal_candidates.append(w)
                    
                    if removal_candidates:
                        # Perform swap and removal of first candidate
                        cover_set.remove(v)
                        cover_set.add(u)
                        
                        # Update counts to new_counts
                        counts = new_counts
                        
                        # Remove w
                        w = removal_candidates[0]
                        cover_set.remove(w)
                        for idx in vertex_to_cliques[w]:
                            counts[idx] -= 1
                        
                        found_swap = True
                        break
                
                if found_swap:
                    break
            
            if not found_swap:
                break

        return sorted(cover_set)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            out.append([])
            continue
        
        edges = instance(n, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find Maximal Cliques
        cliques = bron_kerbosch(adj, n, None)
        
        # Solve Hitting Set
        cover = solve_hitting_set(n, cliques, deadline)
        
        out.append(cover)

    return out
