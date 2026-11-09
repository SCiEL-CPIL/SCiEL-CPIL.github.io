"""Clique partition: hardest-edge greedy expansion with transitive merge refinement."""
import time
from random import Random
from collections import defaultdict

def _random_graph(n, p, seed):
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve_instance(n, p, seed, deadline):
    edges = _random_graph(n, p, seed)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    remaining = set(edges)
    cliques = []
    
    while remaining:
        if time.time() > deadline:
            for e in remaining:
                cliques.append(list(e))
            break
        
        # Find the "hardest" edge: one with the fewest common neighbors
        # This helps form larger cliques in sparse regions
        best_edge = None
        best_common = float('inf')
        for e in remaining:
            u, v = e
            common = len(adj[u] & adj[v])
            if common < best_common:
                best_common = common
                best_edge = e
                if common <= 0:
                    break
        
        if best_edge is None:
            best_edge = next(iter(remaining))
        
        u, v = best_edge
        
        # Build a maximal clique containing edge (u, v)
        # Start with {u, v}, greedily add vertices
        clique = [u, v]
        clique_set = {u, v}
        
        # Candidates: common neighbors of all clique members
        cand_set = adj[u] & adj[v]
        cand_set -= clique_set
        
        # Iterative greedy expansion
        changed = True
        while changed and cand_set:
            changed = False
            if time.time() > deadline:
                break
            
            # Find candidate with most connections to current clique
            best_cand = None
            best_score = -1
            for cand in cand_set:
                # Count edges from cand to clique members that are in remaining
                score = 0
                for x in clique:
                    ek = (min(x, cand), max(x, cand))
                    if ek in remaining:
                        score += 1
                # Tie-break by degree in original graph
                if score > best_score or (score == best_score and len(adj[cand]) > len(adj[best_cand] if best_cand is not None else set())):
                    best_score = score
                    best_cand = cand
            
            if best_cand is None or best_score < len(clique):
                break
            
            # Add this candidate
            clique.append(best_cand)
            clique_set.add(best_cand)
            # Update candidates: must connect to new member
            cand_set = cand_set & adj[best_cand]
            changed = True
        
        # Remove all edges within this clique from remaining
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                ek = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                remaining.discard(ek)
        
        if len(clique) >= 2:
            cliques.append(clique)
    
    # Post-processing: transitive merge using connected components
    # Build a graph where cliques are nodes; edge between two cliques if their union is a clique
    def _transitive_merge(cliques, adj, deadline):
        if len(cliques) <= 1:
            return cliques
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            n_cliques = len(cliques)
            
            # Build adjacency: clique i and j can merge if union is a clique
            merge_adj = [set() for _ in range(n_cliques)]
            for i in range(n_cliques):
                for j in range(i + 1, n_cliques):
                    # Check if all vertices in clique i connect to all in clique j
                    can_merge = True
                    for a in cliques[i]:
                        for b in cliques[j]:
                            if b not in adj[a]:
                                can_merge = False
                                break
                        if not can_merge:
                            break
                    if can_merge:
                        merge_adj[i].add(j)
                        merge_adj[j].add(i)
            
            # Find connected components in merge graph
            visited = [False] * n_cliques
            for start in range(n_cliques):
                if visited[start]:
                    continue
                # BFS to find component
                component = []
                queue = [start]
                visited[start] = True
                while queue:
                    node = queue.pop()
                    component.append(node)
                    for neighbor in merge_adj[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                
                if len(component) > 1:
                    # Merge all cliques in this component
                    merged = []
                    for idx in component:
                        merged.extend(cliques[idx])
                    # Remove old cliques and add merged
                    new_cliques = []
                    for idx in range(n_cliques):
                        if idx not in component:
                            new_cliques.append(cliques[idx])
                    new_cliques.append(merged)
                    cliques = new_cliques
                    changed = True
                    break
        
        return cliques
    
    cliques = _transitive_merge(cliques, adj, deadline)
    
    return cliques

def construct():
    deadline = time.time() + 100  # Safety margin before 120s limit
    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        cliques = _solve_instance(n, p, seed, deadline)
        result.append(cliques)
        if time.time() > deadline - 10:
            # Not enough time for second instance, return edges for second
            edges = _random_graph(n, p, seed)
            result.append([list(e) for e in edges])
            break
    return result
