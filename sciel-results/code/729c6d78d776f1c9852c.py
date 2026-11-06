import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve(edges, time_budget):
        if not edges:
            return []

        n = max(max(e) for e in edges) + 1
        edge_set = set()
        for u, v in edges:
            edge_set.add((u, v) if u < v else (v, u))

        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        # Generate all maximal cliques using Bron-Kerbosch with pivot
        cliques = []
        
        def bron_kerbosch(R, P, X):
            if time.time() > time_budget:
                return
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot with max |P ∩ N(u)|
            best_pivot = None
            max_common = -1
            for u in list(P) + list(X):
                common = len(P & adj[u])
                if common > max_common:
                    max_common = common
                    best_pivot = u
            
            if best_pivot is None:
                candidates = list(P)
            else:
                candidates = [u for u in P if u not in adj[best_pivot]]
            
            for u in candidates:
                if time.time() > time_budget:
                    return
                bron_kerbosch(R | {u}, P & adj[u], X & adj[u])
                P.remove(u)
                X.add(u)

        bron_kerbosch(frozenset(), frozenset(range(n)), frozenset())
        
        # If no cliques found (shouldn't happen with edges), fallback to individual edges
        if not cliques:
            return [list(e) for e in edge_set]

        # Convert to list of lists for easier manipulation
        cliques_list = [list(c) for c in cliques]
        
        # Precompute edge sets for each clique
        clique_edges = []
        for c in cliques_list:
            es = set()
            for u, v in combinations(c, 2):
                es.add((u, v) if u < v else (v, u))
            clique_edges.append(es)

        # Lagrangian relaxation approach
        # We want to select a subset of cliques that covers each edge exactly once
        # This is an exact cover problem. Lagrangian relaxation on the "exactly once" constraint
        # is tricky. Instead, we'll use a heuristic based on reduced cost.
        
        # Actually, a simpler and more robust structural change:
        # Use a greedy set cover that prioritizes cliques covering the most "rare" edges.
        # Edge rarity = 1 / (number of maximal cliques containing that edge).
        
        # Build edge -> list of clique indices map
        edge_to_cliques = {e: [] for e in edge_set}
        for i, es in enumerate(clique_edges):
            for e in es:
                edge_to_cliques[e].append(i)
        
        # Calculate rarity
        edge_rarity = {e: 1.0 / len(cliques_list[i]) if False else 1.0 / max(1, len(edge_to_cliques[e])) for e in edge_set}
        
        # Greedy selection based on rarity-weighted coverage
        remaining = set(edge_set)
        selected = []
        used_cliques = set()
        
        # Sort cliques by average rarity of their edges? No, let's do iterative.
        # At each step, pick the clique that maximizes sum of rarity of uncovered edges it covers,
        # but penalize if it covers edges already covered (shouldn't happen if we remove).
        # Wait, the constraint is "every edge lies in EXACTLY ONE clique".
        # So we cannot pick a clique that contains an edge already covered by another selected clique.
        
        # This makes it a packing problem with exact cover constraint.
        # Greedy:
        # 1. Start with empty selection.
        # 2. Find a clique that contains only uncovered edges.
        # 3. Among those, pick the one that maximizes some score (e.g., size, or rarity sum).
        # 4. Remove its edges from remaining.
        # 5. Repeat until no valid clique exists.
        # 6. For remaining edges (isolated or in non-maximal cliques), add them individually or in pairs.
        
        # Let's refine the greedy:
        # We can only pick a clique C if C's edges are all in `remaining`.
        
        while remaining and time.time() < deadline:
            best_clique_idx = None
            best_score = -1
            
            for i, es in enumerate(clique_edges):
                if i in used_cliques:
                    continue
                # Check if all edges in es are in remaining
                if es.issubset(remaining):
                    # Score: prefer larger cliques, or those covering rare edges
                    score = len(es) * 10 + sum(edge_rarity[e] for e in es)
                    if score > best_score:
                        best_score = score
                        best_clique_idx = i
            
            if best_clique_idx is not None:
                selected.append(cliques_list[best_clique_idx])
                used_cliques.add(best_clique_idx)
                remaining -= clique_edges[best_clique_idx]
            else:
                # No maximal clique fits. Pick an edge and find a clique containing it that fits.
                # Or just pick a single edge as a 2-vertex clique.
                e = next(iter(remaining))
                u, v = e
                # Check if {u, v} is a clique (it is, by definition of edge)
                # But maybe we can find a larger clique containing e that fits?
                # Let's try to find a clique containing e that is a subset of remaining.
                found = False
                for i, es in enumerate(clique_edges):
                    if i in used_cliques:
                        continue
                    if e in es and es.issubset(remaining):
                        selected.append(cliques_list[i])
                        used_cliques.add(i)
                        remaining -= es
                        found = True
                        break
                if not found:
                    # Just add the edge itself
                    selected.append([u, v])
                    remaining.discard(e)
        
        # Verify validity
        covered = set()
        valid = True
        for c in selected:
            if len(c) < 2:
                valid = False
                break
            for u, v in combinations(c, 2):
                e = (u, v) if u < v else (v, u)
                if e not in edge_set:
                    valid = False
                    break
                if e in covered:
                    valid = False
                    break
                covered.add(e)
            if not valid:
                break
        
        if not valid or covered != edge_set:
            # Fallback: individual edges
            return [list(e) for e in edge_set]
            
        return selected

    c1 = solve(random_graph(32, 0.6, 601), deadline)
    c2 = solve(random_graph(64, 0.55, 602), deadline)

    return [c1, c2]
