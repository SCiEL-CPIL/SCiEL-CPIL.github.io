import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        
        if not edges:
            return []

        # Build adjacency list as sets for fast intersection
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        best_cliques = None
        best_count = float('inf')
        
        end_time = start_time + time_budget
        attempts = 0

        while time.time() < end_time:
            attempts += 1
            attempt_rng = random.Random(seed + attempts * 98765 + 111)
            
            remaining_edges = set(edges)
            cliques = []
            
            while remaining_edges:
                # 1. Select seed edge
                # Heuristic: Pick edge with max potential clique size in common neighborhood
                # Approximation: Max number of common neighbors that are also connected in remaining_edges
                
                max_common = -1
                best_edges = []
                
                # Optimization: If remaining_edges is small, just iterate.
                # If large, we can sample? No, n is small.
                
                for u, v in remaining_edges:
                    # Common neighbors in the graph
                    common = adj[u].intersection(adj[v])
                    # Filter: w must be connected to u and v in remaining_edges
                    count = 0
                    for w in common:
                        # Check if edges (u,w) and (v,w) are in remaining_edges
                        # Note: u,v are in remaining_edges, so u!=v.
                        # w is distinct from u,v.
                        if (min(u, w), max(u, w)) in remaining_edges and (min(v, w), max(v, w)) in remaining_edges:
                            count += 1
                            
                    if count > max_common:
                        max_common = count
                        best_edges = [(u, v)]
                    elif count == max_common:
                        best_edges.append((u, v))
                
                if not best_edges:
                    # Fallback
                    u, v = next(iter(remaining_edges))
                    best_edges = [(u, v)]
                
                # Pick one randomly from best_edges
                u, v = attempt_rng.choice(best_edges)
                
                # 2. Build clique starting with {u, v}
                current_clique = [u, v]
                
                # Candidates: vertices w such that (u,w) and (v,w) are in remaining_edges
                candidates = set(adj[u]).intersection(adj[v])
                candidates.discard(u)
                candidates.discard(v)
                
                # Filter candidates: must have edges to u and v in remaining_edges
                candidates = {w for w in candidates 
                              if (min(u, w), max(u, w)) in remaining_edges 
                              and (min(v, w), max(v, w)) in remaining_edges}
                
                while candidates:
                    # Pick the best candidate w
                    # Heuristic: Max degree in the subgraph induced by candidates
                    
                    best_w = None
                    max_deg = -1
                    
                    for w in candidates:
                        deg = 0
                        for z in candidates:
                            if w == z: continue
                            if (min(w, z), max(w, z)) in remaining_edges:
                                deg += 1
                        if deg > max_deg:
                            max_deg = deg
                            best_w = w
                            
                    if best_w is None:
                        break
                    
                    current_clique.append(best_w)
                    candidates.remove(best_w)
                    
                    # Update candidates: must be connected to best_w
                    new_candidates = set()
                    for w in candidates:
                        if (min(w, best_w), max(w, best_w)) in remaining_edges:
                            new_candidates.add(w)
                    candidates = new_candidates
                    
                # 3. Add clique, remove edges
                cliques.append(current_clique)
                
                for i in range(len(current_clique)):
                    for j in range(i + 1, len(current_clique)):
                        e = (min(current_clique[i], current_clique[j]), max(current_clique[i], current_clique[j]))
                        remaining_edges.discard(e)
            
            # Post-processing: Merge cliques if possible
            changed = True
            while changed:
                changed = False
                for i in range(len(cliques)):
                    for j in range(i + 1, len(cliques)):
                        # Check if union is a clique
                        # This requires checking all pairs between clique[i] and clique[j]
                        is_clique = True
                        for u in cliques[i]:
                            for v in cliques[j]:
                                if u == v: continue
                                if (min(u, v), max(u, v)) not in edges:
                                    is_clique = False
                                    break
                            if not is_clique:
                                break
                        if is_clique:
                            cliques[i] = cliques[i] + cliques[j]
                            cliques.pop(j)
                            changed = True
                            break
                    if changed:
                        break
            
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques
                
            if time.time() > end_time - 2:
                break

        return best_cliques

    # Solve both instances
    t1 = time.time()
    # Allocate more time to the larger, harder instance (n=64)
    budget1 = min(20, DEADLINE - (t1 - start_time) - 5)
    cliques1 = solve_instance(32, 0.6, 601, budget1)

    t2 = time.time()
    budget2 = min(110, DEADLINE - (t2 - start_time))
    cliques2 = solve_instance(64, 0.55, 602, budget2)

    return [cliques1, cliques2]
