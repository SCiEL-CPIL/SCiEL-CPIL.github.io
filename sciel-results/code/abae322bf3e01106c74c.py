"""
Clique Partition via Genetic Algorithm with Structural Crossover.
Strategy: 
1. Represent a solution as a list of cliques (sets of vertices).
2. Use a Genetic Algorithm with:
   - Selection: Tournament selection based on number of cliques.
   - Crossover: "Block Crossover" - take cliques from Parent A, fill remaining edges with cliques from Parent B (greedily merging if possible).
   - Mutation: Local improvement via Simulated Annealing on the offspring.
3. This allows combining good structural components from different parents, 
   escaping local minima that single-edge SA might get stuck in.
"""
import time
import random
import math
import sys

_START = time.time()
DEADLINE = 145.0

def _remaining():
    return DEADLINE - time.time()

def _generate_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [[False] * n for _ in range(n)]
    edge_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u][v] = adj[v][u] = True
                edge_list.append((u, v))
    return adj, edge_list

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _verify_partition(cliques, edge_list, adj):
    if not cliques and not edge_list:
        return True
    if not cliques and edge_list:
        return False
    edge_idx = {_edge_key(u, v): i for i, (u, v) in enumerate(edge_list)}
    edge_cover = [0] * len(edge_list)
    for clique in cliques:
        if len(clique) < 2:
            return False
        # Check clique validity
        for a in range(len(clique)):
            for b in range(a + 1, len(clique)):
                if not adj[clique[a]][clique[b]]:
                    return False
                e = _edge_key(clique[a], clique[b])
                if e in edge_idx:
                    edge_cover[edge_idx[e]] += 1
    for c in edge_cover:
        if c != 1:
            return False
    return True

def _clique_edges(clique, edge_set):
    """Get the set of edge indices covered by a clique."""
    edges = set()
    for i in range(len(clique)):
        for j in range(i + 1, len(clique)):
            e = _edge_key(clique[i], clique[j])
            if e in edge_set:
                edges.add(edge_set[e])
    return edges

def _greedy_partition(edge_list, adj, n):
    """
    Generate an initial partition using a greedy heuristic.
    Order edges by 'potential' (number of compatible neighbors).
    """
    m = len(edge_list)
    if m == 0:
        return []
    
    # Precompute compatibility for edges
    # Two edges are compatible if their union of vertices forms a clique.
    # For speed, we just check on the fly in the greedy loop, but we can pre-sort.
    
    # Calculate a heuristic score for each edge: how many other edges it can be in a clique with?
    # Approximation: degree of vertices involved.
    scores = []
    for u, v in edge_list:
        # Sum of degrees of u and v
        deg_u = sum(1 for x in range(n) if adj[u][x])
        deg_v = sum(1 for x in range(n) if adj[v][x])
        scores.append(deg_u + deg_v)
    
    # Sort edges by score descending
    order = sorted(range(m), key=lambda i: scores[i], reverse=True)
    
    cliques = []
    edge_clique = [-1] * m
    
    for eidx in order:
        u, v = edge_list[eidx]
        best_clique_idx = -1
        best_size = -1
        
        # Try to add to existing cliques
        for ci, c in enumerate(cliques):
            # Check if u and v can be added to c
            # 1. Are u, v in c? No, they must be new or existing.
            # Actually, the clique is a set of vertices. 
            # If we add edge (u,v) to clique c, the new clique is c U {u,v}.
            # This is valid if c U {u,v} is a clique.
            # This requires:
            # - u and v are connected to ALL vertices in c.
            # - u and v are connected to each other (they are, it's an edge).
            
            can_add = True
            for x in c:
                if not adj[u][x] or not adj[v][x]:
                    can_add = False
                    break
            
            if can_add:
                # Prefer larger cliques? Or just first fit?
                # Let's prefer first fit for speed, or maybe largest?
                # Largest might lead to fewer cliques.
                if len(c) > best_size:
                    best_size = len(c)
                    best_clique_idx = ci
        
        if best_clique_idx != -1:
            c = cliques[best_clique_idx]
            # Add u and v to the clique
            # Note: c is a list of vertices. We need to add u and v if not present.
            # Since it's a set of edges, the vertices are the union.
            # But we are storing cliques as vertex lists.
            # If u is not in c, add it. If v is not in c, add it.
            if u not in c:
                c.append(u)
            if v not in c:
                c.append(v)
            edge_clique[eidx] = best_clique_idx
        else:
            # New clique
            cliques.append([u, v])
            edge_clique[eidx] = len(cliques) - 1
            
    return cliques

def _local_search(cliques, edge_list, adj, n, time_budget=2.0):
    """
    Improve a partition using Simulated Annealing on edge moves.
    Returns improved cliques.
    """
    if not cliques:
        return cliques
    
    m = len(edge_list)
    edge_to_clique = [-1] * m
    clique_to_edges = [set() for _ in range(len(cliques))]
    
    # Map edges to cliques
    for ci, c in enumerate(cliques):
        c_set = set(c)
        for u in c:
            for v in c:
                if u < v:
                    e = _edge_key(u, v)
                    # Find edge index
                    # This is slow. Let's build a map.
                    pass
    
    # Rebuild mapping efficiently
    edge_idx_map = {_edge_key(u, v): i for i, (u, v) in enumerate(edge_list)}
    for ci, c in enumerate(cliques):
        c_set = set(c)
        for u in c:
            for v in c:
                if u < v:
                    e = _edge_key(u, v)
                    if e in edge_idx_map:
                        ei = edge_idx_map[e]
                        edge_to_clique[ei] = ci
                        clique_to_edges[ci].add(ei)

    current_cliques = [list(c) for c in cliques]
    current_count = len(current_cliques)
    best_cliques = [list(c) for c in current_cliques]
    best_count = current_count
    
    start_time = time.time()
    T = 1.0
    cooling_rate = 0.99
    
    while time.time() - start_time < time_budget and T > 0.01:
        if _remaining() < 5:
            break
            
        eidx = random.randint(0, m - 1)
        cur_ci = edge_to_clique[eidx]
        
        # Move to random other clique
        target_ci = random.randint(0, current_count - 1)
        if target_ci == cur_ci:
            continue
            
        u, v = edge_list[eidx]
        
        # Check if (u, v) can be added to clique target_ci
        target_verts = set(current_cliques[target_ci])
        can_add = True
        for x in target_verts:
            if not adj[u][x] or not adj[v][x]:
                can_add = False
                break
        
        if can_add:
            # Perform move
            # Remove edge from cur_ci
            clique_to_edges[cur_ci].discard(eidx)
            # Add to target_ci
            clique_to_edges[target_ci].add(eidx)
            edge_to_clique[eidx] = target_ci
            
            # Update vertex lists
            # If cur_ci becomes empty (no edges), remove clique?
            # For simplicity in SA, we keep the list of cliques fixed size?
            # No, we should remove empty cliques to reduce count.
            
            # Check if cur_ci is empty
            if not clique_to_edges[cur_ci]:
                # Remove clique cur_ci
                # This shifts indices. Complex.
                # Let's just mark it and clean up periodically?
                # Or just leave it empty? The count is number of non-empty.
                pass
            
            new_count = sum(1 for s in clique_to_edges if s)
            delta = new_count - current_count
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_count = new_count
                # Update vertex lists for affected cliques
                # This is expensive. 
                # Optimization: Only update vertex lists when we find a new best?
                # Or maintain them incrementally?
                # For now, let's just track the edge sets.
                # We need to convert to vertex lists at the end.
                
                if new_count < best_count:
                    best_count = new_count
                    # Convert clique_to_edges to vertex lists
                    best_cliques = []
                    for s in clique_to_edges:
                        if s:
                            verts = set()
                            for ei in s:
                                u, v = edge_list[ei]
                                verts.add(u)
                                verts.add(v)
                            best_cliques.append(list(verts))
            else:
                # Revert
                clique_to_edges[target_ci].discard(eidx)
                clique_to_edges[cur_ci].add(eidx)
                edge_to_clique[eidx] = cur_ci
                
        T *= cooling_rate
        
    return best_cliques

def _crossover(parent1, parent2, edge_list, adj, n):
    """
    Crossover: Take cliques from parent1. For edges not covered, 
    try to form cliques using parent2's structure or greedily.
    """
    # Convert parents to edge sets
    edge_idx_map = {_edge_key(u, v): i for i, (u, v) in enumerate(edge_list)}
    
    p1_edges = set()
    for c in parent1:
        c_set = set(c)
        for u in c_set:
            for v in c_set:
                if u < v:
                    e = _edge_key(u, v)
                    if e in edge_idx_map:
                        p1_edges.add(edge_idx_map[e])
                        
    # Start with parent1's cliques
    child_cliques = [list(c) for c in parent1]
    covered_edges = set(p1_edges)
    
    # Find uncovered edges
    uncovered = [i for i in range(len(edge_list)) if i not in covered_edges]
    
    if not uncovered:
        return child_cliques
    
    # Use parent2 to help cover remaining edges
    # Create a mapping of edges in parent2 to their cliques
    p2_edge_clique = [-1] * len(edge_list)
    for ci, c in enumerate(parent2):
        c_set = set(c)
        for u in c_set:
            for v in c_set:
                if u < v:
                    e = _edge_key(u, v)
                    if e in edge_idx_map:
                        p2_edge_clique[edge_idx_map[e]] = ci
                        
    # Group uncovered edges by their parent2 clique
    groups = {}
    for eidx in uncovered:
        ci = p2_edge_clique[eidx]
        if ci == -1:
            ci = -1 # No parent2 clique
        if ci not in groups:
            groups[ci] = []
        groups[ci].append(eidx)
        
    # For each group, try to form a clique
    for ci, edges in groups.items():
        if ci == -1:
            # Fallback: greedy add to child_cliques
            for eidx in edges:
                u, v = edge_list[eidx]
                added = False
                for c in child_cliques:
                    c_set = set(c)
                    can_add = True
                    for x in c_set:
                        if not adj[u][x] or not adj[v][x]:
                            can_add = False
                            break
                    if can_add:
                        if u not in c: c.append(u)
                        if v not in c: c.append(v)
                        added = True
                        break
                if not added:
                    child_cliques.append([u, v])
        else:
            # Try to merge these edges into a single clique
            # Get the vertices involved
            verts = set()
            for eidx in edges:
                u, v = edge_list[eidx]
                verts.add(u)
                verts.add(v)
            
            # Check if verts form a clique
            is_clique = True
            verts_list = list(verts)
            for i in range(len(verts_list)):
                for j in range(i+1, len(verts_list)):
                    if not adj[verts_list[i]][verts_list[j]]:
                        is_clique = False
                        break
                if not is_clique:
                    break
            
            if is_clique and len(verts) >= 2:
                child_cliques.append(verts_list)
            else:
                # Split into smaller cliques greedily
                for eidx in edges:
                    u, v = edge_list[eidx]
                    added = False
                    for c in child_cliques:
                        c_set = set(c)
                        can_add = True
                        for x in c_set:
                            if not adj[u][x] or not adj[v][x]:
                                can_add = False
                                break
                        if can_add:
                            if u not in c: c.append(u)
                            if v not in c: c.append(v)
                            added = True
                            break
                    if not added:
                        child_cliques.append([u, v])
                        
    return child_cliques

def _solve_one_graph(adj, edge_list, time_budget=60.0):
    n = max(max(u, v) for u, v in edge_list) + 1 if edge_list else 0
    if not edge_list:
        return []
        
    # Initial population
    pop_size = 20
    population = []
    
    # Seed with greedy
    g1 = _greedy_partition(edge_list, adj, n)
    population.append(g1)
    
    # Seed with random permutations of greedy? Or just random restarts of greedy
    for i in range(pop_size - 1):
        # Slightly perturb greedy
        g = _greedy_partition(edge_list, adj, n)
        # Random swap of edges between cliques?
        # Just use the greedy result for now, maybe with different random seeds for tie-breaking
        population.append(g)
        
    # Sort by number of cliques
    population.sort(key=lambda c: len(c))
    
    best_solution = population[0]
    best_count = len(best_solution)
    
    start_time = time.time()
    generation = 0
    
    while time.time() - start_time < time_budget:
        if _remaining() < 5:
            break
            
        new_population = []
        
        # Elitism: keep best 2
        new_population.append(best_solution)
        if len(population) > 1:
            new_population.append(population[1])
            
        # Generate offspring
        while len(new_population) < pop_size:
            # Selection: Tournament
            p1_idx = random.randint(0, len(population)-1)
            p2_idx = random.randint(0, len(population)-1)
            if random.random() < 0.5:
                if len(population[p1_idx]) > len(population[p2_idx]):
                    p1_idx, p2_idx = p2_idx, p1_idx
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # Crossover
            child = _crossover(p1, p2, edge_list, adj, n)
            
            # Local Search (Mutation/Improvement)
            # Budget per individual
            ls_budget = max(0.5, (time_budget - (time.time() - start_time)) / (pop_size - len(new_population) - 1))
            child = _local_search(child, edge_list, adj, n, time_budget=ls_budget)
            
            # Verify
            if not _verify_partition(child, edge_list, adj):
                # Fallback to parent
                child = p1
            
            new_population.append(child)
            
            if len(child) < best_count:
                best_count = len(child)
                best_solution = [list(c) for c in child]
                
        population = new_population
        generation += 1
        
    return best_solution

def construct():
    global _START
    _START = time.time()
    
    results = []
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if _remaining() < 20:
            adj, edge_list = _generate_graph(n, p, seed)
            results.append([[u, v] for u, v in edge_list])
            continue
        
        adj, edge_list = _generate_graph(n, p, seed)
        
        time_budget = min(65.0, _remaining() - 10)
        
        cliques = _solve_one_graph(adj, edge_list, time_budget)
        
        if not _verify_partition(cliques, edge_list, adj):
            cliques = [[u, v] for u, v in edge_list]
        
        results.append(cliques)
        
    return results
