"""
Clique partition via Greedy Maximum Clique Extraction and Local Repair.
Strategy:
1. Greedily extract large cliques from the graph to cover as many edges as possible with few cliques.
2. The remaining graph consists of small components (mostly triangles and edges).
3. Use local search to merge small cliques (triangles into K4s, etc.) where possible.
4. This avoids the O(N^2) overhead of checking invalid edge-label assignments in SA.
"""
import time
import random
from collections import defaultdict

_START = time.time()
DEADLINE = 145.0

def _remaining():
    return DEADLINE - time.time()

def _generate_graph(n, p, seed):
    """Generate the same graph as the verifier."""
    rng = random.Random(seed)
    adj = [[False] * n for _ in range(n)]
    edge_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u][v] = adj[v][u] = True
                edge_list.append((u, v))
    return adj, edge_list

def _find_max_clique_greedy(adj, available_vertices, max_size=10):
    """
    Find a large clique in the subgraph induced by available_vertices.
    Uses a simple greedy expansion: start with a random vertex, add neighbors
    that are connected to all current members.
    """
    if not available_vertices:
        return []
    
    # Sort vertices by degree in the available subgraph (descending)
    # to prefer high-degree vertices for the seed
    degs = []
    for v in available_vertices:
        deg = sum(1 for u in available_vertices if u != v and adj[v][u])
        degs.append((deg, v))
    degs.sort(reverse=True)
    
    best_clique = []
    # Try a few seeds to find a larger clique
    for _ in range(min(5, len(available_vertices))):
        if not available_vertices:
            break
        # Pick a high-degree vertex as seed
        seed_v = degs[random.randint(0, len(degs)-1)][1]
        
        current_clique = [seed_v]
        candidates = [u for u in available_vertices if u != seed_v and adj[seed_v][u]]
        
        while candidates:
            # Pick candidate with highest degree in the remaining candidates
            # that is connected to all in current_clique
            valid_candidates = []
            for u in candidates:
                if all(adj[u][w] for w in current_clique):
                    valid_candidates.append(u)
            
            if not valid_candidates:
                break
            
            # Heuristic: pick the one with highest degree in valid_candidates
            # to keep the clique "dense"
            def get_deg(vc):
                return sum(1 for x in valid_candidates if x != vc and adj[vc][x])
            
            valid_candidates.sort(key=get_deg, reverse=True)
            next_v = valid_candidates[0]
            current_clique.append(next_v)
            
            # Update candidates
            candidates = [u for u in candidates if u != next_v and all(adj[u][w] for w in current_clique)]
            
            if len(current_clique) >= max_size:
                break
        
        if len(current_clique) > len(best_clique):
            best_clique = current_clique
            
    return best_clique

def _greedy_clique_partition(adj, edge_list):
    """
    Repeatedly find a large clique, add it to the partition, and remove its edges.
    """
    n = len(adj)
    # Active edge set
    active_edges = set(edge_list)
    # Active vertex set (vertices incident to at least one active edge)
    active_vertices = set()
    for u, v in active_edges:
        active_vertices.add(u)
        active_vertices.add(v)
        
    cliques = []
    
    while active_edges:
        if _remaining() < 5:
            break
            
        # Find a large clique in the active subgraph
        clique = _find_max_clique_greedy(adj, active_vertices, max_size=8)
        
        if not clique:
            # Fallback: just take any edge
            u, v = next(iter(active_edges))
            clique = [u, v]
            
        # Ensure the clique is actually valid (safety check)
        valid = True
        for i in range(len(clique)):
            for j in range(i+1, len(clique)):
                if not adj[clique[i]][clique[j]]:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid or len(clique) < 2:
             # If greedy failed, pick a random edge
             u, v = next(iter(active_edges))
             clique = [u, v]
        
        cliques.append(clique)
        
        # Remove edges covered by this clique
        clique_set = set(clique)
        edges_to_remove = []
        for u in clique:
            for v in clique:
                if u < v:
                    e = (u, v)
                    if e in active_edges:
                        edges_to_remove.append(e)
                        
        for e in edges_to_remove:
            active_edges.remove(e)
            
        # Update active vertices
        # A vertex is active if it has an incident active edge
        # We can just rebuild or check degrees. Rebuilding is O(E) which is fine for n=64.
        new_active_vertices = set()
        for u, v in active_edges:
            new_active_vertices.add(u)
            new_active_vertices.add(v)
        active_vertices = new_active_vertices
            
    return cliques

def _local_search_merge(cliques, adj, edge_list):
    """
    Try to merge cliques if their union is a clique.
    Repeat until no more merges are possible.
    """
    n_edges = len(edge_list)
    # Map edge to index for quick lookup
    edge_idx = {(min(u,v), max(u,v)): i for i, (u,v) in enumerate(edge_list)}
    
    # Current cliques as sets of vertices
    current_cliques = [set(c) for c in cliques]
    
    changed = True
    while changed:
        if _remaining() < 2:
            break
        changed = False
        for i in range(len(current_cliques)):
            if _remaining() < 2:
                break
            for j in range(i + 1, len(current_cliques)):
                # Check if union is a clique
                union_verts = current_cliques[i] | current_cliques[j]
                valid = True
                verts_list = list(union_verts)
                for a in range(len(verts_list)):
                    for b in range(a + 1, len(verts_list)):
                        if not adj[verts_list[a]][verts_list[b]]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Merge j into i
                    current_cliques[i] = union_verts
                    current_cliques[j] = set()
                    changed = True
                    break
            if changed:
                break
                
    # Filter out empty sets
    final_cliques = [sorted(list(c)) for c in current_cliques if c]
    return final_cliques

def _verify_partition(cliques, edge_list, adj):
    """Verify every edge is in exactly one clique and each clique is valid."""
    n_edges = len(edge_list)
    edge_cover = [0] * n_edges
    edge_idx = {(min(u,v), max(u,v)): i for i, (u,v) in enumerate(edge_list)}
    
    for clique in cliques:
        # Check clique validity
        for a in range(len(clique)):
            for b in range(a + 1, len(clique)):
                if not adj[clique[a]][clique[b]]:
                    return False
        # Mark edges
        for a in range(len(clique)):
            for b in range(a + 1, len(clique)):
                e = (min(clique[a], clique[b]), max(clique[a], clique[b]))
                if e in edge_idx:
                    edge_cover[edge_idx[e]] += 1
    
    for c in edge_cover:
        if c != 1:
            return False
    return True

def construct():
    global _START
    _START = time.time()
    
    results = []
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if _remaining() < 15:
            # Fallback: just output each edge as its own clique
            adj, edge_list = _generate_graph(n, p, seed)
            results.append([[u, v] for u, v in edge_list])
            continue
        
        adj, edge_list = _generate_graph(n, p, seed)
        
        # 1. Greedy extraction
        cliques = _greedy_clique_partition(adj, edge_list)
        
        # 2. Local search merge
        cliques = _local_search_merge(cliques, adj, edge_list)
        
        # 3. Verify
        if not _verify_partition(cliques, edge_list, adj):
            # Fallback to each edge its own clique
            cliques = [[u, v] for u, v in edge_list]
            
        results.append(cliques)
        
    return results
