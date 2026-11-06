"""
Clique partition via simulated annealing on edge-clique assignments.
Each edge is assigned to a clique label. Cliques are derived as connected
components of the hypergraph where edges sharing a label form a clique.
Moves: reassign, split, merge. Cost = number of cliques.
"""
import time
import random
from itertools import combinations

_START = time.time()
DEADLINE = 145.0  # safety margin from 150s

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

def _edge_index(edge_list):
    """Map edge (u,v) with u<v to index."""
    return {(min(u,v), max(u,v)): i for i, (u,v) in enumerate(edge_list)}

def _build_initial_assignment(adj, edge_list):
    """
    Greedy initial assignment: each edge gets its own label, then merge
    adjacent edges into cliques greedily.
    Returns: label array (one per edge), adj, edge_list
    """
    n_edges = len(edge_list)
    labels = list(range(n_edges))  # each edge its own clique initially
    
    # Greedy: try to merge edges into larger cliques
    # Build adjacency of edges: two edges are "compatible" if they share a vertex
    # and the four endpoints (or three if sharing one) form a clique
    # Simpler: iteratively try to merge two cliques if union is a clique
    
    # First pass: for each vertex, try to make a clique from its incident edges
    # Actually, let's do a simple greedy: repeatedly find a set of edges that
    # form a clique and assign them the same label
    
    # Better: start with all separate, then merge
    changed = True
    while changed:
        changed = False
        for i in range(n_edges):
            if _remaining() < 5:
                break
            for j in range(i + 1, n_edges):
                if labels[i] != labels[j]:
                    # Check if merging these two cliques (union of their edges' vertices)
                    # forms a valid clique
                    # Get all vertices in clique i and clique j
                    verts_i = set()
                    verts_j = set()
                    for k in range(n_edges):
                        if labels[k] == labels[i]:
                            u, v = edge_list[k]
                            verts_i.add(u)
                            verts_i.add(v)
                        elif labels[k] == labels[j]:
                            u, v = edge_list[k]
                            verts_j.add(u)
                            verts_j.add(v)
                    union_verts = verts_i | verts_j
                    # Check if union_verts forms a clique
                    valid = True
                    verts_list = list(union_verts)
                    for a in range(len(verts_list)):
                        for b in range(a + 1, len(verts_list)):
                            if not adj[verts_list[a]][verts_list[b]]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid and len(union_verts) >= 2:
                        # Merge: assign all edges in clique j to label i
                        for k in range(n_edges):
                            if labels[k] == labels[j]:
                                labels[k] = labels[i]
                        changed = True
                        break
            if changed:
                break
        if _remaining() < 5:
            break
    
    return labels

def _get_cliques_from_labels(labels, edge_list):
    """
    Given edge labels, derive the actual cliques (sets of vertices).
    Two edges in the same clique must form a valid clique together.
    Returns list of frozensets of vertices.
    """
    from collections import defaultdict
    label_to_edges = defaultdict(list)
    for i, lab in enumerate(labels):
        label_to_edges[lab].append(i)
    
    cliques = []
    for lab, edge_indices in label_to_edges.items():
        verts = set()
        for ei in edge_indices:
            u, v = edge_list[ei]
            verts.add(u)
            verts.add(v)
        if len(verts) >= 2:
            cliques.append(frozenset(verts))
    return cliques

def _is_valid_clique_partition(labels, edge_list, adj):
    """
    Check that each label class forms a valid clique and all edges are covered.
    """
    from collections import defaultdict
    label_to_edges = defaultdict(list)
    for i, lab in enumerate(labels):
        label_to_edges[lab].append(i)
    
    for lab, edge_indices in label_to_edges.items():
        verts = set()
        for ei in edge_indices:
            u, v = edge_list[ei]
            verts.add(u)
            verts.add(v)
        if len(verts) < 2:
            continue
        # Check clique
        verts_list = list(verts)
        for a in range(len(verts_list)):
            for b in range(a + 1, len(verts_list)):
                if not adj[verts_list[a]][verts_list[b]]:
                    return False
    return True

def _simulated_annealing(adj, edge_list, initial_labels, max_time=120.0):
    """
    SA on edge-clique assignments.
    State: labels array (one per edge)
    Cost: number of distinct labels (i.e., number of cliques)
    Moves:
      1. Reassign edge i to a random existing label j (if valid)
      2. Split a clique: pick a label, split its edges into two groups,
         check both form valid cliques
      3. Merge two labels if their union forms a valid clique
    """
    n_edges = len(edge_list)
    labels = initial_labels[:]
    current_cost = len(set(labels))
    best_labels = labels[:]
    best_cost = current_cost
    
    start_time = time.time()
    max_time = min(max_time, _remaining() - 5)
    if max_time <= 0:
        return best_labels
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    rng = random.Random(42)
    
    # Precompute: for each edge, which other edges it can be in a clique with
    # edge_adj[i] = set of edge indices j such that edge i and edge j can be in same clique
    # (i.e., the union of their vertices forms a clique)
    edge_adj = [set() for _ in range(n_edges)]
    for i in range(n_edges):
        ui, vi = edge_list[i]
        for j in range(i + 1, n_edges):
            uj, vj = edge_list[j]
            # Union of vertices
            verts = {ui, vi, uj, vj}
            valid = True
            verts_list = list(verts)
            for a in range(len(verts_list)):
                for b in range(a + 1, len(verts_list)):
                    if not adj[verts_list[a]][verts_list[b]]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                edge_adj[i].add(j)
                edge_adj[j].add(i)
    
    def get_clique_vertices(label, labels, edge_list):
        verts = set()
        for i in range(n_edges):
            if labels[i] == label:
                u, v = edge_list[i]
                verts.add(u)
                verts.add(v)
        return verts
    
    def check_clique_valid(verts, adj):
        if len(verts) < 2:
            return True
        verts_list = list(verts)
        for a in range(len(verts_list)):
            for b in range(a + 1, len(verts_list)):
                if not adj[verts_list[a]][verts_list[b]]:
                    return False
        return True
    
    steps = 0
    while time.time() - start_time < max_time:
        steps += 1
        move_type = rng.random()
        
        if move_type < 0.4:
            # Reassign: pick random edge, assign to random existing label
            i = rng.randrange(n_edges)
            current_label = labels[i]
            # Get all existing labels
            all_labels = list(set(labels))
            if len(all_labels) <= 1:
                continue
            # Pick a different label
            candidates = [l for l in all_labels if l != current_label]
            if not candidates:
                continue
            new_label = rng.choice(candidates)
            
            # Check validity: edge i's vertices must be compatible with clique new_label
            ui, vi = edge_list[i]
            verts_new = get_clique_vertices(new_label, labels, edge_list)
            verts_new.add(ui)
            verts_new.add(vi)
            if check_clique_valid(verts_new, adj):
                # Also check that removing edge i from old clique doesn't break it
                # (it won't, since removing edges can't break clique property)
                old_verts = get_clique_vertices(current_label, labels, edge_list)
                old_verts.discard(ui)
                old_verts.discard(vi)
                # Old clique is still valid (subset of valid clique)
                labels[i] = new_label
                new_cost = len(set(labels))
                delta = new_cost - current_cost
                if delta <= 0 or rng.random() < min(1.0, T / max(1, abs(delta))):
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_labels = labels[:]
                else:
                    labels[i] = current_label
        elif move_type < 0.7:
            # Split: pick a label with >= 2 edges, split into two
            all_labels = list(set(labels))
            if not all_labels:
                continue
            lab = rng.choice(all_labels)
            edges_in_lab = [i for i in range(n_edges) if labels[i] == lab]
            if len(edges_in_lab) < 2:
                continue
            # Random split
            rng.shuffle(edges_in_lab)
            split_point = rng.randrange(1, len(edges_in_lab))
            group1 = edges_in_lab[:split_point]
            group2 = edges_in_lab[split_point:]
            
            # Check both groups form valid cliques
            verts1 = set()
            for ei in group1:
                u, v = edge_list[ei]
                verts1.add(u)
                verts1.add(v)
            verts2 = set()
            for ei in group2:
                u, v = edge_list[ei]
                verts2.add(u)
                verts2.add(v)
            
            if len(verts1) >= 2 and len(verts2) >= 2:
                if check_clique_valid(verts1, adj) and check_clique_valid(verts2, adj):
                    # Assign group2 to a new label
                    new_label = max(set(labels)) + 1
                    for ei in group2:
                        labels[ei] = new_label
                    new_cost = len(set(labels))
                    delta = new_cost - current_cost
                    if delta <= 0 or rng.random() < min(1.0, T / max(1, abs(delta))):
                        current_cost = new_cost
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_labels = labels[:]
                    else:
                        for ei in group2:
                            labels[ei] = lab
        else:
            # Merge: pick two labels, check if union is valid clique
            all_labels = list(set(labels))
            if len(all_labels) < 2:
                continue
            lab1, lab2 = rng.sample(all_labels, 2)
            verts1 = get_clique_vertices(lab1, labels, edge_list)
            verts2 = get_clique_vertices(lab2, labels, edge_list)
            union_verts = verts1 | verts2
            if check_clique_valid(union_verts, adj):
                # Merge
                for i in range(n_edges):
                    if labels[i] == lab2:
                        labels[i] = lab1
                new_cost = len(set(labels))
                delta = new_cost - current_cost
                if delta <= 0 or rng.random() < min(1.0, T / max(1, abs(delta))):
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_labels = labels[:]
                else:
                    # Revert
                    for i in range(n_edges):
                        if labels[i] == lab1:
                            # This is tricky; we need to know which were originally lab2
                            # Let's just skip revert for simplicity and accept
                            pass
                    # Actually, let's just always accept merges since they reduce cost
                    pass
        
        T *= cooling
    
    return best_labels

def _labels_to_clique_list(labels, edge_list):
    """Convert edge labels to list of cliques (vertex lists)."""
    from collections import defaultdict
    label_to_edges = defaultdict(list)
    for i, lab in enumerate(labels):
        label_to_edges[lab].append(i)
    
    cliques = []
    for lab, edge_indices in label_to_edges.items():
        verts = set()
        for ei in edge_indices:
            u, v = edge_list[ei]
            verts.add(u)
            verts.add(v)
        if len(verts) >= 2:
            cliques.append(sorted(verts))
    return cliques

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
        n_edges = len(edge_list)
        
        # Initial greedy assignment
        initial_labels = _build_initial_assignment(adj, edge_list)
        
        # Run SA
        sa_labels = _simulated_annealing(adj, edge_list, initial_labels, max_time=min(60.0, _remaining() - 10))
        
        # Convert to cliques
        cliques = _labels_to_clique_list(sa_labels, edge_list)
        
        # Verify
        if not _verify_partition(cliques, edge_list, adj):
            # Fallback to initial
            cliques = _labels_to_clique_list(initial_labels, edge_list)
            if not _verify_partition(cliques, edge_list, adj):
                # Last resort: each edge its own clique
                cliques = [[u, v] for u, v in edge_list]
        
        results.append(cliques)
    
    return results
