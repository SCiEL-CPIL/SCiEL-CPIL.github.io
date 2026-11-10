import numpy as np
import math
import time

def _jc_dist(a, b):
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / len(a)
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _nj(dist, labels):
    nodes = list(labels)
    trees = {name: name for name in labels}
    while len(nodes) > 3:
        n = len(nodes)
        r = {a: sum(dist[(a, b)] for b in nodes if b != a) for a in nodes}
        best_crit = float('inf')
        best_pair = None
        for i in range(n):
            for j in range(i+1, n):
                a, b = nodes[i], nodes[j]
                crit = (n - 2) * dist[(a, b)] - r[a] - r[b]
                if crit < best_crit:
                    best_crit = crit
                    best_pair = (a, b)
        a, b = best_pair
        joined = f"({trees[a]},{trees[b]})"
        new = f"n{len(nodes)}_{a}_{b}"
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            dist[(new, x)] = dist[(x, new)] = 0.5 * (dist[(a, x)] + dist[(b, x)] - dist[(a, b)])
        nodes = rest + [new]
        trees[new] = joined
    return "(" + ",".join(trees[x] for x in nodes) + ");"

def _parse_newick(newick):
    """Parse Newick to get adjacency list."""
    neighbors = {}
    leaves = set()
    pos = 0
    
    def parse():
        nonlocal pos
        if newick[pos] == '(':
            pos += 1
            children = []
            while True:
                child = parse()
                children.append(child)
                if newick[pos] == ',':
                    pos += 1
                elif newick[pos] == ')':
                    pos += 1
                    break
            # Skip branch length
            while pos < len(newick) and newick[pos] not in '(),;':
                pos += 1
            node = f"_int_{len(neighbors)}"
            neighbors[node] = set(children)
            for c in children:
                neighbors.setdefault(c, set()).add(node)
            return node
        else:
            name = ''
            while pos < len(newick) and newick[pos] not in '(),;':
                name += newick[pos]
                pos += 1
            # Skip branch length
            while pos < len(newick) and newick[pos] not in '(),;':
                pos += 1
            neighbors.setdefault(name, set())
            leaves.add(name)
            return name
    
    newick = newick.rstrip(';')
    root = parse()
    return neighbors, leaves, root

def _nni_improve(newick, sequences, labels, dist, max_time=8.0):
    """NNI local search with distance-based scoring."""
    neighbors, leaves, root = _parse_newick(newick)
    
    # Score function: sum of squared differences between observed and tree distances
    def score_tree(neighbors):
        # Compute all-pairs tree distances
        # For each pair of leaves, find path in tree
        n = len(leaves)
        leaf_list = list(leaves)
        total = 0.0
        for i in range(n):
            for j in range(i+1, n):
                # BFS from leaf_list[i] to leaf_list[j]
                from collections import deque
                q = deque([(leaf_list[i], 0)])
                visited = {leaf_list[i]}
                found = False
                while q:
                    node, d = q.popleft()
                    if node == leaf_list[j]:
                        total += (dist[(leaf_list[i], leaf_list[j])] - d) ** 2
                        found = True
                        break
                    for nb in neighbors[node]:
                        if nb not in visited:
                            visited.add(nb)
                            q.append((nb, d + 1))  # Unit edge lengths
                if not found:
                    total += 100.0
        return total
    
    # This is O(n^2 * n) per score evaluation = O(n^3), and we do O(n) NNI moves per pass.
    # For n=32, that's 32^3 = 32768 per score, times 32 moves = ~1M per pass. Fine.
    
    # Get all internal edges
    def get_internal_edges(neighbors):
        edges = []
        seen = set()
        for node in neighbors:
            for nb in neighbors[node]:
                if (node, nb) not in seen and (nb, node) not in seen:
                    # Check if both are internal
                    if node not in leaves and nb not in leaves:
                        edges.append((node, nb))
                        seen.add((node, nb))
                        seen.add((nb, node))
        return edges
    
    best_score = score_tree(neighbors)
    improved = True
    while improved:
        improved = False
        edges = get_internal_edges(neighbors)
        for (u, v) in edges:
            # u has neighbors: v, a, b
            # v has neighbors: u, c, d
            u_nbs = list(neighbors[u])
            v_nbs = list(neighbors[v])
            a, b = [x for x in u_nbs if x != v]
            c, d = [x for x in v_nbs if x != u]
            
            # Three NNI moves:
            # Original: u-a, u-b, u-v, v-c, v-d
            # Move 1: u-a, u-c, u-v, v-b, v-d
            # Move 2: u-a, u-d, u-v, v-b, v-c
            
            for move in [1, 2]:
                new_neighbors = {k: set(v) for k, v in neighbors.items()}
                if move == 1:
                    # Swap b and c
                    new_neighbors[u].remove(b)
                    new_neighbors[u].add(c)
                    new_neighbors[b].remove(u)
                    new_neighbors[b].add(v)
                    new_neighbors[v].remove(c)
                    new_neighbors[v].add(b)
                    new_neighbors[c].remove(v)
                    new_neighbors[c].add(u)
                else:
                    # Swap b and d
                    new_neighbors[u].remove(b)
                    new_neighbors[u].add(d)
                    new_neighbors[b].remove(u)
                    new_neighbors[b].add(v)
                    new_neighbors[v].remove(d)
                    new_neighbors[v].add(b)
                    new_neighbors[d].remove(v)
                    new_neighbors[d].add(u)
                
                s = score_tree(new_neighbors)
                if s < best_score - 1e-6:
                    neighbors = new_neighbors
                    best_score = s
                    improved = True
                    break
    
    # Convert back to Newick
    return _to_newick(neighbors, leaves)

def _to_newick(neighbors, leaves):
    """Convert adjacency list to Newick string."""
    # Root the tree at an internal node
    internal_nodes = [n for n in neighbors if n not in leaves]
    if not internal_nodes:
        return ""
    root = internal_nodes[0]
    
    parent = {}
    children = {}
    visited = set()
    
    def build(node, par):
        parent[node] = par
        children[node] = []
        for nb in neighbors[node]:
            if nb != par and nb not in visited:
                visited.add(nb)
                children[node].append(nb)
                build(nb, node)
    
    visited.add(root)
    build(root, None)
    
    def to_string(node):
        if node in leaves:
            return node
        if len(children[node]) == 0:
            return f"({node})"
        child_strs = [to_string(c) for c in children[node]]
        return f"({','.join(child_strs)})"
    
    return to_string(root) + ";"

def reconstruct(sequences):
    start_time = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    
    # Compute distance matrix
    dist = {}
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            d = _jc_dist(sequences[labels[i]], sequences[labels[j]])
            dist[(labels[i], labels[j])] = d
            dist[(labels[j], labels[i])] = d
    
    # NJ
    newick = _nj(dist, labels)
    
    elapsed = time.time() - start_time
    if elapsed > 170:
        return newick
    
    # NNI refinement
    try:
        newick = _nni_improve(newick, sequences, labels, dist, max_time=170 - elapsed)
    except:
        pass
    
    return newick
