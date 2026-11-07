"""Packing trees T1..T20 into K20."""
import time
from random import Random

# Fixed deadline: evaluator kills at 150s, we want to return well before.
_DEADLINE = 135.0  # seconds

def _instance_trees(n=20, seed=611):
    """Generate the fixed trees T1..T20."""
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees


def _embed_tree(tree_edges, n, used_edges, rng):
    """
    Try to embed a tree (list of (u,v) edges on vertices 0..k-1) into K_n
    such that no edge of K_n is used more than once.
    Returns a list of length k mapping tree vertex i -> host vertex, or None.
    """
    k = len(tree_edges) + 1 if tree_edges else 1
    # Build adjacency for the tree
    adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Try multiple random orderings / backtracking attempts
    max_attempts = 50
    for attempt in range(max_attempts):
        # Random permutation of tree vertices for backtracking order
        order = list(range(k))
        rng.shuffle(order)
        
        # We'll do DFS/backtracking assigning host vertices
        assign = [-1] * k  # tree vertex -> host vertex
        # Pick a root
        root = order[0]
        assign[root] = rng.randint(0, n - 1)
        
        # Build a traversal order (BFS/DFS) from root
        visited = [False] * k
        queue = [root]
        visited[root] = True
        traversal = [root]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
                    traversal.append(v)
        
        # Backtracking: assign each vertex in traversal order
        def backtrack(idx):
            if idx >= len(traversal):
                return True
            tv = traversal[idx]
            # Get already-assigned neighbors
            assigned_neighbors = [assign[nb] for nb in adj[tv] if assign[nb] != -1]
            # Candidate host vertices: 0..n-1 not causing edge conflict
            candidates = []
            for hv in range(n):
                ok = True
                for nb_hv in assigned_neighbors:
                    if min(hv, nb_hv) in used_edges.get((min(hv, nb_hv), max(hv, nb_hv)), None) or True:
                        # Check if edge (hv, nb_hv) is already used
                        e = (min(hv, nb_hv), max(hv, nb_hv))
                        if e in used_edges:
                            ok = False
                            break
                if ok:
                    candidates.append(hv)
            rng.shuffle(candidates)
            for hv in candidates:
                assign[tv] = hv
                # Add edges to used set
                added_edges = []
                valid = True
                for nb in adj[tv]:
                    if assign[nb] != -1:
                        e = (min(hv, assign[nb]), max(hv, assign[nb]))
                        if e in used_edges:
                            valid = False
                            break
                        added_edges.append(e)
                if not valid:
                    continue
                for e in added_edges:
                    used_edges[e] = True
                if backtrack(idx + 1):
                    return True
                for e in added_edges:
                    del used_edges[e]
            assign[tv] = -1
            return False
        
        if backtrack(1):  # root already assigned
            # Verify and return
            result = assign[:]
            # Double-check all edges are distinct
            seen = set()
            for u, v in tree_edges:
                e = (min(result[u], result[v]), max(result[u], result[v]))
                if e in seen:
                    # Shouldn't happen but just in case
                    return None
                seen.add(e)
            return result
        # Roll back root assignment
        assign[root] = -1
    
    return None


def construct():
    start_time = time.time()
    trees = _instance_trees(20, 611)
    n = 20
    used_edges = set()
    result = [None] * 20
    rng = Random(42)
    
    # Embed trees in order T1..T20 (small first is often easier)
    for k in range(1, 21):
        if time.time() - start_time > _DEADLINE - 10:
            break
        tree_edges = trees[k - 1]
        if not tree_edges:
            # T1: single vertex, no edges. Map to any vertex.
            result[k - 1] = [0]
            continue
        
        # Try multiple times with different random seeds/orderings
        embedded = None
        max_tries = 200
        for t in range(max_tries):
            if time.time() - start_time > _DEADLINE - 5:
                break
            # Save current state of used_edges
            saved_used = set(used_edges)
            emb = _embed_tree(tree_edges, n, used_edges, rng)
            if emb is not None:
                # Verify
                valid = True
                for u, v in tree_edges:
                    e = (min(emb[u], emb[v]), max(emb[u], emb[v]))
                    if e in used_edges:
                        valid = False
                        break
                if valid:
                    embedded = emb
                    for u, v in tree_edges:
                        used_edges.add((min(emb[u], emb[v]), max(emb[u], emb[v])))
                    break
            else:
                # Restore
                used_edges.clear()
                used_edges.update(saved_used)
        
        if embedded is not None:
            result[k - 1] = embedded
        else:
            result[k - 1] = None
            # Continue trying others even if one fails
    
    return result
