import time
import random
from random import Random
from collections import defaultdict

def instance_trees(n=20, seed=611):
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

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    trees = instance_trees(20, 611)
    N = 20
    
    # Pre-calculate tree structures
    tree_structures = []
    for idx, edges in enumerate(trees):
        k = idx + 1
        if k == 1:
            tree_structures.append({
                'k': k,
                'edges': [],
                'adj': [[]],
                'deg': [0],
                'var_order': [0]
            })
            continue
        deg = [0] * k
        adj = [[] for _ in range(k)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].append(v)
            adj[v].append(u)
        
        # Sort vertices by degree descending for MRV
        var_order = sorted(range(k), key=lambda v: -deg[v])
        
        tree_structures.append({
            'k': k,
            'edges': edges,
            'adj': adj,
            'deg': deg,
            'var_order': var_order
        })

    def calculate_host_remaining_deg(used_edges):
        """
        Calculate the number of available edges for each host vertex.
        A host vertex h has N-1 potential edges. 
        remaining_deg[h] = (N - 1) - (number of used edges incident to h)
        """
        host_deg = [0] * N
        for u, v in used_edges:
            host_deg[u] += 1
            host_deg[v] += 1
        return [N - 1 - d for d in host_deg]

    def get_available_edges(h, used_edges):
        """Get the set of vertices that h can connect to (excluding already-used edges)."""
        available = set()
        for v in range(N):
            if v == h:
                continue
            edge = (min(h, v), max(h, v))
            if edge not in used_edges:
                available.add(v)
        return available

    def embed_tree(tree_idx, used_edges, rng, node_limit=200000):
        """
        Try to embed tree_idx into the graph defined by N=20 and used_edges.
        Returns a list of length k with host vertices, or None if failed.
        """
        struct = tree_structures[tree_idx]
        k = struct['k']
        adj = struct['adj']
        var_order = struct['var_order']
        
        if k == 1:
            # Any vertex works, pick 0 for simplicity (or random)
            # Since vertices can be shared, this doesn't consume edges.
            return [0]
            
        mapping = [-1] * k
        used_hosts = set()
        nodes_explored = [0]
        
        # Pre-calculate remaining degrees of hosts to guide candidate ordering
        host_rem_deg = calculate_host_remaining_deg(used_edges)
        
        # Pre-compute which tree vertices have already-mapped neighbors
        # This helps with candidate filtering
        def backtrack(idx):
            if nodes_explored[0] > node_limit:
                return False
            if idx == k:
                return True
            
            v = var_order[idx]
            cands = []
            
            # Determine which neighbors of v are already mapped
            mapped_neighbors = []
            for w in adj[v]:
                if mapping[w] != -1:
                    mapped_neighbors.append(mapping[w])
            
            for h in range(N):
                if h in used_hosts:
                    continue
                # Check if edges to already mapped neighbors are available
                ok = True
                for mw in mapped_neighbors:
                    edge = (min(h, mw), max(h, mw))
                    if edge in used_edges:
                        ok = False
                        break
                if ok:
                    cands.append(h)
            
            if not cands:
                return False
            
            # Heuristic: Sort candidates by remaining degree in host graph (descending)
            # Also consider the number of available edges from h to unmapped tree neighbors
            # This helps pack dense parts of the tree into well-connected host regions
            
            def candidate_score(h):
                # Primary: remaining degree in host
                # Secondary: number of available connections to future tree vertices
                # Tertiary: random jitter
                avail_count = len(get_available_edges(h, used_edges))
                return (-host_rem_deg[h], -avail_count, rng.random())
            
            cands.sort(key=candidate_score)
            
            for h in cands:
                nodes_explored[0] += 1
                if nodes_explored[0] > node_limit:
                    return False
                
                mapping[v] = h
                used_hosts.add(h)
                
                # Record added edges to undo later
                added_edges = []
                for w in adj[v]:
                    mw = mapping[w]
                    if mw != -1:
                        e = (min(h, mw), max(h, mw))
                        used_edges.add(e)
                        added_edges.append(e)
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                for e in added_edges:
                    used_edges.discard(e)
                used_hosts.discard(h)
                mapping[v] = -1
            
            return False
        
        if backtrack(0):
            return [mapping[i] for i in range(k)]
        return None

    best_result = None
    best_edges = -1
    
    # Process trees in order of decreasing size (most constrained first)
    # T20 (19 edges) down to T1 (0 edges)
    order = list(range(19, -1, -1))
    
    # We will try multiple random restarts with different random seeds for the internal heuristics
    # If the first deterministic-like run fails, randomness might help.
    
    # Track how many attempts we've done and adapt based on time remaining
    attempts_done = 0
    max_attempts = 20  # Increased from 10 to give more chances
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
        if best_edges == 190:
            break
            
        rng = Random(611 + attempt)
        used_edges = set()
        embedding = [None] * N
        total_edges = 0
        success = True
        
        for idx in order:
            if time.time() > deadline - 5:
                success = False
                break
                
            k = idx + 1
            if k == 1:
                embedding[idx] = [0]
                continue
                
            # For T2 (1 edge), it's very easy, but we use the same solver for consistency
            # The solver handles k=2 fine.
            
            m = embed_tree(idx, used_edges, rng)
            if m is None:
                success = False
                break
            
            embedding[idx] = m
            # Add edges to used_edges
            tree_edges = trees[idx]
            for u, v in tree_edges:
                host_u = m[u]
                host_v = m[v]
                e = (min(host_u, host_v), max(host_u, host_v))
                used_edges.add(e)
            total_edges += len(tree_edges)
            
        if success and total_edges > best_edges:
            best_edges = total_edges
            best_result = embedding
            if best_edges == 190:
                break
        attempts_done += 1

    if best_result is not None:
        # Ensure types are correct: list of ints or None
        result = [list(e) if e is not None else None for e in best_result]
        # Validate that all entries are either None or lists of distinct ints in 0..19
        for i, entry in enumerate(result):
            if entry is not None:
                if not isinstance(entry, list):
                    entry = list(entry)
                    result[i] = entry
                if len(set(entry)) != len(entry):
                    # Duplicate vertices - invalid
                    result[i] = None
                for v in entry:
                    if not isinstance(v, int) or isinstance(v, bool) or v < 0 or v > 19:
                        result[i] = None
                        break
        return result
    
    # Fallback if nothing worked (should not happen for T20 alone)
    return [None] * 19 + [list(range(20))]

# Test harness / main entry point
if __name__ == "__main__":
    result = construct()
    print(result)
