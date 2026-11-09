"""Improved program: pack trees T1..T20 into K20 via randomized backtracking."""
import time
from random import Random
from collections import defaultdict

def construct():
    deadline = time.time() + 135  # safety margin before 150s limit
    n = 20
    
    # Generate the trees
    trees = _generate_trees(n, seed=611)
    
    # Best result so far: just T20 embedded by identity
    best_edges = n - 1  # 19 edges for T20
    best_result = [None] * (n - 1) + [list(range(n))]
    
    # Try to pack all trees using randomized backtracking
    rng = Random(42)
    
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Randomize tree ordering (larger trees first tends to help)
        tree_order = list(range(1, n + 1))
        # Sort by size descending, with some randomization
        rng.shuffle(tree_order)
        tree_order.sort(key=lambda k: -k)
        
        # Try to embed
        result = _try_pack(trees, tree_order, n, rng, deadline)
        
        if result is not None:
            # Count edges
            total_edges = sum(len(trees[k-1]) for k in range(1, n + 1) if result[k-1] is not None)
            if total_edges > best_edges:
                best_edges = total_edges
                best_result = result
                if total_edges == n * (n - 1) // 2:  # 190
                    break
    
    return best_result


def _generate_trees(n=20, seed=611):
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


def _try_pack(trees, tree_order, n, rng, deadline):
    """
    Try to embed trees in tree_order into K_n with edge-disjoint embeddings.
    Returns a list of length n where index k-1 is None or a list of k distinct ints.
    """
    # Track used edges (frozenset of frozensets)
    used_edges = set()
    
    # Assignment: for each tree k, mapping from tree vertex -> host vertex
    assignments = [None] * (n + 1)
    
    # For backtracking, we need to undo
    def backtrack(idx):
        if time.time() > deadline:
            return False
        
        if idx == len(tree_order):
            return True
        
        k = tree_order[idx]
        tree = trees[k - 1]
        
        if k == 1:
            # Single vertex, any unused host vertex
            for hv in range(n):
                assignments[k] = [hv]
                if backtrack(idx + 1):
                    return True
            assignments[k] = None
            return False
        
        # Build tree adjacency
        adj = defaultdict(list)
        for u, v in tree:
            adj[u].append(v)
            adj[v].append(u)
        
        # Order tree vertices: start from a leaf, do BFS
        # Find a leaf
        leaves = [v for v in range(k) if len(adj[v]) == 1]
        if not leaves:
            start = 0
        else:
            start = rng.choice(leaves)
        
        # BFS ordering
        order = [start]
        visited = {start}
        frontier = [start]
        while frontier:
            next_frontier = []
            for u in frontier:
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        order.append(v)
                        next_frontier.append(v)
            frontier = next_frontier
        
        # Randomize the order a bit
        # Keep root first, shuffle the rest somewhat
        if len(order) > 2:
            rest = order[1:]
            rng.shuffle(rest)
            order = [order[0]] + rest
        
        # Now assign host vertices
        return _assign_tree_vertices(k, order, adj, tree, used_edges, assignments, n, backtrack, idx, rng, deadline)
    
    def _assign_tree_vertices(k, order, adj, tree, used_edges, assignments, n, backtrack, idx, rng, deadline):
        if time.time() > deadline:
            return False
        
        if not order:
            # All vertices assigned, check if all edges are valid (they should be by construction)
            # Commit
            assign_list = [0] * k
            for tv in range(k):
                assign_list[tv] = assignments[k][tv]
            assignments[k] = assign_list
            
            if backtrack(idx + 1):
                return True
            
            # Undo
            assignments[k] = None
            return False
        
        tv = order[0]
        remaining_order = order[1:]
        
        # Determine which host vertices are allowed
        # Must be distinct from already assigned tree vertices
        # And for each edge from tv to an already-assigned neighbor, the host edge must be unused
        
        assigned_host_verts = set()
        # Find already assigned neighbors in the tree
        valid_hosts = []
        
        for hv in range(n):
            # Check if hv is already used by another vertex of this tree
            if hv in assigned_host_verts:
                continue
            
            # Check edges to already-assigned tree neighbors
            conflict = False
            for nb in adj[tv]:
                if nb in assignments[k] and assignments[k][nb] is not None:
                    nh = assignments[k][nb]
                    edge = frozenset((hv, nh))
                    if edge in used_edges:
                        conflict = True
                        break
            
            if not conflict:
                valid_hosts.append(hv)
        
        if not valid_hosts:
            return False
        
        # Shuffle valid hosts for randomization
        rng.shuffle(valid_hosts)
        
        for hv in valid_hosts:
            # Assign
            assignments[k][tv] = hv
            assigned_host_verts.add(hv)
            
            # Mark edges to already-assigned neighbors as used
            new_edges = []
            for nb in adj[tv]:
                if nb in assignments[k] and assignments[k][nb] is not None:
                    nh = assignments[k][nb]
                    edge = frozenset((hv, nh))
                    used_edges.add(edge)
                    new_edges.append(edge)
            
            if _assign_tree_vertices(k, remaining_order, adj, tree, used_edges, assignments, n, backtrack, idx, rng, deadline):
                return True
            
            # Undo
            for edge in new_edges:
                used_edges.discard(edge)
            assigned_host_verts.discard(hv)
            assignments[k][tv] = None
        
        return False
    
    # Initialize assignments for tree 1 (single vertex)
    # Start backtracking
    for k in tree_order:
        assignments[k] = {} if k > 1 else None
    
    # We need a different structure for assignments during search
    # Let me restructure: use a dict for current partial assignment
    
    # Actually, let me redo this more carefully
    
    # Reset
    assignments = {}
    for k in tree_order:
        assignments[k] = {}
    
    # Redo backtracking with proper structure
    def bt(idx):
        if time.time() > deadline:
            return False
        
        if idx == len(tree_order):
            return True
        
        k = tree_order[idx]
        tree = trees[k - 1]
        
        if k == 1:
            for hv in range(n):
                assignments[k] = {0: hv}
                if bt(idx + 1):
                    return True
            assignments[k] = {}
            return False
        
        # Build tree adjacency
        adj = defaultdict(list)
        for u, v in tree:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS ordering from a random leaf
        leaves = [v for v in range(k) if len(adj[v]) == 1]
        start = rng.choice(leaves) if leaves else 0
        
        order = [start]
        visited = {start}
        frontier = [start]
        while frontier:
            next_frontier = []
            for u in frontier:
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        order.append(v)
                        next_frontier.append(v)
            frontier = next_frontier
        
        # Shuffle non-root
        if len(order) > 2:
            rest = order[1:]
            rng.shuffle(rest)
            order = [order[0]] + rest
        
        return _assign(k, 0, order, adj, assignments, used_edges, bt, idx, rng, deadline)
    
    def _assign(k, pos, order, adj, assignments, used_edges, bt, idx, rng, deadline):
        if time.time() > deadline:
            return False
        
        if pos == len(order):
            # Tree fully assigned, commit
            assign_list = [assignments[k][tv] for tv in range(k)]
            # Store as list
            assignments[k] = assign_list
            
            if bt(idx + 1):
                return True
            
            # Undo
            assignments[k] = {}
            return False
        
        tv = order[pos]
        
        # Find valid host vertices
        assigned_hosts = set(assignments[k].values())
        valid_hosts = []
        
        for hv in range(n):
            if hv in assigned_hosts:
                continue
            
            conflict = False
            for nb in adj[tv]:
                if nb in assignments[k]:
                    nh = assignments[k][nb]
                    edge = frozenset((hv, nh))
                    if edge in used_edges:
                        conflict = True
                        break
            
            if not conflict:
                valid_hosts.append(hv)
        
        if not valid_hosts:
            return False
        
        rng.shuffle(valid_hosts)
        
        for hv in valid_hosts:
            assignments[k][tv] = hv
            
            new_edges = []
            for nb in adj[tv]:
                if nb in assignments[k]:
                    nh = assignments[k][nb]
                    edge = frozenset((hv, nh))
                    used_edges.add(edge)
                    new_edges.append(edge)
            
            if _assign(k, pos + 1, order, adj, assignments, used_edges, bt, idx, rng, deadline):
                return True
            
            for edge in new_edges:
                used_edges.discard(edge)
            del assignments[k][tv]
        
        return False
    
    # Run backtracking
    if bt(0):
        # Extract result
        result = [None] * n
        for k in range(1, n + 1):
            if k in assignments:
                if isinstance(assignments[k], list):
                    result[k - 1] = assignments[k]
                elif isinstance(assignments[k], dict):
                    # Should have been converted, but handle anyway
                    if len(assignments[k]) == k:
                        result[k - 1] = [assignments[k][i] for i in range(k)]
                    else:
                        result[k - 1] = None
            else:
                result[k - 1] = None
        
        # Verify
        if _verify(result, trees, n):
            return result
        else:
            return None
    
    return None


def _verify(result, trees, n):
    """Verify that the result is valid: distinct vertices, edge-disjoint."""
    used_edges = set()
    for k in range(1, n + 1):
        if result[k - 1] is None:
            continue
        
        mapping = result[k - 1]
        if not isinstance(mapping, list):
            return False
        if len(mapping) != k:
            return False
        if len(set(mapping)) != k:
            return False
        for v in mapping:
            if not isinstance(v, int) or v < 0 or v >= n:
                return False
        
        for u, w in trees[k - 1]:
            hu, hw = mapping[u], mapping[w]
            edge = frozenset((hu, hw))
            if edge in used_edges:
                return False
            used_edges.add(edge)
    
    return True


# Test
if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    
    trees = _generate_trees(20, seed=611)
    total_edges = sum(len(trees[k-1]) for k in range(1, 21) if result[k-1] is not None)
    valid = _verify(result, trees, 20)
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Total edges: {total_edges}")
    print(f"Valid: {valid}")
    print(f"Trees embedded: {sum(1 for r in result if r is not None)}")
