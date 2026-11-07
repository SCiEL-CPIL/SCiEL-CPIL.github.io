"""Packing trees T1..T20 into K20 - improved backtracking with polish."""
import time
import math
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
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
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 145
    trees = instance_trees(20, 611)
    n = 20
    num_trees = 20

    # Precompute tree edge lists and adjacency
    tree_edges = []
    tree_adj = []
    for t_idx in range(num_trees):
        k = t_idx + 1
        if k <= 2:
            tree_edges.append([] if k == 1 else trees[t_idx])
            tree_adj.append({})
            continue
        edges = trees[t_idx]
        adj = {v: [] for v in range(k)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        tree_edges.append(edges)
        tree_adj.append(adj)

    # Build all edges of K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))

    # Process trees from largest to smallest
    tree_order = list(range(num_trees))
    tree_order.sort(key=lambda idx: -(idx + 1))

    def try_embed(t_idx, avail, rng, deadline):
        """Try to embed tree t_idx into available edges using backtracking."""
        k = t_idx + 1
        if k == 1:
            return [rng.randint(0, n - 1)]
        if k == 2:
            for u, v in avail:
                return [u, v]
            return None

        edges = tree_edges[t_idx]
        adj = tree_adj[t_idx]
        
        # Order vertices by degree descending, then by index
        ordered_verts = sorted(range(k), key=lambda v: (-len(adj[v]), v))
        
        host_map = {}
        used_hosts = set()

        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == k:
                # Verify all tree edges are in available
                for u, v in edges:
                    hu, hv = host_map[u], host_map[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge not in avail:
                        return False
                return True

            tv = ordered_verts[pos]
            candidates = []

            for h in range(n):
                if h in used_hosts:
                    continue
                ok = True
                for nb in adj[tv]:
                    if nb in host_map:
                        hn = host_map[nb]
                        edge = (min(h, hn), max(h, hn))
                        if edge not in avail:
                            ok = False
                            break
                if ok:
                    # Score: prefer hosts with more available connections
                    remaining = sum(1 for j in range(n) if j != h and (min(h, j), max(h, j)) in avail)
                    candidates.append((remaining, h))

            if not candidates:
                return False

            # Sort by remaining connections (ascending - most constrained first)
            candidates.sort(key=lambda x: x[0])
            
            # Randomize among top candidates for diversity
            if len(candidates) > 1:
                top_n = min(len(candidates), 8)
                top = candidates[:top_n]
                rng.shuffle(top)
                rest = candidates[top_n:]
                candidates = top + rest

            for _, h in candidates:
                host_map[tv] = h
                used_hosts.add(h)
                if backtrack(pos + 1):
                    return True
                del host_map[tv]
                used_hosts.remove(h)

            return False

        if backtrack(0):
            return [host_map[i] for i in range(k)]
        return None

    def polish(res, avail, deadline):
        """Try to improve a near-complete solution by local moves."""
        if len(avail) == 0:
            return res, avail
        
        # Count how many edges are still unused
        unused = len(avail)
        if unused == 0:
            return res, avail
        
        # Try to fix by re-embedding some trees
        # Focus on trees that might be using suboptimal edges
        for _ in range(20):  # limited polish attempts
            if time.time() > deadline:
                break
            
            # Find a tree that could potentially use more edges
            # Try re-embedding the largest incomplete tree
            for t_idx in tree_order:
                k = t_idx + 1
                if k <= 1:
                    continue
                
                # Temporarily remove this tree's edges
                temp_avail = set(avail)
                for u, v in tree_edges[t_idx]:
                    if res[t_idx] is not None:
                        hu, hv = res[t_idx][u], res[t_idx][v]
                        edge = (min(hu, hv), max(hu, hv))
                        temp_avail.discard(edge)
                
                # Try to re-embed
                rng = Random(hash((t_idx, len(avail))) % 10000)
                new_embed = try_embed(t_idx, temp_avail, rng, deadline)
                
                if new_embed is not None:
                    # Check if this uses more edges
                    old_edges = set()
                    for u, v in tree_edges[t_idx]:
                        hu, hv = res[t_idx][u], res[t_idx][v]
                        edge = (min(hu, hv), max(hu, hv))
                        old_edges.add(edge)
                    
                    new_edges = set()
                    for u, v in tree_edges[t_idx]:
                        hu, hv = new_embed[u], new_embed[v]
                        edge = (min(hu, hv), max(hu, hv))
                        new_edges.add(edge)
                    
                    # Accept if it uses edges that were previously unused
                    if len(new_edges & avail) > len(old_edges & avail):
                        # Update
                        for u, v in tree_edges[t_idx]:
                            hu, hv = res[t_idx][u], res[t_idx][v]
                            edge = (min(hu, hv), max(hu, hv))
                            avail.discard(edge)
                        
                        res[t_idx] = new_embed
                        for u, v in tree_edges[t_idx]:
                            hu, hv = new_embed[u], new_embed[v]
                            edge = (min(hu, hv), max(hu, hv))
                            avail.discard(edge)
                        break
            
            if len(avail) == 0:
                break
        
        return res, avail

    best_result = None
    best_edges_used = -1

    # Phase 1: Backtracking with multiple restarts
    for attempt in range(1000):
        if time.time() > deadline:
            break

        rng = Random(attempt * 31 + 7)
        avail = set(all_edges)
        res = [None] * num_trees
        success = True

        for t_idx in tree_order:
            if time.time() > deadline:
                success = False
                break
            k = t_idx + 1

            if k == 1:
                res[t_idx] = [rng.randint(0, n - 1)]
                continue

            embedded = None
            # Try multiple times with different randomizations
            for trial in range(5):
                if time.time() > deadline:
                    success = False
                    break
                embedded = try_embed(t_idx, avail, rng, deadline)
                if embedded is not None:
                    break

            if not success:
                break
            if embedded is None:
                success = False
                break

            res[t_idx] = embedded
            if k >= 2:
                for u, v in tree_edges[t_idx]:
                    hu, hv = embedded[u], embedded[v]
                    edge = (min(hu, hv), max(hu, hv))
                    avail.discard(edge)

        if success:
            edges_used = len(all_edges) - len(avail)
            if edges_used > best_edges_used:
                best_edges_used = edges_used
                best_result = [list(r) if r is not None else None for r in res]
            
            # Try to polish if not perfect
            if edges_used < 190:
                polished_res, polished_avail = polish(res, avail, deadline)
                polished_edges = len(all_edges) - len(polished_avail)
                if polished_edges > best_edges_used:
                    best_edges_used = polished_edges
                    best_result = [list(r) if r is not None else None for r in polished_res]
            
            if edges_used == 190:
                break

    if best_result is None:
        # Fallback: return a trivial result
        return [None] * 19 + [list(range(20))]

    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])
    return final
