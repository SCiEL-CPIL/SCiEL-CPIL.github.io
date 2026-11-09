"""Packing trees T1..T20 into K20 using a model-and-optimize approach."""
import sys
import time
import random
from random import Random

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
    """
    Embed trees T1..T20 into K20 with edge-disjoint embeddings.
    Returns a list of 20 entries; entry k-1 is None or a list of k distinct ints in 0..19.
    Position i is the image of vertex i of T_k.
    """
    deadline = time.time() + 145  # safety margin before 150s limit
    trees = instance_trees(20, 611)
    max_possible = sum(len(trees[k-1]) for k in range(1, 21))
    n = 20
    total_vertices = 20

    def solve_one_restart(rng, order, max_attempts_per_tree=300):
        used_edges = set()
        result = [None] * n
        
        for k in order:
            edges_k = trees[k-1]
            if not edges_k:
                result[k-1] = [0]
                continue
            if k == 2:
                found = False
                for _ in range(5000):
                    a = rng.randint(0, 19)
                    b = rng.randint(0, 19)
                    if a != b:
                        key = (min(a,b), max(a,b))
                        if key not in used_edges:
                            result[k-1] = [a, b]
                            used_edges.add(key)
                            found = True
                            break
                if not found:
                    for a in range(20):
                        for b in range(a+1, 20):
                            key = (a, b)
                            if key not in used_edges:
                                result[k-1] = [a, b]
                                used_edges.add(key)
                                found = True
                                break
                        if found:
                            break
                if not found:
                    return None
                continue
            
            # Build adjacency
            adj = [[] for _ in range(k)]
            for (u, v) in edges_k:
                adj[u].append(v)
                adj[v].append(u)
            
            # Compute degrees for heuristic ordering
            deg = [0] * k
            for (u, v) in edges_k:
                deg[u] += 1
                deg[v] += 1
            
            # Find a good root: high degree, or leaf
            # Prefer high degree for better branching factor
            root = max(range(k), key=lambda i: deg[i])
            
            # BFS order starting from root
            visited = [False] * k
            order_list = [root]
            visited[root] = True
            frontier = [root]
            while frontier:
                next_frontier = []
                for node in frontier:
                    for nb in adj[node]:
                        if not visited[nb]:
                            visited[nb] = True
                            order_list.append(nb)
                            next_frontier.append(nb)
                frontier = next_frontier
            
            # If not all visited (shouldn't happen for a tree), add remaining
            for v in range(k):
                if not visited[v]:
                    order_list.append(v)
            
            best_attempt = None
            
            # Try BFS-based embedding with multiple attempts
            for attempt in range(max_attempts_per_tree):
                if time.time() > deadline:
                    return None
                mapping = [None] * k
                mapping[root] = rng.randint(0, 19)
                
                visited_embed = [False] * k
                visited_embed[root] = True
                queue = [root]
                valid_tree = True
                used_hosts = {mapping[root]}
                
                while queue:
                    node = queue.pop(0)
                    for neighbor in adj[node]:
                        if not visited_embed[neighbor]:
                            visited_embed[neighbor] = True
                            candidates = []
                            for h in range(20):
                                if h in used_hosts:
                                    continue
                                edge_key = (min(mapping[node], h), max(mapping[node], h))
                                if edge_key not in used_edges:
                                    candidates.append(h)
                            if candidates:
                                # Heuristic: prefer hosts with fewer used edges incident
                                # to reduce future conflicts
                                def host_score(h):
                                    # Count how many used edges are incident to h
                                    cnt = 0
                                    for (a, b) in used_edges:
                                        if a == h or b == h:
                                            cnt += 1
                                    return cnt
                                candidates.sort(key=host_score)
                                # Take from the top half to balance diversity
                                half = max(1, len(candidates) // 2)
                                chosen = rng.choice(candidates[:half])
                                mapping[neighbor] = chosen
                                used_hosts.add(mapping[neighbor])
                                queue.append(neighbor)
                            else:
                                valid_tree = False
                                break
                    if not valid_tree:
                        break
                
                if valid_tree and all(mapping[i] is not None for i in range(k)):
                    ok = True
                    for (u, v) in edges_k:
                        if (min(mapping[u], mapping[v]), max(mapping[u], mapping[v])) in used_edges:
                            ok = False
                            break
                    if ok:
                        best_attempt = mapping
                        break
            
            if best_attempt is None:
                return None
            
            result[k-1] = list(best_attempt)
            for (u, v) in edges_k:
                used_edges.add((min(best_attempt[u], best_attempt[v]), max(best_attempt[u], best_attempt[v])))
        
        return result
    
    # Generate different orderings to try
    orderings = []
    # Largest first
    orderings.append(list(range(20, 0, -1)))
    # Smallest first
    orderings.append(list(range(1, 21)))
    # Alternating: large, small, second large, second small, etc.
    alt = []
    left, right = 0, 19
    while left <= right:
        if left == right:
            alt.append(left + 1)
            break
        alt.append(left + 1)
        alt.append(right + 1)
        left += 1
        right -= 1
    orderings.append(alt)
    
    # Multiple restarts
    best_solution = None
    best_value = -1
    
    restart = 0
    while time.time() < deadline:
        order_idx = restart % len(orderings)
        order = orderings[order_idx]
        rng = Random(restart * 7919 + 13)
        
        if rng.random() < 0.3 and len(order) > 2:
            i = rng.randint(1, len(order) - 2)
            order[i], order[i+1] = order[i+1], order[i]
        
        # Use fewer attempts for later restarts if we already have a good solution
        attempts = 300 if restart < 5 else 100
        solution = solve_one_restart(rng, order, attempts)
        if solution is not None:
            val = sum(len(trees[k-1]) for k in range(1, 21) if solution[k-1] is not None)
            if val > best_value:
                best_value = val
                best_solution = [list(s) if s is not None else None for s in solution]
        
        if best_value >= max_possible:
            break
        restart += 1
    
    if best_solution is None:
        return [None] * 19 + [list(range(20))]
    
    return best_solution


if __name__ == "__main__":
    res = construct()
    trees = instance_trees(20, 611)
    used_edges = set()
    ok = True
    for k in range(1, 21):
        if res[k-1] is not None:
            r = res[k-1]
            if len(r) != k:
                print(f"T{k}: wrong length {len(r)}")
                ok = False
            if len(set(r)) != k:
                print(f"T{k}: not distinct")
                ok = False
            for v in r:
                if not isinstance(v, int) or v < 0 or v > 19:
                    print(f"T{k}: invalid vertex {v}")
                    ok = False
            for (u, v) in trees[k-1]:
                eu, ev = r[u], r[v]
                key = (min(eu, ev), max(eu, ev))
                if key in used_edges:
                    print(f"T{k}: edge conflict {key}")
                    ok = False
                used_edges.add(key)
    total = sum(len(trees[k-1]) for k in range(1, 21) if res[k-1] is not None)
    print(f"Valid: {ok}, Total edges: {total}")
