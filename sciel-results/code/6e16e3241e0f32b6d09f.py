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
    deadline = time.time() + 140  # safety margin before 150s limit
    trees = instance_trees(20, 611)
    
    # We'll try a greedy randomized approach with multiple restarts
    # Strategy: embed trees from largest to smallest (T20, T19, ..., T1)
    # For each tree, find a vertex mapping that avoids already-used edges.
    
    def solve_one_restart(rng):
        n = 20
        # Track used edges as a set of frozensets or sorted tuples
        used_edges = set()
        result = [None] * 20
        
        # Process trees in order from T20 down to T1 (larger first)
        # Actually, let's process in a smart order
        order = list(range(20, 0, -1))  # 20, 19, ..., 1
        # Shuffle slightly to try different orderings
        if rng.random() < 0.5:
            order = list(range(1, 21))  # 1, 2, ..., 20
        
        for k in order:
            edges_k = trees[k-1]
            if not edges_k:
                result[k-1] = [0]  # T1 is a single vertex
                continue
            if k == 2:
                # Find an unused edge
                found = False
                for _ in range(10000):
                    a = rng.randint(0, 19)
                    b = rng.randint(0, 19)
                    if a != b and (a, b) not in used_edges and (b, a) not in used_edges:
                        result[k-1] = [a, b]
                        used_edges.add((min(a,b), max(a,b)))
                        found = True
                        break
                if not found:
                    # Try systematic
                    for a in range(20):
                        for b in range(a+1, 20):
                            if (a, b) not in used_edges:
                                result[k-1] = [a, b]
                                used_edges.add((a, b))
                                found = True
                                break
                        if found:
                            break
                if not found:
                    return None
                continue
            
            # For tree with k vertices and k-1 edges
            # We need to find a mapping of vertices 0..k-1 to distinct hosts 0..19
            # such that no mapped edge conflicts with used_edges
            
            # Use randomized backtracking / local search
            # Try random permutations and check validity
            best_attempt = None
            num_tries = 50000
            for attempt in range(num_tries):
                mapping = rng.sample(range(20), k)
                valid = True
                for (u, v) in edges_k:
                    eu, ev = mapping[u], mapping[v]
                    if (min(eu, ev), max(eu, ev)) in used_edges:
                        valid = False
                        break
                if valid:
                    best_attempt = mapping
                    break
            
            # Early exit if we've used significant time
            if best_attempt is None and time.time() > deadline - 5:
                return None
            
            if best_attempt is None:
                # Try a smarter approach: build incrementally
                # Start with a random vertex for root, then extend
                # Use the tree structure to guide placement
                success = False
                # Precompute adjacency once (outside the retry loop)
                adj = [[] for _ in range(k)]
                for (u, v) in edges_k:
                    adj[u].append(v)
                    adj[v].append(u)
                
                for try_count in range(200):
                    # Random root
                    mapping = [None] * k
                    used_hosts = set()
                    mapping[0] = rng.randint(0, 19)
                    used_hosts.add(mapping[0])
                    
                    visited = [False] * k
                    visited[0] = True
                    queue = [0]
                    valid_tree = True
                    while queue:
                        node = queue.pop(0)
                        for neighbor in adj[node]:
                            if not visited[neighbor]:
                                visited[neighbor] = True
                                # Find an available host for neighbor
                                candidates = []
                                for h in range(20):
                                    if h in used_hosts:
                                        continue
                                    edge = (min(mapping[node], h), max(mapping[node], h))
                                    if edge not in used_edges:
                                        candidates.append(h)
                                if candidates:
                                    mapping[neighbor] = rng.choice(candidates)
                                    used_hosts.add(mapping[neighbor])
                                    queue.append(neighbor)
                                else:
                                    valid_tree = False
                                    break
                        if not valid_tree:
                            break
                    
                    if valid_tree and all(mapping[i] is not None for i in range(k)):
                        # Verify no internal conflicts
                        ok = True
                        for (u, v) in edges_k:
                            if (min(mapping[u], mapping[v]), max(mapping[u], mapping[v])) in used_edges:
                                ok = False
                                break
                        if ok:
                            best_attempt = mapping
                            success = True
                            break
                
                if not success:
                    return None
            
            if best_attempt is None:
                return None
            
            result[k-1] = list(best_attempt)
            for (u, v) in edges_k:
                used_edges.add((min(best_attempt[u], best_attempt[v]), max(best_attempt[u], best_attempt[v])))
        
        return result
    
    # Multiple restarts with different seeds
    best_solution = None
    best_value = -1
    num_restarts = 50
    start_time = time.time()
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        rng = Random(restart * 7919 + 13)
        solution = solve_one_restart(rng)
        if solution is not None:
            # Compute value
            val = 0
            for k in range(1, 21):
                if solution[k-1] is not None:
                    val += len(trees[k-1])
            if val > best_value:
                best_value = val
                best_solution = [list(s) if s is not None else None for s in solution]
        
        # Check if we achieved maximum possible
        max_possible = sum(len(trees[k-1]) for k in range(1, 21))
        if best_value >= max_possible:
            break
    
    if best_solution is None:
        # Fallback: just embed T20
        return [None] * 19 + [list(range(20))]
    
    return best_solution


if __name__ == "__main__":
    res = construct()
    # Verify
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
