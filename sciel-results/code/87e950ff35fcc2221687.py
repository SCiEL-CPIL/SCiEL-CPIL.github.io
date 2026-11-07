import time
import random
import math
from itertools import combinations

N = 15
DEADLINE = 85.0
START_TIME = time.time()

def _is_valid(adj, n):
    """
    Check if no simple path s-u-v-t (s != t) has deg(s) == deg(t).
    For every edge (u, v), and every s in N(u)\{v}, t in N(v)\{u} with s != t, deg(s) != deg(t).
    """
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        for v in adj[u]:
            if u < v:  # Avoid double checking
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u:
                            continue
                        if s == t:
                            continue
                        if deg[s] == deg[t]:
                            return False
    return True

def _edge_count(adj, n):
    return sum(len(adj[i]) for i in range(n)) // 2

def _build_from_partition(partition):
    """
    Build graph as disjoint union of K_{a,b} components.
    partition: list of (a, b) tuples where a + b = component size, a != b.
    Returns (adj, edge_count) or (None, 0) if degrees collide.
    """
    # Check degree uniqueness
    deg_set = set()
    for a, b in partition:
        if a == 0 or b == 0:
            continue
        if a in deg_set or b in deg_set:
            return None, 0
        deg_set.add(a)
        deg_set.add(b)
    
    adj = [set() for _ in range(N)]
    vertex = 0
    total_edges = 0
    for a, b in partition:
        if a == 0 or b == 0:
            continue
        part_a = list(range(vertex, vertex + a))
        part_b = list(range(vertex + a, vertex + a + b))
        vertex += a + b
        for u in part_a:
            for v in part_b:
                adj[u].add(v)
                adj[v].add(u)
                total_edges += 1
    return adj, total_edges

def _try_local_search(adj, n, seed, max_iter=20000):
    """
    Try to add edges within the same bipartite part (same degree group).
    Only accept if validity is maintained.
    """
    rng = random.Random(seed)
    adj = [set(s) for s in adj]
    current_edges = _edge_count(adj, n)
    
    if not _is_valid(adj, n):
        return adj, current_edges
    
    best_adj = [set(s) for s in adj]
    best_edges = current_edges
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 15:
            break
        
        # Pick a random non-edge
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j or j in adj[i]:
            continue
        
        # Try adding
        adj[i].add(j)
        adj[j].add(i)
        
        if _is_valid(adj, n):
            new_edges = _edge_count(adj, n)
            if new_edges > best_edges:
                best_edges = new_edges
                best_adj = [set(s) for s in adj]
        else:
            # Revert
            adj[i].discard(j)
            adj[j].discard(i)
    
    return best_adj, best_edges

def _try_remove_and_add(adj, n, seed, max_iter=10000):
    """
    Try removing one edge and adding another to enable more edges overall.
    """
    rng = random.Random(seed)
    adj = [set(s) for s in adj]
    current_edges = _edge_count(adj, n)
    
    if not _is_valid(adj, n):
        return adj, current_edges
    
    best_adj = [set(s) for s in adj]
    best_edges = current_edges
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 15:
            break
        
        # Pick a random edge to remove
        all_edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    all_edges.append((i, j))
        if not all_edges:
            break
        
        rem_i, rem_j = rng.choice(all_edges)
        adj[rem_i].discard(rem_j)
        adj[rem_j].discard(rem_i)
        
        # Try adding a random non-edge
        add_i = rng.randint(0, n - 1)
        add_j = rng.randint(0, n - 1)
        if add_i != add_j and add_j not in adj[add_i]:
            adj[add_i].add(add_j)
            adj[add_j].add(add_i)
            
            if _is_valid(adj, n):
                new_edges = _edge_count(adj, n)
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = [set(s) for s in adj]
                else:
                    # Revert both
                    adj[add_i].discard(add_j)
                    adj[add_j].discard(add_i)
                    adj[rem_i].add(rem_j)
                    adj[rem_j].add(rem_i)
            else:
                # Revert both
                adj[add_i].discard(add_j)
                adj[add_j].discard(add_i)
                adj[rem_i].add(rem_j)
                adj[rem_j].add(rem_i)
        else:
            # Couldn't add, revert removal
            adj[rem_i].add(rem_j)
            adj[rem_j].add(rem_i)
    
    return best_adj, best_edges

def construct():
    start = time.time()
    
    # Systematic search over all partitions of 15 into pairs (a, b) with a != b
    # Each component is K_{a,b}
    best_adj = None
    best_count = 0
    
    # Generate all partitions of 15 into components of size >= 2
    # where each component is split as (a, b) with a != b, a >= 1, b >= 1
    
    def gen_partitions(remaining, current):
        if remaining == 0:
            yield current
            return
        # Try component sizes from 2 to remaining
        for size in range(2, remaining + 1):
            if remaining - size < 0:
                break
            # Split size into (a, b) with a != b, a >= 1, b >= 1
            for a in range(1, size):
                b = size - a
                if a == b:
                    continue
                if a > b:
                    continue  # Avoid duplicates (a, b) and (b, a)
                yield from gen_partitions(remaining - size, current + [(a, b)])
    
    all_partitions = list(gen_partitions(N, []))
    
    for partition in all_partitions:
        adj, count = _build_from_partition(partition)
        if adj is not None and count > best_count:
            if _is_valid(adj, N):
                best_count = count
                best_adj = adj
    
    # If no valid partition found, fallback to star
    if best_adj is None:
        best_adj = [set() for _ in range(N)]
        for v in range(1, N):
            best_adj[0].add(v)
            best_adj[v].add(0)
        best_count = 14
    
    # Local search: try to add edges within same part
    for seed in range(30):
        if time.time() - start > DEADLINE - 20:
            break
        result_adj, result_count = _try_local_search(best_adj, N, seed, max_iter=5000)
        if result_count > best_count and _is_valid(result_adj, N):
            best_count = result_count
            best_adj = result_adj
    
    # Remove-and-add search
    for seed in range(20):
        if time.time() - start > DEADLINE - 15:
            break
        result_adj, result_count = _try_remove_and_add(best_adj, N, 10000 + seed, max_iter=5000)
        if result_count > best_count and _is_valid(result_adj, N):
            best_count = result_count
            best_adj = result_adj
    
    # Final validation
    if not _is_valid(best_adj, N):
        best_adj = [set() for _ in range(N)]
        for v in range(1, N):
            best_adj[0].add(v)
            best_adj[v].add(0)
        best_count = 14
    
    edges = []
    for i in range(N):
        for j in best_adj[i]:
            if i < j:
                edges.append([int(i), int(j)])
    
    return edges
