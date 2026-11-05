"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _has_q3(edges_set, n=16):
    """Check if graph with edges_set contains a Q3 subgraph."""
    # Q3 has 8 vertices, 12 edges. Each vertex has degree 3 in Q3.
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges_set:
        adj[u].add(v)
        adj[v].add(u)
    
    # Try all 8-vertex subsets
    for subset in combinations(range(n), 8):
        sset = set(subset)
        # Count edges within subset
        edge_count = 0
        for u in subset:
            for v in adj[u]:
                if v in sset and u < v:
                    edge_count += 1
        if edge_count < 12:
            continue
        if edge_count > 12:
            continue
        # Check if it's exactly Q3: each vertex degree 3 within subset
        ok = True
        for u in subset:
            deg_in = sum(1 for v in adj[u] if v in sset)
            if deg_in != 3:
                ok = False
                break
        if ok:
            # Verify it's connected and isomorphic to Q3
            # Q3 is bipartite, 3-regular, 8 vertices, 12 edges
            # Check bipartite
            color = {}
            queue = [subset[0]]
            color[subset[0]] = 0
            bipartite = True
            while queue:
                u = queue.pop()
                for v in adj[u]:
                    if v not in sset:
                        continue
                    if v not in color:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        bipartite = False
                        break
                if not bipartite:
                    break
            if bipartite:
                # Count vertices in each partition
                c0 = sum(1 for u in subset if color.get(u) == 0)
                c1 = sum(1 for u in subset if color.get(u) == 1)
                if c0 == 4 and c1 == 4:
                    return True
    return False

def _greedy_construct(seed=42):
    """Greedy: add edges in random order, skip if creates Q3."""
    random.seed(seed)
    all_edges = [(u, v) for u in range(16) for v in range(u+1, 16)]
    random.shuffle(all_edges)
    
    edges_set = set()
    for u, v in all_edges:
        edges_set.add((u, v))
        if _has_q3(edges_set):
            edges_set.discard((u, v))
    
    return [[u, v] for u, v in edges_set]

def _local_search(edges_list, time_limit=100):
    """Try to add more edges by swapping or adding."""
    edges_set = set((min(e[0],e[1]), max(e[0],e[1])) for e in edges_list)
    all_edges = [(u, v) for u in range(16) for v in range(u+1, 16)]
    
    start = time.time()
    best = edges_set
    best_size = len(best)
    
    # Try adding each missing edge
    missing = [e for e in all_edges if e not in best]
    random.shuffle(missing)
    
    for u, v in missing:
        if time.time() - start > time_limit:
            break
        trial = best | {(u, v)}
        if not _has_q3(trial):
            best = trial
            best_size += 1
            # Re-shuffle missing since graph changed
            missing = [e for e in all_edges if e not in best]
            random.shuffle(missing)
    
    return [[u, v] for u, v in best]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_edges = _greedy_construct(42)
    best_count = len(best_edges)
    
    # Try multiple random seeds for greedy
    for seed in range(1, 50):
        if time.time() > deadline:
            break
        edges = _greedy_construct(seed)
        if len(edges) > best_count:
            best_edges = edges
            best_count = len(edges)
    
    # Local search on best
    if time.time() < deadline - 5:
        improved = _local_search(best_edges, time_limit=deadline - time.time() - 5)
        if len(improved) > best_count:
            best_edges = improved
            best_count = len(improved)
    
    # More local search iterations
    remaining = deadline - time.time()
    if remaining > 10:
        for _ in range(3):
            if time.time() > deadline:
                break
            improved = _local_search(best_edges, time_limit=min(remaining - 5, 30))
            if len(improved) > best_count:
                best_edges = improved
                best_count = len(improved)
            remaining = deadline - time.time()
            if remaining < 10:
                break
    
    return best_edges
# EVOLVE-BLOCK-END
