import time
import itertools
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_max_bicliques(n, adj, edges_set, max_bicliques=200):
    """Find many large bicliques via greedy expansion from random seeds."""
    bicliques = []
    seen = set()
    
    # Try all pairs as seeds, expand greedily
    for u, v in edges:
        if u > v:
            continue
        # Start with {u} on left, {v} on right
        left = frozenset([u])
        right = frozenset([v])
        
        # Greedy expand: add vertices that connect to all on opposite side
        changed = True
        while changed:
            changed = False
            # Try adding to left
            for x in range(n):
                if x in left:
                    continue
                if all(x in adj[y] for y in right):
                    new_left = frozenset(left | {x})
                    if (new_left, right) not in seen:
                        left = new_left
                        changed = True
                        break
            # Try adding to right
            for x in range(n):
                if x in right:
                    continue
                if all(x in adj[y] for y in left):
                    new_right = frozenset(right | {x})
                    if (left, new_right) not in seen:
                        right = new_right
                        changed = True
                        break
        
        key = (left, right)
        if key not in seen:
            seen.add(key)
            bicliques.append((left, right))
    
    # Sort by size descending
    bicliques.sort(key=lambda x: len(x[0]) * len(x[1]), reverse=True)
    return bicliques[:max_bicliques]

def _greedy_partition(n, adj, edges, bicliques):
    """Greedy: repeatedly pick biclique covering most uncovered edges."""
    edges_set = set()
    for u, v in edges:
        edges_set.add(frozenset([u, v]))
    
    remaining = set(edges_set)
    partition = []
    
    while remaining:
        best = None
        best_count = -1
        for left, right in bicliques:
            count = 0
            for u in left:
                for v in right:
                    if u == v:
                        continue
                    e = frozenset([u, v])
                    if e in remaining:
                        count += 1
            if count > best_count:
                best_count = count
                best = (left, right)
        
        if best is None or best_count == 0:
            # Fallback: make a biclique from remaining edges
            # Just take one edge
            e = next(iter(remaining))
            u, v = list(e)
            partition.append(([u], [v]))
            remaining.discard(e)
        else:
            left, right = best
            # Only include edges that are in remaining
            partition.append((list(left), list(right)))
            for u in left:
                for v in right:
                    if u != v:
                        e = frozenset([u, v])
                        remaining.discard(e)
    
    return partition, remaining

def _validate_partition(n, edges, partition):
    """Check partition is valid: all edges covered exactly once, all cross pairs are edges."""
    adj = _build_adj(n, edges)
    edges_set = set()
    for u, v in edges:
        edges_set.add(frozenset([u, v]))
    
    covered = set()
    for left, right in partition:
        for u in left:
            for v in right:
                if u == v:
                    continue
                e = frozenset([u, v])
                if e not in edges_set:
                    return False  # Not an edge
                if e in covered:
                    return False  # Duplicate
                covered.add(e)
    
    if covered != edges_set:
        return False  # Missing edges
    
    # Check sides are nonempty and disjoint
    for left, right in partition:
        if not left or not right:
            return False
        if set(left) & set(right):
            return False
    
    return True

def _local_search(n, adj, edges, partition, deadline):
    """Try to reduce number of bicliques via merging/splitting."""
    best = partition
    best_count = len(partition)
    
    while time.time() < deadline:
        improved = False
        # Try merging two bicliques
        for i in range(len(best)):
            for j in range(i + 1, len(best)):
                if time.time() >= deadline:
                    break
                li, ri = best[i]
                lj, rj = best[j]
                # Try merging: left = li ∪ lj, right = ri ∪ rj
                # Check if all cross pairs are edges
                merged_left = set(li) | set(lj)
                merged_right = set(ri) | set(rj)
                if merged_left & merged_right:
                    continue
                valid = True
                for u in merged_left:
                    for v in merged_right:
                        if u == v:
                            continue
                        if v not in adj[u]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    new_partition = list(best)
                    del new_partition[j]
                    del new_partition[i]
                    new_partition.append((list(merged_left), list(merged_right)))
                    if len(new_partition) < best_count:
                        best = new_partition
                        best_count = len(best)
                        improved = True
                        break
            if improved or time.time() >= deadline:
                break
        if not improved:
            break
    
    return best

def construct():
    start = time.time()
    deadline = start + 50.0  # 50s safety margin within 60s limit
    
    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)
    
    # Step 1: Find candidate bicliques
    bicliques = _find_max_bicliques(n, adj, set(frozenset(e) for e in edges), max_bicliques=500)
    
    # Step 2: Greedy partition
    partition, remaining = _greedy_partition(n, adj, edges, bicliques)
    
    # Handle any remaining edges
    if remaining:
        for e in remaining:
            u, v = list(e)
            partition.append(([u], [v]))
    
    # Step 3: Local search to improve
    partition = _local_search(n, adj, edges, partition, deadline)
    
    # Final validation
    if not _validate_partition(n, edges, partition):
        # Fallback to seed solution
        partition = [[[u], [v for a, v in edges if a == u]] for u in range(n) if any(a == u for a, v in edges)]
    
    # Convert to proper types
    result = []
    for left, right in partition:
        result.append(([int(x) for x in left], [int(x) for x in right]))
    
    return result
