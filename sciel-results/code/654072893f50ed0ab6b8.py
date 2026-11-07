import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((e[0], e[1]))
        base_edges.add((e[1], e[0]))

    # Build adjacency as sets for fast triangle-free check
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if (u, v) in base_edges and u != v:
                adj[u].add(v)

    def is_triangle_free(u, v):
        """Check if adding edge (u,v) keeps graph triangle-free."""
        if u == v:
            return False
        return len(adj[u] & adj[v]) == 0

    def compute_diameter():
        """Compute diameter via BFS from each vertex."""
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            m = max(dist)
            if m > max_dist:
                max_dist = m
        return max_dist

    def compute_all_distances():
        """Return list of distance lists (one per source)."""
        all_dists = []
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            all_dists.append(dist)
        return all_dists

    def sum_of_all_distances():
        """Compute sum of all pairwise distances."""
        total = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            total += sum(d for d in dist if d != -1)
        return total

    # Generate all candidate non-base edges
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges and (v, u) not in base_edges:
                candidates.append((u, v))

    added = []
    current_diam = compute_diameter()

    # Phase 1: Greedy addition with both diameter and sum-distance metrics
    while current_diam > 4:
        if time.time() > deadline:
            break

        best_edge = None
        best_diam = current_diam
        best_sum_dist = float('inf')
        
        for edge in candidates:
            if time.time() > deadline:
                break
            u, v = edge
            if not is_triangle_free(u, v):
                continue
            
            adj[u].add(v)
            adj[v].add(u)
            d = compute_diameter()
            sum_dist = sum_of_all_distances()
            adj[u].remove(v)
            adj[v].remove(u)
            
            if d < best_diam or (d == best_diam and sum_dist < best_sum_dist):
                best_diam = d
                best_sum_dist = sum_dist
                best_edge = edge
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        added.append([u, v])
        candidates.remove(best_edge)
        current_diam = best_diam

    # Phase 2: Local search to minimize edge count
    # Try to remove each added edge and see if diameter stays <= 4
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(added)):
            if time.time() > deadline:
                break
            u1, v1 = added[i]
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            d = compute_diameter()
            if d <= 4:
                # We can remove this edge!
                added.pop(i)
                candidates.append((u1, v1))
                current_diam = d
                improved = True
                break  # Restart the loop since indices changed
            else:
                # Restore
                adj[u1].add(v1)
                adj[v1].add(u1)

    # Phase 3: Try to swap edges to find even smaller solutions
    # For each pair of added edges, try removing both and adding a different edge
    if time.time() < deadline:
        for _ in range(3):  # Limited iterations
            if time.time() > deadline:
                break
            found_swap = False
            for i in range(len(added)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(added)):
                    if time.time() > deadline:
                        break
                    u1, v1 = added[i]
                    u2, v2 = added[j]
                    # Remove both
                    adj[u1].remove(v1)
                    adj[v1].remove(u1)
                    adj[u2].remove(v2)
                    adj[v2].remove(u2)
                    
                    # Try adding each candidate edge
                    best_new = None
                    best_new_diam = float('inf')
                    for edge in candidates:
                        if time.time() > deadline:
                            break
                        u3, v3 = edge
                        if not is_triangle_free(u3, v3):
                            continue
                        adj[u3].add(v3)
                        adj[v3].add(u3)
                        d = compute_diameter()
                        if d <= 4:
                            best_new = edge
                            best_new_diam = d
                            break
                        adj[u3].remove(v3)
                        adj[v3].remove(u3)
                    
                    # Restore the two removed edges
                    adj[u1].add(v1)
                    adj[v1].add(u1)
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    
                    if best_new is not None:
                        # Perform the swap: remove two, add one
                        u1, v1 = added[i]
                        u2, v2 = added[j]
                        adj[u1].remove(v1)
                        adj[v1].remove(u1)
                        adj[u2].remove(v2)
                        adj[v2].remove(u2)
                        added.pop(j)
                        added.pop(i)
                        u3, v3 = best_new
                        adj[u3].add(v3)
                        adj[v3].add(u3)
                        added.append([u3, v3])
                        candidates.remove(best_new)
                        candidates.append((u1, v1))
                        candidates.append((u2, v2))
                        current_diam = compute_diameter()
                        found_swap = True
                        break
                if found_swap:
                    break
            if not found_swap:
                break

    # Final validation and type conversion
    result = []
    for edge in added:
        result.append([int(edge[0]), int(edge[1])])
    return result
