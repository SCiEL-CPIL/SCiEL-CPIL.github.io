"""Strong edge coloring via iterative MIS extraction with B&B refinement."""
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def build_conflict(edges, adj):
        """Build conflict graph: edge i conflicts with edge j if they share a vertex or are joined by an edge."""
        m = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        conflict = [set() for _ in range(m)]
        
        for i, (u, v) in enumerate(edges):
            for w in adj[u]:
                for x in edges:
                    if (u, w) == x or (w, u) == x:
                        j = edge_to_idx[x]
                        if i != j:
                            conflict[i].add(j)
                            conflict[j].add(i)
            for w in adj[v]:
                for x in edges:
                    if (v, w) == x or (w, v) == x:
                        j = edge_to_idx[x]
                        if i != j:
                            conflict[i].add(j)
                            conflict[j].add(i)
        
        return conflict
    
    def greedy_mis(conflict, remaining):
        """Greedy maximum independent set on conflict graph restricted to remaining nodes."""
        if not remaining:
            return set()
        
        # Order by degree in remaining
        rem_list = sorted(remaining, key=lambda x: len(conflict[x] & remaining))
        mis = set()
        candidates = set(remaining)
        
        for node in rem_list:
            if node not in candidates:
                continue
            # Check if node can be added (no conflict with any node in mis)
            if not (conflict[node] & mis):
                mis.add(node)
                # Remove node and its neighbors from candidates
                candidates -= conflict[node]
                candidates.discard(node)
        
        return mis
    
    def bb_mis(conflict, candidates, best_size, best_set, timeout):
        """Branch and bound for maximum independent set."""
        if time.time() > timeout:
            return best_size, best_set
        if not candidates:
            return len(best_set), best_set
        if len(candidates) + len(best_set) <= best_size:
            return best_size, best_set
        
        # Pick node with max degree in candidates
        node = max(candidates, key=lambda x: len(conflict[x] & candidates))
        
        # Branch 1: include node
        new_candidates = candidates - conflict[node] - {node}
        new_set = best_set | {node}
        s1, set1 = bb_mis(conflict, new_candidates, best_size, new_set, timeout)
        if s1 > best_size:
            best_size, best_set = s1, set1
        
        # Branch 2: exclude node
        new_candidates2 = candidates - {node}
        s2, set2 = bb_mis(conflict, new_candidates2, best_size, best_set, timeout)
        if s2 > best_size:
            best_size, best_set = s2, set2
        
        return best_size, best_set
    
    def solve_instance(edges, adj):
        m = len(edges)
        if m == 0:
            return []
        
        conflict = build_conflict(edges, adj)
        remaining = set(range(m))
        colors = {}
        color_count = 0
        
        while remaining:
            if time.time() > deadline:
                # Fallback: greedy coloring for remaining
                for e_idx in sorted(remaining):
                    taken = set()
                    for other_idx, c in colors.items():
                        if e_idx in conflict[other_idx]:
                            taken.add(c)
                    colors[e_idx] = next(c for c in range(len(colors) + 1) if c not in taken)
                break
            
            # Get greedy MIS first
            greedy_set = greedy_mis(conflict, remaining)
            
            # Try B&B refinement if time permits
            if time.time() < deadline - 5:
                try:
                    best_size, best_set = bb_mis(
                        conflict, remaining, len(greedy_set), greedy_set, 
                        min(time.time() + 3, deadline - 1)
                    )
                    if best_size > len(greedy_set):
                        greedy_set = best_set
                except Exception:
                    pass
            
            if not greedy_set:
                greedy_set = {min(remaining)}
            
            # Assign color
            for idx in greedy_set:
                colors[idx] = color_count
            color_count += 1
            remaining -= greedy_set
        
        # Reindex colors to be consecutive starting from 0
        if colors:
            max_c = max(colors.values())
            if max_c + 1 != color_count:
                mapping = {}
                next_c = 0
                for c in range(max_c + 1):
                    if any(v == c for v in colors.values()):
                        mapping[c] = next_c
                        next_c += 1
                for k in colors:
                    colors[k] = mapping[colors[k]]
        
        return [colors[i] for i in range(m)]
    
    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        if time.time() > deadline:
            # Emergency fallback: simple greedy
            m = len(edges)
            color = {}
            for i, e in enumerate(edges):
                taken = set()
                for j, f in enumerate(edges):
                    if j >= i:
                        break
                    if j in color:
                        # Check conflict
                        u1, v1 = e
                        u2, v2 = f
                        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                            taken.add(color[j])
                        else:
                            # Check if joined by edge
                            if (u1, u2) in adj.get(u1, set()) or (u1, v2) in adj.get(u1, set()) or \
                               (v1, u2) in adj.get(v1, set()) or (v1, v2) in adj.get(v1, set()):
                                taken.add(color[j])
                color[i] = next(c for c in range(len(color) + 1) if c not in taken)
            results.append([color[i] for i in range(m)])
            continue
        
        result = solve_instance(edges, adj)
        results.append(result)
    
    return results
