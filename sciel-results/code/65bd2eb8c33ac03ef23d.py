import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 16
    all_pairs = list(combinations(range(n), 2))
    
    # Baseline: Clebsch graph (40 edges)
    def hamming_distance(a, b):
        return bin(a ^ b).count('1')
    
    clebsch_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if hamming_distance(u, v) in (1, 4):
                clebsch_edges.append([u, v])
    
    best_edges = clebsch_edges[:]
    best_count = len(best_edges)
    
    # Fast 3-colorability check using backtracking with bitmasks
    def is_3_colorable(adj_mask, order):
        """adj_mask: list of int bitmasks, adj_mask[v] has bit u set if (u,v) is edge
           order: list of vertex indices in coloring order
           Returns True if 3-colorable"""
        colors = [0] * n  # 0=uncolored, 1/2/3 = colors
        
        def backtrack(i):
            if i == n:
                return True
            v = order[i]
            # Determine which colors are forbidden
            forbidden = 0
            mask = adj_mask[v]
            # Check neighbors that are already colored (earlier in order)
            for c in range(1, 4):
                # Check if any neighbor has color c
                # We need to check all neighbors, but only colored ones matter
                # Since we color in order, only neighbors with index < i in order matter
                # Actually, we need to check all neighbors for their color
                # Simpler: just iterate neighbors
                pass
            
            # More efficient: precompute for each vertex, the colors used by neighbors
            # Let's use a different approach: for vertex v, check each color
            for c in range(1, 4):
                ok = True
                m = adj_mask[v]
                # Check all neighbors
                # Iterate through bits in m
                mm = m
                while mm:
                    low = mm & (-mm)
                    u = low.bit_length() - 1
                    if colors[u] == c:
                        ok = False
                        break
                    mm ^= low
                if ok:
                    colors[v] = c
                    if backtrack(i + 1):
                        return True
                    colors[v] = 0
            return False
        
        return backtrack(0)
    
    # Even faster: use greedy coloring check first, then backtracking only if needed
    def is_3_colorable_fast(adj_mask, order):
        """Returns True if graph is 3-colorable"""
        colors = [0] * n
        
        # Greedy attempt first
        def greedy():
            for i in range(n):
                v = order[i]
                used = set()
                m = adj_mask[v]
                mm = m
                while mm:
                    low = mm & (-mm)
                    u = low.bit_length() - 1
                    if colors[u] != 0:
                        used.add(colors[u])
                    mm ^= low
                for c in range(1, 4):
                    if c not in used:
                        colors[v] = c
                        break
                else:
                    return False
            return True
        
        if greedy():
            return True
        
        # Backtracking
        colors = [0] * n
        
        def backtrack(i):
            if i == n:
                return True
            v = order[i]
            # Collect forbidden colors
            forbidden = 0
            m = adj_mask[v]
            mm = m
            while mm:
                low = mm & (-mm)
                u = low.bit_length() - 1
                if colors[u] != 0:
                    forbidden |= (1 << colors[u])
                mm ^= low
            
            for c in range(1, 4):
                if not (forbidden & (1 << c)):
                    colors[v] = c
                    if backtrack(i + 1):
                        return True
                    colors[v] = 0
            return False
        
        return backtrack(0)
    
    def build_adj_mask(edge_set):
        adj_mask = [0] * n
        for u, v in edge_set:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        return adj_mask
    
    def is_triangle_free(edge_set):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            neighbors = list(adj[u])
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in adj[neighbors[i]]:
                        return False
        return True
    
    def count_edges(edge_set):
        return len(edge_set)
    
    # Strategy: Start from Clebsch, try to add edges
    def try_add_edges(start_edges, max_time):
        edge_set = set(start_edges)
        adj_mask = build_adj_mask(edge_set)
        
        # Get degree order for coloring
        degrees = [bin(adj_mask[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        # Check if 4-chromatic
        if not is_3_colorable_fast(adj_mask, order):
            best_local = list(edge_set)
            best_local_count = len(edge_set)
        else:
            best_local = None
            best_local_count = 0
        
        # Try to add edges
        remaining = [(u, v) for u, v in all_pairs if (u, v) not in edge_set and (v, u) not in edge_set]
        random.shuffle(remaining)
        
        for u, v in remaining:
            if time.time() > max_time:
                break
            
            # Check if adding (u,v) creates a triangle
            common_neighbors = adj_mask[u] & adj_mask[v]
            if common_neighbors:
                continue  # Would create triangle
            
            # Try adding
            edge_set.add((u, v))
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
            # Update degrees
            degrees[u] += 1
            degrees[v] += 1
            order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
            
            # Check 3-colorability
            if not is_3_colorable_fast(adj_mask, order):
                best_local = list(edge_set)
                best_local_count = len(edge_set)
            else:
                # Revert
                edge_set.discard((u, v))
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                degrees[u] -= 1
                degrees[v] -= 1
        
        return best_local, best_local_count
    
    # Try multiple strategies
    # Strategy 1: Start from Clebsch and add edges
    if time.time() < deadline - 5:
        result, count = try_add_edges(clebsch_edges, deadline - 5)
        if result and count > best_count:
            best_edges = result
            best_count = count
    
    # Strategy 2: Random dense triangle-free graphs
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        
        # Randomly generate a triangle-free graph
        # Start empty, add random edges that don't create triangles
        edge_set = set()
        adj_mask = [0] * n
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if time.time() > deadline - 5:
                break
            common = adj_mask[u] & adj_mask[v]
            if not common:
                edge_set.add((u, v))
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        
        # Check if 4-chromatic
        degrees = [bin(adj_mask[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        if not is_3_colorable_fast(adj_mask, order):
            count = len(edge_set)
            if count > best_count:
                best_edges = list(edge_set)
                best_count = count
            
            # Try to add more edges
            remaining = [(u, v) for u, v in all_pairs if (u, v) not in edge_set]
            random.shuffle(remaining)
            
            for u, v in remaining:
                if time.time() > deadline - 5:
                    break
                common = adj_mask[u] & adj_mask[v]
                if common:
                    continue
                
                edge_set.add((u, v))
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
                
                if not is_3_colorable_fast(adj_mask, order):
                    if len(edge_set) > best_count:
                        best_edges = list(edge_set)
                        best_count = len(edge_set)
                else:
                    edge_set.discard((u, v))
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                    degrees[u] -= 1
                    degrees[v] -= 1
    
    # Convert to list of [u, v] pairs
    result = [[u, v] for u, v in best_edges]
    return result
