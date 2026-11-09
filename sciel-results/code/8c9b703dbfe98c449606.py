import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    # Known best is 17. We aim for 17.
    # If 17 is too hard, fall back to 16, 15...
    
    def has_c6_in_color(edges, n):
        """
        Check if the set of edges (list of (u,v)) contains a C6.
        Returns list of C6s found (as sets of edges or vertices) to help local search.
        For efficiency in local search, we just need to know *if* there is a C6 
        and ideally *which* edges are part of it to guide the flip.
        
        However, for a simple local search, we can just count total C6s.
        To flip, we pick a random edge from the graph and try changing its color.
        We need a fast way to evaluate the change.
        
        Alternative: 
        Maintain adjacency lists for each color.
        Count C6s by iterating all 6-tuples? No, too slow.
        
        Better Local Search Strategy:
        1. Maintain 4 graphs G0, G1, G2, G3 (adjacency lists).
        2. Compute initial C6 count for each color.
        3. To evaluate flipping edge (u,v) from color c to c':
           - Remove (u,v) from Gc, add to Gc'.
           - Calculate change in C6 count for Gc and Gc'.
           - If total C6s decreases (or stays same with some probability), accept.
           - Else, revert.
           
        Calculating C6 count change:
        A C6 involving (u,v) in color c must be of the form u-v-w-x-y-z-u? 
        No, (u,v) is an edge. The C6 is u-v-a-b-c-d-u.
        So we need to check if there is a path of length 5 from u to v in Gc \ {(u,v)}.
        
        This is expensive to recompute from scratch for every flip.
        
        Simpler Heuristic:
        Randomly pick an edge, change its color to a random other color.
        Recalculate total C6 count from scratch? 
        For n=17, edges=136. 
        Checking C6 in a graph of ~34 edges (average) is fast.
        Checking all 4 colors: 4 * O(V^6) is too slow? 
        Actually, checking if a graph contains a C6 can be done by:
        For each edge (u,v), check if there is a path of length 5 between u and v in G \ {(u,v)}.
        With n=17, max path length 5 search is small.
        
        Let's estimate:
        136 edges. For each edge, BFS depth 5.
        BFS depth 5 on sparse graph (deg ~ 2-4) is very fast.
        136 * 4 colors * (BFS cost).
        If we do this from scratch every iteration, and we do 1000 iterations, it might be too slow.
        
        Optimization: 
        Only recalculate C6 count for the colors affected by the flip.
        When flipping (u,v) from c to c':
        1. Remove (u,v) from Gc.
        2. Add (u,v) to Gc'.
        3. Check if Gc still has C6s? We can cache the C6s or just recompute.
        4. Check if Gc' gained C6s?
        
        Let's try a simpler "Min-Conflicts" approach:
        1. Random init.
        2. Count total C6s.
        3. If > 0, pick a random edge (u,v) that is part of at least one C6.
           To find such an edge, we can just pick any random edge and check if it's in a C6?
           Or maintain a list of "bad" edges.
        4. Change its color to the one that minimizes total C6s.
        
        Implementation of C6 count:
        def count_c6s(graph, n):
            count = 0
            edges = list(graph.edges)
            for u, v in edges:
                # Check if path of length 5 exists between u and v in graph without (u,v)
                # Temporarily remove (u,v)
                graph.remove(u, v)
                # BFS from u to v with depth 5
                if has_path_len5(graph, u, v, n):
                    count += 1
                graph.add(u, v)
            return count
        
        This is O(E * BFS). E ~ 34 per color. BFS ~ 5 * deg^5? No, BFS is O(V+E).
        So ~ 34 * 17 ~ 500 ops per color.
        4 colors ~ 2000 ops.
        10,000 iterations ~ 20,000,000 ops. Python can do ~10M-50M simple ops/sec.
        This should fit in 120s.
        
        Let's implement this.
        """
        pass

    def solve_local_search(n, time_budget):
        start = time.time()
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Adjacency lists for 4 colors
        adj = [ [set() for _ in range(n)] for _ in range(4) ]
        colors = [0] * len(edges) # 0,1,2,3
        
        # Random init
        for idx, (u, v) in enumerate(edges):
            c = random.randint(0, 3)
            colors[idx] = c
            adj[c][u].add(v)
            adj[c][v].add(u)
            
        def has_path_len5(u, v, c, exclude_edge=None):
            # BFS from u to v in adj[c], depth 5
            # If exclude_edge is (u,v), we ignore it.
            # Actually, we remove it before calling, so no need to exclude.
            visited = [[False] * n for _ in range(6)]
            queue = [(u, 0)]
            visited[0][u] = True
            
            while queue:
                node, depth = queue.pop(0)
                if depth == 5:
                    if node == v:
                        return True
                    continue
                for neighbor in adj[c][node]:
                    if not visited[depth+1][neighbor]:
                        visited[depth+1][neighbor] = True
                        queue.append((neighbor, depth+1))
            return False

        def count_c6s(c):
            count = 0
            # Iterate over edges in color c
            # We need a list of edges for color c. 
            # Maintaining a list is easier for iteration.
            # Let's maintain edge_lists[c]
            return count # Placeholder

        # To make it efficient, we maintain edge_lists
        edge_lists = [[] for _ in range(4)]
        
        # Re-init with edge_lists
        adj = [ [set() for _ in range(n)] for _ in range(4) ]
        colors = [0] * len(edges)
        for idx, (u, v) in enumerate(edges):
            c = random.randint(0, 3)
            colors[idx] = c
            adj[c][u].add(v)
            adj[c][v].add(u)
            edge_lists[c].append((u, v))
            
        def count_c6s_for_color(c):
            count = 0
            for u, v in edge_lists[c]:
                # Remove edge
                adj[c][u].remove(v)
                adj[c][v].remove(u)
                
                # BFS
                visited = [[False] * n for _ in range(6)]
                queue = [(u, 0)]
                visited[0][u] = True
                found = False
                while queue:
                    node, depth = queue.pop(0)
                    if depth == 5:
                        if node == v:
                            found = True
                            break
                        continue
                    for neighbor in adj[c][node]:
                        if not visited[depth+1][neighbor]:
                            visited[depth+1][neighbor] = True
                            queue.append((neighbor, depth+1))
                
                # Restore edge
                adj[c][u].add(v)
                adj[c][v].add(u)
                
                if found:
                    count += 1
            return count

        # Initial count
        total_c6s = sum(count_c6s_for_color(c) for c in range(4))
        
        while time.time() - start < time_budget:
            if total_c6s == 0:
                return {'n': n, 'colors': colors}
            
            # Pick a random edge that is part of a C6?
            # To be simple, pick a random edge from the whole graph.
            # If it's not in a C6, flipping it might not help or might hurt.
            # Better: Pick a random edge from a color that has C6s.
            
            # Find a color with C6s
            c_counts = [count_c6s_for_color(c) for c in range(4)]
            # If we just recalculated, we have c_counts.
            
            # Pick a random color with count > 0
            bad_colors = [c for c in range(4) if c_counts[c] > 0]
            if not bad_colors:
                break # Should not happen if total_c6s > 0
            
            c_bad = random.choice(bad_colors)
            
            # Pick a random edge from that color
            if not edge_lists[c_bad]:
                continue
            idx_in_list = random.randint(0, len(edge_lists[c_bad]) - 1)
            u, v = edge_lists[c_bad][idx_in_list]
            
            # Find global index of this edge
            # We can precompute a map (u,v) -> idx
            # Or just search. n=17, edges=136. Search is fast.
            # Let's precompute edge_map
            # edge_map = {(u,v): idx for idx, (u,v) in enumerate(edges)}
            
            # Try all other colors
            best_c = -1
            best_total = total_c6s + 1 # Threshold
            
            for c_new in range(4):
                if c_new == c_bad:
                    continue
                
                # Move (u,v) from c_bad to c_new
                # Remove from c_bad
                adj[c_bad][u].remove(v)
                adj[c_bad][v].remove(u)
                edge_lists[c_bad].remove((u, v))
                
                # Add to c_new
                adj[c_new][u].add(v)
                adj[c_new][v].add(u)
                edge_lists[c_new].append((u, v))
                
                # Recalculate counts for c_bad and c_new
                # Note: Other colors are unaffected.
                new_c_bad_count = count_c6s_for_color(c_bad)
                new_c_new_count = count_c6s_for_color(c_new)
                
                # Old counts were c_counts[c_bad] and c_counts[c_new]
                # Total change = (new_c_bad - old_c_bad) + (new_c_new - old_c_new)
                delta = (new_c_bad_count - c_counts[c_bad]) + (new_c_new_count - c_counts[c_new])
                new_total = total_c6s + delta
                
                if new_total < best_total:
                    best_total = new_total
                    best_c = c_new
                    # Save state to revert if not best?
                    # We are in a loop, so we need to revert after each trial.
                    
                # Revert
                adj[c_new][u].remove(v)
                adj[c_new][v].remove(u)
                edge_lists[c_new].remove((u, v))
                
                adj[c_bad][u].add(v)
                adj[c_bad][v].add(u)
                edge_lists[c_bad].append((u, v))
                
            # Apply best move
            if best_c != -1:
                # Move (u,v) from c_bad to best_c
                adj[c_bad][u].remove(v)
                adj[c_bad][v].remove(u)
                edge_lists[c_bad].remove((u, v))
                
                adj[best_c][u].add(v)
                adj[best_c][v].add(u)
                edge_lists[best_c].append((u, v))
                
                # Update counts
                c_counts[c_bad] = count_c6s_for_color(c_bad)
                c_counts[best_c] = count_c6s_for_color(best_c)
                total_c6s = sum(c_counts)
                
                # Update global colors array
                # Find idx of (u,v)
                # We can store idx in the edge list? 
                # Let's just update colors by scanning or using a dict.
                # For n=17, scanning 136 edges is fast.
                for idx, (eu, ev) in enumerate(edges):
                    if (eu == u and ev == v) or (eu == v and ev == u):
                        colors[idx] = best_c
                        break

        return None

    # Try n=17
    res = solve_local_search(17, 100.0)
    if res:
        return res
        
    # Try n=16
    res = solve_local_search(16, 10.0)
    if res:
        return res
        
    # Try n=15
    res = solve_local_search(15, 5.0)
    if res:
        return res
        
    # Fallback n=10
    res = solve_local_search(10, 1.0)
    if res:
        return res
        
    # Hardcoded n=9
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
