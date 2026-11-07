"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import deque, defaultdict
from itertools import combinations

sys.setrecursionlimit(100000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    
    # Precompute shortest paths and predecessors for quick fallback
    INF = float('inf')
    dist = [[INF]*n for _ in range(n)]
    pred = [[-1]*n for _ in range(n)]
    for start in range(n):
        dist[start][start] = 0
        pred[start][start] = -1
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj_list[u]:
                if dist[start][v] == INF:
                    dist[start][v] = dist[start][u] + 1
                    pred[start][v] = u
                    q.append(v)
    
    def get_shortest_path(start, end):
        if dist[start][end] == INF:
            return None
        path = []
        cur = end
        while cur != -1:
            path.append(cur)
            cur = pred[start][cur]
        path.reverse()
        return path

    # Find all simple paths between u and v up to max_len
    # We cache these to avoid recomputation
    path_cache = {}
    def get_all_paths(u, v, max_len=5):
        key = (u, v, max_len)
        if key in path_cache:
            return path_cache[key]
        
        if u == v:
            path_cache[key] = [[u]]
            return path_cache[key]
            
        paths = []
        # DFS to find paths
        def dfs(curr, path, visited):
            if len(path) - 1 > max_len:
                return
            if curr == v:
                paths.append(list(path))
                return
            for nxt in adj_list[curr]:
                if nxt not in visited:
                    visited.add(nxt)
                    path.append(nxt)
                    dfs(nxt, path, visited)
                    path.pop()
                    visited.remove(nxt)
        
        dfs(u, [u], {u})
        # Sort by length, then lexicographically for determinism
        paths.sort(key=lambda p: (len(p), p))
        path_cache[key] = paths
        return paths

    # Bron-Kerbosch for Maximum Clique
    best_clique = []
    def bron_kerbosch(R, P_set, X):
        nonlocal best_clique
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        if P_set:
            pivot = max(P_set, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set
        for v in list(candidates):
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)
            
    bron_kerbosch(set(), set(range(n)), set())
    max_clique_size = len(best_clique)
    
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin for 60s limit
    
    best_k = 0
    best_B = []
    best_paths = {}
    
    # Function to check if a set B has a valid clique subdivision
    # Returns (True, paths_dict) or (False, None)
    def check_subdivision(B, time_limit=2.0):
        start_t = time.time()
        B_set = set(B)
        k = len(B)
        
        # Identify non-adjacent pairs
        non_adj_pairs = []
        for i in range(k):
            for j in range(i+1, k):
                u, v = B[i], B[j]
                if v not in adj[u]:
                    non_adj_pairs.append((u, v))
        
        if not non_adj_pairs:
            return True, {}
        
        # Generate candidate paths for each non-adjacent pair
        # We prioritize shorter paths
        domains = {}
        for u, v in non_adj_pairs:
            # Try lengths 2, 3, 4, 5
            paths = []
            for l in range(2, 6):
                all_p = get_all_paths(u, v, l)
                # Filter out paths that go through B
                valid_p = [p for p in all_p if not any(x in B_set for x in p[1:-1])]
                if valid_p:
                    paths.extend(valid_p)
                    break # Only take the shortest valid ones? Or all?
                    # Taking all shortest is usually better for disjointness
            if not paths:
                # If no short path, try longer?
                # For now, if no path <= 5, assume failure for speed
                # Or extend to 6
                all_p = get_all_paths(u, v, 6)
                valid_p = [p for p in all_p if not any(x in B_set for x in p[1:-1])]
                if not valid_p:
                    return False, None
                paths = valid_p
            
            domains[(u, v)] = paths
        
        # Sort pairs by domain size (MRV heuristic)
        sorted_pairs = sorted(non_adj_pairs, key=lambda p: len(domains[p]))
        
        used_interiors = set()
        assignment = {}
        
        def backtrack(idx):
            if time.time() - start_t > time_limit:
                return False
            if idx == len(sorted_pairs):
                return True
            
            u, v = sorted_pairs[idx]
            # Try paths in order of length (shortest first)
            for path in domains[(u, v)]:
                interior = path[1:-1]
                if any(w in used_interiors for w in interior):
                    continue
                used_interiors.update(interior)
                assignment[(u, v)] = path
                if backtrack(idx + 1):
                    return True
                used_interiors.difference_update(interior)
                del assignment[(u, v)]
            return False
        
        if backtrack(0):
            return True, dict(assignment)
        return False, None

    # 1. Try to extend the max clique
    current_B = list(best_clique)
    ok, paths_dict = check_subdivision(current_B, time_limit=5.0)
    if ok:
        best_k = len(current_B)
        best_B = current_B
        best_paths = paths_dict
    
    # 2. Iterative Deepening: Try to find larger k
    # Start from max_clique_size + 1 up to n
    # We use a heuristic: pick vertices with high degree to the current set?
    # Or random restarts.
    
    # Strategy: Randomized Local Search
    # We will try to find a set of size k for k from max_clique_size+1 to n
    # For each k, we do several random restarts
    
    max_possible_k = n
    found_improvement = False
    
    # First, try a greedy extension from the max clique
    if best_k < n:
        # Sort vertices by degree descending
        vertices_by_degree = sorted(range(n), key=lambda v: -degrees[v])
        
        # Try adding vertices one by one to the current best set
        # This is a heuristic, might get stuck in local optima
        # But it's fast.
        
        # Let's do a more robust search:
        # For k from n down to max_clique_size + 1:
        #   Try to find a valid set of size k.
        
        # Since n=24 is small, we can afford some random sampling.
        
        # Let's try a hybrid:
        # 1. Greedy build up
        # 2. Local search (swap)
        
        # Greedy Build
        B_greedy = list(best_clique)
        paths_greedy = dict(best_paths)
        
        # Try to add vertices
        added = True
        while added and time.time() < deadline - 10:
            added = False
            # Try all non-branch vertices
            # Sort by potential: number of connections to current B
            candidates = [v for v in range(n) if v not in B_greedy]
            # Heuristic: prefer vertices connected to many in B_greedy
            candidates.sort(key=lambda v: -len(adj[v] & set(B_greedy)))
            
            for v in candidates:
                new_B = B_greedy + [v]
                ok, new_paths = check_subdivision(new_B, time_limit=1.0)
                if ok:
                    B_greedy = new_B
                    paths_greedy = new_paths
                    if len(B_greedy) > best_k:
                        best_k = len(B_greedy)
                        best_B = list(B_greedy)
                        best_paths = dict(paths_greedy)
                        found_improvement = True
                    added = True
                    break # Restart with new set
        
        # Local Search: Swap
        # Try to improve by swapping vertices
        if time.time() < deadline - 5:
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                B_local = list(best_B)
                paths_local = dict(best_paths)
                
                for i in range(len(B_local)):
                    if time.time() > deadline - 3:
                        break
                    # Try swapping B_local[i] with some v not in B_local
                    # Sort candidates by degree
                    candidates = [v for v in range(n) if v not in B_local]
                    candidates.sort(key=lambda v: -degrees[v])
                    
                    for v in candidates:
                        new_B = B_local[:i] + [v] + B_local[i+1:]
                        ok, new_paths = check_subdivision(new_B, time_limit=0.5)
                        if ok:
                            # Check if this swap is "better"? 
                            # Since we just want any valid set of size k, and size is same,
                            # we just accept it to potentially allow further swaps.
                            # But we need to ensure we don't loop.
                            # Actually, if size is same, it's neutral.
                            # We only care about increasing size.
                            # So local search on fixed size is only useful to "unlock"
                            # a configuration that allows adding a (k+1)th vertex.
                            B_local = new_B
                            paths_local = dict(new_paths)
                            improved = True
                            break
                    if improved:
                        break
                
                # After swaps, try to add a vertex again
                if time.time() < deadline - 3:
                    candidates = [v for v in range(n) if v not in B_local]
                    candidates.sort(key=lambda v: -len(adj[v] & set(B_local)))
                    for v in candidates:
                        new_B = B_local + [v]
                        ok, new_paths = check_subdivision(new_B, time_limit=1.0)
                        if ok:
                            B_local = new_B
                            paths_local = new_paths
                            if len(B_local) > best_k:
                                best_k = len(B_local)
                                best_B = list(B_local)
                                best_paths = dict(paths_local)
                                found_improvement = True
                            break

    # 3. Random Restarts for larger k
    # If we haven't found a very large k, try random sets
    # Target k from n down to best_k + 1
    
    if time.time() < deadline - 5:
        # Try to find a set of size best_k + 1
        target_k = best_k + 1
        if target_k <= n:
            # Random restarts
            for attempt in range(50): # Up to 50 restarts
                if time.time() > deadline - 3:
                    break
                
                # Pick random set of size target_k
                # Bias towards high degree vertices
                # Weighted sampling
                weights = [degrees[v] + 1 for v in range(n)]
                # Simple random sample for now
                B_rand = random.sample(range(n), target_k)
                
                ok, paths_rand = check_subdivision(B_rand, time_limit=1.0)
                if ok:
                    best_k = target_k
                    best_B = B_rand
                    best_paths = paths_rand
                    found_improvement = True
                    break # Found one, try to extend further?
                    # Actually, let's just continue to see if we can find even larger?
                    # No, we are targeting specific k.
                    # If we found k, we should try k+1 in next loop or outer loop.
                    # But this loop is for fixed target_k.
                    # Let's restart the whole "try to find larger" process.
                    break
            
            # If we found best_k + 1, try best_k + 2?
            # We can loop while best_k < n
            while best_k < n and time.time() < deadline - 3:
                target_k = best_k + 1
                found = False
                for attempt in range(20):
                    if time.time() > deadline - 3:
                        break
                    B_rand = random.sample(range(n), target_k)
                    ok, paths_rand = check_subdivision(B_rand, time_limit=1.0)
                    if ok:
                        best_k = target_k
                        best_B = B_rand
                        best_paths = paths_rand
                        found = True
                        break
                if not found:
                    break # Can't find k+1, stop

    # Final verification and construction
    B = best_B
    k = len(B)
    
    if k < 2:
        # Fallback to a valid small solution
        # Find any edge
        for u, v in P['edges']:
            if u < v:
                B = [u, v]
                k = 2
                break
        if k < 2:
            # Disconnected? Unlikely.
            B = [0, 1]
            k = 2

    paths = []
    B_set = set(B)
    
    # Re-verify and construct paths for the final best_B
    # We must ensure the paths are valid for the returned B.
    # The best_paths dict might be from a previous state if we didn't update it correctly?
    # We updated best_paths whenever best_k was updated.
    # So best_paths should correspond to best_B.
    
    # However, to be safe, let's re-run the check on the final best_B
    # and use the paths from that run.
    
    final_ok, final_paths_dict = check_subdivision(best_B, time_limit=2.0)
    
    if final_ok:
        paths_dict = final_paths_dict
    else:
        # This should not happen if best_B was verified, but just in case
        # Try to recover or fallback
        # If it fails, we might need to shrink B
        # For now, assume it works. If not, the evaluator will fail us.
        # Let's try to shrink B until it works.
        while len(B) > 2 and not final_ok:
            B.pop()
            final_ok, final_paths_dict = check_subdivision(B, time_limit=2.0)
        if not final_ok:
            # Absolute fallback: 2 vertices
            for u, v in P['edges']:
                B = [u, v]
                break
            final_ok, final_paths_dict = check_subdivision(B, time_limit=1.0)
            if not final_ok:
                # Should be impossible for connected graph
                B = [0, 1]
                final_paths_dict = {}
        
        paths_dict = final_paths_dict
        best_k = len(B)
        best_B = B

    # Construct output
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            u, v = B[i], B[j]
            if v in adj[u]:
                paths.append([u, v])
            else:
                if (u, v) in paths_dict:
                    paths.append(list(paths_dict[(u, v)]))
                elif (v, u) in paths_dict:
                    paths.append(list(reversed(paths_dict[(v, u)])))
                else:
                    # Fallback: shortest path
                    p = get_shortest_path(u, v)
                    if p:
                        paths.append(p)
                    else:
                        # Should not happen
                        paths.append([u, v])
    
    # Convert to ensure correct types
    branches_out = [int(b) for b in B]
    paths_out = [[int(x) for x in p] for p in paths]
    
    return {'branches': branches_out, 'paths': paths_out}

# EVOLVE-BLOCK-END
