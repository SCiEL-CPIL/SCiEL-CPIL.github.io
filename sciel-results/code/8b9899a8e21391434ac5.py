"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import random
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # 1. Find Largest Connected Component
    visited = [False] * n
    largest_cc = []
    for i in range(n):
        if not visited[i]:
            comp = []
            queue = deque([i])
            visited[i] = True
            while queue:
                node = queue.popleft()
                comp.append(node)
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
            if len(comp) > len(largest_cc):
                largest_cc = comp
                
    if not largest_cc:
        return [0]
        
    cc_set = set(largest_cc)
    n_cc = len(largest_cc)
    
    # 2. Initialize Path using Greedy (Warnsdorff-like)
    def get_greedy_path(start):
        path = [start]
        seen = {start}
        while True:
            candidates = [nb for nb in adj[path[-1]] if nb in cc_set and nb not in seen]
            if not candidates:
                break
            # Warnsdorff: choose neighbor with fewest unvisited neighbors
            next_v = min(candidates, key=lambda w: (len([x for x in adj[w] if x in cc_set and x not in seen]), w))
            path.append(next_v)
            seen.add(next_v)
        return path

    # Try starting from all nodes in CC, keep best
    best_path = []
    for start in largest_cc:
        p = get_greedy_path(start)
        if len(p) > len(best_path):
            best_path = p
            
    # 3. Local Search: Insertion and 2-Opt Swaps
    # We will run this for a limited number of iterations to stay within time limit
    # Since n=100, O(n^2) operations per iteration is fine.
    
    def can_insert(path, seen, z):
        """Check if z can be inserted into path. Returns index i such that z goes between path[i] and path[i+1], or None."""
        # Check if z is already in path
        if z in seen:
            return None
        # Check if z is connected to any vertex in path
        # We need to find i such that (path[i], z) and (z, path[i+1]) are edges.
        # This is O(L) where L is path length.
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            if z in adj[u] and z in adj[v]:
                return i
        # Also check ends? No, extension is handled separately.
        return None

    def try_extend(path, seen):
        """Try to extend path at start or end."""
        extended = False
        # Extend end
        last = path[-1]
        for nb in adj[last]:
            if nb in cc_set and nb not in seen:
                path.append(nb)
                seen.add(nb)
                extended = True
                break
        # Extend start
        first = path[0]
        for nb in adj[first]:
            if nb in cc_set and nb not in seen:
                path.insert(0, nb)
                seen.add(nb)
                extended = True
                break
        return extended

    def do_2opt(path):
        """
        Perform a 2-opt swap.
        Finds i, j such that we can reverse segment path[i+1..j] to maintain validity.
        Edges (path[i], path[i+1]) and (path[j], path[j+1]) exist in path.
        We want to replace with (path[i], path[j]) and (path[i+1], path[j+1]).
        This requires (path[i], path[j]) in adj and (path[i+1], path[j+1]) in adj.
        """
        L = len(path)
        # We need i < j-1
        # To be efficient, we can just pick random pairs or iterate.
        # Since n is small, we can iterate all pairs O(L^2).
        for i in range(L - 2):
            u = path[i]
            v = path[i+1]
            for j in range(i + 2, L):
                w = path[j-1]
                x = path[j]
                # Check if swap is valid
                if (u in adj[x] and v in adj[w]):
                    # Reverse path[i+1 : j]
                    # Original: ... u, v, ..., w, x, ...
                    # New: ... u, x, ..., v, w, ... 
                    # Wait, standard 2-opt:
                    # Path: p0 ... pi pi+1 ... pj-1 pj ...
                    # Edges: (pi, pi+1) and (pj-1, pj) are internal? No.
                    # Let's say we cut edges (pi, pi+1) and (pj-1, pj).
                    # We reconnect pi-pj-1? No.
                    # Standard 2-opt on path:
                    # Cut (pi, pi+1) and (pj-1, pj).
                    # Reverse segment pi+1 ... pj-1.
                    # New edges: (pi, pj-1) and (pi+1, pj).
                    # So we need (pi, pj-1) in adj and (pi+1, pj) in adj.
                    # In my loop: u=path[i], v=path[i+1], w=path[j-1], x=path[j].
                    # We need (u, w) and (v, x).
                    if (u in adj[w] and v in adj[x]):
                        # Perform swap
                        path[i+1:j] = path[i+1:j][::-1]
                        return True
        return False

    # Main Search Loop
    # We use a simple hill-climbing with random restarts/swaps to escape local optima.
    # Since we want to MAXIMIZE length, we accept any move that increases length.
    # If no insertion/extension is possible, we do a random 2-opt to change structure.
    
    max_iterations = 1000
    no_improve_count = 0
    
    current_path = best_path[:]
    current_seen = set(current_path)
    
    for _ in range(max_iterations):
        improved = False
        
        # 1. Try to extend ends
        if try_extend(current_path, current_seen):
            improved = True
            no_improve_count = 0
            continue
            
        # 2. Try to insert any unvisited vertex
        # To make this faster, we can pick a random unvisited vertex and try to insert it.
        # Or iterate through unvisited vertices.
        unvisited = [v for v in largest_cc if v not in current_seen]
        if unvisited:
            # Shuffle to avoid bias
            random.shuffle(unvisited)
            for z in unvisited:
                idx = can_insert(current_path, current_seen, z)
                if idx is not None:
                    current_path.insert(idx + 1, z)
                    current_seen.add(z)
                    improved = True
                    no_improve_count = 0
                    break
        
        if improved:
            continue
            
        # 3. No improvement. Try a random 2-opt swap to change structure.
        # We only do this if we haven't tried too many swaps without improvement.
        if no_improve_count < 50:
            if do_2opt(current_path):
                no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            # Reset or break? Let's just break if stuck for a while.
            break

    return current_path
# EVOLVE-BLOCK-END
