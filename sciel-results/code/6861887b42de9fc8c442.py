"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    deadline = time.time() + 85
    
    def check_bicolored_cycle(v, c, color_arr):
        """Check if assigning color c to vertex v creates a bicolored cycle."""
        for nb in adj[v]:
            if color_arr[nb] == -1 or color_arr[nb] == c:
                continue
            c2 = color_arr[nb]
            visited = {nb}
            queue = [nb]
            while queue:
                curr = queue.pop(0)
                for nb2 in adj[curr]:
                    if nb2 == v:
                        continue
                    if color_arr[nb2] == -1:
                        continue
                    if color_arr[nb2] not in (c, c2):
                        continue
                    if nb2 not in visited:
                        visited.add(nb2)
                        if color_arr[nb2] == c and nb2 in adj[v]:
                            return True
                        queue.append(nb2)
        return False
    
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    
    for k in range(4, 50):
        if time.time() > deadline - 5:
            break
        
        color_arr = [-1] * n
        
        def backtrack(pos):
            if time.time() > deadline - 2:
                return False
            if pos == n:
                return True
            
            v = order[pos]
            used = set()
            for nb in adj[v]:
                if color_arr[nb] != -1:
                    used.add(color_arr[nb])
            
            for c in range(k):
                if c in used:
                    continue
                if check_bicolored_cycle(v, c, color_arr):
                    continue
                color_arr[v] = c
                if backtrack(pos + 1):
                    return True
                color_arr[v] = -1
            return False
        
        if backtrack(0):
            return [int(x) for x in color_arr]
    
    return [int(x) for x in range(n)]
# EVOLVE-BLOCK-END
