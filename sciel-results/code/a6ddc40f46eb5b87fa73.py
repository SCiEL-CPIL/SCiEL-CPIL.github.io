import time
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    deadline = time.time() + 80
    random.seed(1)
    
    def is_valid(colors, k):
        # Proper coloring
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        # Bicolor cycles
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                parent = list(range(n))
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                for u, v in edges:
                    if (colors[u] == c1 or colors[u] == c2) and (colors[v] == c1 or colors[v] == c2):
                        pu, pv = find(u), find(v)
                        if pu == pv:
                            return False
                        parent[pu] = pv
        return True
    
    # Try i%4 first
    colors4 = [i % 4 for i in range(n)]
    if is_valid(colors4, 4):
        # Try to reduce to 3
        for _ in range(5000):
            if time.time() > deadline:
                break
            c = list(colors4)
            # Pick a random color to merge
            merge_from = random.randint(0, 3)
            merge_to = random.choice([x for x in range(4) if x != merge_from])
            new_c = [merge_to if x == merge_from else x for x in c]
            # Remap to 0,1,2
            present = sorted(set(new_c))
            remap = {v: i for i, v in enumerate(present)}
            new_c = [remap[x] for x in new_c]
            if is_valid(new_c, 3):
                return [int(x) for x in new_c]
        return [int(x) for x in colors4]
    
    # Fallback: random search
    for k in [3, 4, 5, 6, 8, 10, 15, 20, 30, 50]:
        for attempt in range(1000):
            if time.time() > deadline:
                break
            colors = [random.randint(0, k-1) for _ in range(n)]
            if is_valid(colors, k):
                return [int(x) for x in colors]
    
    return list(range(n))
