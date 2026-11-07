import time
import sys
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    deadline = time.time() + 85.0
    n = P['n']
    edges = P['edges']
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def has_bicolored_cycle(colors, c1, c2):
        """Check if subgraph induced by colors c1,c2 has a cycle using union-find."""
        verts = [i for i in range(n) if colors[i] in (c1, c2)]
        if len(verts) < 3:
            return False
        parent = {v: v for v in verts}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        vert_set = set(verts)
        for u in verts:
            for v in adj[u]:
                if v in vert_set and u < v:
                    if not union(u, v):
                        return True
        return False
    
    def count_violations(colors, k):
        """Count number of bicolored cycles + improper colorings."""
        cnt = 0
        for i in range(k):
            for j in range(i+1, k):
                if has_bicolored_cycle(colors, i, j):
                    cnt += 1
        for u, v in edges:
            if colors[u] == colors[v]:
                cnt += 1
        return cnt
    
    def is_valid(colors, k):
        return count_violations(colors, k) == 0
    
    best_colors = list(range(n))
    best_score = n
    
    # Try structured patterns first - these are fast to check
    # For circulant C(50, {1,7}), we need proper coloring AND forest constraint
    
    # 3-coloring: i mod 3 is proper since gcd(1,3)=1 and gcd(7,3)=1
    # But we need to check forest constraint
    for k in range(3, 8):
        for a in range(1, k):
            for b in range(k):
                if time.time() > deadline:
                    break
                colors = [(a * i + b) % k for i in range(n)]
                if is_valid(colors, k):
                    distinct = len(set(colors))
                    if distinct < best_score:
                        best_score = distinct
                        best_colors = colors[:]
    
    # Try 4-coloring patterns from current program
    coloring4 = [(i % 2) * 2 + (i // 2) % 2 for i in range(n)]
    if is_valid(coloring4, 4):
        if 4 < best_score:
            best_score = 4
            best_colors = coloring4[:]
    
    coloring4b = [(i % 2) * 2 + (i // 4) % 2 for i in range(n)]
    if is_valid(coloring4b, 4):
        if 4 < best_score:
            best_score = 4
            best_colors = coloring4b[:]
    
    # Now do local search to try to find better solutions
    rng = random.Random(42)
    
    # Start from best found so far or from a random valid starting point
    if best_score > 3:
        # Try to find a valid 3-coloring via local search
        for _ in range(5):
            if time.time() > deadline:
                break
            colors = [rng.randint(0, 2) for _ in range(n)]
            current_k = 3
            
            # Local search: minimize violations, then minimize distinct colors
            for it in range(3000):
                if time.time() > deadline:
                    break
                
                v = rng.randint(0, n-1)
                old_c = colors[v]
                current_viol = count_violations(colors, current_k)
                
                best_new_c = old_c
                best_new_viol = current_viol
                best_new_distinct = len(set(colors))
                
                for nc in range(current_k):
                    if nc == old_c:
                        continue
                    trial = colors[:]
                    trial[v] = nc
                    trial_viol = count_violations(trial, current_k)
                    if trial_viol < best_new_viol:
                        best_new_viol = trial_viol
                        best_new_c = nc
                        best_new_distinct = len(set(trial))
                    elif trial_viol == best_new_viol:
                        new_distinct = len(set(trial))
                        if new_distinct < best_new_distinct:
                            best_new_distinct = new_distinct
                            best_new_c = nc
                
                if best_new_c != old_c:
                    colors[v] = best_new_c
                    if best_new_viol == 0:
                        distinct = len(set(colors))
                        if distinct < best_score:
                            best_score = distinct
                            best_colors = colors[:]
    
    # Also try starting from a valid 4-coloring and reducing
    if best_score == 4:
        for _ in range(3):
            if time.time() > deadline:
                break
            colors = best_colors[:]
            current_k = 4
            
            for it in range(5000):
                if time.time() > deadline:
                    break
                
                v = rng.randint(0, n-1)
                old_c = colors[v]
                current_viol = count_violations(colors, current_k)
                
                best_new_c = old_c
                best_new_viol = current_viol
                best_new_distinct = len(set(colors))
                
                # Try colors 0..3 and also try to eliminate a color
                for nc in range(current_k):
                    if nc == old_c:
                        continue
                    trial = colors[:]
                    trial[v] = nc
                    trial_viol = count_violations(trial, current_k)
                    if trial_viol < best_new_viol:
                        best_new_viol = trial_viol
                        best_new_c = nc
                        best_new_distinct = len(set(trial))
                    elif trial_viol == best_new_viol:
                        new_distinct = len(set(trial))
                        if new_distinct < best_new_distinct:
                            best_new_distinct = new_distinct
                            best_new_c = nc
                
                if best_new_c != old_c:
                    colors[v] = best_new_c
                    if best_new_viol == 0:
                        distinct = len(set(colors))
                        if distinct < best_score:
                            best_score = distinct
                            best_colors = colors[:]
    
    # Final validation and return
    final_k = max(best_colors) + 1 if best_colors else 0
    if not is_valid(best_colors, final_k):
        # Fallback to identity coloring which is always valid
        return [int(i) for i in range(n)]
    
    return [int(c) for c in best_colors]
# EVOLVE-BLOCK-END
