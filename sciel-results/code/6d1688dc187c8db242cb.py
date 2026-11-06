"""Longest path in G(100, 1.8/100) pinned graph - matching-guided backtracking."""
import time
import sys
from collections import deque
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    adj_list = [list(s) for s in adj]
    degree = [len(adj[i]) for i in range(n)]
    
    deadline = time.time() + 85
    
    best_path = []
    best_len = 0
    
    def compute_matching_slack(seen):
        """Compute slack for each unvisited vertex based on bipartite matching relaxation.
        Slack = number of unvisited neighbors that are 'available' in the matching sense."""
        slack = {}
        for v in range(n):
            if v in seen:
                continue
            unvisited_neighbors = adj[v] - seen
            slack[v] = len(unvisited_neighbors)
        return slack
    
    def greedy_extend(path, seen, slack=None):
        """Greedy extension using Warnsdorff + slack heuristic."""
        path = list(path)
        seen = set(seen)
        while adj[path[-1]] - seen:
            candidates = adj[path[-1]] - seen
            if slack is None:
                slack = compute_matching_slack(seen)
            # Prefer neighbors with lowest slack (most constrained first)
            v = min(candidates, key=lambda w: (slack.get(w, 0), w))
            path.append(v)
            seen.add(v)
            slack = compute_matching_slack(seen)
        return path
    
    # Phase 1: Greedy from every start with slack-guided heuristic
    for start in range(n):
        if time.time() > deadline:
            break
        for seed_val in [0, 1, 2]:
            path = [start]
            seen = {start}
            slack = compute_matching_slack(seen)
            while adj[path[-1]] - seen:
                candidates = adj[path[-1]] - seen
                if seed_val == 0:
                    v = min(candidates, key=lambda w: (slack.get(w, 0), w))
                elif seed_val == 1:
                    v = max(candidates, key=lambda w: (slack.get(w, 0), w))
                else:
                    v = min(candidates, key=lambda w: w)
                path.append(v)
                seen.add(v)
                slack = compute_matching_slack(seen)
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
    
    # Phase 2: Backtracking with matching-slack guided ordering
    def backtrack(path, seen):
        """Backtracking with slack-guided candidate ordering."""
        nonlocal best_path, best_len
        
        if time.time() > deadline:
            return
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        if len(path) == n:
            return
        
        current = path[-1]
        candidates = list(adj[current] - seen)
        if not candidates:
            return
        
        slack = compute_matching_slack(seen)
        # Sort by slack ascending (most constrained first), then by degree
        candidates.sort(key=lambda w: (slack.get(w, 0), -degree[w]))
        
        for v in candidates:
            if time.time() > deadline:
                return
            path.append(v)
            seen.add(v)
            backtrack(path, seen)
            path.pop()
            seen.discard(v)
            
            if time.time() > deadline:
                return
    
    # Phase 3: Randomized restarts with slack-guided greedy
    rng = random.Random(42)
    starts_by_degree = sorted(range(n), key=lambda x: -degree[x])
    
    for start in starts_by_degree[:40]:
        if time.time() > deadline:
            break
        for trial in range(5):
            if time.time() > deadline:
                break
            path = [start]
            seen = {start}
            slack = compute_matching_slack(seen)
            while adj[path[-1]] - seen:
                candidates = list(adj[path[-1]] - seen)
                if not candidates:
                    break
                # Weighted random choice favoring low-slack vertices
                weights = [1.0 / (slack.get(w, 0) + 1) for w in candidates]
                total_w = sum(weights)
                if total_w == 0:
                    v = candidates[0]
                else:
                    r = rng.random() * total_w
                    cum = 0
                    v = candidates[-1]
                    for i, w in enumerate(candidates):
                        cum += weights[i]
                        if r <= cum:
                            v = candidates[i]
                            break
                path.append(v)
                seen.add(v)
                slack = compute_matching_slack(seen)
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
    
    # Phase 4: Local search - extend best path from both ends
    if best_path and time.time() < deadline:
        bp = list(best_path)
        bs = set(bp)
        
        for v in list(adj[bp[0]] - bs):
            if time.time() > deadline:
                break
            new_path = [v] + bp
            new_seen = bs | {v}
            slack = compute_matching_slack(new_seen)
            while adj[new_path[-1]] - new_seen:
                candidates = adj[new_path[-1]] - new_seen
                w = min(candidates, key=lambda x: (slack.get(x, 0), x))
                new_path.append(w)
                new_seen.add(w)
                slack = compute_matching_slack(new_seen)
            if len(new_path) > best_len:
                best_len = len(new_path)
                best_path = list(new_path)
        
        for v in list(adj[bp[-1]] - bs):
            if time.time() > deadline:
                break
            new_path = bp + [v]
            new_seen = bs | {v}
            slack = compute_matching_slack(new_seen)
            while adj[new_path[-1]] - new_seen:
                candidates = adj[new_path[-1]] - new_seen
                w = min(candidates, key=lambda x: (slack.get(x, 0), x))
                new_path.append(w)
                new_seen.add(w)
                slack = compute_matching_slack(new_seen)
            if len(new_path) > best_len:
                best_len = len(new_path)
                best_path = list(new_path)
    
    # Phase 5: Targeted backtracking from promising starts with limited depth
    if time.time() < deadline:
        for start in starts_by_degree[:15]:
            if time.time() > deadline:
                break
            path = [start]
            seen = {start}
            backtrack(path, seen)
            if time.time() > deadline:
                break
    
    # Phase 6: Edge-pair seeding
    if time.time() < deadline - 3:
        for i in range(n):
            if time.time() > deadline:
                break
            for j in adj_list[i]:
                if time.time() > deadline:
                    break
                path = [i, j]
                seen = {i, j}
                slack = compute_matching_slack(seen)
                while adj[path[-1]] - seen:
                    candidates = adj[path[-1]] - seen
                    v = min(candidates, key=lambda w: (slack.get(w, 0), w))
                    path.append(v)
                    seen.add(v)
                    slack = compute_matching_slack(seen)
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = list(path)
    
    return best_path if best_path else [0]
