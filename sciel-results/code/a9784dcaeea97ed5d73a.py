"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    # Build adjacency as bitmask for O(1) edge checks
    adj_mask = [0] * n
    adj_list = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        adj_list[u].add(v)
        adj_list[v].add(u)
    
    def is_edge(a, b):
        return (adj_mask[a] >> b) & 1
    
    deadline = time.time() + 85.0  # 5s safety margin before 90s limit
    
    def warnsdorff_score(v, used_mask):
        """Number of unvisited neighbors - lower is more constrained."""
        score = 0
        m = adj_mask[v] & ~used_mask
        while m:
            m &= m - 1
            score += 1
        return score
    
    def greedy_extend(path, used_mask):
        """Extend path greedily from both ends using Warnsdorff heuristic."""
        changed = True
        while changed:
            changed = False
            # Try extending from the end
            end = path[-1]
            best_v = -1
            best_score = float('inf')
            m = adj_mask[end] & ~used_mask
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                s = warnsdorff_score(v, used_mask)
                if s < best_score or (s == best_score and v < best_v):
                    best_score = s
                    best_v = v
                m &= m - 1
            if best_v >= 0:
                path.append(best_v)
                used_mask |= (1 << best_v)
                changed = True
            # Try extending from the start
            start = path[0]
            best_v = -1
            best_score = float('inf')
            m = adj_mask[start] & ~used_mask
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                s = warnsdorff_score(v, used_mask)
                if s < best_score or (s == best_score and v < best_v):
                    best_score = s
                    best_v = v
                m &= m - 1
            if best_v >= 0:
                path.insert(0, best_v)
                used_mask |= (1 << best_v)
                changed = True
        return path, used_mask
    
    def two_opt_improve(path):
        """Apply 2-opt: reverse a subpath if it maintains validity. Only check 2 edges."""
        improved = True
        while improved:
            improved = False
            lp = len(path)
            for i in range(lp - 2):
                if time.time() > deadline:
                    break
                for j in range(i + 2, lp):
                    if time.time() > deadline:
                        break
                    # Reversing path[i+1..j]: new edges are path[i]-path[j] and path[i+1]-path[j+1] (if j+1 exists)
                    if is_edge(path[i], path[j]):
                        if j + 1 < lp:
                            if is_edge(path[i+1], path[j+1]):
                                path[i+1:j+1] = path[i+1:j+1][::-1]
                                improved = True
                                break
                        else:
                            path[i+1:j+1] = path[i+1:j+1][::-1]
                            improved = True
                            break
                if improved:
                    break
        return path
    
    def iterative_dfs(start, used_mask, time_limit):
        """Iterative DFS with Warnsdorff ordering to find long paths."""
        best_path = [start]
        stack = [(start, 0)]  # (node, next_neighbor_index)
        path = [start]
        cur_mask = used_mask | (1 << start)
        neighbor_lists = []
        
        while stack:
            if time.time() > time_limit:
                break
            node, idx = stack[-1]
            neighbors = [v for v in range(n) if (adj_mask[node] >> v) & 1 and not (cur_mask >> v) & 1]
            neighbors.sort(key=lambda v: (warnsdorff_score(v, cur_mask), v))
            
            if idx < len(neighbors):
                v = neighbors[idx]
                stack[-1] = (node, idx + 1)
                path.append(v)
                cur_mask |= (1 << v)
                if len(path) > len(best_path):
                    best_path = path[:]
                stack.append((v, 0))
            else:
                stack.pop()
                path.pop()
                cur_mask &= ~(1 << node)
        
        return best_path, cur_mask
    
    def splice_paths(path1, path2, used1, used2):
        """Try to splice two paths together by finding connecting edges."""
        # Try: end of path1 connects to start of path2
        if is_edge(path1[-1], path2[0]):
            return path1 + path2
        # Try: end of path1 connects to end of path2 (reverse path2)
        if is_edge(path1[-1], path2[-1]):
            return path1 + path2[::-1]
        # Try: start of path1 connects to start of path2 (reverse path1)
        if is_edge(path1[0], path2[0]):
            return path1[::-1] + path2
        # Try: start of path1 connects to end of path2 (reverse both)
        if is_edge(path1[0], path2[-1]):
            return path1[::-1] + path2[::-1]
        return None
    
    best = [0]
    best_mask = 1
    
    # Phase 1: Greedy from every start vertex
    for start in range(n):
        if time.time() > deadline:
            break
        path = [start]
        mask = 1 << start
        path, mask = greedy_extend(path, mask)
        if len(path) > len(best):
            best = path[:]
            best_mask = mask
    
    # Phase 2: 2-opt on best path
    if time.time() < deadline:
        best = two_opt_improve(best[:])
        # Recompute mask
        best_mask = 0
        for v in best:
            best_mask |= (1 << v)
    
    # Phase 3: Extend after 2-opt
    if time.time() < deadline:
        path, mask = greedy_extend(best[:], best_mask)
        if len(path) > len(best):
            best = path
            best_mask = mask
    
    # Phase 4: Iterative DFS from multiple starts with time budget
    dfs_budget = min(20.0, max(5.0, deadline - time.time() - 10.0))
    if time.time() < deadline:
        per_dfs = dfs_budget / n
        for start in range(n):
            if time.time() > deadline - 10.0:
                break
            t_limit = time.time() + max(0.5, per_dfs)
            dfs_path, _ = iterative_dfs(start, 0, t_limit)
            if len(dfs_path) > len(best):
                best = dfs_path
                best_mask = 0
                for v in best:
                    best_mask |= (1 << v)
    
    # Phase 5: 2-opt + extend again
    if time.time() < deadline:
        best = two_opt_improve(best[:])
        best_mask = 0
        for v in best:
            best_mask |= (1 << v)
        path, mask = greedy_extend(best[:], best_mask)
        if len(path) > len(best):
            best = path
            best_mask = mask
    
    # Phase 6: Random restarts with Warnsdorff DFS
    rng = random.Random(748)
    remaining = deadline - time.time() - 5.0
    if remaining > 2.0:
        n_restarts = int(remaining / 0.5)  # ~0.5s per restart
        for _ in range(n_restarts):
            if time.time() > deadline - 5.0:
                break
            start = rng.randint(0, n - 1)
            # Randomized Warnsdorff greedy
            path = [start]
            mask = 1 << start
            while True:
                end = path[-1]
                candidates = [v for v in range(n) if (adj_mask[end] >> v) & 1 and not (mask >> v) & 1]
                if not candidates:
                    break
                candidates.sort(key=lambda v: (warnsdorff_score(v, mask), v))
                # Pick from top 3 with randomness
                top_k = min(3, len(candidates))
                v = candidates[rng.randint(0, top_k - 1)]
                path.append(v)
                mask |= (1 << v)
            # Extend from both ends
            path, mask = greedy_extend(path, mask)
            if len(path) > len(best):
                best = path
                best_mask = mask
    
    # Phase 7: Final 2-opt polish
    if time.time() < deadline - 2.0:
        best = two_opt_improve(best[:])
        best_mask = 0
        for v in best:
            best_mask |= (1 << v)
        path, mask = greedy_extend(best[:], best_mask)
        if len(path) > len(best):
            best = path
    
    return best
# EVOLVE-BLOCK-END
