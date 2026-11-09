import time
import random
import sys
from collections import defaultdict

def construct():
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e not in edges and degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_bitmasks(edges, adj):
        m = len(edges)
        if m == 0:
            return []
        
        max_node = max(max(u, v) for u, v in edges) + 1
        vert_to_edges = [[] for _ in range(max_node)]
        for i, (u, v) in enumerate(edges):
            vert_to_edges[u].append(i)
            vert_to_edges[v].append(i)
            
        conflict_masks = [0] * m
        
        # 1. Shared vertices
        for u in range(max_node):
            e_list = vert_to_edges[u]
            for i in range(len(e_list)):
                ei = e_list[i]
                for j in range(i + 1, len(e_list)):
                    ej = e_list[j]
                    conflict_masks[ei] |= (1 << ej)
                    conflict_masks[ej] |= (1 << ei)
        
        # 2. Induced matching violations (joined by edge)
        for i, (u, v) in enumerate(edges):
            neighbors_u = adj.get(u, set())
            neighbors_v = adj.get(v, set())
            for j in range(i + 1, m):
                if (conflict_masks[i] >> j) & 1:
                    continue
                w, x = edges[j]
                if w in neighbors_u or x in neighbors_u or w in neighbors_v or x in neighbors_v:
                    conflict_masks[i] |= (1 << j)
                    conflict_masks[j] |= (1 << i)
        
        return conflict_masks

    def dsatur_coloring(m, conflict_masks):
        """
        DSATUR (Degree of Saturation) heuristic for initial coloring.
        Returns list of colors.
        """
        if m == 0:
            return []
        
        # Precompute degrees for tie-breaking
        degrees = [bin(conflict_masks[i]).count('1') for i in range(m)]
        
        assign = [-1] * m
        color_masks = [] # Bitmask of edges in each color
        
        for _ in range(m):
            # Select next vertex (edge) to color
            # Priority: Max number of distinct neighbor colors, then Max degree
            best_idx = -1
            best_sat = -1
            best_deg = -1
            
            for i in range(m):
                if assign[i] != -1:
                    continue
                
                # Count distinct colors among neighbors
                neighbor_colors = set()
                mask = conflict_masks[i]
                for c_idx, c_mask in enumerate(color_masks):
                    if mask & c_mask:
                        neighbor_colors.add(c_idx)
                
                sat = len(neighbor_colors)
                deg = degrees[i]
                
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_idx = i
            
            if best_idx == -1:
                break
                
            # Find valid color
            neighbor_colors = set()
            mask = conflict_masks[best_idx]
            for c_idx, c_mask in enumerate(color_masks):
                if mask & c_mask:
                    neighbor_colors.add(c_idx)
            
            # Try existing colors
            placed = False
            for c in range(len(color_masks)):
                if c not in neighbor_colors:
                    assign[best_idx] = c
                    color_masks[c] |= (1 << best_idx)
                    placed = True
                    break
            
            if not placed:
                assign[best_idx] = len(color_masks)
                color_masks.append(1 << best_idx)
        
        # Compact colors
        if not assign:
            return []
        max_c = max(assign)
        mapping = {}
        next_c = 0
        final = [0] * m
        for i in range(m):
            if assign[i] not in mapping:
                mapping[assign[i]] = next_c
                next_c += 1
            final[i] = mapping[assign[i]]
        return final

    def min_conflicts_coloring(m, conflict_masks, k, max_iters=50000, rng=None):
        """
        Min-Conflicts algorithm to find a valid coloring with at most k colors.
        Returns (success, assignment)
        """
        if m == 0:
            return True, []
        if k <= 0:
            return False, []
        
        if rng is None:
            rng = random.Random()
            
        # Initialize random assignment
        assign = [rng.randrange(k) for _ in range(m)]
        
        # Precompute conflict counts for each edge?
        # We need to evaluate cost of changing edge i to color c.
        # Cost(i, c) = number of neighbors j of i such that assign[j] == c and conflict(i,j)
        
        # To make this fast, we can maintain:
        # conflict_count[i]: total conflicts for edge i
        # But we need per-color conflict counts for the move decision.
        
        # Let's compute initial total conflicts
        total_conflicts = 0
        for i in range(m):
            for j in range(i+1, m):
                if (conflict_masks[i] >> j) & 1:
                    if assign[i] == assign[j]:
                        total_conflicts += 1
        
        if total_conflicts == 0:
            return True, assign
            
        for it in range(max_iters):
            # Pick a random conflicted edge
            # We need a list of conflicted edges.
            # Efficiently finding one:
            # Just pick random i, check if it's conflicted. If not, try again (few times) or just pick random i and hope?
            # Better: maintain a set of conflicted edges.
            
            # For simplicity and speed in Python, let's just pick a random edge.
            # If the graph is dense with conflicts, most edges are conflicted.
            i = rng.randrange(m)
            
            # Check if i is involved in a conflict
            # If not, we might want to pick a different one. 
            # But if total_conflicts > 0, there are conflicted edges.
            # To be safe, let's verify if i is conflicted.
            # Actually, min-conflicts usually picks a random *conflicted* vertex.
            # Finding one requires scanning or maintaining a list.
            # Given m is small (<200), scanning is fine if done occasionally, 
            # but doing it every step is O(m).
            
            # Let's optimize: 
            # We'll just pick a random edge. If it's not conflicted, we might move it anyway? 
            # No, standard min-conflicts picks conflicted.
            
            # Let's find a conflicted edge.
            # We can just iterate a few times.
            conflicted_edge = -1
            for _ in range(10): # try up to 10 times
                candidate = rng.randrange(m)
                # Check if candidate has any conflict
                # We need to know if assign[candidate] == assign[neighbor] for some neighbor.
                # This is O(deg).
                mask = conflict_masks[candidate]
                c = assign[candidate]
                # Check neighbors
                # We can iterate through set bits in mask? Or just loop j?
                # m is small, loop j is fine.
                has_conflict = False
                for j in range(m):
                    if i == j: continue
                    if (conflict_masks[candidate] >> j) & 1:
                        if assign[j] == c:
                            has_conflict = True
                            break
                if has_conflict:
                    conflicted_edge = candidate
                    break
                    
            if conflicted_edge == -1:
                # No conflict found in sample? Maybe total_conflicts is 0?
                # Or we just missed it.
                # If we really can't find one, assume done?
                # Let's recompute total conflicts to be sure.
                tc = 0
                for x in range(m):
                    cx = assign[x]
                    mx = conflict_masks[x]
                    for y in range(x+1, m):
                        if (mx >> y) & 1 and assign[y] == cx:
                            tc += 1
                if tc == 0:
                    return True, assign
                # Else, just pick random and hope
                conflicted_edge = rng.randrange(m)
                
            i = conflicted_edge
            current_color = assign[i]
            
            # Find best new color
            best_color = -1
            min_conf = m + 1
            
            for c in range(k):
                if c == current_color:
                    continue
                # Count conflicts if i takes color c
                conflicts = 0
                mask = conflict_masks[i]
                # Check all neighbors j
                # Optimization: Only check j where assign[j] == c
                # We don't have a list of edges per color.
                # Loop all j
                for j in range(m):
                    if i == j: continue
                    if (mask >> j) & 1:
                        if assign[j] == c:
                            conflicts += 1
                if conflicts < min_conf:
                    min_conf = conflicts
                    best_color = c
            
            if best_color == -1:
                # Should not happen if k >= 1
                best_color = rng.randrange(k)
                
            assign[i] = best_color
            
            # We don't track total_conflicts incrementally here for simplicity,
            # but we can check for termination periodically or just run for max_iters.
            # To check termination, we'd need to update total_conflicts.
            # Let's just run for max_iters and check at the end or every 1000 steps.
            
            if it % 1000 == 0:
                # Quick check
                tc = 0
                for x in range(m):
                    cx = assign[x]
                    mx = conflict_masks[x]
                    for y in range(x+1, m):
                        if (mx >> y) & 1 and assign[y] == cx:
                            tc += 1
                if tc == 0:
                    return True, assign
                    
        # Final check
        tc = 0
        for x in range(m):
            cx = assign[x]
            mx = conflict_masks[x]
            for y in range(x+1, m):
                if (mx >> y) & 1 and assign[y] == cx:
                    tc += 1
        if tc == 0:
            return True, assign
        return False, assign

    def solve_instance(n, d, seed, time_limit=30):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        conflict_masks = build_conflict_bitmasks(edges, adj)
        
        # Get initial upper bound with DSATUR
        init_assign = dsatur_coloring(m, conflict_masks)
        current_k = max(init_assign) + 1 if init_assign else 0
        
        best_assign = init_assign
        best_k = current_k
        
        if best_k <= 1:
            return best_assign
            
        rng = random.Random(seed + 999)
        deadline = time.time() + time_limit
        
        # Try to reduce k
        # Start from best_k - 1 down to 1
        # But we should prioritize smaller k? 
        # If we find k=10, we should check k=9.
        # Iterating downwards is good because if k=10 works, k=11 definitely works (we already have it).
        # We want the MINIMUM k.
        
        # However, finding a valid coloring for small k is harder.
        # So we should give more time to small k? Or just iterate and let time run out?
        # If we iterate k from LB to best_k-1, and find one, we can stop? 
        # No, we want the MINIMUM. So if we find k=10, we must check k=9, 8...
        # So we should iterate k from 1 upwards?
        # If we find a valid coloring for k, that's a candidate.
        # But if k=5 fails, k=6 might succeed.
        # If we iterate 1, 2, 3... and find success at 6, we know 1-5 failed (within time budget).
        # But "failed" might just mean "ran out of time".
        
        # Strategy:
        # We have a budget.
        # We know k=best_k works.
        # We want to find if k < best_k works.
        # Let's try k from best_k - 1 down to 1.
        # Why? Because if we find k=10 works, we update best_k=10. Then we try 9.
        # If 9 works, best_k=9.
        # This ensures we always have a valid solution.
        # The risk is that we spend too much time on k=best_k-1 (which is likely to succeed) 
        # and don't have time for k=best_k-5 (which is the real improvement).
        
        # Better Strategy:
        # Interleaved search.
        # Or:
        # Since best_k is small (e.g., 20), the range is small.
        # Let's just iterate k from best_k - 1 down to 1.
        # Allocate time proportionally? Or fixed time per k?
        
        # Let's give each k a fixed time slice, say 2 seconds.
        # If best_k is 20, we have 19 values. 19 * 2 = 38s. Too long.
        # We have 30s limit.
        
        # Let's prioritize the most promising k.
        # Usually, the optimal k is close to the lower bound.
        # Lower bound: max degree in conflict graph + 1? No, that's upper bound for greedy.
        # Lower bound: size of largest clique in conflict graph.
        # We can estimate clique number? 
        # For induced matchings, a clique in conflict graph corresponds to a set of edges 
        # that are all mutually conflicting.
        # E.g., all edges incident to a vertex v form a clique? 
        # Yes, if e1, e2 share v, they conflict.
        # So LB = max_degree(original_graph) + 1? 
        # If a vertex has degree d, all d edges incident to it form a clique of size d.
        # So we need at least d colors.
        # Actually, if we have a star of degree d, we need d colors for those d edges.
        # So LB = d (max degree of original graph).
        
        lb = d # max degree
        # Also, if there are edges joined by edges, it might be higher.
        # But d is a safe LB.
        
        # We want to try k from lb to best_k - 1.
        # If we find a valid coloring for k, we update best_k = k.
        # We should try smaller k first?
        # If we try k=lb first and it fails, we move to lb+1.
        # If it succeeds, we have a very good result.
        
        # Let's iterate k from lb to best_k - 1.
        # Time budget per k: remaining_time / (best_k - lb)
        
        start_time = time.time()
        
        for k in range(lb, best_k):
            if time.time() > deadline:
                break
            
            # If k >= best_k, skip (we already have best_k)
            # But we are iterating up to best_k - 1.
            
            # Run Min-Conflicts
            # Increase max_iters for smaller k? 
            # Smaller k is harder, so it needs more iterations to converge or fail.
            # But if it's impossible, it will never converge.
            # Let's give it a fixed number of iterations, say 20000.
            
            success, assign = min_conflicts_coloring(m, conflict_masks, k, max_iters=20000, rng=rng)
            
            if success:
                # Verify
                # Check if valid
                valid = True
                for i in range(m):
                    for j in range(i+1, m):
                        if (conflict_masks[i] >> j) & 1:
                            if assign[i] == assign[j]:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    # Check if it uses <= k colors
                    used_colors = set(assign)
                    if len(used_colors) <= k:
                        best_k = k
                        best_assign = assign
                        # We found a better k. 
                        # Should we continue? 
                        # We are iterating upwards. 
                        # If we found k=10, we should check if k=9 works? 
                        # Yes, so we continue the loop.
                        # But we can't break because we want the MINIMUM.
        
        return best_assign

    out = []
    deadline = time.time() + 115
    
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        remaining = deadline - time.time()
        if remaining < 10:
            # Fallback: Fast Greedy
            n, d, seed = spec
            edges = instance(n, d, seed)
            m = len(edges)
            if m == 0:
                out.append([])
                continue
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            conflict_masks = build_conflict_bitmasks(edges, adj)
            # Use DSATUR as fallback too, it's fast
            out.append(dsatur_coloring(m, conflict_masks))
        else:
            t_limit = min(remaining - 5, 35)
            out.append(solve_instance(*spec, time_limit=t_limit))
    
    return out

if __name__ == "__main__":
    # Quick test
    res = construct()
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(set(r)) if r else 0} colors")
