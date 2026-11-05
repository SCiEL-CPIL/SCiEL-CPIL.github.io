import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    edge_index = {}
    for idx, (i, j) in enumerate(edges):
        edge_index[(i, j)] = idx
        edge_index[(j, i)] = idx
        
    k4_edges = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    e1 = edge_index[(a, b)]
                    e2 = edge_index[(a, c)]
                    e3 = edge_index[(a, d)]
                    e4 = edge_index[(b, c)]
                    e5 = edge_index[(b, d)]
                    e6 = edge_index[(c, d)]
                    k4_edges.append((e1, e2, e3, e4, e5, e6))
    
    num_k4 = len(k4_edges)
    
    def evaluate(colors):
        used_colors = set(colors)
        num_used = len(used_colors)
        
        violations = 0
        for e1, e2, e3, e4, e5, e6 in k4_edges:
            c1 = colors[e1]
            c2 = colors[e2]
            c3 = colors[e3]
            c4 = colors[e4]
            c5 = colors[e5]
            c6 = colors[e6]
            # Fast distinct count for 6 items
            # We can just use set, it's C-implemented
            if len({c1, c2, c3, c4, c5, c6}) < 5:
                violations += 1
                
        return num_used, violations

    def try_local_search(K, max_iter=50000):
        colors = np.random.randint(0, K, num_edges)
        
        num_used, violations = evaluate(colors)
        if violations == 0:
            return colors, num_used
        
        current_cost = violations * 1000 + num_used
        best_colors = colors.copy()
        best_num_used = num_used
        
        temp = 1.0
        cooling = 0.9999
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            e_idx = np.random.randint(num_edges)
            old_color = colors[e_idx]
            new_color = np.random.randint(0, K)
            
            if new_color == old_color:
                continue
                
            colors[e_idx] = new_color
            
            new_num_used, new_violations = evaluate(colors)
            new_cost = new_violations * 1000 + new_num_used
            
            if new_cost < current_cost or (temp > 0 and np.random.random() < np.exp(-(new_cost - current_cost) / temp)):
                current_cost = new_cost
                if new_violations == 0 and new_num_used < best_num_used:
                    best_colors = colors.copy()
                    best_num_used = new_num_used
            else:
                colors[e_idx] = old_color
                
            temp *= cooling
            
        return best_colors, best_num_used

    best_result = None
    
    # Try K=13, 14, 15
    for K in [13, 14, 15]:
        if time.time() > deadline:
            break
            
        # Do multiple restarts for each K if time permits
        restarts = 0
        while time.time() < deadline - 5.0:
            restarts += 1
            colors, num_used = try_local_search(K, max_iter=10000)
            if num_used < 100: # Valid
                if best_result is None or num_used < best_result[1]:
                    best_result = (colors, num_used)
                # If we found 13, we can stop trying higher K for this K loop? 
                # But we want the minimum. 13 is the lower bound.
                if num_used == 13:
                    break
            # If we found a valid one, we can break the restart loop for this K 
            # and try a lower K? No, we are iterating K in increasing order.
            # So if we find valid 13, we are done.
            # If we find valid 14, we should try 13.
            # So if we find valid K, we break the inner while loop and move to next K (which is lower? No, K is 13, 14, 15).
            # We should iterate K in increasing order: 13, 14, 15.
            # If we find valid 13, return.
            # If we find valid 14, we can break the inner loop and try 15? No, 15 is worse.
            # So if we find valid 14, we should still try to find 13? 
            # The loop structure `for K in [13, 14, 15]` handles this. 
            # If K=13 fails, it tries K=14. If K=14 succeeds, it updates best_result. 
            # Then it tries K=15. If K=15 finds a better one (impossible, 15>14), it updates.
            # So we just need to ensure we don't waste time on K=15 if we already have 13 or 14.
            if best_result is not None and best_result[1] <= K:
                # We have a valid solution with <= K colors.
                # If K=13, we have the best possible.
                # If K=14, we might still find 13? No, we already tried 13.
                # So we can break the K loop if we found a valid solution?
                # Not necessarily, because we might have found 14, but maybe a different random seed with 13 would have worked? 
                # We already tried 13 with some restarts. 
                # To be safe, we can break the K loop if we found 13.
                # If we found 14, we can continue to 15? No, 15 is worse. 
                # So if we found 14, we should break the K loop and return 14? 
                # Or try more restarts on 13? 
                # Let's just break the K loop if we found a valid solution, as lower K was tried first.
                break
                
        if best_result is not None and best_result[1] == 13:
            break

    if best_result is not None:
        return list(best_result[0])
    else:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

