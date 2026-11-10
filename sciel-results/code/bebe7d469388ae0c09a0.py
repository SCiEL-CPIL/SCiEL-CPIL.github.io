import time
import random
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 120s (or 150s in some contexts, but task says 120s).
    # We aim to finish by 115s.
    deadline = start_time + 115.0
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Helper to get flat index for edge (i, j)
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    # Precompute all K4s and their edge indices
    k4_edges = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    edges = [
                        edge_idx(a, b), edge_idx(a, c), edge_idx(a, d),
                        edge_idx(b, c), edge_idx(b, d), edge_idx(c, d)
                    ]
                    k4_edges.append(edges)
    
    num_k4s = len(k4_edges)
    
    # Map each edge to the list of K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k_idx)

    # Function to check if a K4 is bad (uses < 5 colors)
    def is_bad_k4(edges, colors):
        # Unroll for speed
        c0 = colors[edges[0]]
        c1 = colors[edges[1]]
        c2 = colors[edges[2]]
        c3 = colors[edges[3]]
        c4 = colors[edges[4]]
        c5 = colors[edges[5]]
        # Check distinct count
        # A K4 is bad if distinct colors < 5
        # This is equivalent to:
        # 1. All 6 same
        # 2. 5 same, 1 diff
        # 3. 4 same, 2 diff (if the 2 diff are same or different? No, distinct count)
        # Let's just use set, it's fast enough for 6 items in Py3
        return len({c0, c1, c2, c3, c4, c5}) < 5

    def try_find_valid(K, time_limit):
        # colors is a list of length num_edges
        best_colors = None
        best_bad = float('inf')
        
        # Random initialization
        colors = [random.randint(0, K - 1) for _ in range(num_edges)]
        
        # Initial bad set
        bad_k4s = set()
        for k_idx in range(num_k4s):
            if is_bad_k4(k4_edges[k_idx], colors):
                bad_k4s.add(k_idx)
        
        current_bad = len(bad_k4s)
        if current_bad == 0:
            return colors
        
        best_bad = current_bad
        best_colors = colors[:]
        
        t = 1.0
        cooling = 0.9995
        min_t = 0.001
        
        while time.time() < time_limit:
            # Pick random edge
            e = random.randint(0, num_edges - 1)
            old_c = colors[e]
            new_c = random.randint(0, K - 1)
            if old_c == new_c:
                continue
            
            # Apply change
            colors[e] = new_c
            
            # Update bad_k4s incrementally
            # Only K4s containing edge e need re-checking
            affected_k4s = edge_to_k4s[e]
            delta_bad = 0
            
            for k_idx in affected_k4s:
                was_bad = k_idx in bad_k4s
                is_now_bad = is_bad_k4(k4_edges[k_idx], colors)
                
                if was_bad and not is_now_bad:
                    bad_k4s.remove(k_idx)
                    delta_bad -= 1
                elif not was_bad and is_now_bad:
                    bad_k4s.add(k_idx)
                    delta_bad += 1
            
            new_bad = current_bad + delta_bad
            
            if new_bad < current_bad:
                current_bad = new_bad
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_colors = colors[:]
                    if best_bad == 0:
                        return best_colors
            else:
                # Accept with probability exp(-delta / t)
                if delta_bad > 0:
                    prob = np.exp(-delta_bad / t)
                    if random.random() < prob:
                        current_bad = new_bad
                    else:
                        # Revert
                        colors[e] = old_c
                        # Revert bad_k4s changes
                        for k_idx in affected_k4s:
                            was_bad = k_idx in bad_k4s
                            # Note: We need to revert the set state too
                            # Since we modified bad_k4s in place, we need to undo those specific changes
                            # It's safer to just re-evaluate if we revert, but that's expensive.
                            # Let's trace the logic:
                            # If we revert colors[e], the state of bad_k4s should revert to previous state.
                            # We can just re-check the affected K4s against the reverted colors.
                            pass 
                        # Re-evaluate affected K4s to restore bad_k4s set
                        for k_idx in affected_k4s:
                            # Determine if it should be in bad_k4s based on reverted colors
                            should_be_bad = is_bad_k4(k4_edges[k_idx], colors)
                            if should_be_bad:
                                bad_k4s.add(k_idx)
                            else:
                                bad_k4s.discard(k_idx)
                        current_bad = len(bad_k4s)
                # If delta_bad <= 0, we already accepted or it was neutral/better (handled above)
            
            t *= cooling
            if t < min_t:
                t = min_t
                
            # Occasionally reset if stuck
            if random.random() < 0.0005:
                colors = [random.randint(0, K - 1) for _ in range(num_edges)]
                bad_k4s = set()
                for k_idx in range(num_k4s):
                    if is_bad_k4(k4_edges[k_idx], colors):
                        bad_k4s.add(k_idx)
                current_bad = len(bad_k4s)
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_colors = colors[:]
                    if best_bad == 0:
                        return best_colors
                t = 1.0

        return best_colors if best_bad == 0 else None

    def compact_and_return(colors):
        used = set(colors)
        if not used:
            return [0] * num_edges
        mapping = {c: i for i, c in enumerate(sorted(used))}
        return [mapping[c] for c in colors]

    # Strategy: Try to find a valid coloring with a small number of colors.
    # The theoretical lower bound is 13. We try 13, 14, 15.
    # We allocate time budget to each.
    
    # Try K=13
    time_left = deadline - time.time()
    if time_left > 5:
        result = try_find_valid(13, time_left - 2)
        if result is not None:
            return compact_and_return(result)
            
    # Try K=14
    time_left = deadline - time.time()
    if time_left > 5:
        result = try_find_valid(14, time_left - 2)
        if result is not None:
            return compact_and_return(result)
            
    # Try K=15
    time_left = deadline - time.time()
    if time_left > 5:
        result = try_find_valid(15, time_left - 2)
        if result is not None:
            return compact_and_return(result)
            
    # Fallback: Return a valid coloring even if it uses more colors?
    # The task says "Minimize the number of colours used".
    # If we didn't find a valid one with 15, we might need to return an invalid one?
    # No, "A valid colouring certifies...". An invalid result scores 0.
    # We must return a VALID coloring.
    # If SA fails to find a valid one with K=15, we should try a larger K or a different heuristic.
    # However, f(16) <= 15 is a known bound? "General bounds give 13 <= f(16) <= 15".
    # So a valid coloring with 15 colors MUST exist.
    # If our SA fails, it's likely due to local minima.
    # Let's try K=16 as a last resort, it should be easier to find.
    
    time_left = deadline - time.time()
    if time_left > 2:
        result = try_find_valid(16, time_left - 1)
        if result is not None:
            return compact_and_return(result)
            
    # If all else fails, return a random assignment? No, that will be invalid.
    # We should ensure at least one valid coloring is returned.
    # A trivial valid coloring uses many colors.
    # E.g., color each edge uniquely? 120 colors.
    # Or use a proper edge coloring? No, that's for perfect matchings.
    # We can construct a valid coloring with 120 colors easily:
    # Assign each edge a unique color. Distinct colors in any K4 = 6 >= 5.
    # This is always valid.
    
    # But we want to minimize.
    # If we are about to time out, we return the best valid one found.
    # If no valid one found, we return the unique color assignment (valid, but high score).
    # Wait, the evaluator scores based on "raw value" (number of colors).
    # So returning 120 is better than crashing (0).
    
    # Let's generate a fallback valid coloring with 120 colors
    fallback = []
    for i in range(n):
        for j in range(i + 1, n):
            fallback.append(edge_idx(i, j))
            
    return fallback
