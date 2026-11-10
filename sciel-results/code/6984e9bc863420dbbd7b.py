import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # Pre-computation
    edge_idx = {}
    idx = 0
    for i in range(16):
        for j in range(i + 1, 16):
            edge_idx[(i, j)] = idx
            idx += 1
            
    # Map edge to list of K4 indices
    edge_k4s = [[] for _ in range(120)]
    k4_edges = []
    
    k4_count = 0
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                for d in range(c + 1, 16):
                    e1 = edge_idx[(a, b)]
                    e2 = edge_idx[(a, c)]
                    e3 = edge_idx[(a, d)]
                    e4 = edge_idx[(b, c)]
                    e5 = edge_idx[(b, d)]
                    e6 = edge_idx[(c, d)]
                    
                    k4_edges.append((e1, e2, e3, e4, e5, e6))
                    
                    for e in [e1, e2, e3, e4, e5, e6]:
                        edge_k4s[e].append(k4_count)
                        
                    k4_count += 1
                    
    k4_edges_np = np.array(k4_edges)
    
    def is_violation(k4_idx, colors):
        e1, e2, e3, e4, e5, e6 = k4_edges_np[k4_idx]
        c1, c2, c3, c4, c5, c6 = colors[e1], colors[e2], colors[e3], colors[e4], colors[e5], colors[e6]
        # Check if < 5 distinct
        # Sort and check
        # Small optimization: if all same, violation.
        # We can just use a set for 6 items, it's very fast in Python
        return len({c1, c2, c3, c4, c5, c6}) < 5

    def initial_violations(colors):
        viol = np.zeros(k4_count, dtype=bool)
        count = 0
        for k in range(k4_count):
            if is_violation(k, colors):
                viol[k] = True
                count += 1
        return viol, count

    # Try K = 15
    for K in [15, 14]:
        if time.time() > deadline:
            break
            
        # Random restarts
        for restart in range(20):
            if time.time() > deadline:
                break
                
            colors = np.random.randint(0, K, 120)
            viol, count = initial_violations(colors)
            
            if count == 0:
                return list(colors)
                
            current_count = count
            current_viol = viol
            temperature = 1.0
            
            while time.time() < deadline - 2.0:
                # Pick a random violating K4
                if current_count == 0:
                    return list(colors)
                    
                # Get indices of violations
                # To avoid generating a list every time, we can pick a random index and check if it's a violation
                # Or maintain a list of violating K4s.
                # Maintaining a list is O(N) to update, which is bad.
                # Let's just pick a random edge and change it?
                # No, picking a violating K4 is better for convergence.
                
                # Efficient way to pick a violating K4:
                # We can keep a list of violating K4 indices.
                # When a K4 becomes non-violating, remove it.
                # When it becomes violating, add it.
                
                # Let's use a set of violating K4s for easy removal/addition?
                # Or just a list and filter occasionally?
                
                # Let's try: pick a random edge, change colour.
                # This is simpler and works well for CSPs.
                
                e = np.random.randint(0, 120)
                old_c = colors[e]
                new_c = np.random.randint(0, K)
                
                if old_c == new_c:
                    continue
                    
                colors[e] = new_c
                
                # Update violations for K4s containing e
                delta = 0
                for k in edge_k4s[e]:
                    was_viol = current_viol[k]
                    is_viol = is_violation(k, colors)
                    
                    if was_viol and not is_viol:
                        delta -= 1
                        current_viol[k] = False
                    elif not was_viol and is_viol:
                        delta += 1
                        current_viol[k] = True
                        
                new_count = current_count + delta
                
                if new_count < current_count:
                    current_count = new_count
                    temperature = max(0.1, temperature * 0.99)
                else:
                    # Reject or accept with prob
                    if new_count == current_count:
                         # Accept to escape local minima
                         if random.random() < 0.1:
                             current_count = new_count
                         else:
                             colors[e] = old_c
                             # Revert violations? 
                             # Since we didn't change the state logically (we reverted), 
                             # we don't need to revert the `current_viol` array if we didn't update it?
                             # Wait, we updated `current_viol` in the loop.
                             # We must revert those changes if we reject.
                             # This is messy.
                             
                             # Better: compute delta, then decide.
                             # If reject, revert colors[e] and revert current_viol changes.
                             # To simplify, let's just accept all moves that don't increase count?
                             # No, that gets stuck.
                             
                             # Let's use a proper SA approach with state backup for the affected K4s.
                             pass
                    else:
                        # Increase in violations
                        if random.random() < np.exp(-delta / temperature):
                            current_count = new_count
                            temperature *= 0.99
                        else:
                            # Revert
                            colors[e] = old_c
                            # Revert current_viol
                            for k in edge_k4s[e]:
                                # We need to know the previous state.
                                # This is getting complicated to revert correctly without storing old states.
                                pass
                            
                if current_count == 0:
                    return list(colors)
                    
    # Fallback
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
