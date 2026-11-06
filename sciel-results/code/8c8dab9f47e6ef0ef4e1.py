import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_n = 0
    best_colors = None
    
    for n in range(34, 0, -1):
        if time.time() > deadline - 5:
            break
        
        if n == 34:
            max_time_for_n = 45
        elif n == 33:
            max_time_for_n = 20
        elif n == 32:
            max_time_for_n = 10
        else:
            max_time_for_n = 5
            
        found = False
        start_n_time = time.time()
        
        for attempt in range(20):
            if time.time() > deadline - 5 or time.time() - start_n_time >= max_time_for_n:
                break
                
            time_limit = max(1.0, deadline - 5 - time.time())
            p_red = 0.45 + 0.05 * (attempt % 4)
            
            result = solve_walksat(n, time_limit, p_red=p_red)
            if result is not None:
                best_n = n
                best_colors = result
                found = True
                break
        
        if found:
            break
            
    if best_n == 0:
        n = 9
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        return {'n': n, 'colors': colors}
        
    return {'n': best_n, 'colors': best_colors}

def solve_walksat(n, time_limit=5, p_red=0.45):
    start_time = time.time()
    
    edges = list(combinations(range(n), 3))
    num_edges = len(edges)
    edge_index = {e: i for i, e in enumerate(edges)}
    
    k4s = list(combinations(range(n), 4))
    k5s = list(combinations(range(n), 5))
    
    edge_to_k4s = [[] for _ in range(num_edges)]
    edge_to_k5s = [[] for _ in range(num_edges)]
    
    k4_edges = []
    for k4 in k4s:
        a, b, c, d = k4
        e1 = edge_index[(a,b,c)]
        e2 = edge_index[(a,b,d)]
        e3 = edge_index[(a,c,d)]
        e4 = edge_index[(b,c,d)]
        k4_edges.append([e1, e2, e3, e4])
        for e in [e1, e2, e3, e4]:
            edge_to_k4s[e].append(len(k4_edges) - 1)
            
    k5_edges = []
    for k5 in k5s:
        es = [edge_index[e] for e in combinations(k5, 3)]
        k5_edges.append(es)
        for e in es:
            edge_to_k5s[e].append(len(k5_edges) - 1)
            
    colors = [0 if random.random() < p_red else 1 for _ in range(num_edges)]
    
    red_counts = [0] * len(k4s)
    blue_counts = [0] * len(k5s)
    
    for i, e_list in enumerate(k4_edges):
        for e in e_list:
            if colors[e] == 0:
                red_counts[i] += 1
                
    for i, e_list in enumerate(k5_edges):
        for e in e_list:
            if colors[e] == 1:
                blue_counts[i] += 1
                
    red_violations = set(i for i, c in enumerate(red_counts) if c == 4)
    blue_violations = set(i for i, c in enumerate(blue_counts) if c == 10)
    
    total_violations = len(red_violations) + len(blue_violations)
    
    if total_violations == 0:
        return colors.copy()
        
    p_walk = 0.2
    best_colors = colors.copy()
    best_violations = total_violations
    
    while time.time() < start_time + time_limit:
        if not red_violations and not blue_violations:
            return colors.copy()
            
        # Pick a violation
        if red_violations:
            if random.random() < 0.5 or not blue_violations:
                viol_idx = random.choice(list(red_violations))
                viol_type = 'red'
                candidate_edges = k4_edges[viol_idx]
            else:
                viol_idx = random.choice(list(blue_violations))
                viol_type = 'blue'
                candidate_edges = k5_edges[viol_idx]
        else:
            viol_idx = random.choice(list(blue_violations))
            viol_type = 'blue'
            candidate_edges = k5_edges[viol_idx]
            
        # Choose edge
        if random.random() < p_walk:
            e_idx = random.choice(candidate_edges)
        else:
            best_e = candidate_edges[0]
            best_delta = float('inf')
            
            for e in candidate_edges:
                # Calculate delta without modifying
                delta = 0
                old_color = colors[e]
                
                for k4_idx in edge_to_k4s[e]:
                    if old_color == 0:
                        if red_counts[k4_idx] == 4: delta -= 1
                        # 3->4 is not possible if old was 0 (Red->Blue decreases count)
                        # Wait.
                        # If old is 0 (Red), new is 1 (Blue).
                        # red_count decreases.
                        # If it was 4, it becomes 3. Violation removed. delta -= 1.
                        # If it was 3, it becomes 2. No change in violation status.
                    else:
                        # Old 1 (Blue), New 0 (Red).
                        # red_count increases.
                        # If it was 3, it becomes 4. Violation added. delta += 1.
                        if red_counts[k4_idx] == 3: delta += 1
                        
                for k5_idx in edge_to_k5s[e]:
                    if old_color == 1:
                        # Blue -> Red. blue_count decreases.
                        # If 10 -> 9. Violation removed. delta -= 1.
                        if blue_counts[k5_idx] == 10: delta -= 1
                    else:
                        # Red -> Blue. blue_count increases.
                        # If 9 -> 10. Violation added. delta += 1.
                        if blue_counts[k5_idx] == 9: delta += 1
                        
                if delta < best_delta:
                    best_delta = delta
                    best_e = e
            e_idx = best_e
            
        # Apply flip
        old_color = colors[e_idx]
        new_color = 1 - old_color
        colors[e_idx] = new_color
        
        # Update counts and violation sets
        for k4_idx in edge_to_k4s[e_idx]:
            if old_color == 0:
                # Red -> Blue
                red_counts[k4_idx] -= 1
                if red_counts[k4_idx] == 3:
                    # Was 4, now 3. Remove violation.
                    red_violations.discard(k4_idx)
                # If it was 3->2, no set change.
            else:
                # Blue -> Red
                red_counts[k4_idx] += 1
                if red_counts[k4_idx] == 4:
                    # Was 3, now 4. Add violation.
                    red_violations.add(k4_idx)
                    
        for k5_idx in edge_to_k5s[e_idx]:
            if old_color == 1:
                # Blue -> Red
                blue_counts[k5_idx] -= 1
                if blue_counts[k5_idx] == 9:
                    blue_violations.discard(k5_idx)
            else:
                # Red -> Blue
                blue_counts[k5_idx] += 1
                if blue_counts[k5_idx] == 10:
                    blue_violations.add(k5_idx)
                    
        total_violations = len(red_violations) + len(blue_violations)
        
        if total_violations < best_violations:
            best_violations = total_violations
            best_colors = colors.copy()
            
        if total_violations == 0:
            return colors.copy()

    if best_violations == 0:
        return best_colors
    return None
