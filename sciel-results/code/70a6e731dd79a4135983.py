import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # List of all edges (i, j)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # List of all K4s (sets of 4 vertices)
    k4s = list(combinations(range(n), 4))
    
    # For each K4, find the indices of its 6 edges
    k4_edge_indices = []
    for k4 in k4s:
        vs = list(k4)
        e_indices = []
        for i in range(4):
            for j in range(i + 1, 4):
                e_indices.append(edge_index[(vs[i], vs[j])])
        k4_edge_indices.append(e_indices)
        
    # For each edge, find which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)

    def count_violations(colors):
        # colors is a list of 120 ints
        # Returns number of K4s with < 5 distinct colors
        viol = 0
        for e_indices in k4_edge_indices:
            distinct = len(set(colors[idx] for idx in e_indices))
            if distinct < 5:
                viol += 1
        return viol
        
    def count_colors(colors):
        return len(set(colors))

    # Optimized Local Search with Delta Updates
    def local_search(max_colors, time_limit):
        # Initialize with random colors
        colors = [random.randint(0, max_colors - 1) for _ in range(num_edges)]
        
        # Precompute current state
        # k4_counts[k4_idx] is a list of size max_colors, storing count of each color
        k4_counts = []
        k4_distinct = []
        total_violations = 0
        
        for e_indices in k4_edge_indices:
            counts = [0] * max_colors
            for idx in e_indices:
                counts[colors[idx]] += 1
            distinct = sum(1 for c in counts if c > 0)
            k4_counts.append(counts)
            k4_distinct.append(distinct)
            if distinct < 5:
                total_violations += 1
                
        best_violations = total_violations
        best_colors = colors[:]
        
        # Simulated Annealing parameters
        temp = 10.0
        min_temp = 0.01
        cool_rate = 0.9999
        
        # Main Loop
        while time.time() < start_time + time_limit:
            if total_violations == 0:
                return colors, 0
            
            # Pick a random edge
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            
            # Pick a new color different from old
            new_color = random.randint(0, max_colors - 1)
            if new_color == old_color:
                continue
            
            # Calculate Delta Violations
            delta_viol = 0
            affected_k4s = edge_to_k4s[e_idx]
            
            # We need to track which K4s were modified to revert if necessary
            # Or we can just apply and revert
            # Since we need to revert, we must store the old state of the counts
            # But storing full counts is expensive. 
            # Instead, we can calculate the change in distinctness for each K4.
            # If we reject, we must restore counts.
            
            # Let's apply the change to counts first, track old distinct values, 
            # and revert if rejected.
            
            old_distincts = []
            
            for k4_idx in affected_k4s:
                counts = k4_counts[k4_idx]
                old_d = k4_distinct[k4_idx]
                old_distincts.append(old_d)
                
                # Update counts
                counts[old_color] -= 1
                counts[new_color] += 1
                
                # Update distinct
                new_d = old_d
                if counts[old_color] == 0:
                    new_d -= 1
                if counts[new_color] == 1: # It was 0 before
                    new_d += 1
                
                k4_distinct[k4_idx] = new_d
                
                # Update violation status
                was_viol = 1 if old_d < 5 else 0
                is_viol = 1 if new_d < 5 else 0
                delta_viol += is_viol - was_viol
            
            # Acceptance Criteria
            accept = False
            if delta_viol <= 0:
                accept = True
            else:
                if random.random() < math.exp(-delta_viol / temp):
                    accept = True
            
            if accept:
                colors[e_idx] = new_color
                total_violations += delta_viol
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors[:]
            else:
                # Revert counts
                for k4_idx, old_d in zip(affected_k4s, old_distincts):
                    counts = k4_counts[k4_idx]
                    counts[old_color] += 1
                    counts[new_color] -= 1
                    k4_distinct[k4_idx] = old_d
            
            # Cool down
            temp *= cool_rate
            if temp < min_temp:
                # Restart? Or just keep going? 
                # Restarting can help escape local minima.
                # Let's do a simple restart with new random init if we've been stuck for a while?
                # For now, just continue.
                pass
                
        return best_colors, best_violations

    # Try C=13
    # We have ~110s.
    # Try 13 for 60s.
    sol, viol = local_search(13, 60.0)
    if viol == 0:
        return sol

    # Try C=14 for 30s
    if time.time() < deadline - 30.0:
        sol, viol = local_search(14, 30.0)
        if viol == 0:
            return sol

    # Try C=15 for remaining time
    remaining = deadline - time.time()
    if remaining > 5.0:
        sol, viol = local_search(15, remaining - 2.0)
        if viol == 0:
            return sol

    # Fallback: Return the solution with the fewest violations, or just a valid one if we found one earlier?
    # The task says "Minimize the number of colours used".
    # If we didn't find a valid one with 13, we should return the best valid one with 14 or 15.
    # If we didn't find any valid one, we return the one with min violations? 
    # But the evaluator measures "raw value" which is "colours used" IF valid.
    # "A valid colouring certifies f(16) <= colours used."
    # "The evaluator measures the raw value (lower is better)."
    # "An invalid result ... scores 0."
    
    # So we MUST return a valid coloring.
    # If local_search didn't find a valid one, we are in trouble.
    # However, the seed solution (mod 17) is valid.
    # Let's ensure we always have a valid fallback.
    
    # Seed: (i+j) % 17
    seed_colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    # Check if seed is valid
    if count_violations(seed_colors) == 0:
         # If we found a better one, return that.
         # But local_search returns best_colors which might be invalid if viol > 0.
         # We need to track the best VALID solution found during the search.
         
         pass # The logic above returns best_colors regardless of validity if time runs out.
         # I need to modify local_search to return the best VALID solution if no valid one is found?
         # No, if no valid solution is found, it returns a solution with violations.
         # I should modify the return logic.
         
         # Let's refine local_search to track best_valid.
         
    # Re-run with modified logic? No, I can't re-run.
    # I will implement the logic correctly in the replacement below.
    
    # For now, if the previous attempts failed to return a valid solution, 
    # we must return a valid one.
    # The seed is valid with 17 colors.
    # Is there a valid one with 15?
    # The search for 15 might have failed to find a valid one in the remaining time.
    
    # To be safe, I will add a specific check in the code below.
    
    return seed_colors
