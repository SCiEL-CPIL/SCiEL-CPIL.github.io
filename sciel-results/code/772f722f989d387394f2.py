import time
import random
import math

N = 64

def construct():
        start_time = time.time()
        deadline = start_time + 290
        
        all_points = [(x, y) for x in range(N) for y in range(N)]
        best_set = []
        best_size = 0
        
        def greedy_select_heuristic(points_order, window_size=100):
            selected = []
            dist_from = []
            i = 0
            
            while i < len(points_order) and time.time() < deadline:
                window_end = min(i + window_size, len(points_order))
                best_idx = -1
                best_min_dist = -1
                best_d_p = None
                best_p = None
                
                for j in range(i, window_end):
                    p = points_order[j]
                    px, py = p
                    valid = True
                    d_p = []
                    min_dist = float('inf')
                    
                    for k in range(len(selected)):
                        ax, ay = selected[k]
                        d_sq = (px - ax)**2 + (py - ay)**2
                        d_p.append(d_sq)
                        if d_sq < min_dist:
                            min_dist = d_sq
                    
                    if len(set(d_p)) != len(d_p):
                        continue
                        
                    for k in range(len(selected)):
                        if d_p[k] in dist_from[k]:
                            valid = False
                            break
                            
                    if valid and min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_idx = j
                        best_d_p = d_p
                        best_p = p
                        
                if best_idx != -1:
                    selected.append(best_p)
                    new_dist_set = set()
                    for k in range(len(selected) - 1):
                        new_dist_set.add(best_d_p[k])
                    dist_from.append(new_dist_set)
                    
                    for k in range(len(selected) - 1):
                        dist_from[k].add(best_d_p[k])
                        
                    i = best_idx + 1
                else:
                    i = window_end
                    
            return selected

        # Run multiple restarts
        while time.time() < deadline:
            random.shuffle(all_points)
            current_set = greedy_select_heuristic(all_points)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set
                
        return best_set
