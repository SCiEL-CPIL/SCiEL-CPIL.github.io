import numpy as np
import time
import random
import itertools

def construct():
    start_time = time.time()
    timeout = 145  # Safety margin for 150s limit
    
    n = 16
    num_edges = 120
    num_k4 = 1820
    
    # Precompute edge indices for each K4
    edge_idx = {}
    idx = 0
    for v1 in range(n):
        for v2 in range(v1 + 1, n):
            edge_idx[(v1, v2)] = idx
            idx += 1
            
    # Generate all K4s and their edge indices
    vertices = list(range(n))
    k4_list = list(itertools.combinations(vertices, 4))
    
    k4_edges = []
    for k4 in k4_list:
        v1, v2, v3, v4 = k4
        e1 = edge_idx[(v1, v2)]
        e2 = edge_idx[(v1, v3)]
        e3 = edge_idx[(v1, v4)]
        e4 = edge_idx[(v2, v3)]
        e5 = edge_idx[(v2, v4)]
        e6 = edge_idx[(v3, v4)]
        k4_edges.append([e1, e2, e3, e4, e5, e6])
        
    k4_edges = np.array(k4_edges, dtype=np.int32)  # shape (1820, 6)
    
    # Precompute affected K4s for each edge
    edge_to_k4s = [[] for _ in range(num_edges)]
    for i in range(num_k4):
        for e in k4_edges[i]:
            edge_to_k4s[e].append(i)
    
    def count_bad(coloring):
        """Count number of K4s with fewer than 5 distinct colors."""
        bad_count = 0
        for i in range(num_k4):
            c = coloring[k4_edges[i]]
            if len(set(c)) < 5:
                bad_count += 1
        return bad_count
    
    def run_sa(k, time_limit, initial_coloring=None):
        """Simulated annealing with fast delta updates."""
        if initial_coloring is not None:
            coloring = initial_coloring.copy()
        else:
            coloring = np.random.randint(0, k, size=num_edges)
        
        # Initial bad mask using vectorized approach
        bad_mask = np.zeros(num_k4, dtype=bool)
        for i in range(num_k4):
            c = coloring[k4_edges[i]]
            if len(set(c)) < 5:
                bad_mask[i] = True
        current_bad = int(np.sum(bad_mask))
        best_coloring = coloring.copy()
        best_bad = current_bad
        
        if current_bad == 0:
            return coloring, 0
        
        temp = 1.0
        cooling = 0.9995
        end_time = time.time() + time_limit
        no_improve_count = 0
        
        while time.time() < end_time:
            # Pick random edge
            e_idx = np.random.randint(num_edges)
            old_color = coloring[e_idx]
            
            # Pick new color - try different strategies
            if random.random() < 0.3:
                # Try a color that appears in one of the affected K4s
                affected = edge_to_k4s[e_idx]
                if affected:
                    # Get colors from a random affected K4
                    sample_k4 = random.choice(affected)
                    possible_colors = list(set(coloring[k4_edges[sample_k4]]))
                    if len(possible_colors) > 0:
                        new_color = random.choice(possible_colors)
                    else:
                        new_color = np.random.randint(k)
                else:
                    new_color = np.random.randint(k)
            else:
                new_color = np.random.randint(k)
            
            if new_color == old_color:
                continue
            
            # Identify affected K4s
            affected = edge_to_k4s[e_idx]
            if not affected:
                continue
            
            # Calculate delta by checking affected K4s
            old_bad_in_affected = int(np.sum([bad_mask[i] for i in affected]))
            
            # Apply new color temporarily
            coloring[e_idx] = new_color
            
            new_bad_in_affected = 0
            for i in affected:
                c = coloring[k4_edges[i]]
                if len(set(c)) < 5:
                    new_bad_in_affected += 1
                    bad_mask[i] = True
                else:
                    bad_mask[i] = False
            
            delta = new_bad_in_affected - old_bad_in_affected
            
            # Accept or reject
            if delta < 0 or (temp > 0.001 and random.random() < np.exp(-delta / temp)):
                current_bad += delta
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_coloring = coloring.copy()
                    no_improve_count = 0
                    if best_bad == 0:
                        return best_coloring, 0
                else:
                    no_improve_count += 1
            else:
                # Reject: revert coloring and bad_mask
                coloring[e_idx] = old_color
                for i in affected:
                    c = coloring[k4_edges[i]]
                    if len(set(c)) < 5:
                        bad_mask[i] = True
                    else:
                        bad_mask[i] = False
            
            temp *= cooling
            if temp < 0.001:
                temp = 1.0  # Restart with current best
                coloring = best_coloring.copy()
                bad_mask = np.zeros(num_k4, dtype=bool)
                for i in range(num_k4):
                    c = coloring[k4_edges[i]]
                    if len(set(c)) < 5:
                        bad_mask[i] = True
                current_bad = best_bad
                no_improve_count = 0
        
        return best_coloring, best_bad
    
    def polish(coloring, k, time_limit):
        """Local search polish: try to fix bad K4s by changing edge colors."""
        coloring = coloring.copy()
        bad_mask = np.zeros(num_k4, dtype=bool)
        for i in range(num_k4):
            c = coloring[k4_edges[i]]
            if len(set(c)) < 5:
                bad_mask[i] = True
        current_bad = int(np.sum(bad_mask))
        best_coloring = coloring.copy()
        best_bad = current_bad
        
        if current_bad == 0:
            return coloring, 0
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            if current_bad == 0:
                break
            
            # Find a bad K4
            bad_indices = np.where(bad_mask)[0]
            if len(bad_indices) == 0:
                break
            
            bad_k4_idx = random.choice(bad_indices.tolist())
            affected_edges = k4_edges[bad_k4_idx].tolist()
            
            # Try changing each edge in this K4
            improved = False
            for e_idx in affected_edges:
                old_color = coloring[e_idx]
                
                # Try all colors
                for new_color in range(k):
                    if new_color == old_color:
                        continue
                    
                    # Check if this change would fix the bad K4
                    coloring[e_idx] = new_color
                    c = coloring[k4_edges[bad_k4_idx]]
                    if len(set(c)) >= 5:
                        # Check if this doesn't break other K4s too much
                        affected = edge_to_k4s[e_idx]
                        new_bad_count = 0
                        for i in affected:
                            cc = coloring[k4_edges[i]]
                            if len(set(cc)) < 5:
                                new_bad_count += 1
                                bad_mask[i] = True
                            else:
                                bad_mask[i] = False
                        
                        delta = new_bad_count - int(np.sum([bad_mask[i] for i in affected]))
                        # Actually we need to recalculate properly
                        # Let's just check if total bad decreased
                        total_bad = int(np.sum(bad_mask))
                        if total_bad < best_bad:
                            best_bad = total_bad
                            best_coloring = coloring.copy()
                            current_bad = total_bad
                            improved = True
                            if best_bad == 0:
                                return best_coloring, 0
                            break
                        else:
                            # Revert
                            coloring[e_idx] = old_color
                            for i in affected:
                                cc = coloring[k4_edges[i]]
                                if len(set(cc)) < 5:
                                    bad_mask[i] = True
                                else:
                                    bad_mask[i] = False
                
                if improved:
                    break
            
            if not improved:
                # Try random perturbation
                e_idx = np.random.randint(num_edges)
                coloring[e_idx] = np.random.randint(k)
                for i in edge_to_k4s[e_idx]:
                    c = coloring[k4_edges[i]]
                    if len(set(c)) < 5:
                        bad_mask[i] = True
                    else:
                        bad_mask[i] = False
                current_bad = int(np.sum(bad_mask))
        
        return best_coloring, best_bad
    
    # Try to find a solution with 13 colors first
    if time.time() < start_time + 30:
        sol, bad = run_sa(13, 30)
        if bad == 0:
            return sol.tolist()
        # Polish
        if time.time() < start_time + 40:
            sol, bad = polish(sol, 13, 10)
            if bad == 0:
                return sol.tolist()
    
    # Try 14 colors
    if time.time() < start_time + 70:
        sol, bad = run_sa(14, 30)
        if bad == 0:
            return sol.tolist()
        # Polish
        if time.time() < start_time + 80:
            sol, bad = polish(sol, 14, 10)
            if bad == 0:
                return sol.tolist()
    
    # Try 15 colors
    if time.time() < start_time + 110:
        sol, bad = run_sa(15, 30)
        if bad == 0:
            return sol.tolist()
        # Polish
        if time.time() < start_time + 125:
            sol, bad = polish(sol, 15, 15)
            if bad == 0:
                return sol.tolist()
    
    # Fallback: use a known valid coloring with more colors
    # Use (i + j) % 17 which uses 17 colors
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
