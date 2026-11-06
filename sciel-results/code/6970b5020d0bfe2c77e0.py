import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # We want to maximize n. The known bound is 34.
    # We will try n from 34 downwards.
    for n in range(34, 1, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Precomputation for this n
        edges = list(combinations(range(n), 3))
        m = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        quads = list(combinations(range(n), 4))
        pents = list(combinations(range(n), 5))
        n_quads = len(quads)
        n_pents = len(pents)
        
        # Map each edge to the quads and pents it belongs to
        edge_to_quads = [[] for _ in range(m)]
        edge_to_pents = [[] for _ in range(m)]
        
        # Also map quad/pent to its edges for fast candidate selection
        quad_to_edges = [[] for _ in range(n_quads)]
        pent_to_edges = [[] for _ in range(n_pents)]

        for q_idx, q in enumerate(quads):
            q_edges = []
            for e in combinations(q, 3):
                e_idx = edge_to_idx[e]
                edge_to_quads[e_idx].append(q_idx)
                q_edges.append(e_idx)
            quad_to_edges[q_idx] = q_edges

        for p_idx, p in enumerate(pents):
            p_edges = []
            for e in combinations(p, 3):
                e_idx = edge_to_idx[e]
                edge_to_pents[e_idx].append(p_idx)
                p_edges.append(e_idx)
            pent_to_edges[p_idx] = p_edges

        best_violations = float('inf')
        best_colors = None
        
        # Run local search with restarts
        search_deadline = deadline
        
        restart_count = 0
        
        while time.time() < search_deadline:
            restart_count += 1
            
            # Initialization: Random
            colors = np.random.randint(0, 2, m).astype(np.int8)
            
            # Calculate initial counts
            quad_red_counts = np.zeros(n_quads, dtype=np.int32)
            pent_blue_counts = np.zeros(n_pents, dtype=np.int32)
            
            for i in range(m):
                if colors[i] == 0: # Red
                    for q_idx in edge_to_quads[i]:
                        quad_red_counts[q_idx] += 1
                else: # Blue
                    for p_idx in edge_to_pents[i]:
                        pent_blue_counts[p_idx] += 1
                        
            violations = np.sum(quad_red_counts == 4) + np.sum(pent_blue_counts == 10)
            
            if violations == 0:
                return {'n': n, 'colors': colors.tolist()}
                
            if violations < best_violations:
                best_violations = violations
                best_colors = colors.copy()
                
            # Local search
            no_improve_count = 0
            max_no_improve = 2000 # Restart if stuck
            
            while time.time() < search_deadline and no_improve_count < max_no_improve:
                
                # Identify violating structures
                violating_quads = np.where(quad_red_counts == 4)[0]
                violating_pents = np.where(pent_blue_counts == 10)[0]
                
                if len(violating_quads) == 0 and len(violating_pents) == 0:
                    return {'n': n, 'colors': colors.tolist()}
                    
                # Collect candidate edges from violations
                candidate_edges = set()
                for q_idx in violating_quads:
                    candidate_edges.update(quad_to_edges[q_idx])
                for p_idx in violating_pents:
                    candidate_edges.update(pent_to_edges[p_idx])
                
                if not candidate_edges:
                    break # Should not happen
                
                # Convert to list for random sampling
                cand_list = list(candidate_edges)
                
                # Sample a few candidates and evaluate
                num_samples = min(10, len(cand_list))
                samples = random.sample(cand_list, num_samples)
                
                best_delta = float('inf')
                best_edge = -1
                
                for e_idx in samples:
                    old_c = colors[e_idx]
                    
                    delta_v = 0
                    
                    if old_c == 0: # Red -> Blue
                        # Check quads containing e_idx
                        for q_idx in edge_to_quads[e_idx]:
                            if quad_red_counts[q_idx] == 4:
                                delta_v -= 1 # Violation removed
                            # If it was 3, it becomes 2, no change in violation count
                            # If it was 2, it becomes 1, no change
                        # Check pents containing e_idx
                        for p_idx in edge_to_pents[e_idx]:
                            if pent_blue_counts[p_idx] == 9:
                                delta_v += 1 # Violation added (9->10)
                            # If it was 10, it becomes 11? No, max is 10.
                            # Wait, if it was 10, it's already a violation.
                            # Flipping a Blue edge to Red in a Pent with 10 Blue edges:
                            # Blue count becomes 9. Violation removed? 
                            # No, we are flipping Red->Blue here. 
                            # So old_c=0 (Red). 
                            # In a pent, if blue_count was 9, it becomes 10 (Violation Added).
                            # If blue_count was 10, it becomes 11? No, the edge was Red, so it wasn't counted in Blue.
                            # So blue_count is count of Blue edges. 
                            # If old_c=0 (Red), this edge is NOT blue. 
                            # So if we flip it to Blue, blue_count increases by 1.
                            # If it was 9, it becomes 10 (Violation).
                            # If it was 10, it becomes 11 (Still Violation, count doesn't matter for 0/1 violation flag).
                            # Actually, if it was 10, it's already a violation. 
                            # Does adding a blue edge to a pent that already has 10 blue edges create a NEW violation?
                            # No, the pent is already violating. 
                            # But the count changes from 10 to 11. 
                            # The condition is `count == 10`. 
                            # So if it was 10, it was a violation. After flip, it is 11, so NOT a violation?
                            # Wait. A K5^(3) has 10 edges. If all 10 are Blue, it's a violation.
                            # If 11 are Blue? Impossible, there are only 10 edges in a K5.
                            # So blue_count can never be 11.
                            # So if blue_count was 10, the edge MUST be Blue.
                            # But we are flipping a Red edge. So this edge was NOT Blue.
                            # So blue_count CANNOT be 10 if this edge is Red?
                            # Yes. If an edge is Red, it is not Blue. 
                            # So the max blue count in a pent containing a specific Red edge is 9 (the other 9 edges).
                            # So `pent_blue_counts[p_idx]` for a pent containing a Red edge is at most 9.
                            # Therefore, `if pent_blue_counts[p_idx] == 9` is the only case that adds a violation.
                            
                    else: # Blue -> Red
                        # Check quads containing e_idx
                        for q_idx in edge_to_quads[e_idx]:
                            if quad_red_counts[q_idx] == 3:
                                delta_v += 1 # Violation added (3->4)
                        # Check pents containing e_idx
                        for p_idx in edge_to_pents[e_idx]:
                            if pent_blue_counts[p_idx] == 10:
                                delta_v -= 1 # Violation removed (10->9)
                            # Similar logic: if edge is Blue, it is counted in Blue.
                            # If blue_count is 10, it's a violation.
                            # Flipping to Red makes it 9. Violation removed.
                            # Can blue_count be > 10? No.
                            # Can blue_count be < 10 and flipping to Red remove a violation? No.
                            
                    if delta_v < best_delta:
                        best_delta = delta_v
                        best_edge = e_idx
                        
                # If best_delta is still positive, we are increasing violations.
                # We can still accept it to escape local minima, or reject.
                # Let's accept if best_delta < 0 or (best_delta == 0 and random.random() < 0.5) or (best_delta > 0 and random.random() < 0.1)
                
                if best_edge != -1:
                    if best_delta < 0 or (best_delta == 0 and random.random() < 0.5) or (best_delta > 0 and random.random() < 0.05):
                        # Apply move
                        e_idx = best_edge
                        old_c = colors[e_idx]
                        new_c = 1 - old_c
                        colors[e_idx] = new_c
                        
                        if new_c == 0: # Now Red
                            for q_idx in edge_to_quads[e_idx]:
                                quad_red_counts[q_idx] += 1
                            for p_idx in edge_to_pents[e_idx]:
                                pent_blue_counts[p_idx] -= 1
                        else: # Now Blue
                            for q_idx in edge_to_quads[e_idx]:
                                quad_red_counts[q_idx] -= 1
                            for p_idx in edge_to_pents[e_idx]:
                                pent_blue_counts[p_idx] += 1
                                
                        violations += best_delta
                        no_improve_count = 0
                        
                        if violations == 0:
                            return {'n': n, 'colors': colors.tolist()}
                            
                        if violations < best_violations:
                            best_violations = violations
                            best_colors = colors.copy()
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1

    # If no valid coloring found for n > 1, return n=1
    return {'n': 1, 'colors': []}
