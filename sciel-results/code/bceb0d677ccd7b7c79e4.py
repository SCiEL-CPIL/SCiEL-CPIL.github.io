import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    d = 7
    n = 1 << d
    num_edges = d * (1 << (d - 1))  # 448
    
    # 1. Precompute Edge-Index Mapping
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1
                
    assert idx == num_edges
    
    # 2. Precompute Face Constraints
    # For each edge, store the list of faces it belongs to.
    # For each face, store the 4 edge indices.
    
    faces = []
    edge_to_faces = [[] for _ in range(num_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            for u in range(n):
                if (u & (1 << i)) == 0 and (u & (1 << j)) == 0:
                    v1 = u
                    v2 = u ^ (1 << i)
                    v3 = u ^ (1 << j)
                    v4 = u ^ (1 << i) ^ (1 << j)
                    
                    e1 = (min(v1, v2), max(v1, v2))
                    e2 = (min(v1, v3), max(v1, v3))
                    e3 = (min(v2, v4), max(v2, v4))
                    e4 = (min(v3, v4), max(v3, v4))
                    
                    ie1 = edge_to_idx[e1]
                    ie2 = edge_to_idx[e2]
                    ie3 = edge_to_idx[e3]
                    ie4 = edge_to_idx[e4]
                    
                    face_indices = (ie1, ie2, ie3, ie4)
                    faces.append(face_indices)
                    
                    for ie in face_indices:
                        edge_to_faces[ie].append(len(faces) - 1)
                        
    num_faces = len(faces)
    
    # 3. Simulated Annealing on Hitting Set (Edges to Remove)
    # State: removed_mask (boolean array of size num_edges)
    # Objective: Minimize (num_removed + penalty * num_unhit_faces)
    
    # Precompute face membership for fast updates
    # face_hits[f] = number of edges in face f that are in removed_mask
    # unhit_count = number of faces where face_hits[f] == 0
    
    def initial_state(seed):
        random.seed(seed)
        # Start with a random subset of edges to remove
        # Density around 30%? 448 * 0.3 ~ 134 removed. 448-134 = 314 selected.
        # Best known is 304 selected, so ~144 removed.
        # Let's start with a random set of size ~150
        removed = np.zeros(num_edges, dtype=bool)
        num_to_remove = random.randint(130, 170)
        indices = np.random.choice(num_edges, num_to_remove, replace=False)
        removed[indices] = True
        
        # Calculate initial metrics
        face_hits = np.zeros(num_faces, dtype=int)
        for f_idx, f_edges in enumerate(faces):
            count = 0
            for e in f_edges:
                if removed[e]:
                    count += 1
            face_hits[f_idx] = count
            
        unhit_count = np.sum(face_hits == 0)
        num_removed = np.sum(removed)
        
        return removed, face_hits, num_removed, unhit_count

    def get_cost(num_removed, unhit_count, penalty):
        return num_removed + penalty * unhit_count

    def sa_search(seed, max_steps, T_start, T_end, cooling):
        random.seed(seed)
        removed, face_hits, num_removed, unhit_count = initial_state(seed)
        
        penalty = 10.0 # Penalty for unhit faces
        current_cost = get_cost(num_removed, unhit_count, penalty)
        
        best_removed = removed.copy()
        best_num_removed = num_removed
        best_unhit = unhit_count
        best_cost = current_cost
        
        T = T_start
        cooling_rate = (T_end / T_start) ** (1.0 / max_steps)
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            # Pick a random edge to toggle
            e = random.randint(0, num_edges - 1)
            
            old_removed_val = removed[e]
            
            # Update face hits for faces containing e
            # We need to calculate the change in unhit_count and num_removed
            
            delta_num_removed = 1 if not old_removed_val else -1
            
            # Calculate delta_unhit
            # For each face f containing e:
            # If we add e (old_removed_val == False):
            #   face_hits[f] increases by 1.
            #   If face_hits[f] was 0, it becomes 1 -> unhit_count decreases by 1.
            #   If face_hits[f] > 0, unhit_count unchanged.
            # If we remove e (old_removed_val == True):
            #   face_hits[f] decreases by 1.
            #   If face_hits[f] was 1, it becomes 0 -> unhit_count increases by 1.
            #   If face_hits[f] > 1, unhit_count unchanged.
            
            delta_unhit = 0
            for f_idx in edge_to_faces[e]:
                if not old_removed_val:
                    # Adding e
                    if face_hits[f_idx] == 0:
                        delta_unhit -= 1
                else:
                    # Removing e
                    if face_hits[f_idx] == 1:
                        delta_unhit += 1
            
            new_num_removed = num_removed + delta_num_removed
            new_unhit = unhit_count + delta_unhit
            new_cost = get_cost(new_num_removed, new_unhit, penalty)
            
            delta_cost = new_cost - current_cost
            
            # Acceptance criterion
            if delta_cost < 0 or (T > 0 and random.random() < np.exp(-delta_cost / T)):
                # Accept move
                # Apply changes
                removed[e] = not old_removed_val
                num_removed = new_num_removed
                unhit_count = new_unhit
                current_cost = new_cost
                
                # Update face_hits
                for f_idx in edge_to_faces[e]:
                    if not old_removed_val:
                        face_hits[f_idx] += 1
                    else:
                        face_hits[f_idx] -= 1
                
                # Update best if improved (only consider valid solutions for best tracking? 
                # Or just track lowest cost? 
                # The goal is to minimize removed edges among VALID solutions (unhit=0).
                # If we find a valid solution with fewer removed edges, it's better.
                # If we find an invalid solution, it's not a candidate for final answer, 
                # but might lead to a better valid one.
                if unhit_count == 0 and num_removed < best_num_removed:
                    best_num_removed = num_removed
                    best_unhit = unhit_count
                    best_removed = removed.copy()
                    best_cost = current_cost
                elif unhit_count == 0 and num_removed == best_num_removed and current_cost < best_cost:
                    # Tie-breaker: lower cost (though same removed count implies same cost if unhit=0)
                    best_removed = removed.copy()
                    best_cost = current_cost
            else:
                # Reject move
                pass
            
            T *= cooling_rate
            
        # Final Greedy Reduction Pass
        # Try to remove edges from the best_removed set if they don't cause any face to become unhit
        # This is a local search for a minimal hitting set of the same size or smaller?
        # Actually, if we have a valid hitting set, we can try to remove an edge e if all faces 
        # containing e have at least one OTHER edge in the hitting set.
        
        # We apply this to the best found valid solution
        if best_unhit == 0:
            # Copy to work on
            curr_removed = best_removed.copy()
            curr_face_hits = np.zeros(num_faces, dtype=int)
            for f_idx, f_edges in enumerate(faces):
                count = 0
                for e in f_edges:
                    if curr_removed[e]:
                        count += 1
                curr_face_hits[f_idx] = count
            
            # Try to remove edges
            # Iterate until no more edges can be removed
            changed = True
            while changed:
                changed = False
                # Randomize order to avoid bias
                indices = np.arange(num_edges)
                random.shuffle(indices)
                for e in indices:
                    if curr_removed[e]:
                        # Check if removing e keeps all faces hit
                        safe_to_remove = True
                        for f_idx in edge_to_faces[e]:
                            if curr_face_hits[f_idx] == 1:
                                safe_to_remove = False
                                break
                        if safe_to_remove:
                            curr_removed[e] = False
                            for f_idx in edge_to_faces[e]:
                                curr_face_hits[f_idx] -= 1
                            changed = True
                            best_num_removed -= 1
                            best_removed = curr_removed.copy()
        
        return best_removed, best_num_removed

    best_removed = np.zeros(num_edges, dtype=bool)
    best_num_removed = num_edges
    
    seed = 0
    # Run multiple SA iterations with different parameters
    while time.time() < deadline - 10:
        # Parameters for SA
        # We want to explore the space of hitting sets
        # Start T high, cool down
        T_start = 10.0
        T_end = 0.01
        max_steps = 5000
        
        # Occasionally try longer runs or different T
        if seed % 5 == 0:
            max_steps = 20000
            T_start = 5.0
        
        removed, num_removed = sa_search(seed, max_steps, T_start, T_end, 0.99)
        seed += 1
        
        if num_removed < best_num_removed:
            best_num_removed = num_removed
            best_removed = removed.copy()
            
        # If we reach the theoretical lower bound or a very good number, we can stop or tighten
        # Best known selected is 304, so removed = 448 - 304 = 144.
        # If we find 144, we are at best known.
        # Upper bound on selected is 308, so removed >= 448 - 308 = 140.
        if best_num_removed <= 140:
            break

    # Convert best_removed to list of selected edges
    result = []
    for i in range(num_edges):
        if not best_removed[i]:
            result.append(list(idx_to_edge[i]))
            
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
