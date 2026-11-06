import numpy as np
import time
import random
import sys
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
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
    # For each face, store the 4 edge indices.
    # For each edge, store the list of faces it belongs to.
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
    
    # Helper: Check if adding edge e is valid
    def is_valid_add(selected, e):
        for f_idx in edge_to_faces[e]:
            f_edges = faces[f_idx]
            count = 0
            for oe in f_edges:
                if oe != e and selected[oe]:
                    count += 1
            if count == 3:
                return False
        return True

    # Helper: Greedy fill - add all possible valid edges
    def greedy_fill(selected):
        changed = True
        while changed:
            changed = False
            for e in range(num_edges):
                if not selected[e]:
                    if is_valid_add(selected, e):
                        selected[e] = True
                        changed = True
        return selected

    # Helper: Max-Flow (Dinic's Algorithm)
    # We model the problem of "removing edge e to allow adding others" as a flow problem.
    # Nodes: 2 * num_edges + 2 (Source, Sink)
    # For each edge i in the hypergraph:
    #   Node 2*i is "edge i present"
    #   Node 2*i+1 is "edge i absent"
    #   
    # Constraints:
    # 1. If edge i is present, it contributes to capacity.
    # 2. Face constraint: NOT (a AND b AND c AND d).
    #    This is equivalent to: If 3 are present, the 4th MUST be absent.
    #    
    # Actually, a simpler formulation for "Maximize selected edges given we must remove e":
    # We want to find a set S of edges to remove (including e) such that the remaining set is valid,
    # and |S| is minimized.
    # This is a Minimum Vertex Cover in a 3-uniform hypergraph (since we fix e, the other 3 in a face
    # trigger a constraint if 3 are present... wait).
    
    # Let's use a simpler specific Max-Flow construction for the "1-removal" case.
    # If we remove edge e, we might be able to add edges.
    # The edges that become "available" are those that were blocked ONLY by e.
    # An edge x is blocked by e if there exists a face F containing both e and x, and the other 2 edges in F are selected.
    # If we remove e, x becomes valid IF no OTHER face containing x has 3 selected edges (excluding x).
    
    # Let's try a different structural approach: **Iterative Deepening on "Holes"**
    # The current local search gets stuck because single flips don't work.
    # A "2-for-1" move: Remove 1 edge, Add 2 edges.
    # To find the best edge to remove:
    # For each edge e in the current solution:
    #   1. Temporarily remove e.
    #   2. Identify all edges x that are now valid (were invalid only because of e).
    #   3. If we can add 2 or more of these x's, we gain.
    #   4. To maximize gain, we want to add as many as possible.
    #   5. This is a Max-Set-Packing problem on the "newly valid" edges, subject to their mutual constraints.
    #   6. Since the number of "newly valid" edges is small (usually < 10), we can solve this exactly with brute force or small Max-Flow.
    
    def try_remove_and_add(selected, e, budget_time):
        # Remove e
        selected[e] = False
        count_before = np.sum(selected)
        
        # Find all edges x that are now valid
        candidates = []
        for x in range(num_edges):
            if not selected[x]:
                if is_valid_add(selected, x):
                    candidates.append(x)
        
        # If we can't add at least 2, it's not a gain (since we removed 1)
        # Actually, we want to maximize the number of added edges.
        # Let's try to add as many as possible.
        # Since candidates are small, we can use a greedy or exhaustive search.
        # Greedy: Sort candidates by "conflict count" (how many other candidates they conflict with) and pick.
        
        # Precompute conflicts among candidates
        # Two candidates x, y conflict if they share a face where the other 2 edges are selected.
        
        # For efficiency, just try to add them one by one in a smart order.
        # Order: Prefer edges that don't conflict with other candidates?
        
        # Let's use a simple recursive backtracking for the small candidate set
        best_added = 0
        best_set = set()
        
        # Sort candidates to help pruning? Maybe by degree?
        # For now, just iterate.
        
        def backtrack(idx, current_selected_count, current_set):
            nonlocal best_added, best_set
            if time.time() > budget_time:
                return
            if current_selected_count > best_added:
                best_added = current_selected_count
                best_set = current_set.copy()
            
            if idx == len(candidates):
                return
            
            # Option 1: Skip candidate idx
            backtrack(idx + 1, current_selected_count, current_set)
            
            # Option 2: Add candidate idx if valid
            x = candidates[idx]
            if is_valid_add(selected, x):
                selected[x] = True
                current_set.add(x)
                backtrack(idx + 1, current_selected_count + 1, current_set)
                selected[x] = False
                current_set.remove(x)
        
        backtrack(0, 0, set())
        
        # Apply the best set found
        for x in best_set:
            selected[x] = True
            
        count_after = np.sum(selected)
        return count_after - count_before # Net gain

    # Initialize with a random greedy start
    def greedy_construct(seed):
        random.seed(seed)
        edges = list(range(num_edges))
        random.shuffle(edges)
        selected = np.zeros(num_edges, dtype=bool)
        
        for e in edges:
            if is_valid_add(selected, e):
                selected[e] = True
                
        # Fill remaining
        greedy_fill(selected)
        return selected

    best_count = 0
    best_selected = np.zeros(num_edges, dtype=bool)
    
    seed = 0
    while time.time() < deadline:
        # 1. Greedy Start
        current_selected = greedy_construct(seed)
        seed += 1
        current_count = np.sum(current_selected)
        
        # 2. Local Improvement via "Remove-1-Add-Many"
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Iterate through all selected edges to see if removing one allows a net gain
            # To speed up, we can randomize the order or check all.
            # Checking all 448 edges * backtracking might be slow if backtracking is deep.
            # But candidates are usually few.
            
            sel_edges = np.where(current_selected)[0]
            random.shuffle(sel_edges)
            
            for e in sel_edges:
                if time.time() > deadline:
                    break
                    
                # Try removing e
                original_selected = current_selected.copy()
                gain = try_remove_and_add(current_selected, e, deadline - 0.1)
                
                if gain > 0:
                    current_count += gain
                    improved = True
                    # If we improved, restart the loop to find more opportunities
                    break
                else:
                    # Revert
                    current_selected = original_selected
                    
        if current_count > best_count:
            best_count = current_count
            best_selected = current_selected.copy()
            
        # Occasionally try a more aggressive perturbation: Remove 2 random edges
        if seed % 5 == 0 and time.time() < deadline - 10:
            perturbed = best_selected.copy()
            sel_edges = np.where(perturbed)[0]
            if len(sel_edges) >= 2:
                e1, e2 = random.sample(list(sel_edges), 2)
                perturbed[e1] = False
                perturbed[e2] = False
                # Greedy fill
                greedy_fill(perturbed)
                if np.sum(perturbed) > best_count:
                    best_count = np.sum(perturbed)
                    best_selected = perturbed

    # Convert best_selected to list of edges
    result = []
    for i in range(num_edges):
        if best_selected[i]:
            result.append(list(idx_to_edge[i]))
            
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
