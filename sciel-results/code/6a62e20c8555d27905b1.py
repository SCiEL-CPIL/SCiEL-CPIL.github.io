import numpy as np
import time
import sys
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    d = 7
    n = 1 << d
    num_edges = d * (1 << (d - 1))  # 448
    
    # 1. Precompute edges
    idx_to_edge = []
    edge_to_idx = {}
    idx = 0
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1
                
    # 2. Precompute faces and mappings
    seen_faces = set()
    face_edge_list = []
    
    for i in range(d):
        for j in range(i+1, d):
            for u in range(n):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))
                
                idxs = [edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]]
                key = frozenset(idxs)
                
                if key not in seen_faces:
                    seen_faces.add(key)
                    face_edge_list.append(idxs)
                    
    num_faces = len(face_edge_list)
    
    # Precompute edge -> faces mapping
    edge_to_faces = [[] for _ in range(num_edges)]
    for f_idx, face in enumerate(face_edge_list):
        for e in face:
            edge_to_faces[e].append(f_idx)
            
    # State
    selected = np.ones(num_edges, dtype=np.uint8)
    face_counts = np.zeros(num_faces, dtype=np.int32)
    
    # Initialize counts
    for e in range(num_edges):
        for f_idx in edge_to_faces[e]:
            face_counts[f_idx] += 1
            
    best_count = np.sum(selected)
    best_selected = selected.copy()
    
    rng = np.random.default_rng(42)
    
    # We will run a loop of "Repair" steps.
    # In each step:
    # 1. Find a violated face (count == 4).
    # 2. Pick a random edge from that face to "fix" (remove it).
    # 3. Use Max-Flow to find the minimum set of edges to remove such that all faces containing the removed edge are satisfied?
    # Actually, a better Max-Flow formulation for C4-free subgraph:
    # We want to remove minimum edges to make it C4-free.
    # This is equivalent to finding a Maximum C4-Free Subgraph.
    # This is NP-hard, but we can approximate.
    
    # Alternative Strategy:
    # The "Repair" idea:
    # If we have a C4, we must remove at least one edge.
    # If we remove edge e, we must ensure no other C4 is formed? No, removing edges only helps.
    # So if we remove one edge from a C4, that C4 is broken.
    # But other C4s might still be violated.
    #
    # Let's try a different Max-Flow application:
    # We can model the problem as a Maximum Weight Independent Set on the "Face Graph"?
    # No, that's not right.
    
    # Let's stick to the "Repair" but make it smarter.
    # Current state: Selected edges.
    # Violated faces: V = {f | count[f] == 4}
    # If V is empty, we are done (or locally optimal).
    # If V is not empty, we must remove edges.
    # Which edge to remove?
    # Removing edge e breaks all faces in edge_to_faces[e] that are violated.
    # We want to remove edges such that we minimize the number of removals, or maximize the "potential" of the remaining graph.
    #
    # A simple heuristic: Remove the edge that is in the most violated faces.
    # This is greedy.
    #
    # To use Max-Flow:
    # We can formulate a flow problem where we try to "route" the violations away?
    # Or perhaps use Max-Flow to find a minimum vertex cover in the "conflict graph" of violated faces?
    # The conflict graph has vertices = violated faces.
    # Two faces are connected if they share an edge.
    # We want to remove a set of edges such that every violated face has at least one edge removed.
    # This is a Hitting Set problem on the violated faces.
    # Since the number of violated faces might be large, we can't solve Hitting Set exactly.
    # But we can find a Minimum Vertex Cover in the "Edge-Face Incidence Graph" restricted to violated faces?
    # No, we are selecting edges to remove.
    # Let G_viol = (E_viol, F_viol) where E_viol are edges in violated faces, F_viol are violated faces.
    # We want to select a subset R of E_viol such that every f in F_viol contains at least one edge in R.
    # This is a Set Cover / Hitting Set problem.
    #
    # Approximation: Greedy Hitting Set.
    # Pick the edge that hits the most violated faces.
    # Remove it.
    # Update counts.
    # Repeat.
    
    # This is a structural change: from Local Search (Swaps) to Greedy Repair (Hitting Set).
    
    max_repairs = 5000
    repair_count = 0
    
    while repair_count < max_repairs:
        if time.time() > deadline - 5:
            break
            
        # Find violated faces
        violated_faces = np.where(face_counts == 4)[0]
        
        if len(violated_faces) == 0:
            # No violated faces. We have a valid C4-free set.
            # Can we add edges?
            # Check if any edge is addable.
            found_addable = False
            for e in range(num_edges):
                if selected[e] == 0:
                    can_add = True
                    for f_idx in edge_to_faces[e]:
                        if face_counts[f_idx] == 3:
                            can_add = False
                            break
                    if can_add:
                        selected[e] = 1
                        for f_idx in edge_to_faces[e]:
                            face_counts[f_idx] += 1
                        found_addable = True
                        # Don't break, keep adding if possible?
                        # Actually, if we add, we might create new violations?
                        # No, adding an edge increases counts.
                        # If count was 2, it becomes 3. Not violated.
                        # If count was 3, it becomes 4. Violated.
                        # So we should only add if it doesn't create a violation.
                        # My check `face_counts[f_idx] == 3` ensures it becomes 4?
                        # Wait.
                        # If count is 3, adding makes it 4 (Violated).
                        # So we should NOT add if any face has count 3.
                        # So `can_add` logic is correct: if any face has count 3, we can't add.
                        # If all faces have count < 3, we can add.
                        # If we add, counts become < 4. So no new violations.
                        # So we can greedily add all possible edges.
                        
                        # Let's restart the scan to find more addable edges?
                        # Or just continue.
                        pass
            
            if not found_addable:
                # We are at a local optimum (maximal C4-free set).
                # Break to try a different strategy?
                # Or just break.
                break
            
            # If we added edges, we might have new addable edges?
            # No, adding edges only increases counts, making it harder to add more.
            # So we can scan once more?
            # Actually, if we add edge e, it might make other edges un-addable?
            # No, it makes them MORE blocked.
            # So if we scan and find no addable, we are done.
            
            # However, we might have missed some in the previous pass?
            # The loop `for e in range(num_edges)` scans all.
            # If we add one, we don't re-scan the rest immediately.
            # But adding one edge doesn't make other edges addable.
            # So one pass is sufficient to find ALL addable edges?
            # Wait.
            # If I add edge e1, face f1 count goes 2->3.
            # Edge e2 is in f1. e2 was addable?
            # If e2 was addable, it meant all faces containing e2 had count < 3.
            # So f1 had count < 3.
            # After adding e1, f1 has count 3.
            # So e2 is NO LONGER addable.
            # So adding edges reduces the set of addable edges.
            # So a single pass that adds edges is fine, but we might add an edge that blocks others.
            # Greedy order matters.
            # To maximize, we should perhaps add edges that block the fewest others?
            # That's a heuristic.
            
            # For now, let's just break if no addable found in a full scan.
            # But we didn't do a full scan if we found one and continued?
            # The code above doesn't break on add.
            # So it scans all 448 edges.
            # If it adds any, it continues.
            # At the end of the scan, if it added any, we should loop again?
            # No, because adding edges only makes it harder.
            # So if we scan all and add some, the next scan will find fewer or none.
            # So we can just loop the `while` again?
            # The `while` checks for violated faces.
            # If no violated faces, it enters the add phase.
            # If it adds, it loops back to check violated faces.
            # Since no violated faces were created, it will enter add phase again.
            # This is inefficient.
            
            # Let's optimize:
            # If no violated faces, perform a full scan for addable edges.
            # Add all of them.
            # Then break (since we can't add more without creating violations?
            # No, we can't add more if we've added all currently addable.
            # But adding one might make another un-addable.
            # So we should add them in an order that minimizes blocking?
            # Or just add all and then check again?
            # Actually, if we add ALL edges that are currently addable, we reach a maximal set.
            # We can't add any more without violating.
            # So we can break.
            
            # Let's restructure the add phase:
            # 1. Find all addable edges.
            # 2. Add them all.
            # 3. Break.
            
            # But wait, if I add e1, e2 becomes un-addable.
            # If I add e2, e1 becomes un-addable.
            # Which one to pick?
            # Greedy: Pick one, add it, then re-evaluate.
            # This is what the current loop does (sort of).
            # Let's just break if we found no addable in a full scan.
            # If we found some, we added them.
            # We should loop again to see if more are addable?
            # No, adding edges only increases counts.
            # So the set of addable edges shrinks.
            # So if we scan all and add all we find, we are done.
            # Wait, if I scan e1, add it. Then scan e2.
            # e2 might have been addable before e1 was added, but now it's not.
            # So I skip e2.
            # This is a valid maximal set.
            # So I can break.
            
            break
            
        # Select edge to remove
        # Heuristic: Remove edge in the most violated faces.
        # Count how many violated faces each edge is in.
        # Only consider edges that are in at least one violated face.
        
        # Build a list of (edge, count_violated_faces)
        # We can do this efficiently.
        
        # For each violated face, increment count for its edges.
        # But we only need the max.
        
        # Let's just pick a random violated face and a random edge in it?
        # That's simple but might be slow to converge.
        
        # Better: Pick the edge that is in the most violated faces.
        
        # Create a temporary array for counts of violated faces per edge.
        # This is O(V * 4) where V is number of violated faces.
        
        edge_viol_counts = np.zeros(num_edges, dtype=np.int32)
        for f_idx in violated_faces:
            for e in face_edge_list[f_idx]:
                edge_viol_counts[e] += 1
                
        # Find edge with max count > 0
        # If all are 0, something is wrong.
        
        max_count = np.max(edge_viol_counts)
        if max_count == 0:
            # Should not happen if violated_faces is not empty
            break
            
        # Pick random among those with max count
        candidates = np.where(edge_viol_counts == max_count)[0]
        e_rem = candidates[rng.integers(0, len(candidates))]
        
        # Remove edge
        selected[e_rem] = 0
        for f_idx in edge_to_faces[e_rem]:
            face_counts[f_idx] -= 1
            
        repair_count += 1
        
        # Update best
        current_count = np.sum(selected)
        if current_count > best_count:
            best_count = current_count
            best_selected = selected.copy()
            
    # Convert best_selected to list of edges
    result = []
    if best_selected is not None:
        for i in range(num_edges):
            if best_selected[i] == 1:
                result.append(list(idx_to_edge[i]))
    else:
        # Fallback
        for i in range(num_edges):
            if selected[i] == 1:
                result.append(list(idx_to_edge[i]))
                
    return result

