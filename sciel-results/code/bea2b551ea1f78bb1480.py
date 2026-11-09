import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    d = 7
    n = 1 << d
    
    # Pre-compute all edges and their neighbors (for C4 checking)
    # Edge representation: (u, v) with u < v
    # For an edge (u, v) along dimension i, a C4 is formed if there is another dimension j
    # such that edges (u^j, v^j), (u, u^j), and (v, v^j) are all present.
    # Actually, simpler: An edge e=(u,v) completes a C4 if there exists a dimension j != i
    # such that the other 3 edges of the face are in the set.
    
    all_edges = []
    edge_to_idx = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(all_edges)
                all_edges.append((u, v, i))
                edge_to_idx[(u, v)] = idx
    
    num_edges = len(all_edges)
    
    # For fast C4 check:
    # If we add edge (u, v, i), we must check for each j != i:
    #   Let u' = u ^ (1<<j), v' = v ^ (1<<j)
    #   The face is (u, v, u', v'). The edges are:
    #   (u, v) [current], (u', v') [opposite], (u, u') [side 1], (v, v') [side 2]
    #   If (u', v'), (u, u'), (v, v') are all in `chosen`, then adding (u, v) creates a C4.
    
    # We will maintain a set of chosen edge indices for O(1) lookup.
    # And a list of chosen edges for output.
    
    best_score = -1
    best_edges = []
    
    # Helper to check if adding edge_idx creates a C4 given current chosen set
    # chosen_set: set of edge indices
    # We need to map (u, v) to index quickly.
    # Since edge_to_idx is a dict of tuples, it's O(1) average.
    
    def is_valid_addition(edge_idx, chosen_set):
        u, v, i = all_edges[edge_idx]
        for j in range(d):
            if j == i:
                continue
            u_prime = u ^ (1 << j)
            v_prime = v ^ (1 << j)
            
            # Check the 3 other edges
            # Edge (u, u_prime)
            e1 = (u, u_prime) if u < u_prime else (u_prime, u)
            idx1 = edge_to_idx.get(e1)
            if idx1 is None or idx1 not in chosen_set:
                continue
            
            # Edge (v, v_prime)
            e2 = (v, v_prime) if v < v_prime else (v_prime, v)
            idx2 = edge_to_idx.get(e2)
            if idx2 is None or idx2 not in chosen_set:
                continue
            
            # Edge (u_prime, v_prime)
            e3 = (u_prime, v_prime) if u_prime < v_prime else (v_prime, u_prime)
            idx3 = edge_to_idx.get(e3)
            if idx3 is None or idx3 not in chosen_set:
                continue
            
            # All 3 exist, so adding edge_idx creates a C4
            return False
        return True

    # Helper to check if removing edge_idx breaks any C4?
    # Not needed for greedy add, but for swap we might need to know if removal is safe.
    # Removal is always safe in terms of validity, but we want to maximize count.
    
    # Strategy:
    # 1. Randomized Greedy:
    #    - Shuffle edges
    #    - Iterate, add if valid
    # 2. Local Search / Swap:
    #    - Try to swap a chosen edge with an unchosen edge to increase count?
    #    - Actually, simple greedy might get stuck.
    #    - Better: Try to add more edges by "relaxing" some constraints?
    #    - Alternative: Randomized Restart with different seeds.
    #    - Even better: "Remove and Re-add" heuristic.
    
    # Let's start with multiple randomized greedy passes.
    # The number of edges is 448. Greedy is fast.
    # 448 * 7 checks * 120s. 
    # Check is ~7 dict lookups. Very fast.
    # We can do thousands of restarts.
    
    for seed in range(10000):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        chosen_set = set()
        chosen_list = []
        
        # Shuffle indices
        indices = list(range(num_edges))
        random.shuffle(indices)
        
        for idx in indices:
            if is_valid_addition(idx, chosen_set):
                chosen_set.add(idx)
                chosen_list.append(idx)
        
        # Try to improve with simple local search:
        # Look for unchosen edges that become valid if we remove one specific chosen edge?
        # This is complex. 
        # Simpler improvement: 
        # Sometimes greedy order matters. 
        # Let's try a "repair" phase:
        # Check if any unchosen edge can be added now? No, we already tried all.
        
        # Let's try a different heuristic:
        # Instead of pure random, try to prioritize edges that are "harder" to place?
        # Or just rely on many random restarts.
        
        score = len(chosen_list)
        if score > best_score:
            best_score = score
            best_edges = [all_edges[i][:2] for i in chosen_list]
            # Update best_edges to list of [u, v]
            
        # Optional: Local Swap Search
        # Try to swap one edge out and two in? 
        # Or just one out and one in (to maintain count, but maybe allow adding more later?)
        # Actually, if we remove an edge, we might be able to add 2 others.
        # Let's try a simple hill climbing:
        # 1. Pick a random chosen edge to remove.
        # 2. Try to add as many unchosen edges as possible in random order.
        # 3. If count increases, keep it.
        
        # This might be expensive. Let's see how much time we have.
        # Greedy is ~0.01s per restart? 
        # 10000 restarts * 0.01 = 100s. 
        # So we have little time for local search per restart.
        # Let's limit restarts to ~5000 and add a light local search.
        
        # Light Local Search:
        # Try to add edges that were skipped due to conflicts, by removing the conflicting edges?
        # Too complex.
        
        # Alternative: 
        # Use a deterministic ordering but with random perturbations?
        # No, random shuffle is good.
        
        # Let's check time usage.
        # If we are fast, we can do more.
        
    return best_edges

# Quick sanity check in head:
# Q7 has 448 edges.
# Max independent set of C4s in line graph?
# 304 is best known.
# Greedy usually gets ~300-310 on small hypercubes with randomization.
# Random restarts should find 304+ easily.

# Let's refine the local search to be more effective if time permits.
# Actually, the prompt says "Maximize". 
# A better strategy might be to use a SAT solver or ILP? 
# NumPy/SciPy available. 
# ILP with 448 variables and constraints for each C4.
# Number of C4s in Q7: 
# Faces of Q7. 
# Number of 2D faces = C(7,2) * 2^(7-2) = 21 * 32 = 672 faces.
# Each face has 4 edges. 
# Constraint: sum of 4 edges in face <= 3.
# Maximize sum of x_e.
# This is a 0-1 IP.
# scipy.optimize.milp can solve this.
# 448 variables, 672 constraints.
# This is small enough for a solver to potentially find the optimal or near-optimal quickly.
# However, milp might be slow or not available in all environments? 
# "Python standard library, NumPy and SciPy are available."
# scipy.optimize.milp is available in SciPy >= 1.7.
# Let's try MILP first, with a timeout. If it fails or times out, fallback to greedy.

def construct_milp():
    from scipy.optimize import milp, LinearConstraint, Bounds
    import numpy as np
    
    d = 7
    n = 1 << d
    
    all_edges = []
    edge_to_idx = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(all_edges)
                all_edges.append((u, v, i))
                edge_to_idx[(u, v)] = idx
                
    num_edges = len(all_edges)
    
    # Build constraints for each face
    # Face defined by two dimensions i, j and a base vertex u (with bits i, j = 0)
    # Vertices: u, u^i, u^j, u^i^j
    # Edges:
    # (u, u^i), (u^j, u^i^j) [dimension i]
    # (u, u^j), (u^i, u^i^j) [dimension j]
    
    A_rows = []
    b_upper = []
    b_lower = []
    
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all combinations of other bits
            # There are 2^(d-2) such faces for each pair (i,j)
            mask = 0
            for k in range(d):
                if k != i and k != j:
                    mask |= (1 << k)
                    
            # Iterate over all 2^(d-2) assignments for the other bits
            # We can just iterate u from 0 to n-1 and check if u has 0 in i and j
            for u in range(n):
                if (u & (1 << i)) == 0 and (u & (1 << j)) == 0:
                    v1 = u ^ (1 << i)
                    v2 = u ^ (1 << j)
                    v3 = u ^ (1 << i) ^ (1 << j)
                    
                    e1 = edge_to_idx.get((u, v1))
                    e2 = edge_to_idx.get((v2, v3))
                    e3 = edge_to_idx.get((u, v2))
                    e4 = edge_to_idx.get((v1, v3))
                    
                    if e1 is None or e2 is None or e3 is None or e4 is None:
                        continue # Should not happen
                    
                    row = np.zeros(num_edges)
                    row[e1] = 1
                    row[e2] = 1
                    row[e3] = 1
                    row[e4] = 1
                    A_rows.append(row)
                    
                    b_upper.append(3)
                    b_lower.append(0)
                    
    A = np.array(A_rows)
    b_ub = np.array(b_upper)
    b_lb = np.array(b_lower)
    
    c = -np.ones(num_edges) # Minimize -sum(x)
    
    constraints = LinearConstraint(A, b_lb, b_ub)
    integrality = np.ones(num_edges)
    bounds = Bounds(0, 1)
    
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.success:
            # res.x is the solution
            # Check if it's integer (should be)
            x = np.round(res.x)
            # Filter edges where x[i] > 0.5
            edges = []
            for i in range(num_edges):
                if x[i] > 0.5:
                    u, v, _ = all_edges[i]
                    edges.append([u, v])
            return edges
    except Exception:
        pass
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try MILP first with a short timeout? 
    # milp doesn't have a simple timeout parameter in older scipy, 
    # but we can run it in a thread or just hope it's fast.
    # 448 vars, 672 constraints is small.
    # Let's try it.
    
    milp_result = construct_milp()
    if milp_result is not None:
        # Validate result
        # Check for C4s
        d = 7
        n = 1 << d
        chosen = set()
        for e in milp_result:
            u, v = e
            if u > v: u, v = v, u
            chosen.add((u, v))
            
        valid = True
        for u in range(n):
            for i in range(d):
                v = u ^ (1 << i)
                if u > v: continue
                if (u, v) not in chosen: continue
                # Check C4s involving (u, v)
                for j in range(d):
                    if j == i: continue
                    u2 = u ^ (1 << j)
                    v2 = v ^ (1 << j)
                    e1 = (u, u2) if u < u2 else (u2, u)
                    e2 = (v, v2) if v < v2 else (v2, v)
                    e3 = (u2, v2) if u2 < v2 else (v2, u2)
                    if e1 in chosen and e2 in chosen and e3 in chosen:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if valid:
            return milp_result
    
    # Fallback to Greedy if MILP fails or is invalid
    # (MILP should be valid by construction, but safety check)
    
    # Use the greedy logic from before
    d = 7
    n = 1 << d
    
    all_edges = []
    edge_to_idx = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(all_edges)
                all_edges.append((u, v, i))
                edge_to_idx[(u, v)] = idx
                
    num_edges = len(all_edges)
    
    def is_valid_addition(edge_idx, chosen_set):
        u, v, i = all_edges[edge_idx]
        for j in range(d):
            if j == i:
                continue
            u_prime = u ^ (1 << j)
            v_prime = v ^ (1 << j)
            
            e1 = (u, u_prime) if u < u_prime else (u_prime, u)
            idx1 = edge_to_idx.get(e1)
            if idx1 is None or idx1 not in chosen_set:
                continue
            
            e2 = (v, v_prime) if v < v_prime else (v_prime, v)
            idx2 = edge_to_idx.get(e2)
            if idx2 is None or idx2 not in chosen_set:
                continue
            
            e3 = (u_prime, v_prime) if u_prime < v_prime else (v_prime, u_prime)
            idx3 = edge_to_idx.get(e3)
            if idx3 is None or idx3 not in chosen_set:
                continue
            
            return False
        return True

    best_score = -1
    best_edges = []
    
    for seed in range(5000):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        chosen_set = set()
        chosen_list = []
        
        indices = list(range(num_edges))
        random.shuffle(indices)
        
        for idx in indices:
            if is_valid_addition(idx, chosen_set):
                chosen_set.add(idx)
                chosen_list.append(idx)
        
        score = len(chosen_list)
        if score > best_score:
            best_score = score
            best_edges = [all_edges[i][:2] for i in chosen_list]
            
    return best_edges

# Keep the original entry point name
# The function above is named construct, which matches.
