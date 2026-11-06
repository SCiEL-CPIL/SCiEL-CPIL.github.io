import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        target_min_deg = n - 39
        
        # Try multiple random restarts for this n
        max_restarts = 200  # NumPy is fast, so we can afford more restarts
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize random graph with p=0.5
            # Use int8 for adjacency matrix to allow efficient arithmetic
            A = np.zeros((n, n), dtype=np.int8)
            # Generate random upper triangle
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            rand_vals = (np.random.rand(n, n) < 0.5).astype(np.int8)
            A = np.where(mask, rand_vals, 0)
            A += A.T
            
            # Local search / repair
            improved = True
            max_steps = 500  # Cap steps per restart to avoid infinite loops
            steps = 0
            while improved and steps < max_steps:
                if time.time() > deadline:
                    break
                steps += 1
                improved = False
                
                # Check C4-free condition: |N(u) \cap N(v)| <= 1 for all u != v
                # This is equivalent to (A @ A)[i][j] <= 1 for all i != j
                # Note: A[i][i] is 0, so (A@A)[i][i] = degree(i).
                # We only care about i != j.
                common = A @ A
                
                # Find violations: common[i][j] > 1 for i != j
                # We need to find indices where common > 1 and i != j
                # np.where returns (rows, cols)
                viol_idx = np.where(common > 1)
                if len(viol_idx[0]) > 0:
                    # Pick a random violation
                    # Exclude diagonal if any (though common[i][i] is deg, which can be > 1, 
                    # but diagonal doesn't form C4 with itself. 
                    # However, our condition "common > 1" includes diagonal.
                    # We must filter out i == j.
                    valid_viol = [(u, v) for u, v in zip(viol_idx[0], viol_idx[1]) if u != v]
                    if not valid_viol:
                        # Only diagonal violations (high degrees), which is fine for C4-free check
                        # But wait, if we only check i!=j, we should have filtered before.
                        # Let's re-filter efficiently.
                        pass 
                    if valid_viol:
                        u, v = random.choice(valid_viol)
                        # u and v share >= 2 common neighbors.
                        # Find common neighbors
                        common_neighbors = np.where((A[u] == 1) & (A[v] == 1))[0]
                        if len(common_neighbors) > 0:
                            w = common_neighbors[random.randint(0, len(common_neighbors)-1)]
                            # Remove edge (u, w)
                            A[u, w] = 0
                            A[w, u] = 0
                            improved = True
                    else:
                        # No off-diagonal violations, but maybe we just hit the diagonal?
                        # If no off-diagonal violations, C4 condition (strong) is met.
                        pass
                else:
                    # No C4 violations (strong condition met)
                    # Check min degree
                    degrees = np.sum(A, axis=1)
                    min_deg = np.min(degrees)
                    
                    if min_deg >= target_min_deg:
                        # Valid graph found
                        edges = []
                        # Extract edges from upper triangle
                        iu, ju = np.where(np.triu(A, k=1) == 1)
                        for i, j in zip(iu, ju):
                            edges.append([int(i), int(j)])
                        return {'n': n, 'edges': edges}
                    else:
                        # Try to add an edge to a low-degree vertex
                        low_v = np.argmin(degrees)
                        # Candidates for low_v: non-neighbors
                        candidates = np.where(A[low_v] == 0)[0]
                        candidates = candidates[candidates != low_v]
                        
                        if len(candidates) > 0:
                            # Shuffle candidates
                            random.shuffle(candidates)
                            added = False
                            for u in candidates:
                                # Check if adding (low_v, u) violates C4 condition
                                # New common neighbors for pairs involving low_v or u will increase.
                                # Specifically, for any x:
                                # |N'(low_v) \cap N'(x)| <= 1
                                # |N'(u) \cap N'(x)| <= 1
                                # And |N'(low_v) \cap N'(u)| <= 1
                                
                                # This check is expensive if done naively for all x.
                                # However, adding edge (low_v, u) increases:
                                # 1. Common neighbors of (low_v, x) by 1 if u in N(x).
                                # 2. Common neighbors of (u, x) by 1 if low_v in N(x).
                                # 3. Common neighbors of (low_v, u) by 1 (they become adjacent, 
                                #    wait, common neighbors of adjacent vertices don't form C4 
                                #    in the "strong condition" sense? 
                                #    The strong condition |N(a) \cap N(b)| <= 1 applies to ALL pairs.
                                #    So yes, it applies to (low_v, u) too.
                                #    Before adding, low_v not in N(u) and u not in N(low_v).
                                #    After adding, u in N(low_v) and low_v in N(u).
                                #    Do they share u or low_v? No, simple graph.
                                #    Do they share other neighbors?
                                #    New intersection = Old Intersection.
                                #    So we just need Old Intersection <= 1.
                                
                                old_common_lv_u = int(np.sum(A[low_v] & A[u]))
                                if old_common_lv_u > 1:
                                    continue
                                    
                                # Check neighbors of low_v
                                # For each x in N(low_v), new common(u, x) = old common(u, x) + 1 (if low_v in N(x), which is true)
                                # So we need old common(u, x) == 0 for all x in N(low_v).
                                # This means N(u) and N(low_v) must be disjoint?
                                # Wait.
                                # If x in N(low_v), then low_v in N(x).
                                # Adding (u, low_v) puts low_v in N(u).
                                # So low_v becomes a common neighbor of u and x.
                                # If u and x already shared a neighbor y, then u-y-x-low_v-u is a C4?
                                # No, the condition is just count of common neighbors.
                                # We need |N'(u) \cap N'(x)| <= 1.
                                # N'(u) = N(u) U {low_v}.
                                # N'(x) = N(x). (x is not low_v, and we didn't add edge to x)
                                # Intersection = (N(u) \cap N(x)) U ({low_v} \cap N(x)).
                                # Since x in N(low_v), low_v in N(x). So {low_v} \cap N(x) = {low_v}.
                                # So new intersection size = |N(u) \cap N(x)| + 1.
                                # We need this <= 1, so |N(u) \cap N(x)| must be 0.
                                # This must hold for ALL x in N(low_v).
                                
                                # Similarly for y in N(u), we need |N(low_v) \cap N(y)| == 0.
                                
                                # This implies N(low_v) and N(u) must be disjoint AND
                                # for all x in N(low_v), N(u) \cap N(x) = empty.
                                # This is very restrictive.
                                
                                # Let's just simulate the change on the matrix copy?
                                # Or check conditions.
                                
                                # Condition 1: N(low_v) and N(u) disjoint?
                                # If x in N(low_v) and x in N(u), then x is in both.
                                # Then for pair (low_v, u), x is common neighbor.
                                # Also for pair (low_v, x), u is in N(low_v) and u in N(x)?
                                # If u in N(x), then u is common neighbor of low_v and x.
                                # This is getting complex.
                                
                                # Faster check:
                                # Create a temporary check for the specific edges affected.
                                # The only pairs whose common neighbor count changes are:
                                # (low_v, u), (low_v, x) for x in N(u), (u, x) for x in N(low_v).
                                # And (low_v, y) for y in N(low_v)? No, N(low_v) changes by adding u.
                                # Pairs (low_v, y) where y in N(low_v):
                                # N'(low_v) = N(low_v) U {u}.
                                # N'(y) = N(y).
                                # Intersection = (N(low_v) \cap N(y)) U ({u} \cap N(y)).
                                # If u in N(y), then +1.
                                # So we need |N(low_v) \cap N(y)| + (1 if u in N(y) else 0) <= 1.
                                
                                # Let's just try to add and re-check the whole matrix?
                                # That's O(n^3) per step. n=47. 47^3 ~ 100k ops.
                                # 500 steps * 100k = 50M ops. In NumPy, this is fast (<1s).
                                # Let's do the "add and check" approach for simplicity and correctness.
                                
                                A[low_v, u] = 1
                                A[u, low_v] = 1
                                
                                # Re-check C4 condition
                                common_new = A @ A
                                # Check off-diagonal violations
                                # We only need to check if any new violation appeared.
                                # But checking all is safer.
                                # Optimize: only check rows/cols involving low_v and u?
                                # No, (x,y) not involving low_v/u is unchanged.
                                # So we only need to check pairs (low_v, k) and (u, k) and (low_v, u).
                                
                                valid = True
                                # Check (low_v, u)
                                if common_new[low_v, u] > 1:
                                    valid = False
                                else:
                                    # Check (low_v, k) for all k
                                    # common_new[low_v, :] > 1 for any k != low_v
                                    # Exclude diagonal
                                    row_lv = common_new[low_v, :].copy()
                                    row_lv[low_v] = 0
                                    if np.any(row_lv > 1):
                                        valid = False
                                    else:
                                        # Check (u, k) for all k
                                        row_u = common_new[u, :].copy()
                                        row_u[u] = 0
                                        if np.any(row_u > 1):
                                            valid = False
                                
                                if valid:
                                    added = True
                                    improved = True
                                    break
                                else:
                                    # Revert
                                    A[low_v, u] = 0
                                    A[u, low_v] = 0
                            
                            if not added:
                                # Stuck. Remove a random edge to break symmetry
                                # Pick a random vertex with high degree
                                if np.any(degrees > 0):
                                    v = random.choice(np.where(degrees > 0)[0])
                                    neighbors = np.where(A[v] == 1)[0]
                                    if len(neighbors) > 0:
                                        u = random.choice(neighbors)
                                        A[v, u] = 0
                                        A[u, v] = 0
                                        improved = True
            # End while loop
        # End restart loop
    # Fallback: return empty graph on 39 vertices
    return {'n': 39, 'edges': []}
