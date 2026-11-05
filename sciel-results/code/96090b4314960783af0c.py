"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time

# Precompute squared distances for all pairs of points in [0, N)^2
# Points are indexed 0 to N*N-1, corresponding to (i // N, i % N)
points = [(x, y) for x in range(N) for y in range(N)]
num_points = len(points)

# dist[i][j] is squared distance between points[i] and points[j]
dist = [[0] * num_points for _ in range(num_points)]
for i in range(num_points):
    x1, y1 = points[i]
    for j in range(i + 1, num_points):
        x2, y2 = points[j]
        d = (x1 - x2) ** 2 + (y1 - y2) ** 2
        dist[i][j] = d
        dist[j][i] = d

def is_isosceles(i, j, k):
    """Check if points i, j, k form an isosceles triangle."""
    d_ij = dist[i][j]
    d_ik = dist[i][k]
    d_jk = dist[j][k]
    # Degenerate triangles (collinear with one point between others) have d_ij + d_jk = d_ik etc.
    # But squared distances don't add up linearly.
    # Isosceles means at least two sides are equal.
    # Note: If d_ij == 0, it's not a triangle. But points are distinct in the set.
    return d_ij == d_ik or d_ij == d_jk or d_ik == d_jk

def construct():
    start_time = time.time()
    deadline = start_time + 250  # Safety margin
    
    # Greedy Construction with Blocking Power
    # S is the current set of selected points (indices)
    S = []
    # unblocked: set of indices that can still be added to S
    # A point q is unblocked if it does not form an isosceles triangle with any pair in S.
    unblocked = set(range(num_points))
    
    while unblocked:
        if time.time() > deadline:
            break
            
        best_p = -1
        best_gain = -1
        
        # Calculate gain for each candidate p in unblocked
        # Gain = number of points q in unblocked (q != p) that will be blocked if p is added.
        # q is blocked by S U {p} if exists a in S such that is_isosceles(q, p, a).
        
        candidates = list(unblocked)
        for p in candidates:
            gain = 0
            # Check all q in unblocked, q != p
            # Optimization: If S is empty, gain is 0 for all. We can just pick any.
            if not S:
                # If S is empty, no points are blocked by S.
                # Adding p blocks q if is_isosceles(q, p, a) for a in S (empty). So gain is 0.
                # So we can just pick the first one or any.
                # To make it deterministic, pick the one with highest index or similar.
                pass
            else:
                for q in candidates:
                    if q == p:
                        continue
                    # Check if q is blocked by S U {p}
                    # This requires checking if there exists a in S such that is_isosceles(q, p, a)
                    blocked_by_new = False
                    for a in S:
                        if is_isosceles(q, p, a):
                            blocked_by_new = True
                            break
                    if blocked_by_new:
                        gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_p = p
            # Tie-breaking: if gain is same, we can just keep the first found or randomize.
            # For now, keep first found.
            
        if best_p == -1:
            # Should not happen if unblocked is not empty
            break
            
        # Add best_p to S
        S.append(best_p)
        unblocked.remove(best_p)
        
        # Update unblocked: remove any q that is now blocked by S
        # q is blocked if exists pair (a, b) in S such that is_isosceles(q, a, b)
        new_unblocked = set()
        for q in unblocked:
            is_blocked = False
            # Check all pairs in S
            for i in range(len(S)):
                a = S[i]
                for j in range(i + 1, len(S)):
                    b = S[j]
                    if is_isosceles(q, a, b):
                        is_blocked = True
                        break
                if is_blocked:
                    break
            if not is_blocked:
                new_unblocked.add(q)
        unblocked = new_unblocked

    # Refinement: Local Search for Minimality
    # Try to remove points from S if the remaining set is still maximal.
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(S)):
            p = S[i]
            S_trial = S[:i] + S[i+1:]
            
            # Check if S_trial is maximal
            # S_trial is maximal if no point q not in S_trial can be added to S_trial.
            # i.e., for all q not in S_trial, q is blocked by S_trial.
            
            # Create a set of S_trial for fast lookup
            S_trial_set = set(S_trial)
            is_maximal = True
            for q in range(num_points):
                if q in S_trial_set:
                    continue
                # Check if q can be added to S_trial
                # q can be added if it does NOT form isosceles with any pair in S_trial
                can_add = True
                for a in S_trial:
                    for b in S_trial:
                        if a == b:
                            continue
                        if is_isosceles(q, a, b):
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    # q can be added, so S_trial is NOT maximal
                    is_maximal = False
                    break
            
            if is_maximal:
                # S_trial is maximal and smaller than S
                S = S_trial
                improved = True
                break # Restart the while loop to check again
                
    # Convert indices back to (x, y) coordinates
    result = [points[i] for i in S]
    return result

# EVOLVE-BLOCK-END
