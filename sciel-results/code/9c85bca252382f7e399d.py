"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def _precompute_iso_pairs():
    """
    Precompute for each pair (a, b), the set of points c such that 
    the triangle (a, b, c) is isosceles.
    Note: a, b, c must be distinct.
    """
    points = list(itertools.product(range(N), repeat=2))
    V = len(points)
    
    # dist2[i][j]
    dist2 = np.zeros((V, V), dtype=np.int64)
    for i in range(V):
        xi, yi = points[i]
        for j in range(i + 1, V):
            xj, yj = points[j]
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            dist2[i, j] = d
            dist2[j, i] = d
            
    # iso_pairs[i][j] is a list of k such that {i, j, k} is isosceles
    # We only need i != j != k
    iso_pairs = [[set() for _ in range(V)] for _ in range(V)]
    
    for i in range(V):
        for j in range(i + 1, V):
            d_ij = dist2[i, j]
            # Find k such that d_ik == d_ij or d_jk == d_ij or d_ik == d_jk
            # d_ik == d_ij: k is on circle centered at i with radius d_ij
            # d_jk == d_ij: k is on circle centered at j with radius d_ij
            # d_ik == d_jk: k is on perpendicular bisector of ij
            
            # Iterate k
            for k in range(V):
                if k == i or k == j:
                    continue
                d_ik = dist2[i, k]
                d_jk = dist2[j, k]
                if d_ik == d_ij or d_jk == d_ij or d_ik == d_jk:
                    iso_pairs[i][j].add(k)
                    iso_pairs[j][i].add(k)
                    
    return points, iso_pairs

POINTS, ISO_PAIRS = _precompute_iso_pairs()
V = len(POINTS)

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_size = V
    best_set = []
    
    # We will run multiple restarts.
    # In each restart, we use the greedy set cover approach.
    # To add diversity, we can randomize the tie-breaking or initial state.
    # However, the greedy choice is usually deterministic if gains are unique.
    # If gains are tied, we can pick randomly.
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        # Initialize
        S = set()
        blocked = set()
        
        # Candidates: all points not in S and not blocked.
        # Initially, no points are blocked.
        # But wait, is the empty set valid? Yes.
        # Are any points blocked by empty set? No.
        # So candidates = all points.
        
        # We need to pick points one by one.
        # At each step, for each candidate p, calculate gain.
        
        # To speed up, we can maintain a list of candidates.
        # But the gain of p depends on S.
        
        # Let's just iterate.
        # Max size of S is V.
        # At each step, we iterate over all V points to find the best candidate.
        # Complexity: V steps * V candidates * (cost to calc gain).
        # Cost to calc gain for p:
        #   We need to find { c | exists a in S, c in ISO_PAIRS[a][p] }
        #   and intersect with (All \ Blocked \ S).
        #   This is a union of sets.
        #   If |S| is small, this is fast.
        #   If |S| is large, it's slower.
        #   But S is expected to be small (minimal maximal).
        
        # Let's implement the loop.
        
        # Optimization: Precompute union of ISO_PAIRS[a][p] for a in S?
        # No, S changes.
        
        # Let's try.
        
        for _ in range(V): # Max V points
            if time.time() > deadline:
                break
                
            best_p = -1
            best_gain = -1
            best_new_blocked = None
            
            # Candidates are points not in S and not in blocked
            # We can just iterate all points and skip if in S or blocked
            
            for p in range(V):
                if p in S or p in blocked:
                    continue
                
                # Calculate gain
                # New blocked points from adding p:
                # Union of ISO_PAIRS[a][p] for a in S
                new_blocked = set()
                for a in S:
                    new_blocked.update(ISO_PAIRS[a][p])
                
                # Filter: only points not already blocked and not in S
                # Note: p itself is not in new_blocked (since k != i, j)
                # But p becomes covered by being in S.
                
                valid_new_blocked = {c for c in new_blocked if c not in blocked and c not in S}
                
                gain = len(valid_new_blocked) + 1 # +1 for p itself
                
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_new_blocked = valid_new_blocked
                elif gain == best_gain:
                    # Random tie-breaking
                    if random.random() < 0.5:
                        best_gain = gain
                        best_p = p
                        best_new_blocked = valid_new_blocked
            
            if best_p == -1:
                # No candidates left? That means all points are in S or blocked.
                # So S is maximal.
                break
            
            # Add best_p to S
            S.add(best_p)
            # Update blocked
            if best_new_blocked:
                blocked.update(best_new_blocked)
            # Also, p is now in S, so it's "covered".
            # We don't need to add p to blocked explicitly for the check "p in blocked",
            # because we check "p in S" separately.
            
        # End of construction for this restart
        current_size = len(S)
        if current_size < best_size:
            best_size = current_size
            best_set = list(S)
            
            # If we find a very small set, stop?
            if best_size <= 10:
                break
                
    return [POINTS[i] for i in best_set]

# EVOLVE-BLOCK-END
