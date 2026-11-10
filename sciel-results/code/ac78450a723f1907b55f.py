# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from collections import deque

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than their minimum pairwise distance, aiming to maximize the number of points.

    Returns:
        points: np.ndarray of shape (num_points,11)
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 10 minute budget

    d = 11

    # Generate candidate vectors with squared norm <= 4
    # Squared norm 1: 22 vectors (±1 in one coordinate)
    # Squared norm 2: 11*10 = 110 vectors (±1 in two coordinates)
    # Squared norm 4: 22 vectors (±2 in one coordinate)
    # Total: 22 + 110 + 22 = 154 candidates
    
    candidates = []
    
    # Norm squared = 1
    for i in range(d):
        for sign in [1, -1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = sign
            candidates.append(v)
    
    # Norm squared = 2
    for i in range(d):
        for j in range(i+1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Norm squared = 4
    for i in range(d):
        for sign in [1, -1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = sign * 2
            candidates.append(v)
    
    n = len(candidates)
    candidates = np.array(candidates)
    
    # Compute squared norms
    norms_sq = np.sum(candidates ** 2, axis=1)
    
    # Compute all pairwise squared distances
    # dist_sq[i,j] = ||v_i - v_j||^2
    # = ||v_i||^2 + ||v_j||^2 - 2*v_i·v_j
    dot = candidates @ candidates.T
    dist_sq = norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - 2 * dot
    
    # Find minimum pairwise distance
    # We need min over all pairs i<j of dist_sq[i,j]
    # For the kissing configuration: min_pairwise_dist_sq >= max_norm_sq
    
    # Strategy: try different max_norm_sq values and find max independent set
    # where two vertices are adjacent if dist_sq < threshold
    
    best_k = 0
    best_points = None
    
    # Try different thresholds for max norm squared
    # The max norm squared in our set will be 4 (from the ±2 vectors)
    # But we can choose subsets with smaller max norm
    
    # Let's try to find the maximum clique/independent set
    # We want: for all pairs i,j in S: dist_sq[i,j] >= max_norm_sq(S)
    # where max_norm_sq(S) = max(norms_sq[i] for i in S)
    
    # This is a complex constraint. Let's use a greedy approach with local search.
    
    # First, let's try a simple greedy approach:
    # Sort candidates by norm squared, then greedily add points that satisfy the constraint
    
    # Actually, let's think about this differently.
    # If all points have the same norm squared R, then we need dist_sq[i,j] >= R for all pairs.
    # dist_sq[i,j] = 2R - 2*v_i·v_j >= R
    # => v_i·v_j <= R/2
    
    # For R=1: need v_i·v_j <= 0.5, so v_i·v_j <= 0 (since integer)
    # For R=2: need v_i·v_j <= 1
    # For R=4: need v_i·v_j <= 2
    
    # Let's try to build a set where all points have norm squared = R
    # and pairwise dot products satisfy the constraint
    
    # For R=1: 22 candidates (±e_i). Two are compatible if dot=0, i.e., different coordinates.
    # Max set: all 22? No, e_i and -e_i have dot=-1, which is <= 0, so they're compatible.
    # Wait, we need dot <= R/2 = 0.5, so dot <= 0.
    # e_i · e_j = 0 if i≠j, = 1 if i=j.
    # e_i · (-e_i) = -1 <= 0. OK.
    # e_i · e_j = 0 for i≠j. OK.
    # So all 22 vectors with norm squared 1 are mutually compatible?
    # Let's check: max norm squared = 1, min pairwise dist squared = min over i≠j of ||v_i-v_j||^2
    # ||e_i - e_j||^2 = 2 for i≠j, ||e_i - (-e_i)||^2 = 4
    # So min pairwise dist squared = 2 >= 1 = max norm squared. 
    # So we can have all 22!
    
    # Let's verify: points = all ±e_i for i=1..11
    # max norm squared = 1
    # min pairwise dist squared: between e_i and e_j (i≠j) is 2, between e_i and -e_i is 4
    # So min = 2 >= 1. 
    
    # Can we do better with norm squared = 2?
    # 110 candidates. Need dot <= 1.
    # Let's try to find a large set.
    
    # Actually, let's just compute the answer for norm squared = 1:
    # All 22 vectors work. k=22.
    
    # Can we add vectors with norm squared = 2 to this set?
    # If we add v with ||v||^2=2, then max norm squared becomes 2.
    # We need all pairwise distances >= 2.
    # For v and e_i: ||v - e_i||^2 = 2 + 1 - 2*v·e_i = 3 - 2*v·e_i
    # This needs to be >= 2, so v·e_i <= 0.5, i.e., v·e_i <= 0.
    # v has two nonzero entries, say v_i=a, v_j=b with a^2+b^2=2, so a,b ∈ {±1}.
    # v·e_i = a, v·e_j = b, v·e_k = 0 for k≠i,j.
    # So we need a<=0 and b<=0, meaning v_i=-1 and v_j=-1.
    # So only the vectors with both entries -1 are compatible with all ±e_k.
    # There are C(11,2)=55 such vectors.
    # But we also need pairwise compatibility among these 55 vectors.
    # Two such vectors v (with -1 at i,j) and w (with -1 at k,l):
    # v·w = number of common indices with both -1.
    # If {i,j} and {k,l} are disjoint: v·w=0 <= 1. OK.
    # If they share one index: v·w=-1 <= 1. OK.
    # If they're the same: not allowed (distinct points).
    # So all 55 are mutually compatible!
    # And each has norm squared 2, so max norm squared = 2.
    # Min pairwise distance: among the 55, ||v-w||^2 = 2+2-2*v·w = 4-2*v·w.
    # If disjoint: 4. If share one: 4-2(-1)=6.
    # Between ±e_k and the -1,-1 vectors: computed above, >= 2.
    # Between ±e_k themselves: >= 2.
    # So min pairwise dist squared = 2 >= 2. 
    # Total points: 22 + 55 = 77!
    
    # Can we do even better? Let's try to add more vectors.
    # What about vectors with norm squared = 4?
    # If we add a vector u with ||u||^2=4, max norm squared becomes 4.
    # We need all pairwise distances >= 4.
    # This is much harder. Let's stick with the 77 for now.
    
    # Let's construct the 77 points:
    points_list = []
    
    # All ±e_i
    for i in range(d):
        for sign in [1, -1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = sign
            points_list.append(v)
    
    # All vectors with -1 at two positions
    for i in range(d):
        for j in range(i+1, d):
            v = np.zeros(d, dtype=np.int64)
            v[i] = -1
            v[j] = -1
            points_list.append(v)
    
    points = np.array(points_list, dtype=np.int64)
    
    # Verify the solution
    k = len(points)
    if k >= 2:
        norms_sq = np.sum(points ** 2, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        # Check pairwise distances
        min_dist_sq = np.inf
        for i in range(k):
            for j in range(i+1, k):
                diff = points[i] - points[j]
                d2 = np.sum(diff ** 2)
                if d2 < min_dist_sq:
                    min_dist_sq = d2
        
        if min_dist_sq >= max_norm_sq:
            return points
    
    # If verification fails, fall back to the simple 2-point solution
    return np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)

# EVOLVE-BLOCK-END
