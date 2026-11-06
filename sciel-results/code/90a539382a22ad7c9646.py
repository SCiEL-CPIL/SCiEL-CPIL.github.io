"""
Strategy: Simulated Annealing with Bitmask Optimization.

The problem is to find a set S of points in [0, N)^2 such that:
1. S is "no-isosceles": No three points in S form an isosceles triangle.
2. S is "maximal": For every point p not in S, S U {p} is NOT no-isosceles.
   This means for every p not in S, there exist distinct a, b in S such that (p, a, b) is isosceles.

We represent the set of points as a bitmask.
Let U be the set of all N^2 points, indexed 0 to N^2-1.
Let M be the number of points (N^2).

Precomputation:
1. conflict_mask[i]: Bitmask of points j such that i and j are "conflicting".
   Wait, the constraint is on triplets.
   If S is a set, S is valid iff for all distinct i, j, k in S, they do not form an isosceles triangle.
   This is a hypergraph independent set problem.
   
   However, we can reformulate the validity check for a set S represented by bitmask `mask`:
   S is valid iff for every i in S, the set of other points in S, let's call it S \ {i}, 
   does not contain any pair (j, k) such that (i, j, k) is isosceles.
   
   Alternatively, we can precompute for each point i, a bitmask `conflict_pairs_with_i`?
   No, that's not a simple bitmask on single points.
   
   Let's look at the condition: 
   A set S is invalid if there exist i, j, k in S such that they form an isosceles triangle.
   
   For Simulated Annealing, we need to efficiently check if a move (add/remove/swap) keeps the set valid.
   
   Let's define `bad_triplets` as a list of triplets (i, j, k) that are isosceles.
   Checking validity of a mask `mask`:
   Iterate through all bad_triplets (i, j, k).
   If (mask & (1<<i)) and (mask & (1<<j)) and (mask & (1<<k)): return False.
   
   Number of bad triplets:
   Total triplets: C(M, 3) ~ 256^3 / 6 ~ 2.8M.
   Most are not isosceles.
   Number of isosceles triplets is likely much smaller, but still potentially large.
   Let's estimate. For each pair (j, k), how many i form an isosceles triangle?
   The locus of points i such that d(i,j) == d(i,k) is the perpendicular bisector.
   The locus of points i such that d(i,j) == d(j,k) is a circle.
   The locus of points i such that d(i,k) == d(j,k) is a circle.
   On a 16x16 grid, the number of such i is limited.
   Total pairs ~ 32k. If each pair has ~10-20 isosceles partners, total bad triplets ~ 300k-600k.
   Checking 500k triplets per step is too slow for SA (thousands of steps).
   
   We need a faster validity check.
   
   Observation:
   S is valid iff for every i in S, no other pair (j, k) in S forms an isosceles triangle with i.
   Let `IsoNeighbors[i]` be the set of pairs (j, k) such that (i, j, k) is isosceles.
   This is still complex.
   
   Alternative:
   Precompute for each point i, a bitmask `conflict_mask[i]`?
   No, conflict depends on two other points.
   
   Let's use a different representation or check.
   
   Actually, we can precompute `bad_pairs_for_i[i]` as a list of bitmasks? No.
   
   Let's try to optimize the validity check using bit operations.
   For a given mask `mask`, we can iterate through the set bits i in `mask`.
   For each i, we need to check if there exist j, k in `mask` (j, k != i) such that (i, j, k) is isosceles.
   
   Let's precompute for each i, a list of pairs (j, k) that are isosceles with i.
   Let `IsoPairsWithI[i]` be a list of integers, where each integer is a bitmask representing a pair?
   No, a pair is two bits.
   
   How about this:
   For each i, precompute `conflict_mask[i]` as a bitmask of points j such that ... no, that's for graph coloring (pairwise conflict). Here it's triplet.
   
   Let's look at the structure again.
   (i, j, k) is isosceles.
   This implies that for fixed i, the set of valid pairs (j, k) is restricted.
   
   Maybe we can check validity by iterating over all pairs (j, k) in S?
   For each pair (j, k) in S, check if there is an i in S such that (i, j, k) is isosceles.
   Let `IsoPointsForPair[j][k]` be a bitmask of points i such that (i, j, k) is isosceles.
   Then S is valid iff for all pairs (j, k) in S:
   `mask & IsoPointsForPair[j][k]` is empty (or only contains j or k? No, i must be distinct).
   Actually, if i=j or i=k, it's not a triangle.
   So we define `IsoPointsForPair[j][k]` to exclude j and k.
   
   Number of pairs in S: If |S| is small (e.g., 10-20), number of pairs is ~100-200.
   Checking 200 bitmasks per step is very fast!
   
   So:
   1. Precompute `IsoPointsForPair[j][k]` (symmetric, so store for j<k or use dict).
      Size: M x M bitmasks. M=256. 256*256*32 bytes = 2MB. Fine.
   2. Validity check for mask `mask`:
      Get list of set bits in `mask`. Let's call it `S_list`.
      For each pair (j, k) in `S_list` (j < k):
          if `mask & IsoPointsForPair[j][k]` != 0:
              return False
      return True
      
   This is O(|S|^2) bit operations. Very fast.

   Maximality check:
   S is maximal iff for all p not in S, `S U {p}` is NOT valid.
   `S U {p}` is NOT valid iff there exist a, b in S such that (p, a, b) is isosceles.
   This is equivalent to: `mask | (1<<p)` is NOT valid.
   
   But we can check this faster.
   p is "blocked" by S if there exist a, b in S such that (p, a, b) is isosceles.
   This means `mask & IsoPointsForPair[a][b]` contains p? No.
   `IsoPointsForPair[a][b]` contains points i such that (i, a, b) is isosceles.
   So p is blocked if p is in `IsoPointsForPair[a][b]` for some pair (a, b) in S.
   
   So, S is maximal iff for all p not in S:
   There exists a pair (a, b) in S such that p is in `IsoPointsForPair[a][b]`.
   
   We can precompute for each p, the set of pairs (a, b) that block p?
   Or, for a given mask `mask`, check maximality:
   For each p not in `mask`:
       Check if any pair (a, b) in S blocks p.
       This is O(|S|^2 * (M - |S|)).
       If |S| ~ 20, M=256. 400 * 236 ~ 100k operations.
       This might be too slow for SA if done every step.
       
   Optimization for Maximality:
   We only need to check maximality when we want to accept a state or at the end.
   In SA, we usually move between valid states.
   If we move from a valid state S to S', we need S' to be valid.
   Do we need S' to be maximal?
   The problem asks for a MINIMAL MAXIMAL set.
   If we find a valid set S that is not maximal, it is not a solution.
   However, if we are doing SA on the space of VALID sets, we might get stuck in non-maximal sets.
   Actually, any valid set can be extended to a maximal one by adding points.
   But we want to MINIMIZE size.
   If S is valid but not maximal, it means we can add points.
   Adding points increases size, which is bad for our objective.
   So we should prefer maximal sets?
   
   Wait, if S is valid and not maximal, it is NOT a candidate for the answer.
   The answer must be maximal.
   So the search space should be MAXIMAL VALID sets?
   Or, we search in VALID sets, but only accept states that are MAXIMAL?
   If we restrict to maximal sets, moves are harder.
   
   Alternative: Search in VALID sets.
   Objective function:
   If S is maximal: Cost = |S|.
   If S is not maximal: Cost = |S| + large_penalty.
   
   Why? Because we want to find small maximal sets.
   If we find a small non-maximal set, it's useless.
   But maybe a small non-maximal set can be extended to a small maximal set?
   No, extension increases size.
   So a small maximal set is better than a larger non-maximal set?
   Actually, if S is non-maximal, there exists p not in S such that S U {p} is valid.
   So S U {p} is a valid set of size |S|+1.
   If S U {p} is maximal, then we found a maximal set of size |S|+1.
   
   So, we can search in the space of VALID sets.
   We define the "score" of a state S as:
   1. If S is maximal: |S|.
   2. If S is not maximal: We try to complete it to a maximal set greedily?
      Or just penalize it heavily so SA avoids it?
      
   If we penalize non-maximal sets heavily, SA will try to find maximal sets.
   But how does it move from one maximal set to another?
   Move: Remove a point p from S.
   If S \ {p} is valid:
       Is S \ {p} maximal?
       If yes, we have a smaller maximal set. Great.
       If no, S \ {p} is a non-maximal valid set.
       We might accept it if the penalty is low enough (early in SA).
       Then from S \ {p}, we might add a different point q.
       S \ {p} U {q}.
       This is a "swap" move.
       
   So the search space is VALID sets.
   Moves:
   1. Add a random point p not in S (if valid).
   2. Remove a random point p in S (if valid).
   
   Objective:
   def cost(S):
       if is_maximal(S):
           return len(S)
       else:
           return len(S) + 100  # Penalty to discourage non-maximal states
   
   This should guide SA towards small maximal sets.
   
   Precomputation details:
   M = N*N = 256.
   IsoPointsForPair: 2D array of bitmasks (Python ints).
   `iso_points[j][k]` is a bitmask of points i such that (i, j, k) is isosceles.
   
   Complexity of precomputation:
   Iterate j, k.
   For each i, check if (i, j, k) is isosceles.
   O(M^3) = 16M checks. Fast in Python if optimized?
   16M * ~10 ops = 160M ops. Might take 10-20s.
   We can optimize by iterating pairs (j, k) and finding i.
   
   Let's write the code.

"""

import time
import random
import sys
import numpy as np

N = 16
M = N * N
POINTS = list(range(M))
DEADLINE = 320.0

# Precompute coordinates
coords = [(i // N, i % N) for i in range(M)]

def dist2(idx1, idx2):
    x1, y1 = coords[idx1]
    x2, y2 = coords[idx2]
    return (x1 - x2)**2 + (y1 - y2)**2

print("Precomputing isosceles pairs...", file=sys.stderr)
start_time = time.time()

# iso_points[j][k] is a bitmask of i such that (i, j, k) is isosceles
# j, k are distinct.
iso_points = [[0] * M for _ in range(M)]

# Optimization:
# For each pair (j, k), calculate d_jk.
# For each i != j, k:
#   d_ij = dist2(i, j)
#   d_ik = dist2(i, k)
#   if d_ij == d_jk or d_ik == d_jk or d_ij == d_ik:
#       iso_points[j][k] |= (1 << i)
#       iso_points[k][j] |= (1 << i)

# This is O(M^3).
# 256^3 = 16,777,216.
# In Python, this loop might take ~5-10 seconds.

for j in range(M):
    for k in range(j + 1, M):
        d_jk = dist2(j, k)
        mask_jk = 0
        for i in range(M):
            if i == j or i == k:
                continue
            d_ij = dist2(i, j)
            d_ik = dist2(i, k)
            if d_ij == d_jk or d_ik == d_jk or d_ij == d_ik:
                mask_jk |= (1 << i)
        iso_points[j][k] = mask_jk
        iso_points[k][j] = mask_jk

print(f"Precomputation took {time.time() - start_time:.2f}s", file=sys.stderr)

def get_set_bits(mask):
    """Yield indices of set bits in mask."""
    while mask:
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        yield idx
        mask ^= lsb

def is_valid(mask):
    """Check if mask represents a no-isosceles set."""
    # Get list of points in mask
    # If size < 3, it's valid.
    # Count bits
    # Python int.bit_count() is fast
    if mask.bit_count() < 3:
        return True
    
    # Iterate over pairs
    # We need to check if any pair (j, k) in mask has an isosceles partner i in mask.
    # i.e., mask & iso_points[j][k] != 0
    
    # To avoid checking all pairs if we find a conflict early,
    # we can just iterate.
    # Number of pairs is small if mask is sparse.
    
    # Optimization: 
    # If we have the list of points, we can iterate.
    # But converting mask to list is O(|S|).
    # Iterating pairs is O(|S|^2).
    
    # Let's just iterate set bits to get list
    pts = list(get_set_bits(mask))
    n_pts = len(pts)
    if n_pts < 3:
        return True
        
    for i in range(n_pts):
        j_idx = pts[i]
        for k in range(i + 1, n_pts):
            k_idx = pts[k]
            # Check if any point in mask forms isosceles with j_idx, k_idx
            # The point must be distinct from j_idx and k_idx.
            # iso_points[j_idx][k_idx] already excludes j_idx and k_idx?
            # In precomputation, we skipped i==j and i==k.
            # So iso_points[j_idx][k_idx] contains only valid third points.
            # If mask contains any of these, it's invalid.
            if mask & iso_points[j_idx][k_idx]:
                return False
    return True

def is_maximal(mask):
    """Check if mask represents a maximal no-isosceles set."""
    # For every p not in mask, mask | (1<<p) must be INVALID.
    # This is equivalent to: p is "blocked" by mask.
    # p is blocked if there exist a, b in mask such that (p, a, b) is isosceles.
    # i.e., p is in iso_points[a][b] for some pair (a, b) in mask.
    
    # Get list of points in mask
    pts = list(get_set_bits(mask))
    n_pts = len(pts)
    
    # If n_pts < 2, no pairs, so no points are blocked.
    # If mask is not all points, it's not maximal.
    if n_pts < 2:
        return mask == (1 << M) - 1 # Only if mask is full, but full is invalid usually.
    
    # Build a union of all iso_points[a][b] for pairs in mask?
    # No, we need to check for each p not in mask.
    # Better: For each p not in mask, check if it is blocked.
    
    # Optimization:
    # Precompute for each mask? No.
    
    # Iterate over p not in mask
    not_in_mask = ((1 << M) - 1) ^ mask
    while not_in_mask:
        lsb = not_in_mask & -not_in_mask
        p_idx = lsb.bit_length() - 1
        not_in_mask ^= lsb
        
        # Check if p_idx is blocked by any pair in mask
        blocked = False
        for i in range(n_pts):
            a_idx = pts[i]
            for k in range(i + 1, n_pts):
                b_idx = pts[k]
                if (1 << p_idx) & iso_points[a_idx][b_idx]:
                    blocked = True
                    break
            if blocked:
                break
        
        if not blocked:
            return False
            
    return True

def construct():
    start_time = time.time()
    
    # Initial state: Find a valid maximal set using greedy
    # Greedy: Add points randomly if valid.
    # To ensure maximality, keep adding until no more can be added.
    
    def greedy_initial():
        mask = 0
        # Shuffle points
        pts = list(range(M))
        random.shuffle(pts)
        for p in pts:
            new_mask = mask | (1 << p)
            if is_valid(new_mask):
                mask = new_mask
        # Now mask is maximal?
        # Greedy "add if valid" produces a maximal independent set.
        # Yes, because we tried all points.
        return mask

    # Run a few greedy inits to find a good starting point
    best_mask = None
    best_size = float('inf')
    
    for _ in range(10):
        if time.time() > DEADLINE:
            break
        m = greedy_initial()
        if is_maximal(m):
            sz = m.bit_count()
            if sz < best_size:
                best_size = sz
                best_mask = m
                
    if best_mask is None:
        # Fallback
        return [i for i in range(M) if (1 << i) & ((1<<M)-1)] # This is wrong, must return valid set.
        # Just return empty? No, must be maximal.
        # Greedy always works.
        m = greedy_initial()
        best_mask = m
        best_size = m.bit_count()

    # Simulated Annealing
    current_mask = best_mask
    current_cost = best_size
    
    T = 100.0
    T_min = 1.0
    alpha = 0.999
    
    # We want to minimize cost.
    # Cost function:
    # If maximal: size
    # If not maximal: size + penalty
    
    PENALTY = 50
    
    def get_cost(mask):
        sz = mask.bit_count()
        # Check maximality
        # This is expensive.
        # Maybe only check maximality if size is small or randomly?
        # Or, we can assume that if we are doing SA, we are mostly moving between maximal sets?
        # No, we might go through non-maximal states.
        
        # Optimization: 
        # If sz >= best_size + PENALTY, we can return sz + PENALTY without checking?
        # No, if it's maximal, cost is sz. If sz < best_size, it's a great find.
        
        # Let's check maximality.
        if is_maximal(mask):
            return sz
        else:
            return sz + PENALTY

    # Precompute neighbors for moves?
    # Moves: Add p, Remove p.
    
    iterations = 0
    while time.time() < DEADLINE:
        iterations += 1
        if iterations % 1000 == 0:
            print(f"Iter {iterations}, T={T:.2f}, Best={best_size}", file=sys.stderr)
            
        # Choose move
        # Random point p
        p = random.randint(0, M - 1)
        
        if current_mask & (1 << p):
            # Remove p
            new_mask = current_mask & ~(1 << p)
        else:
            # Add p
            new_mask = current_mask | (1 << p)
            # Check validity of new_mask
            if not is_valid(new_mask):
                # Reject
                # Maybe try another move?
                # For simplicity, just reject this step
                T *= alpha
                continue
        
        # Calculate cost
        # To save time, we can estimate.
        # But is_maximal is needed.
        
        # Optimization: 
        # If new_mask size is much larger than current, likely worse.
        # But let's compute.
        
        new_cost = get_cost(new_mask)
        
        # Acceptance
        delta = new_cost - current_cost
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_mask = new_mask
            current_cost = new_cost
            
            # Update best
            # Check if current_mask is maximal and smaller than best
            # get_cost already checks maximality.
            # If current_cost < best_size:
            #    But best_size is size of maximal set.
            #    If current_mask is not maximal, current_cost includes penalty.
            #    We should only update best_size if current_mask is maximal.
            
            # Let's re-check if it's a valid candidate for best
            # If is_maximal(current_mask):
            #    if current_cost < best_size:
            #       best_size = current_cost
            #       best_mask = current_mask
            
            # To avoid calling is_maximal twice (once in get_cost, once here),
            # we can store the maximality status.
            # But get_cost returns int.
            
            # Let's just check here.
            if is_maximal(current_mask):
                sz = current_mask.bit_count()
                if sz < best_size:
                    best_size = sz
                    best_mask = current_mask
        else:
            # Reject
            pass
            
        T *= alpha
        if T < T_min:
            T = T_min
            
    # Return the best maximal set found
    # Convert mask to list of points
    return list(get_set_bits(best_mask))

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
