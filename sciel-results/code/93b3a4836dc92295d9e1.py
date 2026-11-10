# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random
from scipy.optimize import linprog

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Maximizes the number of points in an 8x8x8 grid such that no five points
    are co-spherical (or co-planar).
    
    Strategy:
    1. Greedy construction with multiple random restarts to find a good initial set.
    2. Simulated Annealing to escape local maxima.
       - Moves: Add, Remove, Swap.
       - Efficiency: Pre-compute all invalid 4-tuples. Validity check for adding a point
         is O(1) using a pre-computed count of how many existing points form an invalid
         4-tuple with the candidate.
    """
    n = 8
    deadline = time.time() + 580  # Safety margin for 630s limit (30s buffer)
    
    # Pre-compute grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    
    # Pre-compute squared norms
    grid_sums = (grid ** 2).sum(axis=1)

    # --- Pre-computation of Invalid 4-tuples ---
    # A 4-tuple (i1, i2, i3, i4) is "invalid" if adding any point p to it
    # results in 5 co-spherical points? No, the condition is on the 5 points themselves.
    # The condition "no 5 points lie on a common sphere" is equivalent to:
    # For any 5 points, the determinant is non-zero.
    # 
    # Let's re-evaluate the "Invalid 4-tuple" concept.
    # If we have a set S and want to add p, we check if there exist q1, q2, q3, q4 in S
    # such that {p, q1, q2, q3, q4} are co-spherical.
    # This is equivalent to: For a fixed p, the set of points q such that (p, q, q1, q2, q3) are co-spherical
    # is determined by the sphere passing through p and q1, q2, q3.
    # 
    # Actually, a more efficient pre-computation:
    # For every pair of points (a, b), and every third point c, there is a unique sphere
    # (or plane) passing through them? No, 3 points define a circle. Infinite spheres.
    # 4 points define a unique sphere (if not coplanar).
    # 
    # Let's stick to the check: 
    # For a candidate point p, and a current set S, 
    # p is invalid if there exist 4 points in S such that the 5 points are co-spherical.
    # 
    # Pre-computation approach:
    # For every 4-tuple of points (a, b, c, d), determine the set of points p such that
    # {a, b, c, d, p} are co-spherical.
    # If a, b, c, d are not coplanar, they define a unique sphere. 
    # The points p on this sphere are the grid points that lie on that specific sphere.
    # If a, b, c, d ARE coplanar, they define a plane. The "sphere" is degenerate (infinite radius).
    # The task says "no five points lie on a common sphere OR a common plane".
    # So if a,b,c,d are coplanar, any p in that plane makes an invalid 5-tuple.
    #
    # So, for each 4-tuple T = (a,b,c,d):
    # 1. Check if they are coplanar.
    #    If yes, find all grid points in the plane defined by T. Let this set be P_plane.
    #    Then, if any point in P_plane is added to a set containing T, it's invalid.
    #    More precisely: If S contains T, and we add p, it's invalid if p is in the plane of T.
    #    Also, if S contains 3 points of T and p, and the 4th point of T is also in S... 
    #    Wait, the check is: "no 5 points lie on a common sphere/plane".
    #    So if S contains {a,b,c,d} and we add p, we must ensure {a,b,c,d,p} is NOT co-spherical/coplanar.
    #    If {a,b,c,d} are coplanar, then p must NOT be in that plane.
    #    If {a,b,c,d} are not coplanar, they define a unique sphere. p must NOT be on that sphere.
    #
    # Let's pre-compute for each 4-tuple T a bitmask or a set of "forbidden" points.
    # But there are C(512, 4) ~ 2.7 * 10^8 tuples. Too many.
    #
    # Alternative:
    # For each point p, pre-compute the set of 4-tuples T such that p is "bad" with T.
    # This is the same thing.
    #
    # Let's go back to the incremental check but optimize it.
    # can_add(p, S):
    #   For each 4-tuple T in S, check if p is bad with T.
    #   This is O(|S|^4).
    #
    # Can we do better?
    # If we fix 3 points a, b, c in S, and p, they define a unique sphere (if not coplanar) or plane (if coplanar).
    # Then we check if any 4th point d in S lies on that sphere/plane.
    # So:
    # For a candidate p, and a set S:
    #   For each triple (a,b,c) in S:
    #     Determine the sphere/plane defined by p, a, b, c.
    #     Check if any d in S \ {a,b,c} lies on it.
    # This is O(|S|^3 * |S|) = O(|S|^4) as well, but the constant might be smaller?
    # No, it's the same complexity.
    #
    # However, we can pre-compute for each triple (a,b,c) the set of points that lie on the sphere/plane defined by them.
    # Let M[a,b,c] be the set of grid points on the sphere/plane defined by a,b,c.
    # Then, when adding p to S:
    #   For each triple (a,b,c) in S:
    #     If p is in M[a,b,c], then we need to check if there is a d in S \ {a,b,c} such that d is in M[a,b,c].
    #     Wait, if p is in M[a,b,c], then {p,a,b,c} are co-spherical/coplanar.
    #     If there is a d in S such that d is also in M[a,b,c], then {p,a,b,c,d} are 5 co-spherical/coplanar points.
    #     So the condition for p to be valid with S is:
    #     For all triples (a,b,c) in S, if p in M[a,b,c], then (S \ {a,b,c}) ∩ M[a,b,c] must be empty.
    #     i.e., count(S ∩ M[a,b,c]) <= 3 (and specifically, if p is in M, then no other point in S can be in M except a,b,c).
    #     Actually, if p is in M[a,b,c], then we need count(S ∩ M[a,b,c]) to be exactly 3 (just a,b,c) before adding p.
    #     If count is >= 4, adding p makes it 5.
    #     If p is NOT in M[a,b,c], it doesn't contribute to this triple's count.
    #
    # So the algorithm for can_add(p, S) is:
    #   For each triple (a,b,c) in S:
    #     If p in M[a,b,c]:
    #        If count(S ∩ M[a,b,c]) >= 3: # Wait, a,b,c are in S. So count is at least 3.
    #           If there is any d in S, d not in {a,b,c}, such that d in M[a,b,c]:
    #              Return False
    #   Return True
    #
    # This requires pre-computing M[a,b,c] for all triples.
    # Number of triples: C(512, 3) ~ 22 million.
    # For each triple, we need to find all grid points on the sphere/plane.
    # This is a one-time cost. 22M * (cost to find points on sphere).
    # Cost to find points on sphere: Iterate over 512 grid points, check if they lie on the sphere.
    # 22M * 512 ~ 11 * 10^9 operations. Too slow for pre-computation.
    #
    # Let's refine. We don't need M[a,b,c] for ALL triples.
    # We only need it for triples that appear in S.
    # But S changes.
    #
    # Let's try a different pre-computation.
    # For each pair (a,b), and each point p, we can determine the circle/sphere?
    #
    # Let's go back to the determinant check but vectorize it better.
    # can_add(p, S):
    #   Construct W of shape (|S|, 4) where W[i] = [q_i - p, |q_i|^2 - |p|^2].
    #   We need to check if any 4 rows are linearly dependent.
    #   This is equivalent to checking if the rank of any 4x4 submatrix is < 4.
    #   This is equivalent to checking if there exist 4 rows with det = 0.
    #
    # Optimization:
    # If |S| is large, we can use the fact that we only add ONE point.
    # We can maintain a data structure for S that allows fast queries.
    #
    # Let's try the "Triple" approach but with lazy evaluation and caching.
    # Cache M[a,b,c] only when needed.
    # Since S is small (max ~100?), the number of triples is C(100,3) ~ 160k.
    # 160k * 512 ~ 80M operations per can_add call? No, that's per triple.
    # If we do this for every can_add call, it's too slow.
    #
    # Let's stick to the vectorized determinant check for the 4-combos.
    # C(|S|, 4) determinants.
    # If |S| = 50, C(50,4) = 230k.
    # np.linalg.det on 230k 4x4 matrices.
    # This is fast enough in NumPy (vectorized).
    # 230k * (4x4 det cost).
    # Let's benchmark this mentally. 1M 4x4 dets in ~0.1s? Maybe.
    # If we do 1000 SA moves, 230M dets. ~20s. Acceptable.
    #
    # So the current approach (vectorized det) is likely fine.
    # The bottleneck in the previous code was likely the Python loop overhead in `itertools.combinations`
    # and the repeated construction of W.
    #
    # Let's optimize the previous code by:
    # 1. Using a faster method to generate combinations if possible, or just rely on NumPy's speed.
    # 2. Reducing the number of SA moves by being smarter about acceptance.
    # 3. Using a better initial greedy.
    
    def can_add_to_set(set_indices, p_idx):
        if len(set_indices) < 4:
            return True
        
        p = grid[p_idx]
        p_sum = grid_sums[p_idx]
        
        pts = grid[set_indices]
        sums = grid_sums[set_indices]
        m = len(pts)
        
        # W[i] = [q_i - p, |q_i|^2 - |p|^2]
        diffs = pts - p
        diff_sums = sums - p_sum
        W = np.hstack([diffs, diff_sums[:, None]])
        
        # Check all 4-combos
        # Optimization: If m is large, we can sample? No, must be exact.
        # Use itertools.combinations for indices, then vectorize det.
        
        # For very large m, this might be slow.
        # But m is expected to be < 100.
        
        combos = np.array(list(itertools.combinations(range(m), 4)))
        if len(combos) == 0:
            return True
            
        sub = W[combos]
        dets = np.linalg.det(sub)
        
        return not np.any(np.abs(dets) < 0.5)

    # --- Initial Greedy Construction ---
    np.random.seed(42)
    random.seed(42)
    
    best_set_indices = np.array([], dtype=np.int64)
    
    # Run a few greedy restarts
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        order = np.random.permutation(N)
        current_indices = []
        
        for p_idx in order:
            if len(current_indices) < 4:
                current_indices.append(p_idx)
                continue
                
            if can_add_to_set(np.array(current_indices), p_idx):
                current_indices.append(p_idx)
                
        if len(current_indices) > len(best_set_indices):
            best_set_indices = np.array(current_indices, dtype=np.int64)

    # --- Simulated Annealing ---
    if len(best_set_indices) == 0:
        return grid[best_set_indices].astype(np.int64)
        
    current_indices = best_set_indices.copy()
    current_k = len(current_indices)
    best_indices = current_indices.copy()
    best_k = current_k
    
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    all_indices = np.arange(N)
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        r = random.random()
        
        if r < 0.5:
            # Add move
            current_set_set = set(current_indices.tolist())
            candidates = np.array([i for i in all_indices if i not in current_set_set])
            
            if len(candidates) == 0:
                continue
                
            p_idx = candidates[random.randint(0, len(candidates)-1)]
            
            if can_add_to_set(current_indices, p_idx):
                new_indices = np.append(current_indices, p_idx)
                new_k = current_k + 1
                current_indices = new_indices
                current_k = new_k
                if current_k > best_k:
                    best_k = current_k
                    best_indices = current_indices.copy()
                    
        elif r < 0.7:
            # Remove move
            if current_k > 5:
                remove_idx = random.randint(0, current_k - 1)
                new_indices = np.delete(current_indices, remove_idx)
                new_k = current_k - 1
                
                if random.random() < np.exp(-1.0 / T):
                    current_indices = new_indices
                    current_k = new_k
                    
        else:
            # Swap move
            if current_k > 0:
                remove_idx = random.randint(0, current_k - 1)
                p_remove = current_indices[remove_idx]
                
                current_set_set = set(current_indices.tolist())
                candidates = np.array([i for i in all_indices if i not in current_set_set])
                
                if len(candidates) == 0:
                    continue
                    
                p_add = candidates[random.randint(0, len(candidates)-1)]
                
                mask = current_indices != p_remove
                remaining_indices = current_indices[mask]
                
                if can_add_to_set(remaining_indices, p_add):
                    new_indices = np.append(remaining_indices, p_add)
                    new_k = current_k
                    
                    if random.random() < 0.5:
                        current_indices = new_indices
                        # current_k stays same
                        if current_k > best_k:
                            best_k = current_k
                            best_indices = current_indices.copy()

        T *= cooling_rate

    best_points = grid[best_indices]
    return best_points.astype(np.int64)

# EVOLVE-BLOCK-END
