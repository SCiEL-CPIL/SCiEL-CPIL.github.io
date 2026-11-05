"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
import random
import time

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import random
import time

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    n_points = len(all_points)
    start_time = time.time()
    deadline = start_time + 290  # safety margin

    # 1. Precompute squared distances
    pts = np.array(all_points, dtype=np.int32)
    diff = pts[:, None, :] - pts[None, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)

    # 2. Precompute iso_with_pair[a, b] = set of p such that (p, a, b) is isosceles
    # Note: p must be distinct from a and b.
    # We store these as sets of indices for fast union operations.
    iso_with_pair = [[set() for _ in range(n_points)] for _ in range(n_points)]
    
    # Optimization: Only compute for a < b? No, isosceles is symmetric in a,b but not p.
    # However, (p, a, b) is isosceles iff d(p,a)==d(p,b) OR d(p,a)==d(a,b) OR d(a,b)==d(p,b).
    # We can iterate p, a, b. 256^3 = 16.7M operations. Might be slow in Python.
    # Let's vectorize or optimize.
    
    # Alternative: For each pair (a, b), find p.
    # d(p,a) = d(p,b) -> p lies on perpendicular bisector of ab.
    # d(p,a) = d(a,b) -> p lies on circle centered a with radius d(a,b).
    # d(a,b) = d(p,b) -> p lies on circle centered b with radius d(a,b).
    
    # Let's use the vectorized check for speed in precomputation if possible, 
    # or just efficient loops.
    # 16M iterations in Python might take ~10-20s. Acceptable.
    
    # To speed up, we can precompute distances and use them.
    # For each a, b:
    #   d_ab = sq_dist[a, b]
    #   p is iso if sq_dist[p, a] == sq_dist[p, b] or sq_dist[p, a] == d_ab or sq_dist[p, b] == d_ab
    #   and p != a and p != b.
    
    for a in range(n_points):
        for b in range(a + 1, n_points):
            d_ab = sq_dist[a, b]
            # Vectorized check for all p
            # d_pa = sq_dist[:, a]
            # d_pb = sq_dist[:, b]
            # mask = (d_pa == d_pb) | (d_pa == d_ab) | (d_pb == d_ab)
            # mask[a] = False; mask[b] = False
            # indices = np.where(mask)[0]
            # iso_with_pair[a][b] = set(indices)
            # iso_with_pair[b][a] = set(indices)
            
            d_pa = sq_dist[:, a]
            d_pb = sq_dist[:, b]
            mask = (d_pa == d_pb) | (d_pa == d_ab) | (d_pb == d_ab)
            mask[a] = False
            mask[b] = False
            indices = np.where(mask)[0]
            s = set(indices.tolist())
            iso_with_pair[a][b] = s
            iso_with_pair[b][a] = s

    def get_blocked(S_set):
        """Return the set of points blocked by S_set."""
        blocked = set()
        S_list = list(S_set)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                blocked.update(iso_with_pair[a][b])
        return blocked

    def is_maximal(S_set):
        """Check if S_set is maximal."""
        blocked = get_blocked(S_set)
        # All points not in S_set must be blocked
        for p in range(n_points):
            if p not in S_set and p not in blocked:
                return False
        return True

    def greedy_construct(random_seed, deadline):
        random.seed(random_seed)
        S_set = set()
        while True:
            if time.time() > deadline:
                return None
            blocked = get_blocked(S_set)
            # Available points: not in S, not blocked
            available = [p for p in range(n_points) if p not in S_set and p not in blocked]
            
            if not available:
                # We are maximal
                return S_set
            
            # Heuristic: Pick point that blocks the most available points
            # Gain(p) = | { q in available | q in blocked(S U {p}) } |
            # blocked(S U {p}) = blocked(S) U { q | (q, a, p) is iso for some a in S } U { q | (q, p, p) is iso? No, distinct }
            # New blocks from p:
            # For each a in S U {p}, blocks added by (p, a) pair.
            # Actually, new blocks are q such that q is in iso_with_pair[p][a] for some a in S, OR q in iso_with_pair[p][p] (empty).
            # Wait, iso_with_pair[p][a] contains q such that (q, p, a) is isosceles.
            # So new_blocked = union(iso_with_pair[p][a] for a in S)
            
            best_p = -1
            max_gain = -1
            
            # Optimization: If available is large, sampling might be better, 
            # but n=256 is small.
            for p in available:
                new_blocked = set()
                for a in S_set:
                    new_blocked.update(iso_with_pair[p][a])
                # Count how many available points are in new_blocked
                gain = len(new_blocked.intersection(available))
                if gain > max_gain:
                    max_gain = gain
                    best_p = p
                # Early break if we found a perfect cover? Unlikely.
            
            S_set.add(best_p)
            
            # Check if we are maximal now
            if is_maximal(S_set):
                return S_set

    def local_search_shrink(S_set, deadline):
        """Try to shrink S_set."""
        S_list = list(S_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(S_list)):
                if time.time() > deadline:
                    break
                p = S_list[i]
                S_minus = S_set - {p}
                
                # Check if S_minus is maximal
                if is_maximal(S_minus):
                    S_set = S_minus
                    S_list = list(S_set)
                    improved = True
                    break
                
                # If not maximal, try to swap p with some q
                # Find unblocked points in S_minus
                blocked_minus = get_blocked(S_minus)
                unblocked = [q for q in range(n_points) if q not in S_minus and q not in blocked_minus]
                
                # Try each q in unblocked
                for q in unblocked:
                    S_new = S_minus | {q}
                    # Check if S_new is maximal
                    if is_maximal(S_new):
                        # Check if this allows further shrinkage? 
                        # For now, just accept if it doesn't increase size (it doesn't)
                        # But we want to find a state that *can* shrink.
                        # Let's just accept the swap and let the outer loop try to shrink again.
                        S_set = S_new
                        S_list = list(S_set)
                        improved = True
                        break
                if improved:
                    break
        return S_set

    best_set = None
    best_size = n_points + 1
    
    # Run multiple greedy restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        S = greedy_construct(42 + i, deadline - 10)
        if S is not None:
            # Shrink it
            S = local_search_shrink(S, deadline - 5)
            if len(S) < best_size:
                best_size = len(S)
                best_set = S
                # If we find a very small set, we can break early?
                # Let's continue to find even smaller ones.

    if best_set is None:
        # Fallback
        best_set = set([0])
        while not is_maximal(best_set):
            blocked = get_blocked(best_set)
            available = [p for p in range(n_points) if p not in best_set and p not in blocked]
            if not available:
                break
            best_set.add(available[0])
    
    return [all_points[i] for i in sorted(best_set)]
# EVOLVE-BLOCK-END
