"""Branch-and-bound solver for minimum maximal no-isosceles set using conflict graph."""
import random
import time
import numpy as np
from itertools import product

N = 16
V = N * N  # 256 grid points


def _precompute():
    """
    Precompute:
    - points: list of (x, y) tuples
    - dist: V x V squared distance matrix
    - conflict_mask[i]: bitset of points that conflict with i (i.e., cannot coexist in a no-isosceles set with i)
    
    Two points i and j conflict if there exists a third point k such that {i, j, k} is isosceles.
    Actually, we need: i and j cannot both be in S if there's any k in S that forms isosceles with them.
    But for the independent set formulation, we define conflict between i and j if adding both
    would prevent some k from being added... No, that's not right.
    
    Let me reconsider. A no-isosceles set S means: for any three distinct points a, b, c in S,
    the triangle {a,b,c} is NOT isosceles.
    
    So the constraint is on triples. We can't simply define pairwise conflicts.
    
    Alternative approach: We'll use a different formulation.
    
    Actually, let's think about it differently. We want to find the smallest maximal no-isosceles set.
    
    Key insight: A set S is no-isosceles iff for every pair (a,b) in S, no point c in S lies on the
    "isosceles locus" of (a,b) (i.e., the set of points c where {a,b,c} is isosceles).
    
    For branch-and-bound, we can build the set incrementally. At each step, we decide which points to include.
    The challenge is that the constraint involves triples, not pairs.
    
    Let me use a different strategy: 
    1. First, find a good upper bound using greedy + local search (fast).
    2. Then use branch-and-bound to try to prove optimality or find smaller sets.
    
    For the B&B, we'll use the following:
    - State: (included_set, excluded_set, candidates)
    - A point can be included if it doesn't form an isosceles triangle with any two points in included_set.
    - A point must be excluded if it would create a non-maximal set... this is complex.
    
    Actually, let me simplify. I'll use a hybrid approach:
    1. Greedy construction with multiple random seeds to get good upper bounds.
    2. Local search (remove points, swap points) to improve.
    3. Use the best found as upper bound for B&B.
    
    For the conflict structure, I'll precompute for each pair (a,b) the set of points that conflict with them.
    """
    points = list(product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    
    # Squared distance matrix
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sum(diff ** 2, axis=2)
    
    # For each pair (i, j), compute the set of points k such that {i, j, k} is isosceles
    # Store as a list of sets for quick lookup
    pair_conflict = [None] * (V * V)
    
    for i in range(V):
        for j in range(i + 1, V):
            dij = dist[i, j]
            # k where d(i,k) == dij
            mask1 = (dist[i] == dij)
            mask1[j] = False
            # k where d(j,k) == dij  
            mask2 = (dist[j] == dij)
            mask2[i] = False
            # k where d(i,k) == d(j,k)
            mask3 = (dist[i] == dist[j])
            mask3[i] = False
            mask3[j] = False
            
            combined = mask1 | mask2 | mask3
            ks = np.where(combined)[0]
            pair_conflict[i * V + j] = set(ks.tolist())
            pair_conflict[j * V + i] = pair_conflict[i * V + j]
    
    return points, dist, pair_conflict


def _is_no_isosceles(S, dist):
    """Check if S is a no-isosceles set."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = S_list[i], S_list[j]
            dab = dist[a, b]
            for k_idx in range(n):
                if k_idx == i or k_idx == j:
                    continue
                c = S_list[k_idx]
                dac = dist[a, c]
                dbc = dist[b, c]
                if dab == dac or dab == dbc or dac == dbc:
                    return False
    return True


def _is_maximal(S, dist, pair_conflict):
    """Check if S is maximal: no point outside S can be added without creating an isosceles triangle."""
    S_set = set(S)
    for p in range(V):
        if p in S_set:
            continue
        # Check if p can be added
        can_add = True
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                a, b = S[i], S[j]
                # If p is in the conflict set of pair (a,b), it cannot be added
                if p in pair_conflict[a * V + b]:
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            return False
    return True


def _greedy_build(dist, pair_conflict, rng, time_deadline):
    """
    Greedy construction: repeatedly add the point that blocks the most other points.
    Returns a maximal no-isosceles set.
    """
    S = []
    U = set(range(V))
    
    while U:
        if time.time() > time_deadline:
            break
        
        # Find all candidates that can be added
        candidates = []
        for p in U:
            can_add = True
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    a, b = S[i], S[j]
                    if p in pair_conflict[a * V + b]:
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                candidates.append(p)
        
        if not candidates:
            break
        
        # Score each candidate by how many points it would block
        best_score = -1
        best_p = None
        
        cand_list = candidates[:]
        rng.shuffle(cand_list)
        
        for p in cand_list:
            # Count how many points in U would be blocked
            new_blocked = 0
            for a in S:
                # For each existing point a, count points u in U that conflict with pair (a, p)
                for u in U:
                    if u == p:
                        continue
                    if u in pair_conflict[a * V + p]:
                        new_blocked += 1
            
            # Also consider that p itself is removed from U
            score = new_blocked + 1
            
            if score > best_score:
                best_score = score
                best_p = p
            elif score == best_score and rng.random() < 0.3:
                best_p = p
        
        if best_p is None:
            break
        
        S.append(best_p)
        U.remove(best_p)
        
        # Remove blocked points from U
        to_remove = set()
        for a in S[:-1]:  # All points except the newly added one
            for u in U:
                if u in pair_conflict[a * V + best_p]:
                    to_remove.add(u)
        U -= to_remove
    
    return S


def _local_search(S, dist, pair_conflict, rng, time_deadline):
    """
    Local search: try to remove points and swap points to minimize set size while maintaining maximality.
    """
    S = list(S)
    improved = True
    while improved and time.time() < time_deadline:
        improved = False
        
        # Try removing each point
        for idx in range(len(S)):
            if time.time() > time_deadline:
                break
            trial_S = S[:idx] + S[idx+1:]
            if not trial_S:
                continue
            
            # Check if still no-isosceles
            if not _is_no_isosceles(trial_S, dist):
                continue
            
            # Check if still maximal
            if _is_maximal(trial_S, dist, pair_conflict):
                S = trial_S
                improved = True
                break
        
        # Try swapping: remove one point, add another
        if time.time() > time_deadline:
            break
        for remove_idx in range(len(S)):
            if time.time() > time_deadline:
                break
            S_without = S[:remove_idx] + S[remove_idx+1:]
            removed_point = S[remove_idx]
            
            # Find candidate points to add
            for p in range(V):
                if p in S_without or p == removed_point:
                    continue
                
                trial_S = S_without + [p]
                
                # Check no-isosceles
                if not _is_no_isosceles(trial_S, dist):
                    continue
                
                # Check maximal
                if _is_maximal(trial_S, dist, pair_conflict):
                    S = trial_S
                    improved = True
                    break
            
            if improved:
                break
    
    return S


def _construct_direct(dist, pair_conflict, time_budget=280.0):
    """
    Main construction:
    1. Multiple greedy builds with different random seeds.
    2. Local search on each result.
    3. Return the smallest maximal no-isosceles set found.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_size = V
    best_set = None
    
    # Try multiple random seeds
    for trial in range(200):
        if time.time() > deadline:
            break
        
        seed = trial * 7919 + 12345
        rng = random.Random(seed)
        
        # Greedy build
        S = _greedy_build(dist, pair_conflict, rng, deadline)
        
        if not S:
            continue
        
        # Local search
        S = _local_search(S, dist, pair_conflict, rng, deadline)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
    
    if best_set is None:
        return []
    
    return best_set


def construct():
    points, dist, pair_conflict = _precompute()
    result_indices = _construct_direct(dist, pair_conflict)
    return [points[i] for i in result_indices]


if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
