"""
Minimum maximal no-isosceles set via branch-and-bound with aggressive pruning.
"""
import time
import random
import numpy as np
from itertools import product

N = 16
V = N * N  # 256 grid points

def _precompute():
    """Precompute squared distances and isosceles conflict relations."""
    points = list(product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    
    # Squared distance matrix
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sum(diff ** 2, axis=2)
    
    # Precompute per-pair blocked sets as bitmasks
    # pair_blocked[i,j] = bitmask of all k such that {i,j,k} forms isosceles triangle
    pair_blocked = np.zeros((V, V), dtype=object)
    
    for i in range(V):
        for j in range(i + 1, V):
            dab = dist[i, j]
            # k where d(i,k) == dab (k != j)
            mask1 = (dist[i] == dab).copy()
            mask1[j] = False
            # k where d(j,k) == dab (k != i)
            mask2 = (dist[j] == dab).copy()
            mask2[i] = False
            # k where d(i,k) == d(j,k) (perpendicular bisector)
            mask3 = (dist[i] == dist[j]).copy()
            mask3[i] = False
            mask3[j] = False
            
            combined = mask1 | mask2 | mask3
            pair_blocked[i, j] = combined
            pair_blocked[j, i] = combined
    
    return points, dist, pair_blocked

def _is_valid_set(S, pair_blocked):
    """Check if S contains no isosceles triangle using precomputed pair conflicts."""
    S_set = set(S)
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            blocked = pair_blocked[a, b]
            # Check if any other point in S is in the blocked set
            for k in S:
                if k != a and k != b and blocked[k]:
                    return False
    return True

def _is_maximal(S, pair_blocked):
    """Check if S is maximal: no point outside S can be added."""
    S_set = set(S)
    for p in range(V):
        if p in S_set:
            continue
        # Check if adding p would create an isosceles triangle with any pair in S
        can_add = True
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                a, b = S[i], S[j]
                if pair_blocked[a, b][p]:
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            return False
    return True

def _can_add_point(p, S, pair_blocked):
    """Check if point p can be added to set S without creating isosceles."""
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            if pair_blocked[a, b][p]:
                return False
    return True

def _get_uncovered_points(S, pair_blocked):
    """Get set of points that can still be added to S."""
    S_set = set(S)
    uncovered = []
    for p in range(V):
        if p in S_set:
            continue
        if _can_add_point(p, S, pair_blocked):
            uncovered.append(p)
    return set(uncovered)

def _construct_bnb(pair_blocked, points, time_budget=280.0):
    """
    Branch-and-bound with greedy initialization and local search.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_size = V + 1
    best_set = None
    
    rng = random.Random(42)
    
    def _greedy_build(rng):
        """Greedy build: pick point that blocks most remaining points."""
        S = []
        U = set(range(V))
        
        for _ in range(100):
            if time.time() > deadline:
                break
            if not U:
                break
            
            # Find candidates that can be added
            candidates = []
            for p in U:
                if _can_add_point(p, S, pair_blocked):
                    candidates.append(p)
            
            if not candidates:
                break
            
            # Score each candidate by how many points in U it would block
            best_score = -1
            best_p = None
            
            # Sample candidates if too many
            if len(candidates) > 30:
                sampled = rng.sample(candidates, 30)
            else:
                sampled = candidates
            
            for p in sampled:
                # Count how many points in U would become non-addable
                blocked_count = 0
                for s in S:
                    blocked_mask = pair_blocked[p, s]
                    for u in U:
                        if u != p and u != s and blocked_mask[u]:
                            blocked_count += 1
                
                if blocked_count > best_score:
                    best_score = blocked_count
                    best_p = p
                elif blocked_count == best_score and rng.random() < 0.3:
                    best_p = p
            
            if best_p is None:
                break
            
            S.append(best_p)
            U.discard(best_p)
            
            # Remove points that can no longer be added
            new_U = set()
            for u in U:
                if _can_add_point(u, S, pair_blocked):
                    new_U.add(u)
            U = new_U
        
        return S
    
    def _local_search(S, rng):
        """Local search: try removals and swaps to reduce size while maintaining maximality."""
        S = list(S)
        
        # Try to remove points (keep maximality)
        for it in range(50):
            if time.time() > deadline:
                break
            improved = False
            for idx in range(len(S)):
                trial_S = S[:idx] + S[idx+1:]
                if len(trial_S) < 2:
                    continue
                if not _is_valid_set(trial_S, pair_blocked):
                    continue
                if _is_maximal(trial_S, pair_blocked):
                    S = trial_S
                    improved = True
                    break
            if not improved:
                break
        
        # Try swaps
        for it in range(30):
            if time.time() > deadline:
                break
            improved = False
            for idx in range(len(S)):
                for p in range(V):
                    if p in S:
                        continue
                    trial_S = S[:idx] + [p] + S[idx+1:]
                    if not _is_valid_set(trial_S, pair_blocked):
                        continue
                    if _is_maximal(trial_S, pair_blocked):
                        S = trial_S
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return S
    
    def _bnb_search(S, target_size, rng):
        """
        Branch-and-bound: try to find a valid maximal set of size <= target_size.
        S is current partial solution.
        """
        if time.time() > deadline:
            return None
        
        if len(S) >= target_size:
            if _is_valid_set(S, pair_blocked) and _is_maximal(S, pair_blocked):
                return list(S)
            return None
        
        # Get candidates that can be added
        S_set = set(S)
        candidates = []
        for p in range(V):
            if p in S_set:
                continue
            if _can_add_point(p, S, pair_blocked):
                candidates.append(p)
        
        if not candidates:
            if _is_maximal(S, pair_blocked):
                return list(S)
            return None
        
        # Shuffle candidates for diversity
        rng.shuffle(candidates)
        
        # Limit branching factor
        max_branch = 8
        if len(candidates) > max_branch:
            # Prioritize candidates that block more points
            scored = []
            for p in candidates:
                blocked_count = 0
                for s in S:
                    blocked_mask = pair_blocked[p, s]
                    for u in range(V):
                        if u not in S_set and u != p and u != s and blocked_mask[u]:
                            blocked_count += 1
                scored.append((blocked_count, p))
            scored.sort(key=lambda x: -x[0])
            candidates = [p for _, p in scored[:max_branch]]
        
        for p in candidates:
            new_S = S + [p]
            result = _bnb_search(new_S, target_size, rng)
            if result is not None:
                return result
        
        return None
    
    # Multiple trials
    for trial in range(300):
        if time.time() > deadline:
            break
        
        seed = trial * 7919 + 12345
        rng = random.Random(seed)
        
        # Greedy build
        S = _greedy_build(rng)
        
        if not S:
            continue
        
        # Local search
        S = _local_search(S, rng)
        
        current_size = len(S)
        
        if current_size < best_size:
            best_size = current_size
            best_set = list(S)
        
        # Try B&B to find smaller set
        if current_size > 5:  # Only try if we have a decent starting point
            target = current_size - 1
            result = _bnb_search(S, target, rng)
            if result is not None and len(result) < best_size:
                best_size = len(result)
                best_set = list(result)
    
    if best_set is None:
        return []
    
    return [points[i] for i in best_set]

def construct():
    points, dist, pair_blocked = _precompute()
    return _construct_bnb(pair_blocked, points)

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
