"""No-isosceles set construction on [0, N)^2 grid."""
import time
import random
import math
import numpy as np

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 275.0  # safety margin before 330s kill
    
    # Pre-compute all grid points and indices
    all_points = [(x, y) for x in range(N) for y in range(N)]
    M = len(all_points)
    point_to_idx = {p: i for i, p in enumerate(all_points)}
    
    # Pre-compute squared distance matrix using uint16 for memory efficiency
    coords = np.array(all_points)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    dx = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
    dy = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
    D = (dx * dx + dy * dy).astype(np.uint16)
    
    # Precompute for each point, the set of distances to all other points
    # This allows O(1) lookup for conflict detection
    # dist_to_points[i] = set of distances from point i to all other points
    # But we need something faster: for each point, precompute which other points
    # are at each distance
    
    # For fast validity check: when adding candidate c to set S,
    # We need to check:
    # 1. No two points in S are at the same distance from c (apex at c)
    # 2. For each point a in S, no other point b in S is at distance D[c,a] from a (apex at a or b)
    
    # Precompute: for each point i and distance d, the list of points j such that D[i,j] == d
    # This is expensive to store fully, so we'll use a different approach
    
    # Alternative: for each point i, precompute a sorted list of (distance, point) pairs
    # Then for a candidate c and set S, we can quickly check conflicts
    
    # Actually, let's use a simpler but faster approach:
    # For each point i, precompute the distance histogram as a dict: dist -> count
    # But we need point-specific info, not just counts
    
    # Best approach: precompute for each pair (i, j) the distance, and for each point i,
    # group other points by distance
    
    # dist_groups[i] = {d: [j1, j2, ...]} where D[i, j] == d
    dist_groups = [{} for _ in range(M)]
    for i in range(M):
        row = D[i]
        for j in range(M):
            if i != j:
                d = int(row[j])
                if d not in dist_groups[i]:
                    dist_groups[i][d] = []
                dist_groups[i][d].append(j)
    
    # Convert to numpy arrays for faster access
    # dist_groups_arr[i] = list of (distance, points_array)
    dist_groups_arr = []
    for i in range(M):
        groups = []
        for d in sorted(dist_groups[i].keys()):
            groups.append((d, np.array(dist_groups[i][d], dtype=np.int32)))
        dist_groups_arr.append(groups)
    
    def is_valid_addition_fast(candidate_idx, current_set_indices):
        """Check if adding candidate_idx to current_set creates any isosceles triangle.
        Uses precomputed distance groups for speed."""
        n = len(current_set_indices)
        if n <= 1:
            return True
        
        cur_arr = np.array(current_set_indices, dtype=np.int32)
        
        # Check 1: No two points in S are at the same distance from candidate
        d_c_s = D[candidate_idx, cur_arr]
        if len(np.unique(d_c_s)) != n:
            return False
        
        # Check 2: For each point a in S, no other point b in S is at distance D[candidate, a] from a
        # For each a in S, check if D[candidate, a] appears in D[a, S\{a}]
        # We can use the precomputed dist_groups for this
        
        for a_idx in current_set_indices:
            d_ca = int(D[candidate_idx, a_idx])
            # Check if any other point b in S (b != a) has D[a, b] == d_ca
            # Using dist_groups: get all points at distance d_ca from a
            if d_ca in dist_groups[a_idx]:
                points_at_d = dist_groups[a_idx][d_ca]
                # Check if any of these points is in current_set (and not a itself)
                # Convert to set for O(1) lookup - but we already have in_set
                # Actually, we can check by seeing if any point in points_at_d is in current_set
                # Since current_set is small, we can just check
                for b in points_at_d:
                    if b != a_idx and b in current_set_indices_set:
                        return False
        
        return True
    
    # We need current_set_indices_set for the above check, so let's restructure
    # Actually, let's use a different approach that's cleaner
    
    def is_valid_addition_v2(candidate_idx, current_set_list, in_set):
        """Check if adding candidate_idx to current_set creates any isosceles triangle."""
        n = len(current_set_list)
        if n <= 1:
            return True
        
        cur_arr = np.array(current_set_list, dtype=np.int32)
        
        # Check 1: No two points in S are at the same distance from candidate
        d_c_s = D[candidate_idx, cur_arr]
        if len(np.unique(d_c_s)) != n:
            return False
        
        # Check 2: For each point a in S, no other point b in S is at distance D[candidate, a] from a
        for a_idx in current_set_list:
            d_ca = int(D[candidate_idx, a_idx])
            # Check if any other point b in S (b != a) has D[a, b] == d_ca
            if d_ca in dist_groups[a_idx]:
                points_at_d = dist_groups[a_idx][d_ca]
                for b in points_at_d:
                    if b != a_idx and b in in_set:
                        return False
        
        return True
    
    def greedy_construct():
        """Greedy construction with random ordering."""
        current_set = []
        in_set = set()
        remaining = list(range(M))
        random.shuffle(remaining)
        
        for candidate_idx in remaining:
            if time.time() > deadline:
                break
            if candidate_idx in in_set:
                continue
            if is_valid_addition_v2(candidate_idx, current_set, in_set):
                current_set.append(candidate_idx)
                in_set.add(candidate_idx)
                
        return current_set
    
    def local_search(initial_set, time_budget=30.0):
        """Local search with add and swap operations."""
        current_set = list(initial_set)
        in_set = set(current_set)
        best_set = list(current_set)
        best_size = len(current_set)
        
        not_in_set = [i for i in range(M) if i not in in_set]
        
        # Track time more carefully
        search_deadline = time.time() + time_budget
        
        while time.time() < search_deadline:
            # Try to add a random point from not_in_set
            if not_in_set:
                # Sample multiple candidates and try the first valid one
                sample_size = min(20, len(not_in_set))
                candidates = random.sample(not_in_set, sample_size)
                for candidate_idx in candidates:
                    if is_valid_addition_v2(candidate_idx, current_set, in_set):
                        current_set.append(candidate_idx)
                        in_set.add(candidate_idx)
                        not_in_set.remove(candidate_idx)
                        if len(current_set) > best_size:
                            best_set = list(current_set)
                            best_size = len(current_set)
                        break
            
            # Try swap: remove one point, try to add another
            if len(current_set) > 0 and random.random() < 0.5:
                idx_to_remove = random.randint(0, len(current_set) - 1)
                removed_idx = current_set.pop(idx_to_remove)
                in_set.discard(removed_idx)
                not_in_set.append(removed_idx)
                
                # Try to add a better point from not_in_set
                added = False
                sample_size = min(30, len(not_in_set))
                if sample_size > 0:
                    candidates = random.sample(not_in_set, sample_size)
                    for candidate_idx in candidates:
                        if is_valid_addition_v2(candidate_idx, current_set, in_set):
                            current_set.append(candidate_idx)
                            in_set.add(candidate_idx)
                            not_in_set.remove(candidate_idx)
                            added = True
                            if len(current_set) > best_size:
                                best_set = list(current_set)
                                best_size = len(current_set)
                            break
                
                if not added:
                    current_set.append(removed_idx)
                    in_set.add(removed_idx)
                    not_in_set.remove(removed_idx)
        
        return best_set
    
    def polish_pass(initial_set, time_budget=10.0):
        """Final polish: try to add points with a more systematic approach."""
        current_set = list(initial_set)
        in_set = set(current_set)
        not_in_set = [i for i in range(M) if i not in in_set]
        
        polish_deadline = time.time() + time_budget
        
        # Try all remaining points in order of "least conflict"
        # Score each candidate by how few conflicts it would create
        while time.time() < polish_deadline:
            if not not_in_set:
                break
            
            # For each candidate, count how many points in current_set it conflicts with
            # A conflict means adding it would create an isosceles triangle
            # We can approximate by checking if it's at the same distance from two points in S
            # or if for some a in S, D[c,a] equals D[a,b] for some b in S
            
            # Try a few random candidates and pick the best
            sample_size = min(10, len(not_in_set))
            candidates = random.sample(not_in_set, sample_size)
            
            best_candidate = None
            best_conflicts = float('inf')
            
            for candidate_idx in candidates:
                # Count conflicts
                cur_arr = np.array(current_set, dtype=np.int32)
                if len(cur_arr) > 0:
                    d_c_s = D[candidate_idx, cur_arr]
                    # Number of duplicate distances
                    conflicts = len(d_c_s) - len(np.unique(d_c_s))
                    
                    # Also check apex-at-other conflicts
                    for a_idx in current_set:
                        d_ca = int(D[candidate_idx, a_idx])
                        if d_ca in dist_groups[a_idx]:
                            points_at_d = dist_groups[a_idx][d_ca]
                            for b in points_at_d:
                                if b != a_idx and b in in_set:
                                    conflicts += 1
                
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_candidate = candidate_idx
            
            if best_candidate is not None and best_conflicts == 0:
                current_set.append(best_candidate)
                in_set.add(best_candidate)
                not_in_set.remove(best_candidate)
            
            elif best_candidate is not None and best_conflicts <= 2 and random.random() < 0.3:
                # Occasionally accept low-conflict candidates to escape local optima
                current_set.append(best_candidate)
                in_set.add(best_candidate)
                not_in_set.remove(best_candidate)
            
            else:
                # No good candidates, break
                break
        
        return current_set
    
    best_overall = []
    best_size_overall = 0
    
    num_restarts = 12
    
    for restart in range(num_restarts):
        if time.time() > deadline - 50:
            break
        
        random.seed(restart * 12345 + 67890)
        
        current_set = greedy_construct()
        if len(current_set) > best_size_overall:
            best_overall = list(current_set)
            best_size_overall = len(current_set)
        
        # Do local search if we have time
        time_remaining = deadline - time.time()
        if time_remaining > 30:
            search_time = min(20.0, time_remaining - 25)
            improved = local_search(current_set, time_budget=search_time)
            if len(improved) > best_size_overall:
                best_overall = list(improved)
                best_size_overall = len(improved)
        
        # Final polish
        time_remaining = deadline - time.time()
        if time_remaining > 15:
            polish_time = min(8.0, time_remaining - 10)
            polished = polish_pass(best_overall, time_budget=polish_time)
            if len(polished) > best_size_overall:
                best_overall = list(polished)
                best_size_overall = len(polished)
    
    # Convert indices back to points
    return [all_points[i] for i in best_overall]
