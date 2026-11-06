import numpy as np
import random
import time

N = 64
NUM_POINTS = N * N

# Precompute coordinates
ys, xs = np.meshgrid(np.arange(N), np.arange(N))
coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.int32)

# Precompute squared distances using uint16 for memory efficiency
# Max distance squared = 2*(63)^2 = 7938, fits in uint16
dx = coords[:, 0][:, np.newaxis] - coords[:, 0][np.newaxis, :]
dy = coords[:, 1][:, np.newaxis] - coords[:, 1][np.newaxis, :]
dist_sq = (dx * dx + dy * dy).astype(np.uint16)
del dx, dy

# Precompute distance groups: dist_groups[i][d] = list of points j where dist_sq[i,j] == d
# This enables fast conflict checking
dist_groups = [{} for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    row = dist_sq[i]
    for j in range(NUM_POINTS):
        if i != j:
            d = int(row[j])
            if d not in dist_groups[i]:
                dist_groups[i][d] = []
            dist_groups[i][d].append(j)

def construct():
    start_time = time.time()
    deadline = start_time + 275.0  # Safety margin before 330s kill
    
    best_set = []
    best_size = 0
    
    def is_valid_addition(candidate_idx, current_set, in_set):
        """Check if adding candidate_idx to current_set creates any isosceles triangle."""
        n = len(current_set)
        if n <= 1:
            return True
        
        # Check 1: No two points in S are at the same distance from candidate
        # Collect distances from candidate to all points in S
        seen_dists = set()
        for a in current_set:
            d = int(dist_sq[candidate_idx, a])
            if d in seen_dists:
                return False
            seen_dists.add(d)
        
        # Check 2: For each point a in S, no other point b in S is at distance D[candidate, a] from a
        for a in current_set:
            d_ca = int(dist_sq[candidate_idx, a])
            # Check if any other point b in S has D[a, b] == d_ca
            if d_ca in dist_groups[a]:
                for b in dist_groups[a][d_ca]:
                    if b != a and b in in_set:
                        return False
        
        return True
    
    def greedy_construct():
        """Greedy construction with random ordering."""
        current_set = []
        in_set = set()
        indices = list(range(NUM_POINTS))
        random.shuffle(indices)
        
        for candidate_idx in indices:
            if time.time() > deadline:
                break
            if candidate_idx in in_set:
                continue
            if is_valid_addition(candidate_idx, current_set, in_set):
                current_set.append(candidate_idx)
                in_set.add(candidate_idx)
        
        return current_set
    
    def local_search(initial_set, time_budget=15.0):
        """Local search with add and swap operations."""
        current_set = list(initial_set)
        in_set = set(current_set)
        best_set_local = list(current_set)
        best_size_local = len(current_set)
        
        not_in_set = [i for i in range(NUM_POINTS) if i not in in_set]
        
        search_deadline = time.time() + time_budget
        
        while time.time() < search_deadline:
            improved = False
            
            # Try to add a random point from not_in_set
            if not_in_set:
                sample_size = min(30, len(not_in_set))
                candidates = random.sample(not_in_set, sample_size)
                for candidate_idx in candidates:
                    if is_valid_addition(candidate_idx, current_set, in_set):
                        current_set.append(candidate_idx)
                        in_set.add(candidate_idx)
                        not_in_set.remove(candidate_idx)
                        improved = True
                        if len(current_set) > best_size_local:
                            best_size_local = len(current_set)
                            best_set_local = list(current_set)
                        break
            
            # Try swap: remove one point, try to add another
            if len(current_set) > 0 and not_in_set and random.random() < 0.4:
                idx_to_remove = random.randint(0, len(current_set) - 1)
                removed_idx = current_set.pop(idx_to_remove)
                in_set.discard(removed_idx)
                
                # Try to find a replacement
                sample_size = min(30, len(not_in_set))
                if sample_size > 0:
                    candidates = random.sample(not_in_set, sample_size)
                    found_replacement = False
                    for candidate_idx in candidates:
                        if is_valid_addition(candidate_idx, current_set, in_set):
                            current_set.append(candidate_idx)
                            in_set.add(candidate_idx)
                            not_in_set.remove(candidate_idx)
                            found_replacement = True
                            if len(current_set) > best_size_local:
                                best_size_local = len(current_set)
                                best_set_local = list(current_set)
                            break
                    
                    if not found_replacement:
                        # Put the removed point back
                        current_set.append(removed_idx)
                        in_set.add(removed_idx)
                        not_in_set.append(removed_idx)
            
            if not improved and time.time() > search_deadline - 2:
                break
        
        return best_set_local
    
    def polish_pass(initial_set):
        """Systematic final improvement: try adding every remaining point."""
        current_set = list(initial_set)
        in_set = set(current_set)
        best_set_local = list(current_set)
        best_size_local = len(current_set)
        
        not_in_set = [i for i in range(NUM_POINTS) if i not in in_set]
        random.shuffle(not_in_set)
        
        for candidate_idx in not_in_set:
            if time.time() > deadline:
                break
            if is_valid_addition(candidate_idx, current_set, in_set):
                current_set.append(candidate_idx)
                in_set.add(candidate_idx)
                if len(current_set) > best_size_local:
                    best_size_local = len(current_set)
                    best_set_local = list(current_set)
        
        return best_set_local
    
    # Run multiple greedy restarts with local search and polish
    while time.time() < deadline - 20:  # Leave 20s for final polish
        greedy_result = greedy_construct()
        
        if len(greedy_result) > best_size:
            best_size = len(greedy_result)
            best_set = greedy_result[:]
        
        # Apply local search if we have enough time
        if time.time() < deadline - 25:
            local_result = local_search(greedy_result, time_budget=min(10.0, deadline - time.time() - 25))
            if len(local_result) > best_size:
                best_size = len(local_result)
                best_set = local_result[:]
        
        # Apply polish pass
        if time.time() < deadline - 5:
            polished = polish_pass(best_set)
            if len(polished) > best_size:
                best_size = len(polished)
                best_set = polished[:]
    
    # Final polish on the best set found
    if time.time() < deadline - 2:
        final_polished = polish_pass(best_set)
        if len(final_polished) > best_size:
            best_set = final_polished
    
    return [(int(coords[p][0]), int(coords[p][1])) for p in best_set]
