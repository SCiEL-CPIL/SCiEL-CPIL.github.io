import random
import time
import math

N = 40
MAX_DIFF = 38  # max difference we care about (1..38)

def construct():
    start_time = time.time()
    timeout = 85.0  # Leave 5s margin for 90s limit
    
    target_diffs = set(range(1, 39))
    n_diffs = len(target_diffs)  # 38
    
    best_S = None
    best_size = 99
    
    # Fast incremental coverage using diff count array
    # diff_counts[d] = number of pairs (i,j) in S with |S[i]-S[j]| = d
    # coverage = number of d in 1..38 where diff_counts[d] > 0
    
    def compute_diff_counts(S):
        """Compute diff_counts array for set S."""
        counts = [0] * (MAX_DIFF + 1)
        S_sorted = sorted(S)
        k = len(S_sorted)
        for i in range(k):
            for j in range(i + 1, k):
                d = S_sorted[j] - S_sorted[i]
                if d <= MAX_DIFF:
                    counts[d] += 1
        return counts
    
    def get_coverage_from_counts(counts):
        """Get coverage from diff_counts array."""
        cov = 0
        for d in range(1, MAX_DIFF + 1):
            if counts[d] > 0:
                cov += 1
        return cov
    
    def update_counts_remove(counts, S_sorted, val):
        """Remove val from S_sorted and update counts. Returns new S_sorted."""
        # Find index of val in S_sorted
        lo, hi = 0, len(S_sorted) - 1
        idx = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if S_sorted[mid] == val:
                idx = mid
                break
            elif S_sorted[mid] < val:
                lo = mid + 1
            else:
                hi = mid - 1
        if idx == -1:
            return S_sorted
        # Subtract contributions
        for i in range(len(S_sorted)):
            if i == idx:
                continue
            d = abs(S_sorted[i] - val)
            if d <= MAX_DIFF:
                counts[d] -= 1
        new_S_sorted = S_sorted[:idx] + S_sorted[idx+1:]
        return new_S_sorted
    
    def update_counts_add(counts, S_sorted, val):
        """Add val to S_sorted and update counts. Returns new S_sorted."""
        # Insert val into S_sorted (maintain sorted order)
        lo, hi = 0, len(S_sorted)
        while lo < hi:
            mid = (lo + hi) // 2
            if S_sorted[mid] < val:
                lo = mid + 1
            else:
                hi = mid
        idx = lo
        # Add contributions
        for i in range(len(S_sorted)):
            d = abs(S_sorted[i] - val)
            if d <= MAX_DIFF:
                counts[d] += 1
        new_S_sorted = S_sorted[:idx] + [val] + S_sorted[idx:]
        return new_S_sorted
    
    # Check if all diffs 1..38 are covered
    def is_full_coverage(counts):
        for d in range(1, MAX_DIFF + 1):
            if counts[d] == 0:
                return False
        return True
    
    # Local search with incremental updates
    def local_search(S, max_iters, time_limit):
        """Run local search starting from S. Returns (S, success)."""
        S_sorted = sorted(S)
        k = len(S_sorted)
        counts = compute_diff_counts(S_sorted)
        current_cov = get_coverage_from_counts(counts)
        
        # Build set for O(1) membership
        S_set = set(S_sorted)
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            
            if is_full_coverage(counts):
                return list(S_sorted), True
            
            # Pick random index
            idx = random.randint(0, k - 1)
            old_val = S_sorted[idx]
            
            # Try a few candidate new values
            best_new_val = None
            best_new_cov = current_cov
            
            for _ in range(8):  # try 8 candidates per step
                new_val = random.randint(1, 39)
                if new_val in S_set and new_val != old_val:
                    continue
                
                # Compute delta coverage efficiently
                # Remove old_val, add new_val, check coverage
                # But this is expensive, so just estimate
                
                # Quick heuristic: check if new_val helps cover missing diffs
                # Find missing diffs
                missing = []
                for d in range(1, MAX_DIFF + 1):
                    if counts[d] == 0:
                        missing.append(d)
                
                # Check if new_val can cover any missing diff with existing elements
                helpful = False
                for d in missing[:10]:  # check first 10 missing
                    if (new_val + d) in S_set or (new_val - d) in S_set:
                        helpful = True
                        break
                
                if helpful:
                    # This candidate is promising
                    best_new_val = new_val
                    break
                elif best_new_val is None:
                    best_new_val = new_val
            
            if best_new_val is None or best_new_val == old_val:
                continue
            
            # Perform the swap
            S_sorted = update_counts_remove(counts, S_sorted, old_val)
            S_set.discard(old_val)
            S_sorted = update_counts_add(counts, S_sorted, best_new_val)
            S_set.add(best_new_val)
            S_sorted = sorted(S_sorted)  # ensure sorted (update_counts_add should maintain it)
            
            new_cov = get_coverage_from_counts(counts)
            
            # Simple hill climbing with occasional random acceptance
            if new_cov >= current_cov:
                current_cov = new_cov
            else:
                # Accept with small probability to escape local optima
                if random.random() < 0.05:
                    current_cov = new_cov
                else:
                    # Revert
                    S_sorted = update_counts_remove(counts, S_sorted, best_new_val)
                    S_set.discard(best_new_val)
                    S_sorted = update_counts_add(counts, S_sorted, old_val)
                    S_set.add(old_val)
                    S_sorted = sorted(S_sorted)
                    current_cov = get_coverage_from_counts(counts)
        
        return list(S_sorted), is_full_coverage(counts)
    
    # Try sizes from 10 upward
    for size in [10, 11, 12]:
        if time.time() - start_time > timeout:
            break
        
        found = False
        num_restarts = 50 if size == 10 else 20
        max_iters_per_restart = 20000
        
        for restart in range(num_restarts):
            if time.time() - start_time > timeout:
                break
            
            # Heuristic initialization: spread out elements
            # Use a structured approach: pick elements that are far apart
            S = []
            # Start with endpoints
            S.append(1)
            S.append(39)
            # Add elements spread across the range
            remaining = size - 2
            if remaining > 0:
                # Use golden-ratio-like spacing
                candidates = []
                for i in range(1, 39):
                    if i not in S:
                        candidates.append(i)
                random.shuffle(candidates)
                S.extend(candidates[:remaining])
            
            S = list(set(S))
            while len(S) < size:
                S.append(random.randint(1, 39))
            S = list(set(S))
            if len(S) != size:
                S = random.sample(range(1, 40), size)
            
            S_found, success = local_search(S, max_iters_per_restart, start_time + timeout)
            
            if success:
                best_S = S_found
                best_size = size
                found = True
                break
        
        if found:
            break
    
    if best_S is None:
        # Fallback: use a larger set that's guaranteed to work
        # With 13+ elements, it's very likely to cover all diffs
        S = random.sample(range(1, 40), 13)
        S_found, success = local_search(S, 50000, start_time + timeout)
        if success:
            best_S = S_found
            best_size = len(S_found)
        else:
            # Dense fallback
            S = list(range(1, 40))
            best_S = S
            best_size = len(S)

    # Construct graph: path + spokes from vertex 0 to each element in best_S
    path_edges = [(i, i+1) for i in range(1, N-1)]
    spoke_edges = [(0, s) for s in best_S]
    edges = path_edges + spoke_edges
    
    S_set = set(best_S)
    cycles = []
    for L in range(3, N+1):
        diff_needed = L - 2
        found_cycle = False
        for u in sorted(best_S):
            v = u + diff_needed
            if v in S_set:
                # Cycle: 0 -> u -> (u+1) -> ... -> v -> 0
                # Length = 1 (0->u) + diff_needed (u to v along path) + 1 (v->0) = diff_needed + 2 = L
                cycle_verts = [0] + list(range(u, v+1))
                cycles.append(cycle_verts)
                found_cycle = True
                break
        if not found_cycle:
            # This shouldn't happen if best_S covers all diffs 1..38
            raise Exception("Cycle not found for length " + str(L))
    
    return {'edges': edges, 'cycles': cycles}
