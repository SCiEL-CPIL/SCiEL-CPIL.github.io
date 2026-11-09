"""Improved B2[2] set construction with scoring-based greedy and local search."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def elapsed_ok():
        return time.time() < deadline
    
    def build_greedy(order, max_time=130):
        """Build a B2[2] set by processing candidates in given order."""
        start = time.time()
        chosen = []
        sum_counts = {}
        chosen_set = set()
        
        for x in order:
            if time.time() - start > max_time:
                break
            if x in chosen_set:
                continue
            # Check if adding x keeps all sums <= 2
            ok = True
            # x + x
            s = 2 * x
            if sum_counts.get(s, 0) >= 2:
                ok = False
            else:
                # x + y for y in chosen, y < x
                for y in chosen:
                    if y >= x:
                        continue
                    s = x + y
                    if sum_counts.get(s, 0) >= 2:
                        ok = False
                        break
            if ok:
                chosen.append(x)
                chosen_set.add(x)
                sum_counts[2 * x] = sum_counts.get(2 * x, 0) + 1
                for y in chosen:
                    if y < x:
                        s = x + y
                        sum_counts[s] = sum_counts.get(s, 0) + 1
        return chosen
    
    def local_search(initial, max_time=120):
        """Try to improve the set by removing and re-adding elements."""
        start = time.time()
        chosen = list(initial)
        chosen_set = set(chosen)
        
        # Build sum counts
        sum_counts = {}
        for i, x in enumerate(chosen):
            for j in range(i + 1):
                s = x + chosen[j]
                sum_counts[s] = sum_counts.get(s, 0) + 1
        
        best_size = len(chosen)
        no_improve_count = 0
        max_no_improve = 2000
        
        while time.time() - start < max_time and no_improve_count < max_no_improve:
            # Try random swaps: remove one element, try to add a different one
            if not chosen:
                break
            
            idx = random.randint(0, len(chosen) - 1)
            removed = chosen[idx]
            
            # Remove from counts
            for i, x in enumerate(chosen):
                if i == idx:
                    continue
                s = x + removed
                sum_counts[s] -= 1
                if sum_counts[s] == 0:
                    del sum_counts[s]
            s2 = 2 * removed
            sum_counts[s2] -= 1
            if sum_counts[s2] == 0:
                del sum_counts[s2]
            
            # Try to add a random unused element
            used = chosen_set - {removed}
            candidates = [x for x in range(1, 1001) if x not in used]
            if not candidates:
                # Put it back
                chosen.insert(idx, removed)
                chosen_set.add(removed)
                for i, x in enumerate(chosen):
                    if i == idx:
                        continue
                    s = x + removed
                    sum_counts[s] = sum_counts.get(s, 0) + 1
                s2 = 2 * removed
                sum_counts[s2] = sum_counts.get(s2, 0) + 1
                no_improve_count += 1
                continue
            
            # Try several candidates
            tried = random.sample(candidates, min(10, len(candidates)))
            added = False
            for cand in tried:
                ok = True
                # Check 2*cand
                if sum_counts.get(2 * cand, 0) >= 2:
                    ok = False
                else:
                    for x in chosen:
                        if x >= cand:
                            continue
                        s = x + cand
                        if sum_counts.get(s, 0) >= 2:
                            ok = False
                            break
                if ok:
                    # Add cand
                    chosen.append(cand)
                    chosen_set.add(cand)
                    sum_counts[2 * cand] = sum_counts.get(2 * cand, 0) + 1
                    for x in chosen:
                        if x < cand:
                            s = x + cand
                            sum_counts[s] = sum_counts.get(s, 0) + 1
                    added = True
                    break
            
            if added:
                if len(chosen) > best_size:
                    best_size = len(chosen)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Put removed back
                chosen.insert(idx, removed)
                chosen_set.add(removed)
                for i, x in enumerate(chosen):
                    if i == idx:
                        continue
                    s = x + removed
                    sum_counts[s] = sum_counts.get(s, 0) + 1
                s2 = 2 * removed
                sum_counts[s2] = sum_counts.get(s2, 0) + 1
                no_improve_count += 1
        
        return chosen
    
    # Phase 1: Multiple greedy passes with different orderings
    best_set = []
    
    # Standard ascending greedy
    t0 = time.time()
    s1 = build_greedy(range(1, 1001), max_time=10)
    if len(s1) > len(best_set):
        best_set = s1
    
    # Descending greedy
    if elapsed_ok():
        s2 = build_greedy(range(1000, 0, -1), max_time=10)
        if len(s2) > len(best_set):
            best_set = s2
    
    # Randomized greedy with scoring
    if elapsed_ok():
        # Score-based: prefer elements that create fewer conflicts
        # Use a priority approach: at each step, pick the element with min marginal conflict
        chosen = []
        chosen_set = set()
        sum_counts = {}
        remaining = set(range(1, 1001))
        
        while remaining and elapsed_ok():
            # For each candidate, compute how many sums would exceed capacity
            # Use a sampled subset for speed
            sample = list(remaining)
            if len(sample) > 200:
                sample = random.sample(sample, 200)
            
            best_cand = None
            best_score = float('inf')
            
            for x in sample:
                # Count conflicts
                conflicts = 0
                s = 2 * x
                if sum_counts.get(s, 0) >= 2:
                    conflicts += 1
                else:
                    for y in chosen:
                        if y < x:
                            s = x + y
                            if sum_counts.get(s, 0) >= 2:
                                conflicts += 1
                                if conflicts > best_score:
                                    break
                # Also consider "slack": sums that are at capacity 1
                # Prefer elements that use less constrained sums
                if conflicts < best_score:
                    best_score = conflicts
                    best_cand = x
                    if conflicts == 0:
                        break
            
            if best_cand is None:
                break
            
            # Add best_cand
            chosen.append(best_cand)
            chosen_set.add(best_cand)
            remaining.discard(best_cand)
            sum_counts[2 * best_cand] = sum_counts.get(2 * best_cand, 0) + 1
            for y in chosen:
                if y < best_cand:
                    s = best_cand + y
                    sum_counts[s] = sum_counts.get(s, 0) + 1
        
        if len(chosen) > len(best_set):
            best_set = chosen
    
    # Phase 2: Local search to improve
    if elapsed_ok() and best_set:
        t1 = time.time()
        remaining_time = deadline - time.time() - 5
        if remaining_time > 5:
            improved = local_search(best_set, max_time=min(remaining_time, 120))
            if len(improved) > len(best_set):
                best_set = improved
    
    # Final verification
    result = sorted(best_set)
    # Verify validity
    sum_counts = {}
    valid = True
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            if sum_counts[s] > 2:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: rebuild with simple greedy
        result = build_greedy(range(1, 1001), max_time=10)
    
    return result
# EVOLVE-BLOCK-END
