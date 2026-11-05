import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s
    
    N = 256
    elements = list(range(1, N + 1))
    
    # Build conflict graph: conflict[a][b] = True if (a*b) % (a+b) == 0
    # Use bitsets for efficiency
    # conflict[i] is a set of j such that i and j conflict
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a].add(b)
                conflict[b].add(a)
    
    # Convert to sorted lists for faster iteration
    conflict_list = [sorted(conflict[i]) for i in range(N + 1)]
    
    # Degree of each node
    degree = [len(conflict_list[i]) for i in range(N + 1)]
    
    def is_valid_subset(subset):
        """Check if subset is a valid independent set."""
        s = set(subset)
        for a in subset:
            for b in conflict_list[a]:
                if b in s and b > a:
                    return False
        return True
    
    def count_conflicts_in_subset(subset):
        """Count number of conflicting pairs in subset."""
        s = set(subset)
        count = 0
        for a in subset:
            for b in conflict_list[a]:
                if b in s and b > a:
                    count += 1
        return count
    
    # Greedy initialization: start with all, remove most-connected first
    def greedy_init():
        # Start with all elements
        included = set(range(1, N + 1))
        # Repeatedly remove the element with most conflicts within included
        while True:
            if time.time() > DEADLINE:
                break
            # Find element with most conflicts within included
            max_conf = -1
            max_elem = None
            for elem in included:
                conf_in = 0
                for c in conflict_list[elem]:
                    if c in included:
                        conf_in += 1
                if conf_in > max_conf:
                    max_conf = conf_in
                    max_elem = elem
            if max_conf <= 0:
                break
            included.remove(max_elem)
        return included
    
    # Local search: try to add back removed elements
    def local_search(subset):
        subset = set(subset)
        removed = set(range(1, N + 1)) - subset
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            # Try to add each removed element
            for elem in list(removed):
                # Check if adding elem creates conflicts
                conflicts_with_included = [c for c in conflict_list[elem] if c in subset]
                if not conflicts_with_included:
                    subset.add(elem)
                    removed.discard(elem)
                    improved = True
            # Try 1-for-1 swaps: remove one from subset, add one from removed
            if not improved:
                for add_elem in list(removed):
                    conflicts_with_included = [c for c in conflict_list[add_elem] if c in subset]
                    if len(conflicts_with_included) == 1:
                        rem_elem = conflicts_with_included[0]
                        # Check if removing rem_elem and adding add_elem is valid
                        # add_elem must not conflict with (subset - {rem_elem})
                        ok = True
                        for c in conflict_list[add_elem]:
                            if c in subset and c != rem_elem:
                                ok = False
                                break
                        if ok:
                            subset.discard(rem_elem)
                            subset.add(add_elem)
                            removed.discard(add_elem)
                            removed.add(rem_elem)
                            improved = True
                            if time.time() > DEADLINE:
                                break
        return subset
    
    # Try 1-for-2 swaps
    def local_search_2(subset):
        subset = set(subset)
        removed = set(range(1, N + 1)) - subset
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            # Try 1-for-2: remove one from subset, add two from removed
            for add1 in list(removed):
                if time.time() > DEADLINE:
                    break
                conflicts1 = [c for c in conflict_list[add1] if c in subset]
                if len(conflicts1) == 1:
                    rem_elem = conflicts1[0]
                    # Check add1 doesn't conflict with subset - {rem_elem}
                    ok1 = True
                    for c in conflict_list[add1]:
                        if c in subset and c != rem_elem:
                            ok1 = False
                            break
                    if not ok1:
                        continue
                    # Now try to also add another element from removed
                    for add2 in list(removed):
                        if add2 == add1:
                            continue
                        # add2 must not conflict with (subset - {rem_elem}) or add1
                        ok2 = True
                        for c in conflict_list[add2]:
                            if c in subset and c != rem_elem:
                                ok2 = False
                                break
                            if c == add1:
                                ok2 = False
                                break
                        if ok2:
                            # Valid 1-for-2 swap
                            subset.discard(rem_elem)
                            subset.add(add1)
                            subset.add(add2)
                            removed.discard(add1)
                            removed.discard(add2)
                            removed.add(rem_elem)
                            improved = True
                            break
                    if improved:
                        break
        return subset
    
    # Run multiple strategies
    best_subset = set()
    best_size = 0
    
    # Strategy 1: Greedy + local search
    if time.time() < DEADLINE:
        s1 = greedy_init()
        s1 = local_search(s1)
        s1 = local_search_2(s1)
        if len(s1) > best_size:
            best_size = len(s1)
            best_subset = s1
    
    # Strategy 2: Start with all, do aggressive local search
    if time.time() < DEADLINE:
        s2 = set(range(1, N + 1))
        # Remove conflicts greedily
        while True:
            if time.time() > DEADLINE:
                break
            found = False
            for a in sorted(s2):
                for b in conflict_list[a]:
                    if b in s2 and b > a:
                        # Remove the one with higher degree
                        if degree[a] >= degree[b]:
                            s2.discard(b)
                        else:
                            s2.discard(a)
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        s2 = local_search(s2)
        s2 = local_search_2(s2)
        if len(s2) > best_size:
            best_size = len(s2)
            best_subset = s2
    
    # Strategy 3: Randomized greedy
    for trial in range(5):
        if time.time() > DEADLINE - 5:
            break
        random.seed(42 + trial)
        s3 = set(range(1, N + 1))
        # Random order removal
        order = list(range(1, N + 1))
        random.shuffle(order)
        for elem in order:
            if elem not in s3:
                continue
            # Check if elem conflicts with any in s3
            has_conflict = False
            for c in conflict_list[elem]:
                if c in s3 and c != elem:
                    has_conflict = True
                    break
            if has_conflict:
                s3.discard(elem)
        s3 = local_search(s3)
        s3 = local_search_2(s3)
        if len(s3) > best_size:
            best_size = len(s3)
            best_subset = s3
    
    # Strategy 4: Simulated annealing style
    if time.time() < DEADLINE - 10:
        s4 = set(best_subset)
        removed4 = set(range(1, N + 1)) - s4
        current_size = len(s4)
        best_s4 = set(s4)
        best_size4 = current_size
        temperature = 1.0
        for it in range(5000):
            if time.time() > DEADLINE - 5:
                break
            if not removed4:
                break
            # Pick a random element from removed4 to try to add
            add_elem = random.choice(list(removed4))
            conflicts = [c for c in conflict_list[add_elem] if c in s4]
            if not conflicts:
                s4.add(add_elem)
                removed4.discard(add_elem)
                if len(s4) > best_size4:
                    best_size4 = len(s4)
                    best_s4 = set(s4)
            elif len(conflicts) == 1:
                rem_elem = conflicts[0]
                # Check if valid swap
                ok = True
                for c in conflict_list[add_elem]:
                    if c in s4 and c != rem_elem:
                        ok = False
                        break
                if ok:
                    # Accept with probability based on temperature
                    if random.random() < temperature:
                        s4.discard(rem_elem)
                        s4.add(add_elem)
                        removed4.discard(add_elem)
                        removed4.add(rem_elem)
                        if len(s4) > best_size4:
                            best_size4 = len(s4)
                            best_s4 = set(s4)
            elif len(conflicts) == 2:
                # Try 1-for-2
                rem1, rem2 = conflicts
                ok = True
                for c in conflict_list[add_elem]:
                    if c in s4 and c not in (rem1, rem2):
                        ok = False
                        break
                if ok:
                    if random.random() < temperature * 0.5:
                        s4.discard(rem1)
                        s4.discard(rem2)
                        s4.add(add_elem)
                        removed4.discard(add_elem)
                        removed4.add(rem1)
                        removed4.add(rem2)
                        if len(s4) > best_size4:
                            best_size4 = len(s4)
                            best_s4 = set(s4)
            temperature *= 0.999
            if temperature < 0.01:
                break
        
        if best_size4 > best_size:
            best_size = best_size4
            best_subset = best_s4
    
    # Final polish: try to add any remaining elements
    if time.time() < DEADLINE - 2:
        for elem in range(1, N + 1):
            if elem in best_subset:
                continue
            conflicts = [c for c in conflict_list[elem] if c in best_subset]
            if not conflicts:
                best_subset.add(elem)
    
    if time.time() > DEADLINE:
        # Emergency: return what we have
        pass
    
    return sorted(best_subset)
