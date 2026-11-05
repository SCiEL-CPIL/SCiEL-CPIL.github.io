import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s limit
    n = 256

    # Precompute: for each a, the list of (b,c) pairs where a | (b+c), b<=c
    # Only need to check pairs where b,c are in [1,n]
    # We'll precompute a bitmask or just a set of (b,c) pairs for each a
    # Actually, let's precompute for each a, the set of sums that are multiples of a
    # sums range from 2 to 2n
    bad_sums = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for s in range(2 * a, 2 * n + 1, a):
            bad_sums[a].add(s)

    def count_conflicts(S):
        """Count the number of violating triples (a,b,c) in S."""
        S_sorted = sorted(S)
        m = len(S_sorted)
        count = 0
        for i in range(m):
            a = S_sorted[i]
            for j in range(i + 1, m):
                b = S_sorted[j]
                for k in range(j, m):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        count += 1
        return count

    def get_conflicts(S):
        """Get all conflicting triples in S."""
        S_sorted = sorted(S)
        m = len(S_sorted)
        conflicts = []
        for i in range(m):
            a = S_sorted[i]
            for j in range(i + 1, m):
                b = S_sorted[j]
                for k in range(j, m):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        conflicts.append((a, b, c))
        return conflicts

    def is_valid_set(S):
        """Check if the entire set S is valid."""
        S_sorted = sorted(S)
        m = len(S_sorted)
        for i in range(m):
            a = S_sorted[i]
            for j in range(i + 1, m):
                b = S_sorted[j]
                for k in range(j, m):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True

    best_set = []
    best_size = 0

    def try_improve(S):
        """Try to improve set S using local search."""
        nonlocal best_set, best_size
        S = S[:]
        S_set = set(S)
        
        # Phase 1: Add elements greedily
        for x in range(1, n + 1):
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            # Check if x can be added
            valid = True
            for a in S:
                if a < x:
                    # x is b or c
                    # If x is b: need c >= x, a | (x+c)
                    for c in S:
                        if c >= x:
                            if (x + c) % a == 0:
                                valid = False
                                break
                if not valid:
                    break
                # If x is a: need b,c > x, x | (b+c)
                # This is harder to check incrementally, so we'll do full check
            if valid:
                # Full check
                S_test = S + [x]
                if is_valid_set(S_test):
                    S.append(x)
                    S_set.add(x)
        
        # Phase 2: Local search - try to swap elements
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_conflicts = get_conflicts(S)
            
            # Try removing elements that are in conflicts
            conflict_elements = set()
            for (a, b, c) in current_conflicts:
                conflict_elements.add(a)
                conflict_elements.add(b)
                conflict_elements.add(c)
            
            # Try swapping: remove an element, add another
            S_set = set(S)
            for to_remove in conflict_elements:
                if time.time() > deadline:
                    break
                for to_add in range(1, n + 1):
                    if time.time() > deadline:
                        break
                    if to_add in S_set:
                        continue
                    S_test = [x for x in S if x != to_remove] + [to_add]
                    if len(S_test) == len(S):  # Ensure no duplicate
                        if is_valid_set(S_test):
                            S = S_test
                            S_set = set(S)
                            improved = True
                            break
                if time.time() > deadline:
                    break
            
            # Also try just removing elements to reduce conflicts
            if not improved and current_conflicts:
                # Try removing each conflict element
                for elem in conflict_elements:
                    if time.time() > deadline:
                        break
                    S_test = [x for x in S if x != elem]
                    if len(S_test) < len(S):  # Element was present
                        # Try to add something else
                        S_set = set(S_test)
                        for to_add in range(1, n + 1):
                            if time.time() > deadline:
                                break
                            if to_add in S_set:
                                continue
                            S_try2 = S_test + [to_add]
                            if is_valid_set(S_try2):
                                S = S_try2
                                improved = True
                                break
                        if improved:
                            break
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
        
        return S

    # Strategy 1: Greedy from large to small
    S_greedy = []
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
        S_test = S_greedy + [x]
        if is_valid_set(S_test):
            S_greedy.append(x)
    
    S_greedy = try_improve(S_greedy)

    # Strategy 2: Start with dense upper tail, prune, improve
    for start_val in [128, 140, 150, 160, 180, 200, 100, 110, 130]:
        if time.time() > deadline:
            break
        
        S_try = list(range(start_val, n + 1))
        
        # Prune S_try to make it valid
        # Find conflicts and remove elements
        changed = True
        while changed and time.time() < deadline:
            changed = False
            conflicts = get_conflicts(S_try)
            if not conflicts:
                break
            
            # Remove the element that appears in the most conflicts
            from collections import Counter
            conflict_count = Counter()
            for (a, b, c) in conflicts:
                conflict_count[a] += 1
                conflict_count[b] += 1
                conflict_count[c] += 1
            
            # Remove the element with highest conflict count
            max_count = max(conflict_count.values())
            candidates_to_remove = [elem for elem, count in conflict_count.items() if count == max_count]
            random.shuffle(candidates_to_remove)
            
            for elem in candidates_to_remove:
                if elem in S_try:
                    S_try.remove(elem)
                    changed = True
                    break
        
        S_try = try_improve(S_try)

    # Strategy 3: Random restarts with local improvement
    for attempt in range(40):
        if time.time() > deadline:
            break
        
        # Randomize the order of candidates
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        S_try = []
        for x in candidates:
            if time.time() > deadline:
                break
            S_test = S_try + [x]
            if is_valid_set(S_test):
                S_try.append(x)
        
        S_try = try_improve(S_try)

    # Strategy 4: Simulated annealing
    if time.time() < deadline - 10:
        S_sa = S_greedy[:]
        S_sa = try_improve(S_sa)
        
        current_temp = 1.0
        cooling_rate = 0.95
        min_temp = 0.01
        sa_set = S_sa[:]
        
        while current_temp > min_temp and time.time() < deadline:
            # Generate neighbor: remove one element, add another
            if len(sa_set) == 0:
                break
            
            # Choose an operation
            op = random.random()
            S_set = set(sa_set)
            
            if op < 0.5 and len(sa_set) > 0:
                # Remove and add
                to_remove = random.choice(sa_set)
                to_add = random.randint(1, n)
                if to_add in S_set:
                    continue
                S_test = [x for x in sa_set if x != to_remove] + [to_add]
            elif op < 0.75 and len(sa_set) > 0:
                # Just remove
                to_remove = random.choice(sa_set)
                S_test = [x for x in sa_set if x != to_remove]
            else:
                # Just add
                to_add = random.randint(1, n)
                if to_add in S_set:
                    continue
                S_test = sa_set + [to_add]
            
            if len(S_test) != len(sa_set) + (1 if op >= 0.75 else 0) + (1 if op < 0.5 and len(sa_set) > 0 else 0):
                # Skip if size changed unexpectedly
                pass
            
            # Evaluate
            conflicts_new = count_conflicts(S_test)
            conflicts_old = count_conflicts(sa_set)
            
            # Accept if better, or with probability based on temperature
            if conflicts_new < conflicts_old:
                sa_set = S_test
            elif conflicts_new == conflicts_old:
                if random.random() < 0.5:
                    sa_set = S_test
            else:
                # Worse: accept with probability exp(-(delta)/temp)
                delta = conflicts_new - conflicts_old
                if random.random() < math.exp(-delta / max(current_temp, 0.001)):
                    sa_set = S_test
            
            # Try to improve
            if random.random() < 0.3:
                sa_set = try_improve(sa_set)
            
            current_temp *= cooling_rate
        
        if len(sa_set) > best_size:
            best_size = len(sa_set)
            best_set = sa_set[:]

    # Final polish: Try to add any remaining numbers to best_set
    S_final = best_set[:]
    S_final_set = set(S_final)
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if x in S_final_set:
            continue
        S_test = S_final + [x]
        if is_valid_set(S_test):
            S_final.append(x)
            S_final_set.add(x)
    
    if len(S_final) > best_size:
        best_size = len(S_final)
        best_set = S_final[:]

    return best_set
# EVOLVE-BLOCK-END
