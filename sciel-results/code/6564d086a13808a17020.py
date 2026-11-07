import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin before 90s kill

    def is_valid(sums):
        """Check if any two distinct sums divide each other."""
        if len(sums) < 2:
            return True
        sum_list = sorted(sums)
        max_sum = sum_list[-1]
        for u in sum_list:
            if u * 2 > max_sum:
                break
            for v in range(u * 2, max_sum + 1, u):
                if v in sums:
                    return False
        return True

    def try_add_element(current_sums, c):
        """Try adding element c. Returns new_sums if valid, else None."""
        # Optimization: if adding c causes a collision (s1 + c == s2 for existing s1, s2),
        # it's invalid. But we also need to check divisibility.
        # The collision check is: (current_sums + c) intersects current_sums.
        # i.e., exists s in current_sums such that s - c is in current_sums (and s-c > 0).
        # Or simpler: just build the set and check size? 
        # Building the set is O(N). Checking size is O(1).
        # But we also need to check divisibility, which is expensive.
        # Let's do the collision check first to fail fast.
        
        # Quick collision check
        # We want to know if any (s + c) is already in current_sums.
        # This is equivalent to checking if (current_sums - c) intersects current_sums.
        # But we can't just subtract a set.
        # Let's just build the new set. It's fast enough.
        
        new_sums = set(current_sums)
        for s in current_sums:
            ns = s + c
            if ns in new_sums:
                return None # Collision: two subsets have same sum
            new_sums.add(ns)
        new_sums.add(c)
        
        if not is_valid(new_sums):
            return None
            
        return new_sums

    def get_subset_sums(elements):
        """Return set of all nonempty subset sums."""
        sums = set()
        n = len(elements)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            sums.add(s)
        return sums

    best_set = []
    best_size = 0

    # Strategy 1: Deterministic Greedy (Large to Small)
    # Large numbers have fewer multiples, making divisibility less likely.
    # Also, large numbers are less likely to cause sum collisions with smaller sets?
    # Actually, collisions are about additive structure.
    
    # Try different orderings
    orderings = [
        list(range(256, 0, -1)),
        list(range(256, 128, -1)) + list(range(127, 0, -1)),
        [x for x in range(1, 257) if x % 2 == 1], # Odd numbers only?
        [x for x in range(1, 257) if x % 2 == 0], # Even numbers only?
    ]
    
    for ordering in orderings:
        if time.time() > deadline - 5.0:
            break
        current = []
        current_sums = set()
        for c in ordering:
            if len(current) >= 30:
                break
            new_sums = try_add_element(current_sums, c)
            if new_sums is not None:
                current.append(c)
                current_sums = new_sums
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            if best_size >= 30:
                break

    # Strategy 2: Randomized Greedy with Restart
    # This is the main workhorse.
    # We try to build a set by picking random candidates, biased towards larger values initially.
    
    # Pre-compute some useful ranges
    all_candidates = list(range(1, 257))
    
    for attempt in range(2000):
        if time.time() > deadline - 3.0:
            break
        
        current = []
        current_sums = set()
        
        # Create a shuffled list of candidates, but with a bias
        # We want to try adding elements.
        # Let's shuffle the candidates.
        candidates = all_candidates[:]
        random.shuffle(candidates)
        
        # To improve success rate, we can try to "greedily" pick from the shuffled list.
        # If an element fails, we just skip it.
        
        for c in candidates:
            if len(current) >= 30:
                break
            new_sums = try_add_element(current_sums, c)
            if new_sums is not None:
                current.append(c)
                current_sums = new_sums
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            if best_size >= 30:
                break

    # Strategy 3: Local Search / Hill Climbing on Best Set
    # If we have a good set, try to improve it by swapping elements.
    # Or try to add elements by removing others.
    
    if best_size > 0 and time.time() < deadline - 2.0:
        # We try to extend the best set.
        # The greedy method got stuck because no single element could be added.
        # Maybe swapping one element allows adding two?
        # Or maybe removing one element allows adding two?
        
        # Let's try a simple "remove one, add two" move.
        # Or just "swap one for one" to change the landscape, then try adding.
        
        current = best_set[:]
        current_sums = get_subset_sums(current)
        
        # Try to add any element from 1..256 not in current
        # If we can't, try swapping.
        
        # Actually, a more powerful move is:
        # 1. Pick a random element 'old' from current.
        # 2. Remove it.
        # 3. Try to add a random element 'new'.
        # 4. If valid, accept.
        # 5. Then try to add another element.
        
        # Let's do a few iterations of this.
        
        for _ in range(1000):
            if time.time() > deadline - 1.0:
                break
            
            # Try to add a random element first
            c = random.choice(all_candidates)
            if c in current:
                continue
            new_sums = try_add_element(current_sums, c)
            if new_sums is not None:
                current.append(c)
                current_sums = new_sums
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = current[:]
                continue
            
            # If adding failed, try a swap
            if len(current) == 0:
                continue
                
            idx = random.randint(0, len(current) - 1)
            old_elem = current[idx]
            
            # Try to swap old_elem with a random new_elem
            new_elem = random.choice(all_candidates)
            if new_elem == old_elem or new_elem in current:
                continue
                
            # Remove old_elem
            # This is tricky to do incrementally on sums.
            # We have to recompute sums for the set without old_elem.
            # For size ~15, 2^15 = 32k sums. Recomputing is fast.
            
            reduced_set = current[:idx] + current[idx+1:]
            red_sums = get_subset_sums(reduced_set)
            
            # Now try to add new_elem to reduced_set
            new_sums = try_add_element(red_sums, new_elem)
            if new_sums is not None:
                # Swap accepted
                current = reduced_set + [new_elem]
                current_sums = new_sums
                # Note: size stays same. But composition changed.
                # This might allow adding an element in the next iteration.
                # We don't update best_size here because size didn't increase.
                # But we update best_set if we want to keep the best composition?
                # No, we only care about size for the final answer.
                # But keeping a "good" composition helps.
                # Since size is same, we don't update best_set unless we find a larger one later.
                # However, we should keep 'current' as the active state for further swaps/adds.
                
                # Try to add an element immediately after a successful swap
                for c2 in all_candidates:
                    if c2 in current:
                        continue
                    new_sums2 = try_add_element(current_sums, c2)
                    if new_sums2 is not None:
                        current.append(c2)
                        current_sums = new_sums2
                        if len(current) > best_size:
                            best_size = len(current)
                            best_set = current[:]
                        break

    # Ensure types are correct
    # The problem requires Python ints.
    return [int(x) for x in best_set]

if __name__ == "__main__":
    # Quick self-test
    result = construct()
    print(f"Result size: {len(result)}")
    print(f"Result: {result}")
    
    # Verify
    if result:
        sums = set()
        n = len(result)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += result[i]
            if s in sums:
                print("INVALID: Duplicate sum")
                break
            sums.add(s)
        
        sum_list = sorted(sums)
        valid = True
        for u in sum_list:
            if u * 2 > sum_list[-1]:
                break
            for v in range(u * 2, sum_list[-1] + 1, u):
                if v in sums:
                    print(f"INVALID: {u} divides {v}")
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            print("VALID")
        else:
            print("INVALID")
    else:
        print("Empty set")
