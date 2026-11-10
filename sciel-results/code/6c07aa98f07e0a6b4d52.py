import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_set = []
    best_size = 0
    
    def greedy_build(order):
        """Build a B2[2] set by greedily adding elements in the given order."""
        current_set = []
        counts = {}
        
        for x in order:
            valid = True
            
            for y in current_set:
                s = x + y
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            
            if valid:
                s2 = 2 * x
                if counts.get(s2, 0) >= 2:
                    valid = False
            
            if valid:
                for y in current_set:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                
                s2 = 2 * x
                counts[s2] = counts.get(s2, 0) + 1
                
                current_set.append(x)
        
        return current_set
    
    def build_counts(s):
        """Build sum-count dictionary for a set."""
        counts = {}
        sorted_s = sorted(s)
        for i in range(len(sorted_s)):
            a = sorted_s[i]
            for j in range(i, len(sorted_s)):
                b = sorted_s[j]
                sm = a + b
                counts[sm] = counts.get(sm, 0) + 1
        return counts
    
    def can_add(x, s, counts):
        """Check if x can be added to set s with given counts."""
        for y in s:
            sm = x + y
            if counts.get(sm, 0) >= 2:
                return False
        s2 = 2 * x
        if counts.get(s2, 0) >= 2:
            return False
        return True
    
    def add_element(x, s, counts):
        """Add x to set s, updating counts in place."""
        s.add(x)
        for y in s:
            if y != x:
                sm = x + y
                counts[sm] = counts.get(sm, 0) + 1
        s2 = 2 * x
        counts[s2] = counts.get(s2, 0) + 1
    
    def local_search(initial_set):
        """Improve the set by trying to add elements and swap out elements."""
        current_set = set(initial_set)
        counts = build_counts(current_set)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding elements not in the set
            for x in range(1, 1001):
                if x in current_set:
                    continue
                
                if can_add(x, current_set, counts):
                    add_element(x, current_set, counts)
                    improved = True
                    break
            
            if not improved:
                # Try swapping: remove one element, add another
                sorted_set = sorted(current_set)
                found_swap = False
                for x in sorted_set:
                    if time.time() >= deadline:
                        break
                    
                    # Remove x and rebuild counts
                    trial_set = current_set - {x}
                    trial_counts = build_counts(trial_set)
                    
                    # Find a candidate to add
                    for y in range(1, 1001):
                        if y in trial_set:
                            continue
                        
                        if can_add(y, trial_set, trial_counts):
                            # Perform the swap
                            current_set = trial_set
                            counts = trial_counts
                            add_element(y, current_set, counts)
                            found_swap = True
                            break
                    
                    if found_swap:
                        improved = True
                        break
        
        return list(current_set)
    
    # Try different greedy orderings
    orderings = []
    
    # Random orderings
    for _ in range(40):
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        orderings.append(candidates)
    
    # Ascending
    orderings.append(list(range(1, 1001)))
    
    # Descending
    orderings.append(list(range(1000, 0, -1)))
    
    # Weighted: prefer smaller numbers
    candidates = list(range(1, 1001))
    random.shuffle(candidates)
    candidates.sort(key=lambda x: x + random.random() * 100)
    orderings.append(candidates)
    
    # Sparse: prefer numbers with large gaps
    candidates = list(range(1, 1001))
    random.shuffle(candidates)
    candidates.sort(key=lambda x: random.random())
    orderings.append(candidates)
    
    # Geometric spacing
    candidates = list(range(1, 1001))
    candidates.sort(key=lambda x: -random.random() * x)
    orderings.append(candidates)
    
    # Multi-start: collect top greedy results, then local search each
    greedy_results = []
    for order in orderings:
        if time.time() >= deadline - 5:
            break
        
        result = greedy_build(order)
        greedy_results.append(result)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Apply local search to top greedy results (not just the best)
    greedy_results.sort(key=len, reverse=True)
    unique_results = []
    seen = set()
    for r in greedy_results:
        key = frozenset(r)
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
    
    for r in unique_results[:5]:  # Local search top 5 distinct results
        if time.time() >= deadline - 10:
            break
        
        improved_set = local_search(r)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set[:]
    
    return best_set
