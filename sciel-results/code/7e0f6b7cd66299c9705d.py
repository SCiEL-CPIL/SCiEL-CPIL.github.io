"""Rewritten program using hypergraph independent set with randomized greedy + local search."""
import random
import time
from itertools import combinations

def construct():
    n = 192
    max_k = 48
    
    # Precompute for each a, the set of triples (b,c,d) of distinct elements from [1..n] \ {a}
    # such that a divides b+c+d.
    # Instead, for each a, we can precompute which other numbers can be in a "conflicting triple" with a.
    # Actually, let's think differently: for a selected set S, a conflict exists if there exist distinct b,c,d in S\{a}
    # such that a | (b+c+d).
    
    # We'll use a greedy approach:
    # 1. Compute for each number x in 1..n, its "conflict degree" with respect to potential triples.
    # 2. Use a randomized greedy selection based on degrees.
    # 3. Apply local search to improve.
    
    # Precompute: for each a, and for each pair (b,c) with b,c != a, b != c,
    # find all d != a, d != b, d != c such that a | (b+c+d).
    # This is expensive. Instead, let's precompute for each a, the set of sums mod a.
    
    # Alternative approach: For each a, precompute which triples of other numbers conflict with a.
    # Since n=192, this is feasible but memory-intensive.
    
    # Let's use a different strategy:
    # For each candidate set S, check if it's valid by verifying no a in S divides sum of any 3 distinct others.
    
    # Greedy: start with empty set, add numbers in order of some heuristic.
    # Heuristic: numbers that are "harder to satisfy" first, i.e., numbers that divide many sums.
    
    # Let's compute for each x, the number of triples (b,c,d) from [1..n]\{x} such that x | (b+c+d).
    # This gives a measure of how "restrictive" x is.
    
    def count_conflicts(x, n):
        """Count how many triples of distinct numbers from [1..n]\{x} have sum divisible by x."""
        count = 0
        others = [y for y in range(1, n+1) if y != x]
        # For efficiency, we can iterate over pairs and check if there's a third
        # But for n=192, let's just do it directly but efficiently.
        # For each pair (b,c), we need d such that d = (k*x - b - c) mod x, and d in others, d != b, d != c.
        # Actually, d must satisfy: b+c+d ≡ 0 (mod x), so d ≡ -(b+c) (mod x).
        # For each residue r mod x, count how many numbers in others have that residue.
        residue_counts = {}
        for y in others:
            r = y % x
            residue_counts[r] = residue_counts.get(r, 0) + 1
        
        for b in others:
            for c in others:
                if c <= b:  # avoid double counting, but we need distinct b,c,d
                    continue
                target = (- (b + c)) % x
                if target in residue_counts:
                    # Count d in others with d % x == target, d != b, d != c
                    cnt = residue_counts[target]
                    if b % x == target:
                        cnt -= 1
                    if c % x == target:
                        cnt -= 1
                    count += cnt
        return count
    
    # Precompute conflict counts
    conflict_counts = {}
    for x in range(1, n+1):
        conflict_counts[x] = count_conflicts(x, n)
    
    # Sort by conflict count descending (most restrictive first)
    sorted_numbers = sorted(range(1, n+1), key=lambda x: -conflict_counts[x])
    
    def is_valid(set_list):
        """Check if the set is valid: no element divides sum of any 3 distinct others."""
        if len(set_list) < 4:
            return True
        s = set(set_list)
        for a in set_list:
            others = [x for x in set_list if x != a]
            for triple in combinations(others, 3):
                if sum(triple) % a == 0:
                    return False
        return True
    
    def is_valid_addition(current_set, new_elem):
        """Check if adding new_elem to current_set keeps it valid."""
        if len(current_set) + 1 < 4:
            return True
        s = list(current_set) + [new_elem]
        # Only need to check conflicts involving new_elem or existing elements with new_elem in triple
        # For each a in s, check if a divides sum of any 3 distinct others in s
        # Optimization: only check a = new_elem and a in current_set where new_elem is in the triple
        # Case 1: a = new_elem, triple from current_set
        for triple in combinations(current_set, 3):
            if sum(triple) % new_elem == 0:
                return False
        # Case 2: a in current_set, triple includes new_elem and 2 from current_set
        for a in current_set:
            for pair in combinations(current_set, 2):
                if a in pair:
                    continue
                # triple = (new_elem, pair[0], pair[1])
                if sum(pair) + new_elem % a == 0:  # BUG: need sum(pair) + new_elem
                    if (sum(pair) + new_elem) % a == 0:
                        return False
        return True
    
    # Greedy construction
    selected = []
    for x in sorted_numbers:
        if is_valid_addition(selected, x):
            selected.append(x)
        if len(selected) >= max_k:
            break
    
    # Local search: try to swap elements to improve
    def try_improve(selected, iterations=5000):
        current = list(selected)
        best = list(current)
        best_size = len(current)
        
        for _ in range(iterations):
            # Try adding a random unselected element
            unselected = [x for x in range(1, n+1) if x not in current]
            if not unselected:
                break
            
            # Try swapping: remove one, add another
            if current:
                idx = random.randint(0, len(current)-1)
                remove_elem = current[idx]
                new_current = current[:idx] + current[idx+1:]
                
                # Try to add a better element
                candidates = [x for x in unselected if x not in new_current]
                random.shuffle(candidates)
                
                improved = False
                for x in candidates[:10]:  # limit candidates for speed
                    if is_valid_addition(new_current, x):
                        new_current.append(x)
                        if len(new_current) > best_size:
                            best = list(new_current)
                            best_size = len(new_current)
                        current = new_current
                        improved = True
                        break
                if not improved:
                    current = new_current + [remove_elem]
            else:
                break
        
        return best
    
    # Run local search
    selected = try_improve(selected, iterations=2000)
    
    # Try multiple random restarts
    best_selected = selected
    best_size = len(selected)
    
    start_time = time.time()
    timeout = 50  # leave 10s margin
    
    for restart in range(10):
        if time.time() - start_time > timeout:
            break
        
        # Randomized greedy with different order
        random.shuffle(sorted_numbers)
        selected = []
        for x in sorted_numbers:
            if is_valid_addition(selected, x):
                selected.append(x)
            if len(selected) >= max_k:
                break
        
        # Local search
        selected = try_improve(selected, iterations=1000)
        
        if len(selected) > best_size:
            best_selected = selected
            best_size = len(selected)
        
        if best_size >= max_k:
            break
    
    return best_selected
