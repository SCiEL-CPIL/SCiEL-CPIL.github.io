import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for 90s limit
    
    # Helper to check if adding x to current_sums is valid
    # Returns (is_valid, new_sums_set)
    def check_addition(current_sums, x):
        # Generate all new subset sums when adding x
        # New sums are: x, and for each s in current_sums, s+x
        new_sums = set()
        for s in current_sums:
            new_sums.add(s + x)
        new_sums.add(x)
        
        # Check for duplicate sums (union should have size len(current_sums) + len(new_sums))
        # If there's an intersection, it means two different subsets have the same sum.
        # This is invalid per the problem statement ("Two different subsets with the same sum still make the set invalid").
        if current_sums & new_sums:
            return False, None
        
        all_sums = current_sums.union(new_sums)
        
        # Check divisibility: no two distinct sums in all_sums should divide each other
        # We only need to check pairs where at least one is in new_sums
        # Because current_sums is already valid.
        
        # Optimization: Sort new_sums and check against all_sums
        # For each s in new_sums:
        #   For each t in all_sums:
        #     if s == t: continue
        #     if s > t and s % t == 0: return False
        #     if t > s and t % s == 0: return False
        
        # To speed this up, we can pre-sort all_sums.
        # But since we do this many times, let's just iterate.
        
        # Further optimization: 
        # If s is small, it divides many things. If s is large, it divides few things.
        
        for s in new_sums:
            # Check s against all other sums
            for t in all_sums:
                if s == t:
                    continue
                if s > t:
                    if s % t == 0:
                        return False, None
                else:
                    if t % s == 0:
                        return False, None
                        
        return True, all_sums

    # Beam Search
    # State: (elements_tuple, sums_set)
    # We want to maximize size.
    
    # Candidates: 1..256
    # We iterate candidates in descending order to prefer larger numbers.
    candidates = list(range(256, 0, -1))
    
    # Beam width
    beam_width = 100
    
    # Initial beam: empty set
    beam = [([], set())]
    
    best_set = []
    best_size = 0
    
    # We will process candidates one by one.
    # At each step, we decide whether to include the current candidate in each state of the beam.
    # This is a standard beam search for subset selection.
    
    for x in candidates:
        if time.time() > deadline:
            break
            
        new_beam = []
        
        for elements, sums in beam:
            # Option 1: Do not include x
            new_beam.append((elements, sums))
            
            # Option 2: Include x
            valid, new_sums = check_addition(sums, x)
            if valid:
                new_elements = elements + (x,)
                new_beam.append((new_elements, new_sums))
                
                if len(new_elements) > best_size:
                    best_size = len(new_elements)
                    best_set = list(new_elements)
        
        # Prune the beam: keep top beam_width states
        # Ranking criteria:
        # 1. Size (number of elements) - Primary
        # 2. Sum of elements - Secondary (larger sums are generally "safer" as they are less likely to be divisors of other sums, though this is a heuristic)
        # 3. Random tie-breaker to maintain diversity
        
        if len(new_beam) > beam_width:
            # Sort by size descending, then by sum of elements descending
            # To avoid expensive sum calculation, we can store sum in state or compute on fly.
            # Let's compute sum on fly for sorting.
            
            # For efficiency, we can just take the first beam_width if we don't care about optimality of the beam composition.
            # But we want the "best" states to survive.
            
            # Custom sort
            # Key: (-len(elements), -sum(elements), random.random())
            # Since we want diversity, we can just randomly sample if the beam is too large, 
            # but keeping the largest is safer for finding a large set.
            
            # Let's sort by size, then random.
            new_beam.sort(key=lambda state: (-len(state[0]), random.random()))
            new_beam = new_beam[:beam_width]
            
        beam = new_beam
        
        # Early termination if we found a very large set?
        # Max possible is 30. If we hit 30, we can stop.
        if best_size >= 30:
            break

    # After processing all candidates, we have a set of potential solutions in the beam.
    # The best one is already tracked in best_set.
    
    # However, the beam search might have pruned a path that leads to a slightly smaller set but allows a much larger extension later?
    # No, because we process candidates in a fixed order. The beam contains states after considering the first i candidates.
    # So the final beam contains the best states considering ALL candidates.
    # We should check the final beam for the absolute best.
    
    for elements, sums in beam:
        if len(elements) > best_size:
            best_size = len(elements)
            best_set = list(elements)

    # Fallback if beam search didn't find anything good (unlikely)
    if not best_set:
        # Try a simple greedy
        current_elements = []
        current_sums = set()
        for x in range(256, 0, -1):
            valid, new_sums = check_addition(current_sums, x)
            if valid:
                current_elements.append(x)
                current_sums = new_sums
        best_set = current_elements
        best_size = len(best_set)
        
    # Ensure all elements are Python int and in valid range
    best_set = [int(x) for x in best_set if 1 <= x <= 256]
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for x in best_set:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    best_set = unique
    
    return best_set
