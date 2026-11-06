import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    def is_valid_with_incremental(S, x):
        """Check if adding x to valid set S keeps it valid.
        S is assumed valid. x is not in S.
        Returns True if S ∪ {x} is valid."""
        # Check for each a in S: does a divide sum of any 3 distinct from (S ∪ {x}) \ {a}?
        # New triples involving x: x + b + c where b,c distinct in S\{a}
        for a in S:
            # Triples with x: x + b + c, b,c in S\{a}, b != c
            others = [v for v in S if v != a]
            for i in range(len(others)):
                for j in range(i+1, len(others)):
                    if (x + others[i] + others[j]) % a == 0:
                        return False
        # Check for a = x: does x divide sum of any 3 distinct from S?
        for triple in combinations(S, 3):
            if sum(triple) % x == 0:
                return False
        return True
    
    # Greedy phase: try from high to low
    selected = []
    for x in range(192, 0, -1):
        if time.time() > deadline - 10:
            break
        if is_valid_with_incremental(selected, x):
            selected.append(x)
    
    # Local search phase: try swaps to improve
    best_set = list(selected)
    best_size = len(best_set)
    
    # Try to add more elements by removing some
    # Or try swaps
    while time.time() < deadline - 5:
        improved = False
        # Try adding an element not in the set
        for x in range(192, 0, -1):
            if x in best_set:
                continue
            # Try adding x
            if is_valid_with_incremental(best_set, x):
                best_set.append(x)
                best_size += 1
                improved = True
                break
        if improved:
            continue
        
        # Try swaps: remove one, add one
        for i in range(len(best_set)):
            if time.time() > deadline - 5:
                break
            removed = best_set[i]
            temp_set = [v for j, v in enumerate(best_set) if j != i]
            for x in range(192, 0, -1):
                if x in temp_set or x == removed:
                    continue
                # Check if temp_set ∪ {x} is valid
                # temp_set is valid (subset of valid set)
                if is_valid_with_incremental(temp_set, x):
                    best_set = temp_set + [x]
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try more aggressive: remove two, add one or two
            break
    
    return best_set
