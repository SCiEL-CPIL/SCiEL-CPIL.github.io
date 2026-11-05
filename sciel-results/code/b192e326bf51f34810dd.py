"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    def is_valid_addition(x, S):
        """Check if adding x to set S is valid."""
        # S is a list of selected integers
        S_set = set(S)
        # Check that x doesn't divide sum of any 3 distinct other elements in S
        for a in S:
            # Sum of 3 distinct elements from S (excluding a)
            others = [v for v in S if v != a]
            if len(others) >= 3:
                for tri in combinations(others, 3):
                    if (sum(tri) % x == 0):
                        return False
            # Also check x against sums of 3 elements that include x? No, x is new
            # Check that existing elements don't divide sum of any 3 others including x
        # Check that no existing element divides sum of any 3 distinct OTHER selected elements
        # The "other" elements must be distinct from the tested element
        # For each a in S, we need to check sums of 3 distinct elements from S ∪ {x} excluding a
        # If the 3 elements don't include x, this was already checked when a was added
        # If the 3 elements include x, we need to check: a does not divide (x + b + c) for b,c in S, b≠a, c≠a, b≠c
        for a in S:
            others_no_a = [v for v in S if v != a]
            # Triples including x and two from others_no_a
            for pair in combinations(others_no_a, 2):
                s = x + pair[0] + pair[1]
                if s % a == 0:
                    return False
            # Triples of 3 from others_no_a (already checked when those were added, but let's be safe)
            # Actually those were checked, so we can skip
        return True

    def is_valid_set(S):
        """Check if the entire set S is valid."""
        S_set = set(S)
        for a in S:
            others = [v for v in S if v != a]
            if len(others) >= 3:
                for tri in combinations(others, 3):
                    if sum(tri) % a == 0:
                        return False
        return True

    # Greedy approach with randomization
    best = []
    best_size = 0
    
    # First, try a deterministic greedy from large to small
    selected = []
    for x in range(192, 0, -1):
        if is_valid_addition(x, selected):
            selected.append(x)
    
    if len(selected) > best_size:
        best = selected[:]
        best_size = len(selected)
    
    # Try randomized restarts
    random.seed(42)
    attempts = 0
    while time.time() < deadline and attempts < 50:
        attempts += 1
        # Randomly shuffle the order of consideration
        candidates = list(range(192, 0, -1))
        random.shuffle(candidates)
        
        selected = []
        for x in candidates:
            if time.time() > deadline - 5:
                break
            if is_valid_addition(x, selected):
                selected.append(x)
        
        if len(selected) > best_size:
            best = selected[:]
            best_size = len(selected)
        
        # Try local search: try to add more elements or swap
        # Simple local search: try to add each unselected element
        unselected = [x for x in range(1, 193) if x not in set(selected)]
        for x in unselected:
            if time.time() > deadline - 3:
                break
            if is_valid_addition(x, selected):
                selected.append(x)
                if len(selected) > best_size:
                    best = selected[:]
                    best_size = len(selected)
        
        # Try swapping: remove one element, try to add others
        for i in range(len(selected)):
            if time.time() > deadline - 2:
                break
            removed = selected[i]
            temp = selected[:i] + selected[i+1:]
            added = 0
            for x in unselected:
                if time.time() > deadline - 1:
                    break
                if x == removed:
                    continue
                if is_valid_addition(x, temp):
                    temp.append(x)
                    added += 1
            if len(temp) > len(selected):
                selected = temp
                unselected = [x for x in range(1, 193) if x not in set(selected)]
                if len(selected) > best_size:
                    best = selected[:]
                    best_size = len(selected)
    
    return best
# EVOLVE-BLOCK-END
