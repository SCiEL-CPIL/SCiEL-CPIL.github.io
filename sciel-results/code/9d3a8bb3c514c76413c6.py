"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from collections import Counter

def _can_form_target(target, elements, max_size=10):
    """Check if target can be formed as sum of reciprocals of a subset of elements.
    Uses DFS with pruning."""
    if not elements:
        return target == Fraction(0, 1)
    
    # Sort elements by value (larger reciprocals first) for better pruning
    elements = sorted(elements, key=lambda y: -1/y)
    
    def dfs(idx, current_sum, used_count):
        if current_sum == target:
            return True
        if idx >= len(elements) or current_sum > target or used_count >= max_size:
            return False
        # Pruning: even adding all remaining can't reach target
        remaining_max = current_sum
        for i in range(idx, len(elements)):
            remaining_max += Fraction(1, elements[i])
            if remaining_max >= target:
                break
        if remaining_max < target:
            return False
        
        # Option 1: skip element idx
        if dfs(idx + 1, current_sum, used_count):
            return True
        # Option 2: include element idx
        if dfs(idx + 1, current_sum + Fraction(1, elements[idx]), used_count + 1):
            return True
        return False
    
    return dfs(0, Fraction(0, 1), 0)

def _is_valid(S):
    """Check if set S has no reciprocal-subsum violations."""
    S = set(S)
    for x in S:
        others = [y for y in S if y != x]
        target = Fraction(1, x)
        if _can_form_target(target, others):
            return False
    return True

def _find_violations(S):
    """Find all violations in S."""
    S = set(S)
    violations = []
    for x in S:
        others = [y for y in S if y != x]
        target = Fraction(1, x)
        if _can_form_target(target, others):
            violations.append(x)
    return violations

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    # Start with all 24 elements
    S_set = set(range(1, 25))
    
    # Remove elements that cause violations
    while True:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        violations = _find_violations(S_set)
        if not violations:
            break
        
        # Remove one violating element (the smallest one, as it's most likely to cause issues)
        worst = min(violations)
        S_set.remove(worst)
    
    # Try to add back elements
    for _ in range(5):
        changed = False
        candidates = sorted([x for x in range(1, 25) if x not in S_set])
        for x in candidates:
            elapsed = time.time() - start_time
            if elapsed > deadline:
                break
            test_set = S_set | {x}
            if _is_valid(test_set):
                S_set = test_set
                changed = True
        if not changed:
            break
    
    return sorted(S_set)
# EVOLVE-BLOCK-END
