"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

MAX_SUM = 256 * 30  # Maximum possible subset sum for 30 elements each <= 256
MAX_ELEMENTS = 30

def _precompute_divisibility_forbidden():
    """Precompute which pairs of sums (a, b) with a < b have b % a == 0."""
    # forbidden[s] is a set of sums t > s such that t % s == 0
    # We'll use a different approach: for each sum s, store the set of multiples
    forbidden = {}
    for s in range(1, MAX_SUM + 1):
        multiples = set()
        for k in range(2, MAX_SUM // s + 1):
            multiples.add(s * k)
        forbidden[s] = multiples
    return forbidden

def _is_valid_set(s, forbidden=None):
    """Check if a set of integers has the required properties."""
    if forbidden is None:
        forbidden = _precompute_divisibility_forbidden()
    
    n = len(s)
    if n == 0:
        return True
    
    # Generate all nonempty subset sums
    subset_sums = set()
    for mask in range(1, 1 << n):
        total = 0
        for i in range(n):
            if mask & (1 << i):
                total += s[i]
        subset_sums.add(total)
    
    # Check divisibility constraint: no two distinct sums divide each other
    sums_list = sorted(subset_sums)
    for i, s_val in enumerate(sums_list):
        if s_val in forbidden and any(t in forbidden[s_val] for t in sums_list[i+1:]):
            return False
    
    return True

def _get_subset_sums(elements):
    """Get all nonempty subset sums as a set."""
    n = len(elements)
    sums = set()
    for mask in range(1, 1 << n):
        total = 0
        for i in range(n):
            if mask & (1 << i):
                total += elements[i]
        sums.add(total)
    return sums

def _check_validity_incremental(current_sums, new_element, forbidden):
    """Check if adding new_element to a set with existing subset sums maintains validity."""
    # New sums will be: old_sums, {new_element}, and {old_sum + new_element for old_sum in current_sums}
    new_sums = set(current_sums)
    new_sums.add(new_element)
    for s in current_sums:
        new_sums.add(s + new_element)
    
    # Check for divisibility conflicts among all pairs in new_sums
    sorted_sums = sorted(new_sums)
    for i, a in enumerate(sorted_sums):
        if a in forbidden:
            for j in range(i + 1, len(sorted_sums)):
                b = sorted_sums[j]
                if b in forbidden[a]:
                    return None  # Invalid
    return new_sums

def _backtrack_search(deadline, forbidden):
    """Backtracking search for a valid set of maximum size using incremental validity."""
    best = []
    best_size = 0
    nodes_visited = 0
    
    def backtrack(current_elements, current_sums, start_val):
        nonlocal best, best_size, nodes_visited
        if time.time() > deadline:
            return
        
        nodes_visited += 1
        if len(current_elements) > best_size:
            best = current_elements[:]
            best_size = len(current_elements)
        
        # Try to add more elements
        for val in range(start_val, 257):
            if time.time() > deadline:
                return
            
            new_sums = _check_validity_incremental(current_sums, val, forbidden)
            if new_sums is not None:
                backtrack(current_elements + [val], new_sums, val + 1)
    
    backtrack([], set(), 2)
    return best

def _try_multiple_starts(deadline, forbidden):
    """Try backtracking from multiple different starting points."""
    best_overall = []
    
    # Try starting with different initial elements
    for start in range(2, 100):  # Limit to reasonable starting values
        if time.time() > deadline:
            break
        
        initial_sums = {start}
        # Check if singleton is valid (always is)
        result = _backtrack_search_with_start([start], initial_sums, deadline, forbidden)
        if len(result) > len(best_overall):
            best_overall = result
        if len(best_overall) >= 6:
            break
    
    return best_overall

def _backtrack_search_with_start(initial_elements, initial_sums, deadline, forbidden):
    """Backtracking search starting from given initial elements."""
    best = initial_elements[:]
    best_size = len(initial_elements)
    
    def backtrack(current_elements, current_sums, start_val):
        nonlocal best, best_size
        if time.time() > deadline:
            return
        
        if len(current_elements) > best_size:
            best = current_elements[:]
            best_size = len(current_elements)
        
        # Try to add more elements
        for val in range(start_val, 257):
            if time.time() > deadline:
                return
            
            new_sums = _check_validity_incremental(current_sums, val, forbidden)
            if new_sums is not None:
                backtrack(current_elements + [val], new_sums, val + 1)
    
    backtrack(initial_elements, initial_sums, initial_elements[-1] + 1 if initial_elements else 2)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    # Precompute forbidden divisibility pairs
    forbidden = _precompute_divisibility_forbidden()
    
    # Try multiple starting points
    result = _try_multiple_starts(deadline, forbidden)
    
    # Ensure we return a list of Python ints
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
