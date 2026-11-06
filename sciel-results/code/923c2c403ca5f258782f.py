"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _precompute_triples(N=180):
    """
    Precompute all forbidden triples (a, b, c) with distinct a, b, c in [1, N]
    such that 1/a = 1/b + 1/c.
    
    This means a = bc/(b+c), so b*c must be divisible by (b+c), and a < b, a < c.
    We return a list of sets, where triples[i] is the set of pairs (j, k) 
    (with j < k, j != i, k != i) such that {i, j, k} is a forbidden triple.
    
    Actually, for efficiency in a greedy algorithm, let's store:
    - For each element x, the list of pairs (y, z) with y < z, y != x, z != x,
      such that {x, y, z} is a forbidden triple.
    """
    # For each pair (b, c) with b < c, check if a = bc/(b+c) is a valid integer < min(b,c)
    # and distinct from b and c.
    triples_by_element = [set() for _ in range(N + 1)]
    
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            # a = bc / (b+c)
            numerator = b * c
            denominator = b + c
            if numerator % denominator == 0:
                a = numerator // denominator
                if a >= 1 and a <= N and a != b and a != c:
                    # We have a triple {a, b, c}
                    # Add to each element's conflict set
                    triples_by_element[a].add((b, c))
                    triples_by_element[b].add((a, c))
                    triples_by_element[c].add((a, b))
    
    return triples_by_element

def _is_valid_subset(subset, triples_by_element):
    """
    Check if subset has no forbidden triple.
    subset is a set of integers.
    """
    subset_set = set(subset)
    for x in subset:
        for (y, z) in triples_by_element[x]:
            if y in subset_set and z in subset_set:
                return False
    return True

def _greedy_max_subset(N=180, triples_by_element=None):
    """
    Greedy approach: try to include as many elements as possible.
    We process elements in some order and include them if they don't create a triple.
    """
    if triples_by_element is None:
        triples_by_element = _precompute_triples(N)
    
    # Try different orderings
    best_subset = set()
    
    # Strategy 1: Descending order (include large numbers first)
    for order in [range(N, 0, -1), range(1, N + 1)]:
        current = set()
        for x in order:
            # Check if adding x creates any triple
            conflict = False
            for (y, z) in triples_by_element[x]:
                if y in current and z in current:
                    conflict = True
                    break
            if not conflict:
                current.add(x)
        
        if len(current) > len(best_subset):
            best_subset = current
    
    return best_subset

def _local_search_improve(subset, N=180, triples_by_element=None, max_iters=1000):
    """
    Try to improve the subset by swapping elements or removing/adding.
    """
    if triples_by_element is None:
        triples_by_element = _precompute_triples(N)
    
    current = set(subset)
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # Try removing one element and adding another
        for x in list(current):
            # Try adding an element not in current
            for y in range(1, N + 1):
                if y in current:
                    continue
                # Try: remove x, add y
                new_set = current - {x} | {y}
                if len(new_set) != len(current):
                    continue
                if _is_valid_subset(new_set, triples_by_element):
                    current = new_set
                    improved = True
                    break
            if improved:
                break
        
        # Try adding an element without removing
        for y in range(1, N + 1):
            if y in current:
                continue
            new_set = current | {y}
            if _is_valid_subset(new_set, triples_by_element):
                current = new_set
                improved = True
                break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline with 10s safety margin
    
    N = 180
    triples_by_element = _precompute_triples(N)
    
    # Initial greedy solution
    best_subset = _greedy_max_subset(N, triples_by_element)
    
    # Try to improve with local search
    if time.time() < deadline:
        improved = _local_search_improve(best_subset, N, triples_by_element)
        if len(improved) > len(best_subset):
            best_subset = improved
    
    # Convert to sorted list
    result = sorted(best_subset)
    
    return result
# EVOLVE-BLOCK-END
