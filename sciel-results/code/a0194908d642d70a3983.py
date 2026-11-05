"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

def _gcd(a, b):
    return math.gcd(a, b)

def _is_bad_triple(a, b, c):
    """Check if three distinct elements have all three pairwise gcds equal."""
    g1 = _gcd(a, b)
    g2 = _gcd(a, c)
    g3 = _gcd(b, c)
    return g1 == g2 == g3

def _count_conflicts(x, selected_set, all_numbers):
    """Count how many bad triples would be formed if x were added to selected_set."""
    conflicts = 0
    for i, a in enumerate(selected_set):
        for b in selected_set[i+1:]:
            if _is_bad_triple(x, a, b):
                conflicts += 1
    return conflicts

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin
    
    all_numbers = list(range(1, 129))
    n = len(all_numbers)
    
    # Precompute gcd matrix for speed
    gcd_matrix = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            gcd_matrix[i][j] = _gcd(i, j)
            gcd_matrix[j][i] = gcd_matrix[i][j]
    
    def is_bad(a, b, c):
        return gcd_matrix[a][b] == gcd_matrix[a][c] == gcd_matrix[b][c]
    
    # Phase 1: Greedy construction with degree-based selection
    # We want to maximize the size of the set with no bad triples.
    # Strategy: iteratively add the element that creates the fewest conflicts
    # with the current set, preferring elements that are "safe".
    
    # Actually, a better greedy: start with all elements, then remove elements
    # that are in the most bad triples. This is like a maximal independent set
    # in a 3-uniform hypergraph.
    
    # Let's try: start with all 128 elements, then iteratively remove the
    # element that is in the most bad triples.
    
    current_set = set(all_numbers)
    
    def count_triples_with(x, s):
        """Count bad triples that include x and two elements from s."""
        cnt = 0
        s_list = list(s)
        for i in range(len(s_list)):
            a = s_list[i]
            if a == x:
                continue
            for j in range(i + 1, len(s_list)):
                b = s_list[j]
                if b == x:
                    continue
                if is_bad(x, a, b):
                    cnt += 1
        return cnt
    
    # Iterative removal: remove the element in the most bad triples
    # This is a greedy approximation to maximum independent set
    removed = []
    while current_set:
        if time.time() > deadline:
            break
        
        # Find the element with the most bad triples
        max_conflicts = -1
        worst = None
        for x in current_set:
            cnt = count_triples_with(x, current_set)
            if cnt > max_conflicts:
                max_conflicts = cnt
                worst = x
            if cnt == 0:
                break  # No conflicts, can keep all remaining
        
        if worst is None or max_conflicts == 0:
            break
        
        current_set.remove(worst)
        removed.append(worst)
    
    # Phase 2: Try to add back removed elements
    # For each removed element, check if it can be added without creating bad triples
    # Try in order of least conflicts
    for x in removed:
        if time.time() > deadline:
            break
        # Check if x can be added to current_set
        can_add = True
        s_list = list(current_set)
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i + 1, len(s_list)):
                b = s_list[j]
                if is_bad(x, a, b):
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            current_set.add(x)
    
    # Phase 3: Local search - try swaps to improve
    # Try removing one element and adding two others, or swapping
    improved = True
    while improved and time.time() < deadline:
        improved = False
        s_list = list(current_set)
        for i, x in enumerate(s_list):
            if time.time() > deadline:
                break
            # Try removing x and adding a different element y not in current_set
            current_set.remove(x)
            for y in all_numbers:
                if y in current_set:
                    continue
                # Check if y can be added
                can_add = True
                s2 = list(current_set)
                for a in s2:
                    for b in s2:
                        if a >= b:
                            continue
                        if is_bad(y, a, b):
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    current_set.add(y)
                    improved = True
                    break
            if not improved:
                current_set.add(x)
    
    result = sorted(current_set)
    if not result:
        result = [1]
    return result

# EVOLVE-BLOCK-END
