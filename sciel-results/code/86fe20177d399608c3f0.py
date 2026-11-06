import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    def elapsed():
        return time.time() - start_time
    
    # Precompute GCD matrix
    n = 128
    nums = list(range(1, 129))
    gcd_mat = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
    
    def is_bad_triple(a, b, c):
        """Check if three distinct elements have all pairwise GCDs equal."""
        g1 = gcd_mat[a][b]
        g2 = gcd_mat[a][c]
        g3 = gcd_mat[b][c]
        return g1 == g2 == g3
    
    def is_valid_addition(x, selected):
        """Check if adding x to selected creates no bad triple."""
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                a, b = selected[i], selected[j]
                if is_bad_triple(a, b, x):
                    return False
        return True
    
    def greedy_add(order, selected):
        """Try to add elements in given order."""
        for x in order:
            if x in selected:
                continue
            if is_valid_addition(x, selected):
                selected.append(x)
        return selected
    
    def local_search(selected, max_iters=5000):
        """Try swaps to improve the set."""
        selected_set = set(selected)
        not_in_set = [x for x in range(1, 129) if x not in selected_set]
        improved = True
        iters = 0
        while improved and iters < max_iters and elapsed() < DEADLINE - 5:
            improved = False
            iters += 1
            for i, out_elem in enumerate(list(selected)):
                for in_elem in not_in_set:
                    if in_elem == out_elem:
                        continue
                    # Try swap
                    new_selected = list(selected)
                    new_selected[i] = in_elem
                    new_set = set(new_selected)
                    # Check validity
                    valid = True
                    for a, b in combinations(new_selected, 2):
                        for c in new_selected:
                            if c == a or c == b:
                                continue
                            if is_bad_triple(a, b, c):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid and len(new_selected) > len(selected):
                        selected = new_selected
                        selected_set = new_set
                        not_in_set = [x for x in range(1, 129) if x not in selected_set]
                        improved = True
                        break
                if improved or elapsed() > DEADLINE - 5:
                    break
        return selected
    
    # Multiple strategies for initial construction
    best_selected = []
    
    # Strategy 1: Sequential greedy (original)
    selected = []
    for x in range(1, 129):
        if is_valid_addition(x, selected):
            selected.append(x)
    if len(selected) > len(best_selected):
        best_selected = list(selected)
    
    # Strategy 2: Reverse greedy
    selected = []
    for x in range(128, 0, -1):
        if is_valid_addition(x, selected):
            selected.append(x)
    if len(selected) > len(best_selected):
        best_selected = list(selected)
    
    # Strategy 3: Primes first, then composites
    def is_prime(x):
        if x < 2:
            return False
        for d in range(2, int(x**0.5)+1):
            if x % d == 0:
                return False
        return True
    
    primes = [x for x in range(1, 129) if is_prime(x)]
    composites = [x for x in range(1, 129) if not is_prime(x)]
    selected = []
    for x in primes + composites:
        if is_valid_addition(x, selected):
            selected.append(x)
    if len(selected) > len(best_selected):
        best_selected = list(selected)
    
    # Strategy 4: Composites first
    selected = []
    for x in composites + primes:
        if is_valid_addition(x, selected):
            selected.append(x)
    if len(selected) > len(best_selected):
        best_selected = list(selected)
    
    # Strategy 5: Greedy removal from full set
    selected = list(range(1, 129))
    # Remove elements that are in bad triples
    changed = True
    while changed and elapsed() < DEADLINE - 10:
        changed = False
        # Find all bad triples
        bad_elems = set()
        for a, b, c in combinations(selected, 3):
            if is_bad_triple(a, b, c):
                bad_elems.update([a, b, c])
        if bad_elems:
            # Remove one element from a bad triple (the one with most conflicts)
            # Count conflicts for each element
            conflict_count = {}
            for a, b, c in combinations(selected, 3):
                if is_bad_triple(a, b, c):
                    for e in [a, b, c]:
                        conflict_count[e] = conflict_count.get(e, 0) + 1
            # Remove element with most conflicts
            if conflict_count:
                worst = max(conflict_count, key=conflict_count.get)
                selected.remove(worst)
                changed = True
    
    if len(selected) > len(best_selected):
        best_selected = list(selected)
    
    # Local search on best found so far
    if elapsed() < DEADLINE - 10:
        best_selected = local_search(best_selected)
    
    # More local search iterations if time permits
    while elapsed() < DEADLINE - 3:
        prev_len = len(best_selected)
        best_selected = local_search(best_selected, max_iters=2000)
        if len(best_selected) <= prev_len:
            break
    
    # Ensure nonempty
    if not best_selected:
        best_selected = [1]
    
    return best_selected
