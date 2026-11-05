import time
from itertools import combinations
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    """
    
    # Precompute GCDs for all pairs in [1, 128]
    # gcd_table[i][j] = gcd(i, j)
    # Using 1-based indexing for simplicity, size 129x129
    gcd_table = [[0] * 129 for _ in range(129)]
    for i in range(1, 129):
        for j in range(i, 129):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def is_valid_triple_fast(a, b, c):
        """Check if three elements form a bad triple using precomputed table."""
        g_ab = gcd_table[a][b]
        g_ac = gcd_table[a][c]
        g_bc = gcd_table[b][c]
        return g_ab == g_ac == g_bc

    def count_conflicts_fast(x, selected):
        """Count how many bad triplets would be formed if x were added to selected."""
        conflicts = 0
        # selected is a set of ints
        # We need to check pairs (i, j) in selected
        # Optimization: Iterate over selected
        sel_list = list(selected)
        n = len(sel_list)
        for i in range(n):
            a = sel_list[i]
            g_ax = gcd_table[x][a]
            for j in range(i+1, n):
                b = sel_list[j]
                # Check if gcd(x,a) == gcd(x,b) == gcd(a,b)
                if g_ax == gcd_table[x][b] == gcd_table[a][b]:
                    conflicts += 1
        return conflicts

    def greedy_build(order):
        """Build a valid set by adding elements in the given order."""
        selected = set()
        for x in order:
            if count_conflicts_fast(x, selected) == 0:
                selected.add(x)
        return selected

    def local_search_fast(selected, max_iters=100):
        """Try to improve the set by removing/adding elements."""
        selected = set(selected)
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            
            # 1. Try adding any excluded element
            excluded = [x for x in range(1, 129) if x not in selected]
            for x in excluded:
                if count_conflicts_fast(x, selected) == 0:
                    selected.add(x)
                    improved = True
                    break
            
            # 2. If no add, try 2-for-1 exchange: remove 1, add 2
            if not improved:
                found_improvement = False
                for x in list(selected):
                    temp = selected - {x}
                    # Try to add as many as possible from excluded
                    added = []
                    for y in excluded:
                        if y == x:
                            continue
                        if count_conflicts_fast(y, temp | set(added)) == 0:
                            added.append(y)
                    if len(added) >= 2:
                        new_set = temp | set(added)
                        if len(new_set) > len(selected):
                            selected = new_set
                            improved = True
                            found_improvement = True
                            break
                    elif len(added) == 1:
                        # Try 1-for-1 swap as fallback
                        new_set = temp | set(added)
                        if len(new_set) == len(selected):
                            # Check if this swap helps by trying further additions
                            further_added = []
                            for z in excluded:
                                if z == x or z in added:
                                    continue
                                if count_conflicts_fast(z, new_set | set(further_added)) == 0:
                                    further_added.append(z)
                            if further_added:
                                selected = new_set | set(further_added)
                                improved = True
                                found_improvement = True
                                break
                if found_improvement:
                    continue

            # 3. If still no improvement, try random perturbation
            if not improved:
                if random.random() < 0.15:
                    if selected:
                        x = random.choice(list(selected))
                        selected.remove(x)
                        improved = True

        return selected

    deadline = time.time() + 87.0  # Safety margin (limit 90s)
    best_set = set()
    best_size = 0
    
    import random
    random.seed(42)

    # Strategy 1: Greedy in natural order
    order1 = list(range(1, 129))
    s1 = greedy_build(order1)
    s1 = local_search_fast(s1)
    if len(s1) > best_size:
        best_size = len(s1)
        best_set = s1

    if time.time() > deadline:
        return sorted(best_set) if best_set else [1]

    # Strategy 2: Greedy in descending order
    order2 = list(range(128, 0, -1))
    s2 = greedy_build(order2)
    s2 = local_search_fast(s2)
    if len(s2) > best_size:
        best_size = len(s2)
        best_set = s2

    if time.time() > deadline:
        return sorted(best_set) if best_set else [1]

    # Strategy 3: Iterated Local Search with perturbation
    for trial in range(300):
        if time.time() > deadline:
            break
        
        # Perturb the best set found so far
        if best_set and random.random() < 0.5:
            # Start from a perturbed version of best_set
            current = set(best_set)
            # Remove 1-3 random elements
            num_remove = random.randint(1, min(3, len(current)))
            to_remove = random.sample(list(current), num_remove)
            for x in to_remove:
                current.remove(x)
        else:
            # Fresh randomized greedy
            order3 = list(range(1, 129))
            random.shuffle(order3)
            current = greedy_build(order3)
        
        # Local search
        s3 = local_search_fast(current)
        if len(s3) > best_size:
            best_size = len(s3)
            best_set = s3

    if time.time() > deadline:
        return sorted(best_set) if best_set else [1]

    # Strategy 4: Final polish with more iterations on best_set
    s4 = local_search_fast(best_set, max_iters=200)
    if len(s4) > best_size:
        best_size = len(s4)
        best_set = s4

    if not best_set:
        best_set = {1}
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
