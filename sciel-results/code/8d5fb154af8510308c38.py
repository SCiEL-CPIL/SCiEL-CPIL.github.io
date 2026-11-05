import time
import random
from math import gcd

def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    """
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin (limit is 60s)
    
    N = 128
    
    # Precompute GCD matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute Conflict Lists
    # conflict[x] is a list of pairs (a, b) such that (x, a, b) is a bad triple.
    # i.e., gcd(x,a) == gcd(x,b) == gcd(a,b)
    conflict = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                g_ab = gcd_mat[a][b]
                g_ac = gcd_mat[a][c]
                g_bc = gcd_mat[b][c]
                if g_ab == g_ac and g_ab == g_bc:
                    # Triple (a, b, c) is bad
                    # For x=a, the other two are b, c
                    conflict[a].append((b, c))
                    # For x=b, the other two are a, c
                    conflict[b].append((a, c))
                    # For x=c, the other two are a, b
                    conflict[c].append((a, b))

    def can_add(x, in_set):
        """
        Check if x can be added to the set represented by in_set (list of bools).
        Uses precomputed conflict lists for O(1) per pair check.
        """
        for a, b in conflict[x]:
            if in_set[a] and in_set[b]:
                return False
        return True

    def greedy_fill(in_set, mask, size):
        """
        Greedily add as many elements as possible to in_set.
        Returns (mask, in_set, size) after filling.
        """
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if not in_set[x]:
                    if can_add(x, in_set):
                        in_set[x] = True
                        mask |= (1 << x)
                        size += 1
                        changed = True
                        break
        return mask, in_set, size

    def degree_greedy_build(in_set=None):
        """
        Build set by repeatedly adding the element with lowest conflict degree.
        Degree = number of bad triples involving that element.
        This prioritizes 'safe' elements that have fewer constraints.
        """
        if in_set is None:
            in_set = [False] * (N + 1)
        mask = 0
        size = 0
        
        # Precompute hypergraph degree: number of bad triples each vertex is in
        degree = [0] * (N + 1)
        for x in range(1, N + 1):
            degree[x] = len(conflict[x])
        
        # Repeatedly pick the available element with lowest degree
        while True:
            best_x = -1
            best_deg = float('inf')
            for x in range(1, N + 1):
                if not in_set[x]:
                    if can_add(x, in_set):
                        if degree[x] < best_deg:
                            best_deg = degree[x]
                            best_x = x
            
            if best_x == -1:
                break
            
            in_set[best_x] = True
            mask |= (1 << best_x)
            size += 1
            
            # Update degrees: removing best_x from consideration
            # (we don't actually remove it, but we mark it as in_set)
        
        return mask, in_set, size

    def try_2_1_exchange(in_set, mask, size, deadline):
        """
        Try to find a 2-1 exchange: remove 1 element, add 2 new elements.
        This can escape local optima that 1-1 exchanges cannot.
        Returns (mask, in_set, size) if improvement found, else None.
        """
        current_elements = [i for i in range(1, N + 1) if in_set[i]]
        
        for u in current_elements:
            if time.time() > deadline:
                return None
            
            # Save state
            old_in_set = in_set[:]
            old_mask = mask
            old_size = size
            
            # Remove u
            in_set[u] = False
            mask ^= (1 << u)
            size -= 1
            
            # Try to add two elements that can't be added individually but work together
            # First pass: greedy fill
            mask, in_set, size = greedy_fill(in_set, mask, size)
            
            # Check if we gained net elements
            if size > old_size:
                return mask, in_set, size
            
            # If not improved, revert
            in_set = old_in_set
            mask = old_mask
            size = old_size
        
        return None

    def greedy_build(order, in_set=None):
        """Build set greedily following given order."""
        if in_set is None:
            in_set = [False] * (N + 1)
        mask = 0
        size = 0
        for x in order:
            if not in_set[x]:
                if can_add(x, in_set):
                    in_set[x] = True
                    mask |= (1 << x)
                    size += 1
        return mask, in_set, size

    def local_search(mask, in_set, size, max_iters=500):
        """
        Improve the set via additions, kicks, and 2-1 exchanges.
        Returns the best (mask, in_set, size) found.
        """
        best_mask = mask
        best_in_set = in_set[:]
        best_size = size
        
        iters = 0
        while iters < max_iters and time.time() < deadline:
            iters += 1
            improved = False
            
            # 1. Try to add any element
            mask, in_set, size = greedy_fill(in_set, mask, size)
            
            if size > best_size:
                best_mask = mask
                best_in_set = in_set[:]
                best_size = size
                continue

            # 2. Local maximum reached. Try kicks.
            current_elements = [i for i in range(1, N + 1) if in_set[i]]
            if not current_elements:
                break
            
            # Try random kicks to escape local optimum
            found_better = False
            for _ in range(30):
                if time.time() > deadline:
                    break
                
                # Randomly pick an element to remove
                u = random.choice(current_elements)
                
                # Save state
                old_in_set = in_set[:]
                old_mask = mask
                old_size = size
                
                # Remove u
                in_set[u] = False
                mask ^= (1 << u)
                size -= 1
                
                # Greedily add back as many as possible
                mask, in_set, size = greedy_fill(in_set, mask, size)
                
                # If we improved
                if size > best_size:
                    best_mask = mask
                    best_in_set = in_set[:]
                    best_size = size
                    found_better = True
                    current_elements = [i for i in range(1, N + 1) if in_set[i]]
                    break
                else:
                    # Revert
                    in_set = old_in_set
                    mask = old_mask
                    size = old_size
            
            # 3. Try 2-1 exchange if still stuck
            if not found_better and time.time() < deadline:
                result = try_2_1_exchange(in_set, mask, size, deadline)
                if result is not None:
                    new_mask, new_in_set, new_size = result
                    if new_size > best_size:
                        best_mask = new_mask
                        best_in_set = new_in_set[:]
                        best_size = new_size
                        found_better = True
            
            if not found_better:
                break
                
        return best_mask, best_in_set, best_size

    def polish(mask, in_set, size):
        """
        Final polish: try removing each element one at a time and re-greedy fill.
        This is a systematic 1-kick search.
        """
        best_mask = mask
        best_in_set = in_set[:]
        best_size = size
        
        for u in range(1, N + 1):
            if time.time() > deadline:
                break
            if not in_set[u]:
                continue
            
            # Save state
            old_in_set = in_set[:]
            old_mask = mask
            old_size = size
            
            # Remove u
            in_set[u] = False
            mask ^= (1 << u)
            size -= 1
            
            # Greedily fill
            mask, in_set, size = greedy_fill(in_set, mask, size)
            
            if size > best_size:
                best_mask = mask
                best_in_set = in_set[:]
                best_size = size
            else:
                # Revert
                in_set = old_in_set
                mask = old_mask
                size = old_size
        
        return best_mask, best_in_set, best_size

    best_mask = 0
    best_in_set = None
    best_size = 0
    
    def update_best(mask, in_set, size):
        nonlocal best_mask, best_in_set, best_size
        if size > best_size:
            best_mask = mask
            best_in_set = in_set[:]
            best_size = size

    # Strategy 1: Degree-based greedy (prioritize low-conflict elements)
    mask, in_set, size = degree_greedy_build()
    mask, in_set, size = local_search(mask, in_set, size)
    update_best(mask, in_set, size)
    
    # Strategy 2: Greedy with natural order
    if time.time() < deadline:
        mask, in_set, size = greedy_build(range(1, N + 1))
        mask, in_set, size = local_search(mask, in_set, size)
        update_best(mask, in_set, size)
    
    # Strategy 3: Greedy with reverse order
    if time.time() < deadline:
        mask, in_set, size = greedy_build(range(N, 0, -1))
        mask, in_set, size = local_search(mask, in_set, size)
        update_best(mask, in_set, size)

    # Strategy 4: Iterated Local Search / Random Restarts
    while time.time() < deadline - 5:
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        mask, in_set, size = greedy_build(random_order)
        mask, in_set, size = local_search(mask, in_set, size)
        update_best(mask, in_set, size)
        
    # Final polish pass on the best solution found
    if best_in_set is not None and time.time() < deadline - 2:
        best_mask, best_in_set, best_size = polish(best_mask, best_in_set, best_size)
        
    # Convert mask to list
    result = []
    for i in range(1, N + 1):
        if best_mask & (1 << i):
            result.append(i)
            
    if not result:
        result = [1]
        
    return result
