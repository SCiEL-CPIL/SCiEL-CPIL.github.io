def construct():
    import time
    import random
    from math import gcd

    deadline = time.time() + 85  # Safety margin for 90s limit

    N = 128
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Build conflict structure:
    # conflicts[x] contains pairs (a, b) such that {x, a, b} is a bad triple.
    # a < b, a != x, b != x
    conflicts = {x: [] for x in range(1, N + 1)}
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            g = gcd_table[x][a]
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                if gcd_table[x][b] == g and gcd_table[a][b] == g:
                    conflicts[x].append((a, b))

    def can_add(x, S):
        """Check if x can be added to set S without creating a bad triple."""
        for a, b in conflicts[x]:
            if a in S and b in S:
                return False
        return True

    def potential(S):
        """Count number of elements in [1, 128] \\ S that can be added to S."""
        count = 0
        for x in range(1, N + 1):
            if x not in S:
                if can_add(x, S):
                    count += 1
        return count

    def greedy_init(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    def local_search(S):
        """Improve S using Greedy Add and Potential-Guided 1-Swaps."""
        S = set(S)
        while time.time() < deadline:
            improved = False
            
            # 1. Try to add any element (Greedy Add)
            for x in range(1, N + 1):
                if x not in S:
                    if can_add(x, S):
                        S.add(x)
                        improved = True
                        break # Restart loop to try adding more
            
            if improved:
                continue
            
            # 2. If no add possible, try 1-Swap to maximize potential
            current_pot = potential(S)
            best_pot = current_pot
            best_swap = None
            
            S_list = list(S)
            for a in S_list:
                S_minus_a = S - {a}
                for x in range(1, N + 1):
                    if x in S:
                        continue
                    if can_add(x, S_minus_a):
                        S_new = S_minus_a | {x}
                        # Calculate potential of new state
                        # Optimization: We can estimate potential, but exact is safer
                        pot = potential(S_new)
                        if pot > best_pot:
                            best_pot = pot
                            best_swap = (a, x)
            
            if best_swap:
                a, x = best_swap
                S.remove(a)
                S.add(x)
                # Do not break, loop to try adding more
            else:
                # Stuck in local optimum for this initialization
                break
        
        return S

    best_set = set()
    best_size = 0
    
    # Different initialization orders
    orders = []
    orders.append(list(range(1, N + 1))) # Ascending
    orders.append(list(range(N, 0, -1))) # Descending
    
    # Random orders
    random.seed(42)
    for _ in range(10):
        o = list(range(1, N + 1))
        random.shuffle(o)
        orders.append(o)
    
    # Degree-sorted orders (high conflict degree first or last)
    degrees = {x: len(conflicts[x]) for x in range(1, N + 1)}
    orders.append(sorted(range(1, N + 1), key=lambda x: degrees[x], reverse=True))
    orders.append(sorted(range(1, N + 1), key=lambda x: degrees[x]))

    for order in orders:
        if time.time() > deadline:
            break
        
        S = greedy_init(order)
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
            # Early termination if we found a very good set?
            # Theoretically max is around 20-30?
            # If we hit 30, it's likely optimal or close.
            if best_size >= 30:
                break

    # Final check and return
    # Ensure valid
    # The local_search maintains validity by construction (can_add checks)
    # But just in case, we could verify.
    # Verification is O(N^3) which is fine for N=128 once.
    def is_valid(S):
        elems = sorted(S)
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                g_ij = gcd_table[elems[i]][elems[j]]
                for k in range(j + 1, len(elems)):
                    if gcd_table[elems[i]][elems[k]] == g_ij and gcd_table[elems[j]][elems[k]] == g_ij:
                        return False
        return True

    if is_valid(best_set):
        return list(best_set)
    else:
        return [1]
