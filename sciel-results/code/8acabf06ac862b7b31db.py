"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    n = 256
    # Build conflict bitsets
    conflict = [0] * n
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Greedy independent set with multiple strategies
    def greedy_min_degree(conf, n):
        """Pick vertex with min conflicts, add to IS, remove neighbors."""
        remaining = list(range(n))
        result = []
        while remaining:
            best_v = min(remaining, key=lambda v: bin(conf[v]).count('1'))
            result.append(best_v)
            # Remove best_v and its neighbors
            new_remaining = []
            for r in remaining:
                if r == best_v:
                    continue
                if conf[best_v] & (1 << r):
                    continue
                new_remaining.append(r)
            remaining = new_remaining
        return result
    
    def greedy_max_independent(conf, n):
        """Try to build large IS by considering vertices in order of low conflict."""
        # Sort by number of conflicts
        order = sorted(range(n), key=lambda v: bin(conf[v]).count('1'))
        in_set = [False] * n
        result = []
        for v in order:
            # Check if v can be added
            can_add = True
            for u in result:
                if conf[v] & (1 << u):
                    can_add = False
                    break
            if can_add:
                in_set[v] = True
                result.append(v)
        return result
    
    # Get initial solutions
    sol1 = greedy_min_degree(conflict, n)
    sol2 = greedy_max_independent(conflict, n)
    
    best_set = [v + 1 for v in sol1]
    best_size = len(sol1)
    if len(sol2) > best_size:
        best_set = [v + 1 for v in sol2]
        best_size = len(sol2)
    
    # Local search with swaps
    def local_search(current_set, conf, n, deadline):
        current = set(current_set)
        in_set = [False] * n
        for v in current:
            in_set[v] = True
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding a vertex
            for v in range(n):
                if in_set[v]:
                    continue
                can_add = True
                for u in current:
                    if conf[v] & (1 << u):
                        can_add = False
                        break
                if can_add:
                    current.add(v)
                    in_set[v] = True
                    improved = True
                    break
            if improved:
                continue
            # Try 1-swap: remove one, add one
            for v in range(n):
                if in_set[v]:
                    continue
                # Find vertices in set that conflict with v
                conflicts = [u for u in current if conf[v] & (1 << u)]
                if not conflicts:
                    continue
                for u in conflicts:
                    # Check if v can be added after removing u
                    can_add = True
                    for w in current:
                        if w == u:
                            continue
                        if conf[v] & (1 << w):
                            can_add = False
                            break
                    if can_add:
                        current.discard(u)
                        in_set[u] = False
                        current.add(v)
                        in_set[v] = True
                        improved = True
                        break
                if improved:
                    break
        return [v + 1 for v in sorted(current)]
    
    improved = local_search(best_set, conflict, n, DEADLINE)
    if len(improved) > best_size:
        best_set = improved
        best_size = len(best_set)
    
    # Try 2-swap local search
    def local_search_2swap(current_set, conf, n, deadline):
        current = set(current_set)
        in_set = [False] * n
        for v in current:
            in_set[v] = True
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding
            for v in range(n):
                if in_set[v]:
                    continue
                can_add = True
                for u in current:
                    if conf[v] & (1 << u):
                        can_add = False
                        break
                if can_add:
                    current.add(v)
                    in_set[v] = True
                    improved = True
                    break
            if improved:
                continue
            # Try 1-swap
            for v in range(n):
                if in_set[v]:
                    continue
                conflicts = [u for u in current if conf[v] & (1 << u)]
                if not conflicts:
                    continue
                for u in conflicts:
                    can_add = True
                    for w in current:
                        if w == u:
                            continue
                        if conf[v] & (1 << w):
                            can_add = False
                            break
                    if can_add:
                        current.discard(u)
                        in_set[u] = False
                        current.add(v)
                        in_set[v] = True
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # Try 2-swap: remove two, add one or two
            # For speed, only try removing one and adding two
            for v in range(n):
                if in_set[v]:
                    continue
                conflicts = [u for u in current if conf[v] & (1 << u)]
                if len(conflicts) > 2:
                    continue
                for u in conflicts:
                    # Try adding v and another w after removing u
                    for w in range(n):
                        if in_set[w] or w == v or w == u:
                            continue
                        # Check if v and w can both be added after removing u
                        can_add_v = True
                        can_add_w = True
                        for x in current:
                            if x == u:
                                continue
                            if conf[v] & (1 << x):
                                can_add_v = False
                            if conf[w] & (1 << x):
                                can_add_w = False
                        if conf[v] & (1 << w):
                            can_add_v = False
                            can_add_w = False
                        if can_add_v and can_add_w:
                            current.discard(u)
                            in_set[u] = False
                            current.add(v)
                            in_set[v] = True
                            current.add(w)
                            in_set[w] = True
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return [v + 1 for v in sorted(current)]
    
    improved2 = local_search_2swap(best_set, conflict, n, DEADLINE)
    if len(improved2) > best_size:
        best_set = improved2
        best_size = len(best_set)
    
    return best_set
# EVOLVE-BLOCK-END
