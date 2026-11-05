import math
import time

def construct():
    N = 512
    start_time = time.time()
    time_limit = 82.0  # Leave 8s margin for 90s limit

    # 1. Precompute LCM conflict graph
    # adj[i] is a bitmask of j such that lcm(i, j) > N
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            l = i * j // g
            if l > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # comp_adj[i] = bitmask of vertices compatible with i (excluding i itself)
    full_mask = (1 << (N + 1)) - 2
    comp_adj = [0] * (N + 1)
    for i in range(1, N + 1):
        comp_adj[i] = full_mask & ~adj[i] & ~(1 << i)

    def is_compatible_with_set(x, mask):
        """Check if x is compatible with all elements in the set represented by mask."""
        return (mask & ~comp_adj[x]) == 0

    def greedy_color_bound(candidates_mask, adj_list):
        """
        Greedy coloring upper bound for maximum independent set.
        candidates_mask: bitmask of candidate vertices
        Returns an upper bound on the max independent set size within candidates.
        """
        # Extract candidate vertices
        candidates = []
        m = candidates_mask
        while m:
            v = (m & -m).bit_length() - 1
            candidates.append(v)
            m &= m - 1

        if not candidates:
            return 0

        # Greedy coloring: assign colors to vertices
        # A vertex's color = 1 + max color of its neighbors already colored
        color_of = {}
        max_color = 0
        for v in candidates:
            # Get neighbors of v that are in candidates and already colored
            neighbor_colors = set()
            m2 = adj_list[v] & candidates_mask
            while m2:
                u = (m2 & -m2).bit_length() - 1
                if u in color_of:
                    neighbor_colors.add(color_of[u])
                m2 &= m2 - 1
            c = max(neighbor_colors) + 1 if neighbor_colors else 1
            color_of[v] = c
            max_color = max(max_color, c)

        return max_color

    def branch_and_bound(initial_set, time_limit):
        """
        B&B to maximize independent set size.
        initial_set: a valid independent set to start from
        """
        # State: current solution set, candidate set (vertices that could be added)
        # We use a simple recursive B&B with greedy coloring bound
        
        best_size = len(initial_set)
        best_set = list(initial_set)
        
        # Convert initial_set to mask
        sol_mask = 0
        for v in initial_set:
            sol_mask |= (1 << v)
        
        # Candidates: all vertices not in solution that are compatible with solution
        candidates_mask = 0
        for v in range(1, N + 1):
            if not (sol_mask >> v) & 1:
                if is_compatible_with_set(v, sol_mask):
                    candidates_mask |= (1 << v)
        
        # B&B state
        # We'll do iterative deepening or just a single B&B run
        
        # Use a stack for iterative B&B
        # Each state: (solution_mask, candidates_mask, depth)
        # But this can blow up. Let's use a simpler approach:
        # Greedy construction + local search with restarts
        
        # Actually, let's implement a proper B&B with recursion
        # but limit depth and use the coloring bound
        
        def bb_solve(sol_mask, cand_mask, current_size):
            nonlocal best_size, best_set
            
            if time.time() > time_limit:
                return
            
            if not cand_mask:
                if current_size > best_size:
                    best_size = current_size
                    # Reconstruct set from sol_mask
                    s = []
                    m = sol_mask
                    while m:
                        v = (m & -m).bit_length() - 1
                        s.append(v)
                        m &= m - 1
                    best_set = s
                return
            
            # Upper bound: current_size + greedy_color_bound(cand_mask)
            ub = current_size + greedy_color_bound(cand_mask, adj)
            if ub <= best_size:
                return
            
            # Pick a vertex to branch on
            # Heuristic: pick vertex with highest degree in cand_mask
            # or just pick the first one
            m = cand_mask
            v = (m & -m).bit_length() - 1
            
            # Branch 1: include v
            new_sol_mask = sol_mask | (1 << v)
            new_cand_mask = cand_mask & comp_adj[v] & ~(1 << v)
            bb_solve(new_sol_mask, new_cand_mask, current_size + 1)
            
            if time.time() > time_limit:
                return
            
            # Branch 2: exclude v
            new_cand_mask2 = cand_mask & ~(1 << v)
            bb_solve(sol_mask, new_cand_mask2, current_size)
        
        bb_solve(sol_mask, candidates_mask, len(initial_set))
        
        return best_set

    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return divs

    def local_search(initial_set, time_limit):
        """
        Local search with add and swap moves.
        """
        current_set = set(initial_set)
        current_mask = 0
        for v in current_set:
            current_mask |= (1 << v)
        
        best_size = len(current_set)
        best_set = list(current_set)
        
        candidates_order = list(range(1, N + 1))
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # 1. Try to Add
            added_any = False
            for x in candidates_order:
                if (current_mask >> x) & 1:
                    continue
                if is_compatible_with_set(x, current_mask):
                    current_set.add(x)
                    current_mask |= (1 << x)
                    improved = True
                    added_any = True
                    break
            
            if added_any:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                continue
            
            # 2. Try to Swap
            swap_found = False
            for y in list(current_set):
                if (time.time() - start_time) > time_limit:
                    return best_set
                
                temp_mask = current_mask & ~(1 << y)
                
                for x in candidates_order:
                    if x == y:
                        continue
                    if (temp_mask >> x) & 1:
                        continue
                    if is_compatible_with_set(x, temp_mask):
                        current_set.remove(y)
                        current_set.add(x)
                        current_mask = temp_mask | (1 << x)
                        improved = True
                        swap_found = True
                        break
                
                if swap_found:
                    break
            
            if swap_found:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)

        return best_set

    # 3. Generate Initial Solutions
    candidates = [360, 480, 420, 336, 300, 240, 210, 180, 252, 288, 360, 420, 480]
    best_initial = []
    
    for m in candidates:
        divs = get_divisors(m)
        if len(divs) > len(best_initial):
            best_initial = divs
    
    simple_set = list(range(1, 23))
    if len(simple_set) > len(best_initial):
        best_initial = simple_set

    # 4. Run Local Search first (fast)
    result_ls = local_search(best_initial, time_limit)
    
    # 5. Run B&B with the best result from local search as initial
    if time.time() - start_time < time_limit - 5:
        result_bb = branch_and_bound(result_ls, time_limit - 2)
        if len(result_bb) > len(result_ls):
            result_ls = result_bb
    
    # 6. Final Validation
    result_set = set(result_ls)
    valid = True
    for i in range(len(result_ls)):
        for j in range(i+1, len(result_ls)):
            a, b = result_ls[i], result_ls[j]
            l = a * b // math.gcd(a, b)
            if l > 512:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return sorted(best_initial)
    
    return sorted(result_ls)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            l = a * b // math.gcd(a, b)
            if l > 512:
                print(f"Invalid: {a}, {b}, lcm={l}")
                break
