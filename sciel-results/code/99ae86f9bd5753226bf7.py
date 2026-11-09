import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    N = 256

    # Build conflict graph using the gcd characterization:
    # a,b conflict iff gcd(x,y)=1 and (x+y)|g where a=g*x, b=g*y
    # conflict[a] is a bitset (Python int) with bit (b-1) set if a and b conflict.
    conflict = [0] * (N + 1)  # 1-indexed

    for x in range(1, N + 1):
        for y in range(x + 1, N + 1):
            if math.gcd(x, y) != 1:
                continue
            s = x + y
            if s > N:
                continue
            # g must be a multiple of s, and g*x <= N, g*y <= N
            # max g = N // max(x, y) = N // y (since y > x)
            max_g = N // y
            for g in range(s, max_g + 1, s):
                a = g * x
                b = g * y
                if a <= N and b <= N:
                    conflict[a] |= (1 << (b - 1))
                    conflict[b] |= (1 << (a - 1))

    # Compute degrees
    degrees = [0] * (N + 1)
    for v in range(1, N + 1):
        degrees[v] = conflict[v].bit_count()

    # Greedy independent set for initial bound
    # Try multiple orderings: by degree ascending, random, etc.
    all_nodes = list(range(1, N + 1))

    def greedy_is(order):
        chosen_bits = 0
        chosen = []
        for v in order:
            if (conflict[v] & chosen_bits) == 0:
                chosen_bits |= (1 << (v - 1))
                chosen.append(v)
        return chosen_bits, chosen, len(chosen)

    # Order by degree ascending (low degree first tends to give larger IS)
    order_deg_asc = sorted(all_nodes, key=lambda v: degrees[v])
    bits, lst, sz = greedy_is(order_deg_asc)
    best_size = sz
    best_set = set(lst)

    # Try a few random orderings for better initial bound
    import random
    rng = random.Random(42)
    for _ in range(20):
        order = all_nodes[:]
        rng.shuffle(order)
        _, _, sz2 = greedy_is(order)
        if sz2 > best_size:
            best_size = sz2
            _, lst2, _ = greedy_is(order)
            best_set = set(lst2)

    # Branch and bound for maximum independent set
    # State: current independent set (list), available candidates (bitset)
    # A candidate v can be added if (conflict[v] & current_bits) == 0
    # Upper bound: len(current) + popcount(available)
    # Branching: pick vertex with max degree within available
    
    # Precompute: for each vertex, its neighbors as a bitset (already in conflict[v])
    
    # We'll use a recursive B&B with bitsets
    # To speed up, precompute the order of vertices by degree for branching
    
    # Sort vertices by degree descending for branching priority
    vertices_by_deg_desc = sorted(all_nodes, key=lambda v: -degrees[v])
    
    # For branching, we want to pick the vertex in 'available' with highest degree
    # We can iterate vertices_by_deg_desc and find the first one in available
    
    best_size_global = best_size
    best_set_global = set(best_set)
    
    # Convert best_set to bitset
    best_bits_global = 0
    for v in best_set_global:
        best_bits_global |= (1 << (v - 1))
    
    # B&B function
    # current_bits: bitset of current independent set
    # available: bitset of vertices that can still be added (compatible with current set)
    # current_size: number of vertices in current set
    
    # We'll implement iteratively with a stack to avoid recursion depth issues
    # But for N=256, recursion depth is at most 256, which is fine.
    
    # Actually, let's use recursion with a global deadline check.
    
    def branch_and_bound(current_bits, available, current_size):
        nonlocal best_size_global, best_set_global, best_bits_global
        
        if time.time() > deadline - 5:
            return
        
        # Upper bound
        ub = current_size + available.bit_count()
        if ub <= best_size_global:
            return
        
        # If no candidates left
        if available == 0:
            if current_size > best_size_global:
                best_size_global = current_size
                # Reconstruct set from current_bits
                s = set()
                for v in range(1, N + 1):
                    if (current_bits >> (v - 1)) & 1:
                        s.add(v)
                best_set_global = s
                best_bits_global = current_bits
            return
        
        # Choose branching vertex: highest degree in available
        # Iterate vertices_by_deg_desc and find first in available
        branch_v = -1
        for v in vertices_by_deg_desc:
            if (available >> (v - 1)) & 1:
                branch_v = v
                break
        
        if branch_v == -1:
            return
        
        # Branch 1: Include branch_v
        # New available = available & ~conflict[branch_v] & ~(1 << (branch_v - 1))
        # But we also need to remove branch_v itself from available
        new_available = available & ~(conflict[branch_v] | (1 << (branch_v - 1)))
        new_current_bits = current_bits | (1 << (branch_v - 1))
        new_size = current_size + 1
        
        # Check upper bound for this branch
        if new_size + new_available.bit_count() > best_size_global:
            branch_and_bound(new_current_bits, new_available, new_size)
        
        # Branch 2: Exclude branch_v
        # New available = available & ~(1 << (branch_v - 1))
        new_available2 = available & ~(1 << (branch_v - 1))
        
        # Check upper bound
        if current_size + new_available2.bit_count() > best_size_global:
            branch_and_bound(current_bits, new_available2, current_size)
    
    # Start B&B from empty set with all vertices available
    # But we already have a good initial bound, so this should prune well
    all_available = 0
    for v in range(1, N + 1):
        all_available |= (1 << (v - 1))
    
    branch_and_bound(0, all_available, 0)
    
    # Return the best set found
    result = list(best_set_global)
    if not result:
        # Fallback: return all odd numbers (should be a valid independent set)
        result = [v for v in range(1, N + 1, 2)]
    
    return result
