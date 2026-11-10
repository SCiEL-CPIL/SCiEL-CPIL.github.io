import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 88.0  # Safety margin before 90s kill

    N = 256
    recip = [0.0] + [1.0 / i for i in range(1, N + 1)]

    # Precompute conflicts
    # For each x, conflict_pairs[x] contains (y, z) such that {x, y, z} forms a 3-AP
    conflict_pairs = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c = 2 * b - a
            if c > N or c <= b:
                continue
            # a, b, c is a 3-AP
            conflict_pairs[a].append((b, c))
            conflict_pairs[b].append((a, c))
            conflict_pairs[c].append((a, b))

    def can_add(x, S_mask):
        """Check if x can be added to set S_mask."""
        for y, z in conflict_pairs[x]:
            if (S_mask & (1 << y)) and (S_mask & (1 << z)):
                return False
        return True

    def get_float_sum(S_mask):
        """Compute floating-point reciprocal sum from bitmask."""
        total = 0.0
        for x in range(1, N + 1):
            if S_mask & (1 << x):
                total += recip[x]
        return total

    def greedy_fill(forbidden_mask, S_mask, order=None):
        """Fill S by adding elements in a given order."""
        S = S_mask
        if order is None:
            order = range(1, N + 1)
        for x in order:
            if forbidden_mask & (1 << x):
                continue
            if S & (1 << x):
                continue
            if can_add(x, S):
                S |= (1 << x)
        return S

    def branch_and_bound(available_mask, current_sum, best_sum_ref):
        """
        Branch and bound for maximum weight independent set.
        available_mask: bitmask of vertices that can still be added
        current_sum: sum of reciprocals in current solution
        best_sum_ref: [best_sum] mutable reference
        Returns (best_mask, best_sum)
        """
        nonlocal best_S, best_sum
        
        if time.time() > deadline - 2:
            return best_S, best_sum
        
        if available_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
            return best_S, best_sum
        
        # Upper bound: current_sum + sum of all available weights
        # Quick upper bound using sum of reciprocals of available elements
        upper_bound = current_sum
        temp_mask = available_mask
        while temp_mask:
            lsb = temp_mask & (-temp_mask)
            x = lsb.bit_length() - 1
            upper_bound += recip[x]
            temp_mask ^= lsb
        
        if upper_bound <= best_sum + 1e-15:
            return best_S, best_sum
        
        # Choose vertex to branch on: pick the one with highest weight
        # among available vertices
        best_vertex = -1
        best_weight = -1
        temp_mask = available_mask
        while temp_mask:
            lsb = temp_mask & (-temp_mask)
            x = lsb.bit_length() - 1
            if recip[x] > best_weight:
                best_weight = recip[x]
                best_vertex = x
            temp_mask ^= lsb
        
        if best_vertex == -1:
            return best_S, best_sum
        
        # Branch 1: Include best_vertex
        if can_add(best_vertex, current_mask):
            new_mask = current_mask | (1 << best_vertex)
            # Remove all vertices that conflict with best_vertex
            # A vertex y conflicts with best_vertex if there exists z such that
            # (y, z, best_vertex) or (best_vertex, y, z) or (y, best_vertex, z) is a 3-AP
            # More precisely, y cannot be added if both y and some z are in the set
            # such that they form a 3-AP with best_vertex.
            # Actually, we need to remove y from available if adding y would create
            # a 3-AP with best_vertex and some other element.
            # This is complex. Let's use a simpler approach:
            # After adding best_vertex, the available set should exclude any vertex
            # that would form a 3-AP with best_vertex and another vertex in current_mask.
            new_available = available_mask & ~(1 << best_vertex)
            for y, z in conflict_pairs[best_vertex]:
                # If y is in current_mask, then z cannot be added (and vice versa)
                if current_mask & (1 << y):
                    new_available &= ~(1 << z)
                if current_mask & (1 << z):
                    new_available &= ~(1 << y)
            
            S1, sum1 = branch_and_bound(new_available, current_sum + recip[best_vertex], best_sum_ref)
            best_S, best_sum = S1, sum1
        
        # Branch 2: Exclude best_vertex
        new_available = available_mask & ~(1 << best_vertex)
        S2, sum2 = branch_and_bound(new_available, current_sum, best_sum_ref)
        best_S, best_sum = S2, sum2
        
        return best_S, best_sum

    # Step 1: Get initial solution via greedy
    S_asc = greedy_fill(0, 0, order=range(1, N + 1))
    initial_sum = get_float_sum(S_asc)
    
    best_S = S_asc
    best_sum = initial_sum
    
    # Step 2: Branch and bound
    # Start with all vertices available
    full_mask = (1 << (N + 1)) - 2  # bits 1 through N set
    branch_and_bound(full_mask, 0.0, [best_sum])
    
    # Step 3: Local search polish on the B&B result
    if best_S != 0 and time.time() < deadline - 10:
        # Run local search
        S = best_S
        sum_float = get_float_sum(S)
        S_elements = [x for x in range(1, N + 1) if S & (1 << x)]
        
        improved = True
        while improved and (time.time() - start_time) < deadline - 5:
            improved = False
            
            # Try 1-Swap: remove x, add y
            best_gain = 0
            best_x = -1
            best_y = -1
            
            for y in range(1, N + 1):
                if (time.time() - start_time) > deadline - 5:
                    break
                if S & (1 << y):
                    continue
                
                for x in range(y + 1, N + 1):
                    if not (S & (1 << x)):
                        continue
                    
                    gain = recip[y] - recip[x]
                    if gain <= 1e-15:
                        continue
                    
                    S_new = S & ~(1 << x)
                    if can_add(y, S_new):
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                            best_y = y
            
            if best_y != -1:
                S = S & ~(1 << best_x)
                S |= (1 << best_y)
                sum_float += best_gain
                improved = True
                continue
            
            # Try adding a single element if possible
            added = False
            for y in range(1, N + 1):
                if (time.time() - start_time) > deadline - 5:
                    break
                if S & (1 << y):
                    continue
                if can_add(y, S):
                    S |= (1 << y)
                    sum_float += recip[y]
                    S_elements.append(y)
                    improved = True
                    added = True
                    break
            
            if not added:
                # Try remove + greedy
                for x in S_elements:
                    if (time.time() - start_time) > deadline - 5:
                        break
                    if not (S & (1 << x)):
                        continue
                    
                    S_no_x = S & ~(1 << x)
                    S_new = greedy_fill(1 << x, S_no_x)
                    sum_new = get_float_sum(S_new)
                    gain = sum_new - sum_float
                    
                    if gain > 1e-15:
                        S = S_new
                        sum_float = sum_new
                        S_elements = [x for x in range(1, N + 1) if S & (1 << x)]
                        improved = True
                        break
        
        if sum_float > best_sum:
            best_S = S
            best_sum = sum_float
    
    # Convert bitmask to list
    result = [x for x in range(1, N + 1) if best_S & (1 << x)]
    
    # Validate
    S_set = set(result)
    valid = True
    for a in S_set:
        for b in S_set:
            if a >= b:
                continue
            s = a + b
            if s % 2 == 0:
                mid = s // 2
                if mid in S_set and mid != a and mid != b:
                    valid = False
                    break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid set if validation fails
        result = [x for x in range(1, N + 1) if S_asc & (1 << x)]
        
    return result
