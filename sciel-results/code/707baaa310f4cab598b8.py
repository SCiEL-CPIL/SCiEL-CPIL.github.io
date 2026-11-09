import time
import random
import sys

sys.setrecursionlimit(10000)

def is_triangle_free_mask(n, adj_masks):
    # adj_masks[i] is an int bitmask of neighbors of i
    for u in range(n):
        # Get neighbors of u
        mask_u = adj_masks[u]
        # For each pair of neighbors v, w
        # We can iterate through set bits
        # Or just check: for each v in neighbors(u), (adj_masks[v] & mask_u) should not contain any other neighbor?
        # Actually, triangle exists if there is v in N(u) such that N(v) intersects N(u) at a vertex w != v.
        # Since graph is simple, v is not in N(v).
        # So if (adj_masks[v] & mask_u) is not empty, there is a triangle.
        # Wait, adj_masks[v] contains u. So (adj_masks[v] & mask_u) will always contain u?
        # No, mask_u is N(u). v is in N(u).
        # adj_masks[v] contains u.
        # So (adj_masks[v] & mask_u) contains u.
        # We need to check if there is w in N(u) \ {v} such that w in N(v).
        # So (adj_masks[v] & (mask_u ^ (1 << v))) should be empty.
        
        # Let's iterate v in N(u)
        temp = mask_u
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            # Check if any other neighbor of u is neighbor of v
            # N(u) \ {v}
            others = mask_u ^ lsb
            if adj_masks[v] & others:
                return False
            temp ^= lsb
    return True

def find_independent_set_mask(n, adj_masks, k, deadline=None):
    # adj_masks[i] is bitmask of neighbors of i
    # Independent set: no two vertices are adjacent.
    # If we pick v, we must exclude N(v).
    
    def backtrack(current_set, available_mask):
        if deadline and time.time() > deadline:
            return "TIMEOUT"
        if len(current_set) == k:
            return current_set
        if bin(available_mask).count("1") < k - len(current_set):
            return None
            
        # Pick vertex with min degree in available
        # We need to find v in available_mask with min popcount(adj_masks[v] & available_mask)
        # Iterate through bits in available_mask
        temp = available_mask
        min_deg = float('inf')
        chosen = None
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = bin(adj_masks[v] & available_mask).count("1")
            if deg < min_deg:
                min_deg = deg
                chosen = v
            temp ^= lsb
            
        if chosen is None:
            return None
            
        chosen_bit = 1 << chosen
        # Branch 1: Include chosen
        new_current = current_set + [chosen]
        # New available: available_mask without chosen and without neighbors of chosen
        new_available = available_mask & ~(chosen_bit | adj_masks[chosen])
        
        res = backtrack(new_current, new_available)
        if res == "TIMEOUT":
            return "TIMEOUT"
        if res:
            return res
            
        # Branch 2: Exclude chosen
        res = backtrack(current_set, available_mask & ~chosen_bit)
        if res == "TIMEOUT":
            return "TIMEOUT"
        if res:
            return res
            
        return None

    initial_available = (1 << n) - 1
    res = backtrack([], initial_available)
    if res == "TIMEOUT":
        return None
    return res

def construct():
    deadline = time.time() + 110.0
    
    for n in range(39, 25, -1):
        if time.time() > deadline:
            break
            
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            random.seed(attempt + n)
            
            adj_masks = [0] * n
            
            # Generate random triangle-free graph
            # Add edges (u,v) if N(u) & N(v) is empty
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj_masks[u] & (1 << v)) and not (adj_masks[v] & (1 << u)):
                         # This check is wrong. adj_masks[u] doesn't contain v yet.
                         # We need to check if N(u) and N(v) have common neighbors.
                         # Common neighbors = adj_masks[u] & adj_masks[v]
                         if not (adj_masks[u] & adj_masks[v]):
                             if random.random() < 0.5:
                                 adj_masks[u] |= (1 << v)
                                 adj_masks[v] |= (1 << u)
            
            if not is_triangle_free_mask(n, adj_masks):
                continue
                
            ind_set = find_independent_set_mask(n, adj_masks, 10, deadline)
            
            if ind_set is None:
                # Valid
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj_masks[u] & (1 << v):
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
            else:
                # Try to fix
                # Add edges within ind_set
                ind_list = list(ind_set)
                changed = True
                while changed and time.time() < deadline:
                    changed = False
                    for i in range(len(ind_list)):
                        for j in range(i + 1, len(ind_list)):
                            u = ind_list[i]
                            v = ind_list[j]
                            if adj_masks[u] & (1 << v):
                                continue
                            # Check if adding (u,v) creates triangle
                            # N(u) & N(v) should be empty
                            if not (adj_masks[u] & adj_masks[v]):
                                adj_masks[u] |= (1 << v)
                                adj_masks[v] |= (1 << u)
                                changed = True
                                
                ind_set = find_independent_set_mask(n, adj_masks, 10, deadline)
                if ind_set is None:
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj_masks[u] & (1 << v):
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}

    # Fallback
    n = 26
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
    return {'n': n, 'edges': edges}
