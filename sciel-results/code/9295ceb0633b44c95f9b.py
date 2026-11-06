import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 90s limit
    
    N = 256
    
    # Precompute conflict masks
    # conflicts[i] is a bitmask of j (1-indexed positions) such that (i*j) % (i+j) == 0
    conflicts = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    # Full mask of all vertices 1..N
    full_mask = 0
    for i in range(1, N + 1):
        full_mask |= (1 << i)
    
    def popcount(x):
        return x.bit_count()
    
    def mask_to_set(mask):
        return [i for i in range(1, N + 1) if (mask >> i) & 1]
    
    # Greedy Min-Degree for initial solution
    def greedy_min_degree(seed=None):
        if seed is not None:
            random.seed(seed)
        current_available = full_mask
        solution_mask = 0
        
        while current_available:
            # Find node with min degree in current_available
            # Tie-breaking: random
            best_node = -1
            min_deg = N + 1
            best_candidates = []
            
            temp = current_available
            while temp:
                lsb = temp & -temp
                x = lsb.bit_length() - 1
                deg = popcount(conflicts[x] & current_available)
                if deg < min_deg:
                    min_deg = deg
                    best_candidates = [x]
                elif deg == min_deg:
                    best_candidates.append(x)
                temp -= lsb
                
            if best_node == -1:
                break
            
            # Pick randomly from best candidates to diversify
            if best_candidates:
                best_node = random.choice(best_candidates)
            else:
                # Fallback
                lsb = current_available & -current_available
                best_node = lsb.bit_length() - 1
            
            solution_mask |= (1 << best_node)
            remove_mask = (1 << best_node) | conflicts[best_node]
            current_available &= ~remove_mask
            
        return solution_mask

    # Get initial solution
    best_mask = greedy_min_degree(42)
    best_size = popcount(best_mask)
    
    # Try a few random greeds to get a better initial bound
    for s in range(100):
        if time.time() > deadline - 5:
            break
        m = greedy_min_degree(s)
        sz = popcount(m)
        if sz > best_size:
            best_size = sz
            best_mask = m

    # Tomita's algorithm for Maximum Independent Set with Greedy Coloring Bound
    # We use recursion with branching
    
    def greedy_color_bound(candidates):
        """
        Returns an upper bound on the size of the MIS in the subgraph induced by 'candidates'.
        Uses greedy coloring: the number of colors used is an upper bound.
        """
        if candidates == 0:
            return 0
        
        # Get list of vertices in candidates
        verts = []
        temp = candidates
        while temp:
            lsb = temp & -temp
            x = lsb.bit_length() - 1
            verts.append(x)
            temp -= lsb
            
        # Sort by degree in the candidate subgraph (descending) for better coloring
        # Degree of v in candidates is popcount(conflicts[v] & candidates)
        def get_deg(v):
            return popcount(conflicts[v] & candidates)
            
        verts.sort(key=get_deg, reverse=True)
        
        # Greedy coloring
        # color[v] = color assigned to v
        # colors_used = max color + 1
        color = {}
        colors_used = 0
        
        for v in verts:
            # Find smallest color not used by neighbors
            neighbor_colors = set()
            # Get neighbors of v within candidates
            neigh_mask = conflicts[v] & candidates
            temp_neigh = neigh_mask
            while temp_neigh:
                lsb = temp_neigh & -temp_neigh
                u = lsb.bit_length() - 1
                if u in color:
                    neighbor_colors.add(color[u])
                temp_neigh -= lsb
                
            # Assign smallest available color
            c = 0
            while c in neighbor_colors:
                c += 1
            color[v] = c
            if c + 1 > colors_used:
                colors_used = c + 1
                
        return colors_used

    def branch_and_bound(candidates, current_mask, current_size):
        nonlocal best_mask, best_size
        
        if time.time() > deadline - 1:
            return
        
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
        
        # Pruning: Use greedy coloring bound
        # If current_size + upper_bound(candidates) <= best_size, prune
        ub = greedy_color_bound(candidates)
        if current_size + ub <= best_size:
            return
        
        # Select vertex to branch on
        # Heuristic: pick vertex with highest degree in candidates subgraph
        # This tends to reduce the candidate set faster in the "include" branch
        v = -1
        max_deg = -1
        temp = candidates
        while temp:
            lsb = temp & -temp
            x = lsb.bit_length() - 1
            deg = popcount(conflicts[x] & candidates)
            if deg > max_deg:
                max_deg = deg
                v = x
            temp -= lsb
        
        if v == -1:
            return
        
        # Branch 1: Include v
        # New candidates = candidates - {v} - neighbors(v)
        new_candidates = candidates & ~(1 << v) & ~conflicts[v]
        branch_and_bound(new_candidates, current_mask | (1 << v), current_size + 1)
        
        # Branch 2: Exclude v
        # New candidates = candidates - {v}
        new_candidates = candidates & ~(1 << v)
        branch_and_bound(new_candidates, current_mask, current_size)
    
    # Run B&B with time limit
    # Start with the full set. The initial best_mask provides the lower bound.
    branch_and_bound(full_mask, 0, 0)
    
    # If we still have time, we might try to improve with local search, 
    # but B&B with coloring bound is usually very effective for N=256.
    # We will just return the best found.
    
    return mask_to_set(best_mask)

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(f"Found set of size {len(res)}")
    # Verify
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            if (a * b) % (a + b) == 0:
                valid = False
                print(f"Conflict: {a}, {b}")
    print(f"Valid: {valid}")
