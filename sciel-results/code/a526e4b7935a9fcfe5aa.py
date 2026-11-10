# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero
    quadratic residue modulo 205.
    
    Approach:
    1. Build conflict graph using bitmasks
    2. Use branch-and-bound with greedy coloring for exact MIS on the conflict graph
    3. Supplement with fast local search (add/remove/swap) using bitmask ops
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 600  # 600s safety margin for 630s limit

    # Compute forbidden differences (nonzero quadratic residues mod 205)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # Build conflict graph: conflict[i] is a bitmask of vertices conflicting with i
    # Two vertices a,b conflict iff (a-b) mod m is in forbidden
    # Since forbidden is symmetric (if d is forbidden, m-d is also forbidden because
    # (m-d)^2 = d^2 mod m), the graph is undirected.
    
    # Precompute: for each vertex v, which other vertices conflict with it
    # conflict_mask[v] has bit u set iff v and u conflict
    conflict_mask = [0] * m
    
    # For each vertex v, for each forbidden difference d, v+d and v-d conflict with v
    for v in range(m):
        mask = 0
        for d in forbidden:
            u1 = (v + d) % m
            u2 = (v - d) % m
            if u1 != v:
                mask |= (1 << u1)
            if u2 != v and u2 != u1:
                mask |= (1 << u2)
        conflict_mask[v] = mask

    # Fast check: can vertex v be added to set represented by bitmask 'selected'?
    # v can be added iff no vertex in selected conflicts with v
    # i.e., (conflict_mask[v] & selected) == 0
    def can_add(v, selected_mask):
        return (conflict_mask[v] & selected_mask) == 0

    # Count bits in a Python int (bit_count available in 3.10+)
    def popcount(x):
        return x.bit_count()

    # Greedy coloring for upper bound in branch-and-bound
    # Given a set of candidate vertices (bitmask), color the subgraph
    # and return the number of colors used (upper bound on independent set size)
    def greedy_color_upper_bound(candidates_mask):
        """
        Returns an upper bound on the size of an independent set in the
        subgraph induced by candidates_mask using greedy coloring.
        """
        # Extract vertices
        verts = []
        temp = candidates_mask
        while temp:
            # Get lowest set bit
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            verts.append(v)
            temp ^= lb
        
        if not verts:
            return 0
        
        # Greedy coloring: assign each vertex the smallest color not used by its colored neighbors
        color_of = {}
        num_colors = 0
        for v in verts:
            # Find colors used by already-colored neighbors of v in candidates
            used_colors = set()
            neighbors = conflict_mask[v] & candidates_mask
            for u in verts:
                if u != v and (neighbors >> u) & 1:
                    if u in color_of:
                        used_colors.add(color_of[u])
            
            # Assign smallest available color
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            num_colors = max(num_colors, c + 1)
        
        return num_colors

    # Branch and bound for Maximum Independent Set
    # We work on the conflict graph; independent set in conflict graph = valid set
    def branch_and_bound(candidates_mask, current_size, best_size, best_set):
        """
        Recursively find maximum independent set.
        candidates_mask: bitmask of remaining candidate vertices
        current_size: size of current independent set
        best_size: best size found so far
        best_set: best set found so far (bitmask)
        Returns (best_size, best_set)
        """
        if popcount(candidates_mask) + current_size <= best_size[0]:
            return

        if candidates_mask == 0:
            return

        # Upper bound using greedy coloring
        ub = greedy_color_upper_bound(candidates_mask)
        if current_size + ub <= best_size[0]:
            return

        # Get a vertex to branch on
        # Pick vertex with highest degree in the candidate subgraph (heuristic)
        verts = []
        temp = candidates_mask
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            verts.append(v)
            temp ^= lb
        
        # Pick vertex with max degree in candidate subgraph
        max_deg = -1
        pivot = verts[0]
        for v in verts:
            deg = popcount(conflict_mask[v] & candidates_mask)
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        # Branch 1: Include pivot
        # If we include pivot, we remove all its neighbors from candidates
        new_candidates = candidates_mask & ~(conflict_mask[pivot] | (1 << pivot))
        new_size = current_size + 1
        branch_and_bound(new_candidates, new_size, best_size, best_set)
        
        # Branch 2: Exclude pivot
        new_candidates = candidates_mask & ~(1 << pivot)
        branch_and_bound(new_candidates, current_size, best_size, best_set)

    # Fast local search using bitmask
    def local_search(initial_set, max_iters=200000):
        """
        Local search with add, remove+add, swap moves using bitmask operations.
        """
        current_mask = 0
        for v in initial_set:
            current_mask |= (1 << v)
        
        current_size = popcount(current_mask)
        best_mask = current_mask
        best_size = current_size
        
        # Precompute: for each vertex, its conflict mask
        # can_add check is just (conflict_mask[v] & current_mask) == 0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try adding any vertex
            for v in range(m):
                if (current_mask >> v) & 1:
                    continue
                if can_add(v, current_mask):
                    current_mask |= (1 << v)
                    current_size += 1
                    if current_size > best_size:
                        best_mask = current_mask
                        best_size = current_size
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try removing one vertex and adding another (swap)
            # Or removing one and adding two
            found_swap = False
            for v_remove in range(m):
                if not ((current_mask >> v_remove) & 1):
                    continue
                
                temp_mask = current_mask & ~(1 << v_remove)
                temp_size = current_size - 1
                
                # Try to add one vertex
                for v_add in range(m):
                    if (temp_mask >> v_add) & 1:
                        continue
                    if v_add == v_remove:
                        continue
                    if can_add(v_add, temp_mask):
                        new_mask = temp_mask | (1 << v_add)
                        new_size = temp_size + 1
                        if new_size > best_size:
                            current_mask = new_mask
                            current_size = new_size
                            best_mask = new_mask
                            best_size = new_size
                            found_swap = True
                            break
                if found_swap:
                    break
                
                # Try to add two vertices
                if not found_swap:
                    added = []
                    for v_add1 in range(m):
                        if (temp_mask >> v_add1) & 1:
                            continue
                        if v_add1 == v_remove:
                            continue
                        if can_add(v_add1, temp_mask):
                            temp2_mask = temp_mask | (1 << v_add1)
                            added.append(v_add1)
                            for v_add2 in range(m):
                                if (temp2_mask >> v_add2) & 1:
                                    continue
                                if v_add2 == v_remove or v_add2 == v_add1:
                                    continue
                                if can_add(v_add2, temp2_mask):
                                    new_mask = temp2_mask | (1 << v_add2)
                                    new_size = temp_size + 2
                                    if new_size > best_size:
                                        current_mask = new_mask
                                        current_size = new_size
                                        best_mask = new_mask
                                        best_size = new_size
                                        found_swap = True
                                        break
                            break
                    if found_swap:
                        break
            
            if not found_swap:
                # No improvement possible with single add or swap
                break
        
        return best_mask, best_size

    # Greedy initialization
    def greedy_init(order):
        selected_mask = 0
        for v in order:
            if can_add(v, selected_mask):
                selected_mask |= (1 << v)
        return selected_mask

    # Main search
    best_mask = 0
    best_size = 0
    
    # 1. Try greedy with natural order
    mask = greedy_init(list(range(m)))
    sz = popcount(mask)
    if sz > best_size:
        best_size = sz
        best_mask = mask
    
    # 2. Try greedy with various random orders
    rng = random.Random(12345)
    for trial in range(500):
        if time.time() > deadline - 120:
            break
        order = list(range(m))
        rng.shuffle(order)
        mask = greedy_init(order)
        sz = popcount(mask)
        if sz > best_size:
            best_size = sz
            best_mask = mask
    
    # 3. Local search from best greedy solution
    if time.time() < deadline - 60:
        best_set_list = [i for i in range(m) if (best_mask >> i) & 1]
        lm, ls = local_search(best_set_list, max_iters=100000)
        if ls > best_size:
            best_size = ls
            best_mask = lm
    
    # 4. Local search from several random greedy solutions
    for trial in range(100):
        if time.time() > deadline - 30:
            break
        order = list(range(m))
        rng.shuffle(order)
        mask = greedy_init(order)
        sz = popcount(mask)
        if sz >= best_size - 2:  # Only search if close to best
            set_list = [i for i in range(m) if (mask >> i) & 1]
            lm, ls = local_search(set_list, max_iters=50000)
            if ls > best_size:
                best_size = ls
                best_mask = lm
    
    # 5. Branch and bound for exact solution (if time permits)
    # Use B&B on the conflict graph to try to find the true maximum
    if time.time() < deadline - 10:
        # Run B&B with a time limit
        best_bb = [best_size, best_mask]
        
        # We need to handle the time limit within recursion
        # Use a simpler approach: run B&B with node limit
        node_count = [0]
        max_nodes = 500000  # Limit nodes to avoid timeout
        
        def bbb(candidates_mask, current_size):
            if time.time() > deadline - 5 or node_count[0] > max_nodes:
                return
            node_count[0] += 1
            
            if popcount(candidates_mask) + current_size <= best_bb[0]:
                return
            if candidates_mask == 0:
                return
            
            ub = greedy_color_upper_bound(candidates_mask)
            if current_size + ub <= best_bb[0]:
                return
            
            # Pick pivot: vertex with max degree in candidate subgraph
            verts = []
            temp = candidates_mask
            while temp:
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                verts.append(v)
                temp ^= lb
            
            if not verts:
                return
            
            max_deg = -1
            pivot = verts[0]
            for v in verts:
                deg = popcount(conflict_mask[v] & candidates_mask)
                if deg > max_deg:
                    max_deg = deg
                    pivot = v
            
            # Branch 1: Include pivot
            new_candidates = candidates_mask & ~(conflict_mask[pivot] | (1 << pivot))
            bbb(new_candidates, current_size + 1)
            if time.time() > deadline - 5 or node_count[0] > max_nodes:
                return
            
            # Branch 2: Exclude pivot
            new_candidates = candidates_mask & ~(1 << pivot)
            bbb(new_candidates, current_size)
        
        # Start B&B from full set
        all_mask = (1 << m) - 1
        bbb(all_mask, 0)
        
        if best_bb[0] > best_size:
            best_size = best_bb[0]
            best_mask = best_bb[1]
    
    # Final local search polish on the best solution
    if time.time() < deadline - 2:
        best_set_list = [i for i in range(m) if (best_mask >> i) & 1]
        lm, ls = local_search(best_set_list, max_iters=200000)
        if ls > best_size:
            best_size = ls
            best_mask = lm
    
    # Convert bitmask to sorted array
    result = []
    for i in range(m):
        if (best_mask >> i) & 1:
            result.append(i)
    
    return np.array(sorted(result), dtype=np.int64)


# EVOLVE-BLOCK-END
