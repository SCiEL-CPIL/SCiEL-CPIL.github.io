import time
import random
import sys
import math

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # 5 seconds margin for 60s limit

    N = 256
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Identify vertices that cannot be in the set (2*v is square)
    forbidden = set()
    for v in range(1, N + 1):
        if 2 * v in squares:
            forbidden.add(v)

    valid_vertices = [v for v in range(1, N + 1) if v not in forbidden]
    M = len(valid_vertices)
    vertex_map = valid_vertices  # i -> v
    
    # Build adjacency list for valid vertices
    # adj[i] is a set of j such that vertex_map[i] + vertex_map[j] is a square
    adj = [set() for _ in range(M)]
    
    for i in range(M):
        vi = vertex_map[i]
        for j in range(i + 1, M):
            vj = vertex_map[j]
            if vi + vj in squares:
                adj[i].add(j)
                adj[j].add(i)
                
    # Convert adjacency to bitmasks for faster B&B
    # neighbors[i] is a bitmask where bit j is set if i and j are adjacent
    neighbors = [0] * M
    for i in range(M):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        neighbors[i] = mask

    # Initial lower bound using greedy with random restarts
    best_size = 0
    best_set_indices = set()
    
    def greedy_random(seed):
        random.seed(seed)
        order = list(range(M))
        random.shuffle(order)
        selected = set()
        for v in order:
            conflict = any(u in selected for u in adj[v])
            if not conflict:
                selected.add(v)
        return selected

    # Run several greedy restarts to get a good initial bound
    for i in range(100):
        s = greedy_random(i)
        if len(s) > best_size:
            best_size = len(s)
            best_set_indices = set(s)
            
    # Also try a deterministic greedy based on degree
    def greedy_deterministic():
        verts = sorted(range(M), key=lambda x: len(adj[x]))
        selected = set()
        for v in verts:
            if v in selected: continue
            conflict = any(u in selected for u in adj[v])
            if not conflict:
                selected.add(v)
        return selected
        
    s = greedy_deterministic()
    if len(s) > best_size:
        best_size = len(s)
        best_set_indices = set(s)
        
    # Local search to improve the initial solution
    def local_search(initial_set):
        selected = set(initial_set)
        improved = True
        while improved:
            improved = False
            # Try adding each unselected vertex
            for v in range(M):
                if v in selected:
                    continue
                if not any(u in selected for u in adj[v]):
                    selected.add(v)
                    improved = True
            # Try swapping
            for v in list(selected):
                without_v = selected - {v}
                candidates = [u for u in range(M) if u not in without_v and not any(w in without_v for w in adj[u])]
                for u in candidates:
                    if len(without_v) + 1 > len(selected):
                        selected = without_v | {u}
                        improved = True
                        break
        return selected

    improved_set = local_search(best_set_indices)
    if len(improved_set) > best_size:
        best_size = len(improved_set)
        best_set_indices = improved_set

    # Branch and Bound with Bitsets
    sys.setrecursionlimit(10000)
    
    def greedy_matching_size(mask):
        """
        Compute a greedy matching size for the subgraph induced by mask.
        Used for upper bound: |V| - matching_size.
        """
        remaining = mask
        match_count = 0
        
        while remaining:
            # Find vertex with highest degree in remaining
            best_v = -1
            best_deg = -1
            temp = remaining
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (neighbors[v] & remaining).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v == -1:
                break
                
            # Find a neighbor in remaining
            neigh_mask = neighbors[best_v] & remaining
            if neigh_mask:
                # Pick the first available neighbor
                lsb_n = neigh_mask & -neigh_mask
                u = lsb_n.bit_length() - 1
                
                match_count += 1
                remaining &= ~(1 << best_v)
                remaining &= ~(1 << u)
            else:
                # No neighbor, remove best_v
                remaining &= ~(1 << best_v)
                
        return match_count

    def get_max_degree_vertex(mask):
        best_v = -1
        best_deg = -1
        temp = mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (neighbors[v] & mask).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb
        return best_v

    def solve(mask, current_size):
        nonlocal best_size, best_set_indices
        
        if time.time() - start_time > DEADLINE:
            return
            
        if mask == 0:
            if current_size > best_size:
                best_size = current_size
                # Reconstruct set is expensive, but we only need the size for the score?
                # No, we need to return the set.
                # We can reconstruct the set from the path in B&B, but we don't store the path.
                # To keep it simple and fast, we can store the best mask found.
                # But we need to return the values.
                # Let's store the best_mask.
            return
            
        # Cheap bound
        if current_size + mask.bit_count() <= best_size:
            return
            
        # Tight bound
        gm = greedy_matching_size(mask)
        ub = mask.bit_count() - gm
        if current_size + ub <= best_size:
            return
            
        # Pick vertex with highest degree
        v_choice = get_max_degree_vertex(mask)
        
        if v_choice == -1:
            return
            
        # Branch 1: Include v_choice
        # Remove v_choice and its neighbors
        new_mask = mask & ~(neighbors[v_choice] | (1 << v_choice))
        solve(new_mask, current_size + 1)
        
        if time.time() - start_time > DEADLINE:
            return
            
        # Branch 2: Exclude v_choice
        new_mask2 = mask & ~(1 << v_choice)
        solve(new_mask2, current_size)

    # We need to track the best set found by B&B to return it.
    # Since B&B doesn't naturally track the set, we can run it to find the max size,
    # and then if it finds a size larger than our initial greedy, we need the set.
    # A common trick: If B&B finds a larger size, we can try to reconstruct or just run a
    # targeted search. However, for this problem size, the initial greedy + local search
    # is often very close to optimal.
    # To be safe, let's modify B&B to store the best mask.
    
    best_mask = 0
    
    def solve_tracked(mask, current_size, current_mask):
        nonlocal best_size, best_mask
        
        if time.time() - start_time > DEADLINE:
            return
            
        if mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        if current_size + mask.bit_count() <= best_size:
            return
            
        gm = greedy_matching_size(mask)
        ub = mask.bit_count() - gm
        if current_size + ub <= best_size:
            return
            
        v_choice = get_max_degree_vertex(mask)
        if v_choice == -1:
            return
            
        # Include
        new_mask = mask & ~(neighbors[v_choice] | (1 << v_choice))
        new_current_mask = current_mask | (1 << v_choice)
        solve_tracked(new_mask, current_size + 1, new_current_mask)
        
        if time.time() - start_time > DEADLINE:
            return
            
        # Exclude
        new_mask2 = mask & ~(1 << v_choice)
        solve_tracked(new_mask2, current_size, current_mask)

    # Run B&B starting from empty set
    # Initial mask is all valid vertices
    full_mask = (1 << M) - 1
    solve_tracked(full_mask, 0, 0)
    
    # Determine final result
    if best_mask != 0:
        # Convert best_mask to set of indices
        result_indices = set()
        m = best_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            result_indices.add(idx)
            m ^= lsb
        # If B&B found a better solution, use it
        if best_size > len(best_set_indices):
            best_set_indices = result_indices
    else:
        # If B&B didn't find anything better (or didn't run), use the greedy result
        pass

    result_values = [vertex_map[i] for i in best_set_indices]
    result_values.sort()

    return result_values

if __name__ == "__main__":
    res = construct()
    print(f"Found set size: {len(res)}")
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
    valid = True
    for a in res:
        if 2 * a in squares:
            valid = False
            print(f"Invalid diagonal: {a}")
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            if res[i] + res[j] in squares:
                valid = False
                print(f"Invalid pair: {res[i]}, {res[j]}")
    if valid:
        print("Valid")
    else:
        print("Invalid")
