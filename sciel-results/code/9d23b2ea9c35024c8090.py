import time
import random
import sys

def construct():
    """
    Construct a K4-free graph on 18 vertices minimizing the largest triangle-free induced subgraph.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 18
    
    def get_edges_from_adj(adj_mask):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj_mask[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def is_k4_free(adj_mask):
        """
        Check if graph represented by adj_mask is K4-free.
        adj_mask[i] is a bitmask of neighbors of i.
        """
        for u in range(n):
            for v in range(u+1, n):
                if (adj_mask[u] >> v) & 1:
                    # Common neighbors of u and v
                    common = adj_mask[u] & adj_mask[v]
                    # Remove u and v from common (they shouldn't be in their own neighbor list usually, but just in case)
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    
                    # Check if any two vertices in common are connected
                    # Iterate through bits in common
                    c1 = common
                    while c1:
                        # Get lowest set bit
                        lsb = c1 & -c1
                        c1 ^= lsb
                        w = lsb.bit_length() - 1
                        # Check if w is connected to any other vertex in common (excluding w)
                        # neighbors of w in common
                        w_neigh_in_common = adj_mask[w] & common
                        if w_neigh_in_common:
                            return False
        return True

    def eval_max_tf(adj_mask):
        """
        Find the largest triangle-free induced subgraph size.
        Uses DP over all subsets.
        """
        # Precompute triangles conflicts
        # tri_conflicts[v] is a list of bitmasks, each representing a pair {u, w}
        # such that (u, v, w) is a triangle.
        tri_conflicts = [[] for _ in range(n)]
        
        for v in range(n):
            # Neighbors of v
            neigh = adj_mask[v]
            # Iterate through neighbors u
            u_bit = neigh
            while u_bit:
                lsb_u = u_bit & -u_bit
                u_bit ^= lsb_u
                u = lsb_u.bit_length() - 1
                # For each neighbor u, find common neighbors w of u and v
                # w must be in neigh (since w is neighbor of v) and in adj_mask[u]
                common_w = neigh & adj_mask[u]
                # Remove u and v
                common_w &= ~(1 << u)
                common_w &= ~(1 << v)
                
                # For each w in common_w, (u, v, w) is a triangle
                w_bit = common_w
                while w_bit:
                    lsb_w = w_bit & -w_bit
                    w_bit ^= lsb_w
                    w = lsb_w.bit_length() - 1
                    # The pair is {u, w}
                    pair_mask = (1 << u) | (1 << w)
                    tri_conflicts[v].append(pair_mask)
        
        # DP over all masks
        # valid[mask] is True if mask is triangle-free
        # We only need to track the max size, but we need valid[] for DP
        # 2^18 = 262144 booleans. This fits in memory.
        valid = [False] * (1 << n)
        valid[0] = True
        
        max_size = 0
        
        for mask in range(1, 1 << n):
            # Get lowest set bit
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            prev_mask = mask ^ lsb
            
            if not valid[prev_mask]:
                valid[mask] = False
                continue
            
            # Check if adding v creates a triangle
            # A triangle is created if there exists a pair {u, w} in tri_conflicts[v]
            # such that both u and w are in prev_mask.
            conflict_found = False
            for pair_mask in tri_conflicts[v]:
                # Check if prev_mask contains both bits in pair_mask
                if (prev_mask & pair_mask) == pair_mask:
                    conflict_found = True
                    break
            
            if not conflict_found:
                valid[mask] = True
                # Update max size
                size = mask.bit_count()
                if size > max_size:
                    max_size = size
            else:
                valid[mask] = False
                
        return max_size

    def local_search(adj_mask, time_limit):
        """
        Local search using edge flips.
        """
        current_adj = adj_mask[:]
        current_score = eval_max_tf(current_adj)
        best_adj = current_adj[:]
        best_score = current_score
        
        # Precompute all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        while time.time() < time_limit:
            improved = False
            # Shuffle edges for randomness
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                if time.time() > time_limit:
                    break
                
                # Try flipping edge (u, v)
                old_val = (current_adj[u] >> v) & 1
                # Flip
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
                
                # Check K4-free
                if not is_k4_free(current_adj):
                    # Revert
                    current_adj[u] ^= (1 << v)
                    current_adj[v] ^= (1 << u)
                    continue
                
                new_score = eval_max_tf(current_adj)
                
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj[:]
                    improved = True
                else:
                    # Revert
                    current_adj[u] ^= (1 << v)
                    current_adj[v] ^= (1 << u)
                    
            if not improved:
                # If no improvement in a full pass, try a random jump or break
                # To avoid getting stuck, we can accept equal scores with some probability
                # But for simplicity, let's just break if no improvement found in a pass
                # Or, we can continue to explore equal-score moves
                # Let's add a mechanism to accept equal moves to escape local minima
                pass_count = 0
                while pass_count < 5 and time.time() < time_limit:
                    pass_count += 1
                    random.shuffle(all_edges)
                    found_equal = False
                    for u, v in all_edges:
                        if time.time() > time_limit:
                            break
                        old_val = (current_adj[u] >> v) & 1
                        current_adj[u] ^= (1 << v)
                        current_adj[v] ^= (1 << u)
                        
                        if not is_k4_free(current_adj):
                            current_adj[u] ^= (1 << v)
                            current_adj[v] ^= (1 << u)
                            continue
                        
                        new_score = eval_max_tf(current_adj)
                        
                        if new_score < current_score:
                            current_score = new_score
                            if new_score < best_score:
                                best_score = new_score
                                best_adj = current_adj[:]
                            found_equal = True # Actually improved
                            break
                        elif new_score == current_score and random.random() < 0.5:
                            # Accept equal move
                            found_equal = True
                            break
                        else:
                            current_adj[u] ^= (1 << v)
                            current_adj[v] ^= (1 << u)
                    
                    if found_equal:
                        break
                    else:
                        break
                        
        return best_adj, best_score

    def generate_random_k4_free():
        """
        Generate a random K4-free graph.
        """
        adj = [0] * n
        # Start with random graph p=0.5
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Remove edges to make it K4-free
        changed = True
        while changed:
            changed = False
            # Find a K4
            # Iterate through all 4-subsets? Too slow.
            # Use the is_k4_free logic to find a violation?
            # Or just iterate through edges and check if adding it (or keeping it) causes K4?
            # Simpler: iterate through all quadruples? No.
            
            # Let's just check if it's K4-free. If not, find a K4 and remove an edge.
            if is_k4_free(adj):
                break
            
            # Find a K4
            # We can reuse the logic from is_k4_free but stop at first found
            k4_found = None
            for u in range(n):
                for v in range(u+1, n):
                    if (adj[u] >> v) & 1:
                        common = adj[u] & adj[v]
                        common &= ~(1 << u)
                        common &= ~(1 << v)
                        c1 = common
                        while c1:
                            lsb = c1 & -c1
                            c1 ^= lsb
                            w = lsb.bit_length() - 1
                            w_neigh_in_common = adj[w] & common
                            if w_neigh_in_common:
                                # Found K4: u, v, w, and some x in w_neigh_in_common
                                lsb_x = w_neigh_in_common & -w_neigh_in_common
                                x = lsb_x.bit_length() - 1
                                k4_found = (u, v, w, x)
                                break
                        if k4_found:
                            break
                if k4_found:
                    break
            
            if k4_found:
                u, v, w, x = k4_found
                # Remove a random edge from the K4
                edges = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                eu, ev = random.choice(edges)
                adj[eu] &= ~(1 << ev)
                adj[ev] &= ~(1 << eu)
                changed = True
        
        return adj

    # Main search loop
    best_adj = None
    best_score = n + 1
    
    # Try a few random starts
    num_starts = 5
    
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        adj = generate_random_k4_free()
        
        # Evaluate initial
        score = eval_max_tf(adj)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            
        # Local search
        remaining = deadline - time.time() - 5
        if remaining > 2:
            adj_ls, score_ls = local_search(adj, remaining)
            if score_ls < best_score:
                best_score = score_ls
                best_adj = adj_ls
                
    # Fallback: Turan graph T(18,3)
    # Parts: 0-5, 6-11, 12-17
    turan_adj = [0] * n
    for i in range(6):
        for j in range(6, 12):
            turan_adj[i] |= (1 << j)
            turan_adj[j] |= (1 << i)
        for j in range(12, 18):
            turan_adj[i] |= (1 << j)
            turan_adj[j] |= (1 << i)
    for i in range(6, 12):
        for j in range(12, 18):
            turan_adj[i] |= (1 << j)
            turan_adj[j] |= (1 << i)
            
    turan_score = eval_max_tf(turan_adj)
    if turan_score < best_score:
        best_score = turan_score
        best_adj = turan_adj
        
    if best_adj is None:
        best_adj = turan_adj
        
    return get_edges_from_adj(best_adj)

if __name__ == "__main__":
    edges = construct()
    print(f"Number of edges: {len(edges)}")
