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
    all_vertices = list(range(n))
    
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
        """
        for u in range(n):
            for v in range(u+1, n):
                if (adj_mask[u] >> v) & 1:
                    common = adj_mask[u] & adj_mask[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    c1 = common
                    while c1:
                        lsb = c1 & -c1
                        c1 ^= lsb
                        w = lsb.bit_length() - 1
                        w_neigh_in_common = adj_mask[w] & common
                        if w_neigh_in_common:
                            return False
        return True

    def has_triangle_free_induced_subgraph_of_size_k(adj_mask, k=11):
        """
        Check if there exists a subset of k vertices inducing a triangle-free graph.
        Returns True if such a subset exists.
        """
        # Generate all k-combinations of vertices
        # For n=18, k=11, there are 31824 combinations.
        # We can iterate through them.
        
        # Optimization: Precompute triangles?
        # For a given mask, checking if it's triangle-free:
        # A subset S is triangle-free if for every edge (u,v) in S, there is no w in S connected to both.
        
        # Let's just iterate and check.
        # To speed up, we can use a recursive check or bitmasks.
        
        # We will use a simple check for each combination.
        # There are 31824 combinations.
        # Checking one combination:
        # Get the induced subgraph.
        # Check for triangles.
        
        # Efficient check for triangle in subset S:
        # For each u in S:
        #   For each v in S, v > u:
        #     if (u,v) is edge:
        #       common = adj[u] & adj[v] & S
        #       if common: return False (has triangle)
        
        # This is O(|S|^3) worst case, but with bitmasks it's fast.
        
        # We iterate over all 11-subsets.
        # We can generate them using itertools.combinations.
        
        from itertools import combinations
        
        for combo in combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            # Check if mask is triangle-free
            is_tf = True
            # Iterate u in combo
            for i, u in enumerate(combo):
                # Iterate v in combo, v > u
                for j in range(i+1, len(combo)):
                    v = combo[j]
                    if (adj_mask[u] >> v) & 1:
                        # Check common neighbors in mask
                        common = adj_mask[u] & adj_mask[v] & mask
                        if common:
                            is_tf = False
                            break
                if not is_tf:
                    break
            
            if is_tf:
                return True
                
        return False

    def generate_turan_3(n=18):
        """
        Generate the Turan graph T_3(18).
        Parts: 0-5, 6-11, 12-17.
        """
        adj = [0] * n
        # Connect all pairs of vertices from different parts
        for i in range(n):
            part_i = i // 6
            for j in range(i+1, n):
                part_j = j // 6
                if part_i != part_j:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    # Main search
    best_adj = generate_turan_3(n)
    best_score = 11 # Initial known value
    
    # Check if Turan graph has TF induced subgraph of size 11
    # It does (e.g., P1 U P2).
    
    # We want to find a graph where has_triangle_free_induced_subgraph_of_size_k(adj, 11) is False.
    # If we find one, the score is at most 10.
    
    current_adj = best_adj[:]
    
    # Precompute all non-edges for adding
    # We will try to add edges.
    
    # To make the search efficient, we can track which 11-subsets are still triangle-free.
    # But re-evaluating all 31824 subsets is fast enough (~0.1s).
    # So we can just re-evaluate.
    
    # However, we need to ensure K4-free.
    
    # Strategy:
    # 1. Start with Turan.
    # 2. Randomly add edges that don't create K4.
    # 3. If adding an edge reduces the number of TF 11-subsets, keep it.
    # 4. If we reach 0, we are done.
    
    # To count TF 11-subsets, we modify the function to return the count.
    
    def count_tf_induced_subgraphs_of_size_k(adj_mask, k=11):
        count = 0
        from itertools import combinations
        for combo in combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            is_tf = True
            for i, u in enumerate(combo):
                for j in range(i+1, len(combo)):
                    v = combo[j]
                    if (adj_mask[u] >> v) & 1:
                        common = adj_mask[u] & adj_mask[v] & mask
                        if common:
                            is_tf = False
                            break
                if not is_tf:
                    break
            
            if is_tf:
                count += 1
        return count

    current_count = count_tf_induced_subgraphs_of_size_k(current_adj)
    # For Turan, this count is high.
    
    # We want to reduce current_count to 0.
    
    # List of all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # We will perform a local search.
    # At each step, pick a random non-edge.
    # Add it.
    # If K4 created, revert.
    # If K4 not created, check new count.
    # If new count < old count, accept.
    # If new count == old count, accept with small probability (to escape local minima).
    # If new count > old count, revert.
    
    # To speed up K4 check, we can just run is_k4_free.
    
    # To speed up count, we can just run count_tf...
    
    # Since count is expensive (31824 * checks), we should minimize calls.
    # But 31824 is small.
    
    # Let's try this loop.
    
    random.seed(42)
    
    while time.time() < deadline - 5:
        # Pick a random non-edge
        u, v = random.choice(all_edges)
        
        if (current_adj[u] >> v) & 1:
            continue # Already an edge
            
        # Try adding
        current_adj[u] |= (1 << v)
        current_adj[v] |= (1 << u)
        
        if not is_k4_free(current_adj):
            # Revert
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            continue
        
        new_count = count_tf_induced_subgraphs_of_size_k(current_adj)
        
        if new_count < current_count:
            current_count = new_count
            if current_count == 0:
                # Found a graph with max TF induced subgraph size <= 10
                best_adj = current_adj[:]
                best_score = 10
                break
        else:
            # Revert
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            
            # Occasionally, try a 2-edge move or reset?
            # For now, just continue.
            
            # If we get stuck, we might need to remove an edge.
            # But removing edges increases TF sets.
            # So we should only add edges.
            
            # If we can't add any more edges without K4 or without increasing TF count,
            # we are stuck.
            
            # To escape, we can try to swap edges.
            # Remove an edge (u,v), add an edge (x,y).
            
            # For simplicity, let's just restart with a different random seed or structure if stuck for a while.
            # But we have time.
            
    # If we didn't find 0, return the best we have (which is Turan, score 11)
    # Or maybe we found a graph with count > 0 but less than Turan?
    # The score depends on the MAX size.
    # If count(11) > 0, max size is at least 11.
    # If count(11) == 0, max size is at most 10.
    
    # So we only improve if we reach 0.
    
    # If we didn't reach 0, we return Turan.
    
    if best_score > 10:
        best_adj = generate_turan_3(n)
        
    return get_edges_from_adj(best_adj)

if __name__ == "__main__":
    edges = construct()
    print(f"Number of edges: {len(edges)}")
