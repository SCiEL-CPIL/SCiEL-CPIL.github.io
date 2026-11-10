import random
import time

def _has_c6_through(u, v, adj):
    """Check if adding edge (u, v) creates a C6.
    This is true iff there is a simple path of length 5 from u to v in the
    current graph.
    """
    def dfs(node, depth, visited):
        if depth == 5:
            return node == v
        for nb in adj[node]:
            if nb == v and depth == 4:
                return True
            if nb not in visited and nb != u:
                visited.add(nb)
                if dfs(nb, depth + 1, visited):
                    return True
                visited.remove(nb)
        return False
    
    visited = {u}
    return dfs(u, 0, visited)

def construct():
    n = 40
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)
    
    best_edges = []
    best_count = 0
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    # We will run multiple trials of Randomized Greedy + Local Search
    trial = 0
    while time.time() - start_time < time_limit:
        trial += 1
        adj = [set() for _ in range(n)]
        
        # 1. Randomized Greedy Construction
        # Shuffle edges
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() - start_time > time_limit:
                break
            # Check if adding (u,v) creates a C6
            if not _has_c6_through(u, v, adj):
                adj[u].add(v)
                adj[v].add(u)
        
        # Convert to list for counting and storage
        current_edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    current_edges.add((u, v))
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = [list(e) for e in current_edges]
        
        # 2. Local Search (Swaps)
        # We try to swap an existing edge with a non-existing edge
        # to potentially allow more edges later, or just rearrange.
        # Note: Swapping keeps the edge count constant.
        # To increase edge count, we need to find a sequence of swaps
        # that leads to a state where we can ADD an edge.
        # So, after swaps, we should try to ADD edges again.
        
        # Let's do a few rounds of swaps, then try to add.
        swaps_per_round = 200
        rounds = 5
        
        for _ in range(rounds):
            if time.time() - start_time > time_limit:
                break
                
            # Perform random swaps
            # Get list of existing and non-existing edges
            existing_list = list(current_edges)
            non_existing_list = [e for e in all_edges if e not in current_edges]
            
            if not non_existing_list:
                break
                
            for _ in range(swaps_per_round):
                if time.time() - start_time > time_limit:
                    break
                if not existing_list or not non_existing_list:
                    break
                
                e_rem = random.choice(existing_list)
                e_add = random.choice(non_existing_list)
                
                u_rem, v_rem = e_rem
                u_add, v_add = e_add
                
                # Temporarily remove e_rem
                adj[u_rem].remove(v_rem)
                adj[v_rem].remove(u_rem)
                
                # Check if e_add can be added
                if not _has_c6_through(u_add, v_add, adj):
                    # Accept swap
                    adj[u_add].add(v_add)
                    adj[v_add].add(u_add)
                    
                    # Update sets
                    current_edges.remove(e_rem)
                    current_edges.add(e_add)
                    
                    existing_list.remove(e_rem)
                    existing_list.append(e_add)
                    non_existing_list.remove(e_add)
                    non_existing_list.append(e_rem)
                else:
                    # Revert
                    adj[u_rem].add(v_rem)
                    adj[v_rem].add(u_rem)
            
            # Try to add more edges after swaps
            # Re-evaluate non-existing edges
            non_existing_list = [e for e in all_edges if e not in current_edges]
            random.shuffle(non_existing_list)
            
            for u, v in non_existing_list:
                if time.time() - start_time > time_limit:
                    break
                if not _has_c6_through(u, v, adj):
                    adj[u].add(v)
                    adj[v].add(u)
                    current_edges.add((u, v))
            
            # Update best
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = [list(e) for e in current_edges]
        
        # If we are running low on time, break outer loop
        if time.time() - start_time > time_limit - 5:
            break

    return best_edges
