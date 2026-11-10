import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {edge: i for i, edge in enumerate(all_edges)}
    
    # Adjacency as list of sets for fast intersection
    adj = [set() for _ in range(n)]
    
    def creates_q3(u, v):
        # Check if adding (u, v) creates a Q3
        # u, v are not currently connected
        
        # 1. Find common neighbors x in N(u) ∩ N(v)
        common_uv = adj[u] & adj[v]
        if len(common_uv) < 2:
            return False
            
        squares = []
        for x in common_uv:
            # Find w in N(u) ∩ N(x) \ {v}
            # w must be such that (u,w), (w,x) exist
            # And w != v
            candidates_w = (adj[u] & adj[x]) - {v}
            for w in candidates_w:
                if w == u or w == x:
                    continue
                # Check if (x, w) is an edge? 
                # w is in N(x), so (x,w) exists.
                # Check if (u, w) is an edge?
                # w is in N(u), so (u,w) exists.
                # So we have square u-v-x-w-u.
                # Vertices: u, v, x, w.
                squares.append((x, w))
                
        if len(squares) < 2:
            return False
            
        # Check pairs of squares
        # squares[i] = (x_i, w_i)
        for i in range(len(squares)):
            x1, w1 = squares[i]
            for j in range(i + 1, len(squares)):
                x2, w2 = squares[j]
                
                # Check distinctness of x1, w1, x2, w2
                if x1 == x2 or x1 == w2 or w1 == x2 or w1 == w2:
                    continue
                    
                used = {u, v, x1, w1, x2, w2}
                
                # Find Z1 = N(x1) ∩ N(x2) \ {v}
                z1_candidates = (adj[x1] & adj[x2]) - {v}
                # Find Z2 = N(w1) ∩ N(w2) \ {u}
                z2_candidates = (adj[w1] & adj[w2]) - {u}
                
                for z1 in z1_candidates:
                    if z1 in used:
                        continue
                    for z2 in z2_candidates:
                        if z2 in used or z2 == z1:
                            continue
                        # Check if (z1, z2) is an edge
                        if z2 in adj[z1]:
                            return True
                            
        return False

    best_edges = 0
    best_mask = 0
    
    # Initial Greedy
    def greedy_init():
        nonlocal adj, best_edges, best_mask
        adj = [set() for _ in range(n)]
        mask = 0
        edges_list = all_edges[:]
        random.shuffle(edges_list)
        
        for edge in edges_list:
            u, v = edge
            if not creates_q3(u, v):
                adj[u].add(v)
                adj[v].add(u)
                mask |= (1 << edge_index[edge])
                
        return mask

    # Local Search (Tabu)
    def local_search(initial_mask):
        nonlocal adj, best_edges, best_mask
        
        # Rebuild adj from mask
        adj = [set() for _ in range(n)]
        for i, edge in enumerate(all_edges):
            if initial_mask & (1 << i):
                u, v = edge
                adj[u].add(v)
                adj[v].add(u)
                
        current_mask = initial_mask
        current_edges = bin(initial_mask).count('1')
        tabu_list = []
        tabu_tenure = 10
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_move = None
            best_score = -1
            best_add = -1
            best_rem = -1
            
            # Try all possible moves: Add 1, Remove 1, Swap 1
            # To keep it fast, we only try Add and Remove.
            
            # 1. Try Add
            for i, edge in enumerate(all_edges):
                if current_mask & (1 << i):
                    continue
                if i in tabu_list:
                    continue
                u, v = edge
                if not creates_q3(u, v):
                    # Valid move
                    score = current_edges + 1
                    if score > best_score:
                        best_score = score
                        best_add = i
                        best_rem = -1
                        best_move = (i, -1)
                        
            # 2. Try Remove
            for i, edge in enumerate(all_edges):
                if not (current_mask & (1 << i)):
                    continue
                if i in tabu_list:
                    continue
                u, v = edge
                # Removing always valid (unless we want to maintain connectivity, but we don't)
                # But removing reduces edge count.
                # Only useful if it allows adding multiple edges later?
                # For 1-opt, removing alone is bad.
                # So we skip pure removals unless we are stuck.
                pass
                
            if best_add != -1:
                # Apply move
                u, v = all_edges[best_add]
                adj[u].add(v)
                adj[v].add(u)
                current_mask |= (1 << best_add)
                current_edges += 1
                tabu_list.append(best_add)
                if len(tabu_list) > tabu_tenure:
                    tabu_list.pop(0)
                improved = True
            else:
                # Stuck, try a random removal to escape
                if current_edges > 0:
                    removed_edges = [i for i in range(len(all_edges)) if current_mask & (1 << i)]
                    if removed_edges:
                        rem_idx = random.choice(removed_edges)
                        u, v = all_edges[rem_idx]
                        adj[u].remove(v)
                        adj[v].remove(u)
                        current_mask &= ~(1 << rem_idx)
                        current_edges -= 1
                        tabu_list.append(rem_idx)
                        if len(tabu_list) > tabu_tenure:
                            tabu_list.pop(0)
                        # After removal, try to add
                        for i, edge in enumerate(all_edges):
                            if current_mask & (1 << i):
                                continue
                            if i in tabu_list:
                                continue
                            u2, v2 = edge
                            if not creates_q3(u2, v2):
                                adj[u2].add(v2)
                                adj[v2].add(u2)
                                current_mask |= (1 << i)
                                current_edges += 1
                                tabu_list.append(i)
                                if len(tabu_list) > tabu_tenure:
                                    tabu_list.pop(0)
                                break
                else:
                    break
                    
        if current_edges > best_edges:
            best_edges = current_edges
            best_mask = current_mask

    # Main Loop
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        init_mask = greedy_init()
        local_search(init_mask)
        
        # Random Restart with perturbation
        if r > 0 and r % 5 == 0:
             # Perturb best solution
             adj = [set() for _ in range(n)]
             mask = best_mask
             for i, edge in enumerate(all_edges):
                 if mask & (1 << i):
                     u, v = edge
                     adj[u].add(v)
                     adj[v].add(u)
             
             # Remove 5 random edges
             edges_in_mask = [i for i in range(len(all_edges)) if mask & (1 << i)]
             random.shuffle(edges_in_mask)
             for i in edges_in_mask[:5]:
                 u, v = all_edges[i]
                 adj[u].remove(v)
                 adj[v].remove(u)
                 mask &= ~(1 << i)
             
             local_search(mask)

    # Reconstruct best graph
    result = []
    for i, edge in enumerate(all_edges):
        if best_mask & (1 << i):
            result.append(list(edge))
            
    return result
