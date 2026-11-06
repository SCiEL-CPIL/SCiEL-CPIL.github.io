import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 148  # Safety margin before 150s kill
    
    n = 16
    m_target = 65
    
    # Precompute bitmasks for all pairs? 
    # adj_mask[i] is an integer where bit j is 1 if edge (i,j) exists.
    
    def calc_score(adj_mask, deg):
        """
        Fast score calculation using bitmasks.
        adj_mask: list of integers, adj_mask[i] has bits set for neighbors of i.
        deg: list of integers.
        """
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    # Common neighbors of u and v
                    common = adj_mask[u] & adj_mask[v]
                    # Remove u and v from common (though they aren't neighbors of themselves in simple graph mask usually, 
                    # but adj_mask[u] doesn't have bit u set. 
                    # However, v is in adj_mask[u] and u is in adj_mask[v].
                    # So common might contain u and v? 
                    # adj_mask[u] has bit v. adj_mask[v] has bit u.
                    # common = adj_mask[u] & adj_mask[v].
                    # Bit u in common? adj_mask[v] has bit u. adj_mask[u] has bit u? No. So bit u is 0 in common.
                    # Bit v in common? adj_mask[u] has bit v. adj_mask[v] has bit v? No. So bit v is 0 in common.
                    # So common contains only w != u, v.
                    
                    if common:
                        # Iterate through set bits in common
                        w_mask = common
                        while w_mask:
                            # Get lowest set bit
                            lsb = w_mask & -w_mask
                            w = lsb.bit_length() - 1
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
                            w_mask ^= lsb
        return max_sum

    def generate_k610_structure():
        """Generate K_{6,10} + 5 matching edges structure with random vertex labeling."""
        adj_mask = [0] * n
        deg = [0] * n
        vertices = list(range(n))
        random.shuffle(vertices)
        A = vertices[:6]
        B = vertices[6:]
        
        # Bipartite edges A-B
        for u in A:
            for v in B:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
        
        # 5 disjoint edges in B (perfect matching on 10 vertices)
        random.shuffle(B)
        for i in range(5):
            u, v = B[2*i], B[2*i+1]
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            
        return adj_mask, deg

    def local_search(adj_mask, deg, max_iter=5000, time_limit=10):
        """Hill climbing with random restarts within a phase."""
        current_score = calc_score(adj_mask, deg)
        best_score = current_score
        best_adj_mask = adj_mask.copy()
        best_deg = deg.copy()
        
        edges_set = set()
        non_edges_set = set()
        for u in range(n):
            for v in range(u+1, n):
                if (adj_mask[u] >> v) & 1:
                    edges_set.add((u, v))
                else:
                    non_edges_set.add((u, v))
        
        edges_list = list(edges_set)
        non_edges_list = list(non_edges_set)
        
        T = 0.5
        cooling = 0.999
        no_improve_count = 0
        tabu_list = []
        tabu_tenure = 50
        
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            if not edges_list or not non_edges_list:
                break
                
            # Avoid tabu moves
            valid_edges = [e for e in edges_list if e not in tabu_list]
            valid_non_edges = [ne for ne in non_edges_list if ne not in tabu_list]
            
            if not valid_edges or not valid_non_edges:
                tabu_list = []
                valid_edges = edges_list
                valid_non_edges = non_edges_list
            
            u, v = random.choice(valid_edges)
            x, y = random.choice(valid_non_edges)
            
            move = ((u, v), (x, y))
            
            # Apply swap
            adj_mask[u] ^= (1 << v)
            adj_mask[v] ^= (1 << u)
            adj_mask[x] ^= (1 << y)
            adj_mask[y] ^= (1 << x)
            
            deg[u] -= 1
            deg[v] -= 1
            deg[x] += 1
            deg[y] += 1
            
            new_score = calc_score(adj_mask, deg)
            
            accept = False
            if new_score < current_score:
                accept = True
                no_improve_count = 0
                if new_score < best_score:
                    best_score = new_score
                    best_adj_mask = adj_mask.copy()
                    best_deg = deg.copy()
            elif new_score == current_score:
                if random.random() < 0.3:
                    accept = True
                    no_improve_count += 1
            else:
                delta = new_score - current_score
                if random.random() < np.exp(-delta / T):
                    accept = True
                    no_improve_count += 1
            
            if accept:
                # Update sets
                edges_set.remove((u, v))
                non_edges_set.add((u, v))
                edges_set.add((x, y))
                non_edges_set.remove((x, y))
                
                edges_list = list(edges_set)
                non_edges_list = list(non_edges_set)
                
                # Update tabu
                tabu_list.append(move)
                if len(tabu_list) > tabu_tenure:
                    tabu_list.pop(0)
            else:
                # Revert
                adj_mask[u] ^= (1 << v)
                adj_mask[v] ^= (1 << u)
                adj_mask[x] ^= (1 << y)
                adj_mask[y] ^= (1 << x)
                
                deg[u] += 1
                deg[v] += 1
                deg[x] -= 1
                deg[y] -= 1
            
            T *= cooling
            if T < 0.05:
                T = 0.05
                
            # If stuck for too long, perturb
            if no_improve_count > 200:
                # Do a few random perturbations
                for _ in range(3):
                    if not edges_list or not non_edges_list:
                        break
                    u2, v2 = random.choice(edges_list)
                    x2, y2 = random.choice(non_edges_list)
                    
                    adj_mask[u2] ^= (1 << v2)
                    adj_mask[v2] ^= (1 << u2)
                    adj_mask[x2] ^= (1 << y2)
                    adj_mask[y2] ^= (1 << x2)
                    
                    deg[u2] -= 1
                    deg[v2] -= 1
                    deg[x2] += 1
                    deg[y2] += 1
                    
                    edges_set.remove((u2, v2))
                    non_edges_set.add((u2, v2))
                    edges_set.add((x2, y2))
                    non_edges_set.remove((x2, y2))
                    
                    edges_list = list(edges_set)
                    non_edges_list = list(non_edges_set)
                
                current_score = calc_score(adj_mask, deg)
                no_improve_count = 0
                T = 0.5
                tabu_list = []
        
        return best_adj_mask, best_deg, best_score

    # Run multiple independent searches with different initializations
    best_overall_score = float('inf')
    best_overall_adj_mask = None
    best_overall_deg = None
    
    num_restarts = 10
    time_per_restart = (deadline - (time.time() - start_time)) / num_restarts
    time_per_restart = min(time_per_restart, 12)  # Cap per restart
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2:
            break
        
        # Generate a diverse initial structure
        if restart == 0:
            # Use the known good K_{6,10} structure
            adj_mask, deg = generate_k610_structure()
        else:
            # For other restarts, start from K_{6,10} but with random perturbations
            adj_mask, deg = generate_k610_structure()
            # Apply a few random swaps to diversify
            edges_set = set((u, v) for u in range(n) for v in range(u+1, n) if (adj_mask[u] >> v) & 1)
            non_edges_set = set((u, v) for u in range(n) for v in range(u+1, n) if not ((adj_mask[u] >> v) & 1))
            
            for _ in range(random.randint(2, 5)):
                if not edges_set or not non_edges_set:
                    break
                u, v = random.choice(list(edges_set))
                x, y = random.choice(list(non_edges_set))
                
                adj_mask[u] ^= (1 << v)
                adj_mask[v] ^= (1 << u)
                adj_mask[x] ^= (1 << y)
                adj_mask[y] ^= (1 << x)
                
                deg[u] -= 1
                deg[v] -= 1
                deg[x] += 1
                deg[y] += 1
                
                edges_set.remove((u, v))
                non_edges_set.add((u, v))
                edges_set.add((x, y))
                non_edges_set.remove((x, y))
        
        # Run local search
        adj_ls, deg_ls, score_ls = local_search(adj_mask, deg, max_iter=5000, time_limit=time_per_restart)
        
        if score_ls < best_overall_score:
            best_overall_score = score_ls
            best_overall_adj_mask = adj_ls
            best_overall_deg = deg_ls
            
            # Early termination if we hit the theoretical target
            if best_overall_score <= 23:
                break
    
    # Final answer
    if best_overall_adj_mask is None:
        # Fallback to simple structure
        adj_mask, deg = generate_k610_structure()
        best_overall_adj_mask = adj_mask
        best_overall_deg = deg
        best_overall_score = calc_score(adj_mask, deg)
    
    # Reconstruct edge list
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_overall_adj_mask[i] >> j) & 1:
                final_edges.append([i, j])
                
    return {'n': n, 'edges': final_edges}
