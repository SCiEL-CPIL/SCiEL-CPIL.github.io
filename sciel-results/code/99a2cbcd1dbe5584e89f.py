import random
import time
import math

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # Generate all K4s
    k4s = []
    k4_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    k4s.append((i, j, k, l))
                    k4_edges.append([
                        edge_index[(i, j)], edge_index[(i, k)], edge_index[(i, l)],
                        edge_index[(j, k)], edge_index[(j, l)],
                        edge_index[(k, l)]
                    ])
                    
    num_k4s = len(k4s)
    
    # Precompute which K4s contain each edge
    edge_to_k4s = [[] for _ in range(len(edges))]
    for k_idx, k4_e in enumerate(k4_edges):
        for e_idx in k4_e:
            edge_to_k4s[e_idx].append(k_idx)
            
    def count_distinct(colors_subset):
        return len(set(colors_subset))
        
    def evaluate(colours):
        # colours is a list of 120 ints
        k4_counts = [0] * num_k4s
        violations = 0
        for k_idx, k4_e in enumerate(k4_edges):
            c_set = set()
            for e_idx in k4_e:
                c_set.add(colours[e_idx])
            cnt = len(c_set)
            k4_counts[k_idx] = cnt
            if cnt < 5:
                violations += 1
        return violations, k4_counts

    def try_k(K, time_limit):
        # Initialize random colours in 0..K-1
        colours = [random.randint(0, K - 1) for _ in range(len(edges))]
        violations, k4_counts = evaluate(colours)
        
        if violations == 0:
            return colours, K
            
        # Simulated Annealing
        T = 1.0
        cooling = 0.999
        best_violations = violations
        best_colours = colours[:]
        
        while T > 0.01 and time.time() - start_time < time_limit:
            # Pick random edge
            e_idx = random.randint(0, len(edges) - 1)
            old_colour = colours[e_idx]
            new_colour = random.randint(0, K - 1)
            if new_colour == old_colour:
                continue
                
            # Calculate delta cost
            delta_violations = 0
            old_counts_snapshot = []
            
            for k_idx in edge_to_k4s[e_idx]:
                # Get current colours in this K4
                k4_e = k4_edges[k_idx]
                # We need to know the set of colours to update count
                # This is slightly expensive if we rebuild set.
                # Optimization: maintain color counts per K4?
                # For now, let's just re-evaluate the 6 edges for the affected K4s.
                # It's 91 * 6 = 546 operations. Acceptable.
                
                # Actually, let's just re-calculate the distinct count for the 91 K4s.
                # It's simpler and less bug-prone.
                pass
            
            # Let's stick to the simpler re-evaluation of affected K4s for correctness first, 
            # but optimize by only checking the 91 K4s.
            
            # We need to update k4_counts for the 91 K4s.
            # We don't strictly need to maintain k4_counts globally if we just 
            # compute the new total violations by re-checking the 91 K4s.
            # But we need to know if the change is accepted.
            
            # Let's compute new violations for the affected K4s.
            # Current violations for these K4s:
            old_local_violations = 0
            for k_idx in edge_to_k4s[e_idx]:
                if k4_counts[k_idx] < 5:
                    old_local_violations += 1
            
            # Apply change
            colours[e_idx] = new_colour
            
            new_local_violations = 0
            for k_idx in edge_to_k4s[e_idx]:
                k4_e = k4_edges[k_idx]
                c_set = set()
                for e_idx2 in k4_e:
                    c_set.add(colours[e_idx2])
                cnt = len(c_set)
                k4_counts[k_idx] = cnt
                if cnt < 5:
                    new_local_violations += 1
            
            delta_violations = new_local_violations - old_local_violations
            violations += delta_violations
            
            if violations < best_violations:
                best_violations = violations
                best_colours = colours[:]
                if violations == 0:
                    return colours, K

            # Accept or reject
            if delta_violations <= 0:
                T *= cooling
            else:
                if random.random() < math.exp(-delta_violations / T):
                    T *= cooling
                else:
                    # Reject
                    colours[e_idx] = old_colour
                    # Revert k4_counts
                    for k_idx in edge_to_k4s[e_idx]:
                        k4_e = k4_edges[k_idx]
                        c_set = set()
                        for e_idx2 in k4_e:
                            c_set.add(colours[e_idx2])
                        k4_counts[k_idx] = len(c_set)
                    violations -= delta_violations
                    T *= cooling

        if best_violations == 0:
            return best_colours, K
        return None, None

    # Try K=13, 14, 15
    for K in [13, 14, 15]:
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
        # Allocate time proportional to likelihood? 
        # Just give each 1/3 of remaining time roughly.
        t_limit = min(remaining_time, 30.0) 
        res, used_K = try_k(K, t_limit)
        if res:
            return res
            
    # Fallback to seed solution
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
