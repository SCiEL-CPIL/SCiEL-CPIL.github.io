"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency, dtype=np.int8)
    
    n = adjacency.shape[0]
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Pre-compute neighbors list for faster iteration
    # neighbors[v] = list of indices u where adjacency[v,u] == 1
    neighbors = [np.nonzero(adjacency[v])[0] for v in range(n)]
    degrees = np.array([len(nei) for nei in neighbors])
    
    def get_initial_cut(labels):
        # Efficient calculation using matrix product
        s = 2 * (1 - labels) - 1 
        val = s @ adjacency @ s
        total_edges = int(np.sum(adjacency)) // 2
        return total_edges - 0.5 * val

    def run_sa(labels_init, t_start, cooling_rate, max_steps):
        labels = labels_init.copy()
        current_cut = get_initial_cut(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        # Initialize sum_0 and sum_1
        # sum_0[v] = sum of adjacency[v, u] for u in S_0
        # sum_1[v] = sum of adjacency[v, u] for u in S_1
        sum_0 = np.zeros(n, dtype=np.int32)
        sum_1 = np.zeros(n, dtype=np.int32)
        
        for v in range(n):
            if labels[v] == 0:
                # sum_0[v] is sum of neighbors in S_0
                # sum_1[v] is sum of neighbors in S_1
                # We can compute these by iterating neighbors
                for u in neighbors[v]:
                    if labels[u] == 0:
                        sum_0[v] += 1
                    else:
                        sum_1[v] += 1
            else:
                for u in neighbors[v]:
                    if labels[u] == 0:
                        sum_0[v] += 1
                    else:
                        sum_1[v] += 1
        
        T = t_start
        
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            
            v = np.random.randint(n)
            l = labels[v]
            
            # Get d_same
            if l == 0:
                d_same = sum_0[v]
            else:
                d_same = sum_1[v]
                
            d_diff = degrees[v] - d_same
            delta = d_same - d_diff
            
            accept = False
            if delta > 0:
                accept = True
            else:
                if T > 1e-10:
                    r = np.random.random()
                    if r == 0:
                        accept = True
                    else:
                        accept = np.log(r) < delta / T
            
            if accept:
                # Flip v
                if l == 0:
                    # v moves 0 -> 1
                    # Update neighbors
                    for u in neighbors[v]:
                        if labels[u] == 0:
                            # u is in S_0, v left S_0
                            sum_0[u] -= 1
                        else:
                            # u is in S_1, v joined S_1
                            sum_1[u] += 1
                    labels[v] = 1
                else:
                    # v moves 1 -> 0
                    for u in neighbors[v]:
                        if labels[u] == 0:
                            # u is in S_0, v joined S_0
                            sum_0[u] += 1
                        else:
                            # u is in S_1, v left S_1
                            sum_1[u] -= 1
                    labels[v] = 0
                
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            
            T *= cooling_rate
            if T < 1e-10:
                T = 1e-10

        return best_labels, best_cut

    best_labels_global = None
    best_cut_global = -1
    
    restart_count = 0
    while time.time() < deadline - 5.0:
        restart_count += 1
        
        if restart_count % 2 == 1:
            labels_init = np.random.randint(0, 2, size=n)
        else:
            labels_init = np.arange(n) % 2
        
        # Tuning SA params
        # With O(degree) updates, we can do more steps.
        # Let's try 200,000 steps.
        max_steps = 200000
        t_start = 10.0
        cooling_rate = 0.9995
        
        labels_sol, cut_sol = run_sa(labels_init, t_start, cooling_rate, max_steps)
        
        if cut_sol > best_cut_global:
            best_cut_global = cut_sol
            best_labels_global = labels_sol
            
    return best_labels_global if best_labels_global is not None else np.arange(n) % 2
# EVOLVE-BLOCK-END
