import numpy as np
import time

def partition(adjacency):
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    start_time = time.time()
    DEADLINE = start_time + 145.0

    def time_left():
        return DEADLINE - time.time()

    # Precompute neighbor lists for faster updates
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj[i, j] == 1:
                neighbors[i].append(j)

    def run_local_search(x, S, current_cut):
        """
        Greedy Local Search using vectorized delta calculation.
        x: spins (-1, 1)
        S: sum of neighbor spins
        """
        improved = True
        while improved:
            if time_left() < 1.0:
                break
            # Delta for flipping v is x[v] * S[v]
            deltas = x * S
            max_idx = np.argmax(deltas)
            max_delta = deltas[max_idx]
            
            if max_delta <= 0:
                break
            
            v = max_idx
            old_x_v = x[v]
            x[v] = -x[v]
            current_cut += max_delta
            
            # Update S for neighbors of v
            # S[u] -= 2 * old_x_v * adj[u, v]
            for u in neighbors[v]:
                S[u] -= 2 * old_x_v * adj[u, v]
                
        return x, S, current_cut

    def run_sa(x, S, current_cut, T_start, T_end, alpha, max_iters):
        """
        Simulated Annealing
        """
        T = T_start
        best_cut = current_cut
        best_x = x.copy()
        best_S = S.copy()
        
        for i in range(max_iters):
            if time_left() < 2.0:
                break
                
            v = np.random.randint(n)
            
            delta = x[v] * S[v]
            
            if delta >= 0:
                accept = True
            else:
                if T > 0.01:
                    if np.random.random() < np.exp(delta / T):
                        accept = True
                    else:
                        accept = False
                else:
                    accept = False
            
            if accept:
                old_x_v = x[v]
                x[v] = -x[v]
                current_cut += delta
                
                for u in neighbors[v]:
                    S[u] -= 2 * old_x_v * adj[u, v]
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_x = x.copy()
                    best_S = S.copy()
                    
            T *= alpha
            if T < T_end:
                T = T_end
                
        return best_x, best_S, best_cut

    best_global_cut = -1
    best_global_labels = None

    # Multiple restarts
    while time_left() > 5.0:
        # Initialize random spins
        x = np.random.choice([-1, 1], size=n).astype(np.int8)
        
        # Initialize S: S[v] = sum(x[u] * adj[v][u])
        # x @ adj works because adj is symmetric
        S = (x @ adj).astype(np.int16)
        
        # Calculate initial cut
        # Cut = E - 0.5 * x^T A x
        xAx = x @ (adj @ x)
        E_edges = np.sum(adj) // 2
        current_cut = E_edges - 0.5 * xAx
        
        # Local Search Phase 1
        x, S, current_cut = run_local_search(x, S, current_cut)
        
        # Simulated Annealing Phase
        T_start = 15.0
        T_end = 0.1
        alpha = 0.9999
        # Run SA for a fixed number of steps, adjusted by remaining time if needed
        # But fixed 2000 steps is fast for N=100
        x, S, current_cut = run_sa(x, S, current_cut, T_start, T_end, alpha, 2000)
        
        # Local Search Phase 2 (Polish)
        x, S, current_cut = run_local_search(x, S, current_cut)
        
        if current_cut > best_global_cut:
            best_global_cut = current_cut
            # Convert spins to 0/1 labels
            # x = 1 -> 0, x = -1 -> 1
            best_global_labels = (1 - x) // 2
            best_global_labels = best_global_labels.astype(int)

    if best_global_labels is None:
        return np.random.randint(0, 2, size=n).astype(int)

    return best_global_labels
