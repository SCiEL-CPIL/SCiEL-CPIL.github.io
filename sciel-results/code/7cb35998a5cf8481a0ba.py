# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    n = len(adjacency)
    A = np.asarray(adjacency, dtype=np.int8)
    total_weight = A.sum()
    
    # Deadline guard: 150s wall-clock limit, return best with safety margin
    deadline = time.time() + 140.0
    best_cut = -1
    best_labels = np.zeros(n, dtype=np.int8)
    
    rng = np.random.default_rng(42)
    
    def cut_size_from_x(x):
        # x is +/-1
        val = x @ (A @ x)
        return (total_weight - val) // 2

    def vectorized_local_search(x, Ax):
        # Flip all vertices with positive gain
        improved = True
        while improved:
            gains = x * Ax
            mask = gains > 0
            if not np.any(mask):
                break
            x[mask] *= -1
            Ax = A @ x
            improved = True
        return x, Ax

    def tabu_search(x, Ax, tabu_len, max_iters):
        tabu_list = {}
        current_cut = cut_size_from_x(x)
        best_local_cut = current_cut
        best_local_x = x.copy()
        
        for step in range(max_iters):
            if time.time() > deadline:
                break
            
            gains = x * Ax
            
            valid = np.ones(n, dtype=bool)
            for v, exp in tabu_list.items():
                if step - exp < tabu_len:
                    valid[v] = False
            
            masked_gains = np.where(valid, gains, -np.inf)
            best_v = np.argmax(masked_gains)
            best_gain = masked_gains[best_v]
            
            # If all are tabu, pick best overall
            if best_gain == -np.inf:
                best_v = np.argmax(gains)
                best_gain = gains[best_v]

            x[best_v] *= -1
            # Update Ax: Ax = Ax + 2 * x_new * A[:, best_v]
            Ax += 2 * x[best_v] * A[:, best_v]
            
            current_cut += best_gain
            
            if current_cut > best_local_cut:
                best_local_cut = current_cut
                best_local_x = x.copy()
            
            tabu_list[best_v] = step
            
        return best_local_x

    while time.time() < deadline:
        # Random init
        labels = rng.integers(0, 2, n).astype(np.int8)
        x = (2 * labels - 1).astype(np.int8)
        Ax = A @ x
        
        # Local Search
        x, Ax = vectorized_local_search(x, Ax)
        cur_cut = cut_size_from_x(x)
        
        if cur_cut > best_cut:
            best_cut = cur_cut
            best_labels = ((x + 1) // 2).astype(np.int8)
        
        # Tabu Search
        if time.time() < deadline - 5:
            x_tabu = tabu_search(x, Ax, tabu_len=15, max_iters=100)
            # Polish
            x_tabu, _ = vectorized_local_search(x_tabu, A @ x_tabu)
            final_cut = cut_size_from_x(x_tabu)
            
            if final_cut > best_cut:
                best_cut = final_cut
                best_labels = ((x_tabu + 1) // 2).astype(np.int8)

    return best_labels
# EVOLVE-BLOCK-END
