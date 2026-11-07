import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0  # Safety margin from 330s limit

def _run_aco(D, p, n_ants=20, n_iter=30, seed=42):
    """Run ACO with heuristic 1/d^p and return best tour length."""
    n = D.shape[0]
    rng = np.random.RandomState(seed)
    
    # Heuristic: 1/d^p, handle diagonal
    eta = np.ones((n, n))
    mask = D > 0
    eta[mask] = 1.0 / np.power(D[mask], p)
    eta[~mask] = 1e-9  # diagonal and zero distances
    
    tau = np.ones((n, n))
    best_len = float('inf')
    
    for it in range(n_iter):
        # Construct tours
        for ant in range(n_ants):
            current = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            path = [0]
            tour_len = 0.0
            
            for step in range(1, n):
                # Get available nodes
                available = np.where(~visited)[0]
                
                # Compute probabilities: tau * eta
                probs = tau[current, available] * eta[current, available]
                total = probs.sum()
                
                if total <= 0:
                    next_node = available[0]
                else:
                    probs = probs / total
                    r = rng.random()
                    cumsum = np.cumsum(probs)
                    next_node = available[np.searchsorted(cumsum, r)]
                
                tour_len += D[current, next_node]
                current = next_node
                visited[current] = True
                path.append(current)
            
            # Return to start
            tour_len += D[current, 0]
            path.append(0)
            
            if tour_len < best_len:
                best_len = tour_len
            
            # Store for pheromone update
            path_ant = path
            len_ant = tour_len
            # We need to store all paths for pheromone update
            # Let's collect them in lists
            if not hasattr(_run_aco, '_paths'):
                _run_aco._paths = []
                _run_aco._lens = []
            _run_aco._paths.append(path_ant)
            _run_aco._lens.append(len_ant)
        
        # Pheromone update
        tau *= 0.9
        for ant_idx in range(n_ants):
            l = _run_aco._lens[ant_idx]
            if l <= 0:
                continue
            reward = 1.0 / l
            path = _run_aco._paths[ant_idx]
            for k in range(len(path) - 1):
                i = path[k]
                j = path[k+1]
                tau[i, j] += reward
        
        # Clear stored paths
        _run_aco._paths = []
        _run_aco._lens = []
    
    return best_len

def heuristics(distance_matrix):
    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a reasonable default heuristic
        eta = 1.0 / (D + 1e-9)
        eta[D == 0] = 1e-9
        return eta
    
    # Allocate time for search
    # We'll spend up to ~90 seconds on search per call (3 calls * 90s = 270s, leaving 30s buffer)
    search_deadline = time.time() + 90.0
    
    # Initial candidates for p
    candidates = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
    
    best_p = 1.0
    best_score = float('inf')
    
    # Evaluate each candidate
    for p in candidates:
        if time.time() > search_deadline:
            break
        try:
            score = _run_aco(D, p, seed=42)
            if score < best_score:
                best_score = score
                best_p = p
        except Exception:
            continue
    
    # Refine around best_p with smaller steps
    if time.time() < search_deadline - 10:
        step = 0.1
        current_p = best_p
        for _ in range(15):
            if time.time() > search_deadline:
                break
            improved = False
            for delta_p in [-step, step]:
                new_p = current_p + delta_p
                if new_p <= 0.1:
                    continue
                try:
                    score = _run_aco(D, new_p, seed=42)
                    if score < best_score:
                        best_score = score
                        best_p = new_p
                        current_p = new_p
                        improved = True
                        break
                except Exception:
                    continue
            if not improved:
                step *= 0.7
            if step < 0.01:
                break
    
    # Final refinement with even smaller steps if time permits
    if time.time() < search_deadline - 5:
        step = 0.02
        current_p = best_p
        for _ in range(10):
            if time.time() > search_deadline:
                break
            improved = False
            for delta_p in [-step, step]:
                new_p = current_p + delta_p
                if new_p <= 0.05:
                    continue
                try:
                    score = _run_aco(D, new_p, seed=42)
                    if score < best_score:
                        best_score = score
                        best_p = new_p
                        current_p = new_p
                        improved = True
                        break
                except Exception:
                    continue
            if not improved:
                break
    
    # Generate final heuristic
    eta = np.ones((n, n))
    mask = D > 0
    eta[mask] = 1.0 / np.power(D[mask], best_p)
    eta[~mask] = 1e-9
    
    # Ensure all entries are positive
    eta[eta <= 0] = 1e-9
    
    return eta
