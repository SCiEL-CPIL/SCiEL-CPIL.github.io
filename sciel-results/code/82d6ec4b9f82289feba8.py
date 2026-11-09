import numpy as np
import time

def _run_aco(prizes, dist, max_len, heuristic, seed):
    n = len(prizes)
    rng = np.random.RandomState(seed)
    pheromone = np.ones((n, n), dtype=float)
    heur = np.where(heuristic > 0, heuristic, 1e-9)
    best_overall = 0.0
    for it in range(30):
        best_iter = 0.0
        for ant in range(20):
            visited = set([0])
            current = 0
            length = 0.0
            total_prize = 0.0
            while True:
                feasible = []
                for j in range(n):
                    if j in visited:
                        continue
                    if length + dist[current, j] + dist[j, 0] <= max_len:
                        feasible.append(j)
                if not feasible:
                    break
                probs = np.array([pheromone[current, j] * heur[current, j] for j in feasible])
                probs = probs / probs.sum()
                idx = rng.choice(len(feasible), p=probs)
                j = feasible[idx]
                length += dist[current, j]
                total_prize += prizes[j]
                visited.add(j)
                current = j
            if total_prize > best_iter:
                best_iter = total_prize
            if total_prize > best_overall:
                best_overall = total_prize
        pheromone *= 0.9
        pheromone += best_iter / 10.0
    return best_overall

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 320  # 10s safety margin
    n = len(prizes)
    
    # Generate candidate heuristics
    candidates = []
    
    # Candidate 1: current baseline
    candidates.append((prizes[None, :] + 1e-9) / (distance_matrix + 1e-9))
    
    # Candidate 2: include return distance
    candidates.append((prizes[None, :] + 1e-9) / (distance_matrix + distance_matrix[:, 0][None, :] + 1e-9))
    
    # Candidate 3: squared prize
    candidates.append((prizes[None, :]**2 + 1e-9) / (distance_matrix + distance_matrix[:, 0][None, :] + 1e-9))
    
    # Candidate 4: exponential decay
    candidates.append(prizes[None, :] * np.exp(-(distance_matrix + distance_matrix[:, 0][None, :])))
    
    # Candidate 5: prize / (d + d0)^1.5
    candidates.append((prizes[None, :] + 1e-9) / (distance_matrix + distance_matrix[:, 0][None, :] + 1e-9)**1.5)
    
    # Candidate 6: prize^3 / (d + d0)
    candidates.append((prizes[None, :]**3 + 1e-9) / (distance_matrix + distance_matrix[:, 0][None, :] + 1e-9))
    
    # Evaluate each candidate
    best_score = -1
    best_heuristic = candidates[0]
    
    for i, heur in enumerate(candidates):
        if time.time() > deadline - 30:  # Leave 30s margin
            break
        # Run ACO with this heuristic
        score = _run_aco(prizes, distance_matrix, max_len, heur, seed=42)
        if score > best_score:
            best_score = score
            best_heuristic = heur
    
    # Ensure positive values
    best_heuristic = np.where(best_heuristic > 0, best_heuristic, 1e-9)
    
    return best_heuristic
