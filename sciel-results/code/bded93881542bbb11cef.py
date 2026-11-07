import numpy as np

def heuristics(prizes, distance_matrix, max_len):
        # Ensure inputs are numpy arrays
        prizes = np.asarray(prizes, dtype=np.float64)
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        
        # Distance from each node to the start (node 0)
        dist_to_start = distance_matrix[:, 0]
        
        # Effective distance: cost to go from i to j and then j to 0
        # This is a lower bound on the additional path length if we go i->j->0
        # Actually, the constraint is L + d_ij + d_j0 <= max_len.
        # So d_ij + d_j0 is the "marginal cost" of visiting j from i.
        
        # Heuristic: Reward / Cost
        # We use a power law to sharpen the preference.
        # alpha=1.5 for prize, beta=1.5 for effective distance.
        
        # Add epsilon to avoid division by zero
        eps = 1e-9
        effective_dist = distance_matrix + dist_to_start[None, :] + eps
        prize_part = np.power(prizes[None, :] + eps, 1.5)
        
        heuristic = prize_part / np.power(effective_dist, 1.5)
        
        # Ensure all values are finite and positive
        heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
        heuristic = np.maximum(heuristic, 1e-9)
        
        return heuristic
