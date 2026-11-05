import numpy as np

def priority(adjacency):
        # Ensure float for operations
        adj = np.asarray(adjacency, dtype=float)
        
        # Degrees
        degrees = adj.sum(axis=1)
        
        # Sum of degrees of neighbors
        # A * degrees gives for each vertex v, sum of deg(u) for u in N(v)
        neighbor_degree_sums = adj @ degrees
        
        # Mean degree of neighbors
        # Avoid division by zero for isolated nodes
        # For isolated nodes, mean is 0.
        mean_neighbor_degrees = np.zeros_like(degrees)
        mask = degrees > 0
        mean_neighbor_degrees[mask] = neighbor_degree_sums[mask] / degrees[mask]
        
        # Score: Reward high neighbor degree, penalize own degree
        # The knowledge suggests "average neighbor degree minus own degree"
        # Let's use a weighted combination to balance scale.
        # In ER graphs, degrees are O(1) to O(log n).
        # If we just do mean_nb - deg, the scale is similar.
        
        scores = mean_neighbor_degrees - degrees
        
        # Optional: Add a small random tie-breaker or keep deterministic?
        # The task says "fixed greedy decoder". Deterministic is safer for reproducibility 
        # unless the evaluator shuffles. The prompt implies a fixed order based on score.
        # Ties in score: The decoder order for ties is unspecified. 
        # Usually, numpy sort is stable, so original index order is kept.
        # To be safe, we can add a tiny epsilon based on index to break ties deterministically 
        # in a way that might favor lower indices? Or just rely on stable sort.
        # Given the "fixed greedy decoder", it likely just does `np.argsort(-scores)`.
        
        return scores
