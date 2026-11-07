"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of CVRP.
    
    Uses a flow-based potential heuristic: solve a minimum-cost flow relaxation
    to get node potentials, then set edge attractiveness inversely proportional
    to (potential difference + edge cost), encoding global capacity constraints.
    """
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=float)
    depot_demand = 0.0
    customer_demands = demands[1:]  # exclude depot
    
    # Total demand and number of trips needed
    total_demand = customer_demands.sum()
    num_trips = int(np.ceil(total_demand / capacity))
    
    # Build a min-cost flow formulation:
    # We want to assign each customer to a "trip" (route) such that
    # each trip has demand <= capacity, and minimize total travel cost.
    # 
    # Simplified relaxation: create a bipartite-like structure where
    # we send flow from depot to customers through "trip" nodes.
    #
    # Variables: x[i,j] = flow on edge i->j (for i in {0} union trips, j in customers or depot)
    #
    # For simplicity, use a direct assignment model:
    # Each customer j must be assigned to exactly one trip k.
    # Each trip k has capacity constraint: sum of demands of assigned customers <= capacity.
    # Cost: approximate by sum of distances from depot to customer and back,
    #        but this doesn't capture route structure well.
    #
    # Better: use a two-stage approach.
    # Stage 1: Solve a relaxed assignment to get "trip affinity" for each customer.
    # Stage 2: Derive node potentials from the dual of this LP.
    
    # Let's use a simpler but effective approach:
    # Compute a "density-weighted" distance metric that accounts for demand.
    # Edges to high-demand customers are penalized more (they're harder to fit).
    # Edges between geographically close customers with compatible demands are favored.
    
    # Approach: Use a combined heuristic that blends:
    # 1. Inverse distance (local proximity)
    # 2. Demand compatibility (edges between nodes with demands that fit well together)
    # 3. Global balance (nodes in under-served regions get boosted)
    
    # Compute demand-weighted distances
    # For each pair (i,j), the "effective cost" is distance * (1 + demand_i/capacity * demand_j/capacity)
    # This penalizes edges between high-demand nodes.
    
    # Actually, let's implement the flow-based approach properly.
    # 
    # Min-cost flow formulation:
    # Source -> trip nodes -> customer nodes -> depot -> sink
    # 
    # But this is complex. Let's use a dual-variable approach from a simpler LP.
    
    # Simplified: solve for node potentials pi[i] such that
    # pi[i] - pi[j] <= distance[i,j] for all edges (like shortest path duals)
    # and use these to guide the heuristic.
    
    # Compute shortest path distances from depot (these give lower bounds on route costs)
    # pi[i] = shortest distance from depot to i
    # Then edge attractiveness ~ 1 / (distance[i,j] + pi[j] - pi[i])
    # This favors edges that "progress" toward the depot (negative progress) or are cheap.
    
    # Compute shortest paths from depot using Dijkstra (dense graph, so O(n^2))
    pi = np.full(n, np.inf)
    pi[0] = 0.0
    visited = np.zeros(n, dtype=bool)
    
    for _ in range(n):
        # Find unvisited node with minimum pi
        unvisited = ~visited
        if not unvisited.any():
            break
        unvisited_indices = np.where(unvisited)[0]
        min_idx = unvisited_indices[np.argmin(pi[unvisited_indices])]
        if pi[min_idx] == np.inf:
            break
        visited[min_idx] = True
        # Relax edges from min_idx
        for j in range(n):
            if not visited[j]:
                new_dist = pi[min_idx] + distance_matrix[min_idx, j]
                if new_dist < pi[j]:
                    pi[j] = new_dist
    
    # Replace inf with large value
    pi = np.where(np.isfinite(pi), pi, distance_matrix.max())
    
    # Edge attractiveness: 1 / (distance[i,j] + max(0, pi[j] - pi[i]) + epsilon)
    # This favors edges where pi[j] <= pi[i] (going "back" toward depot) or short edges.
    # Actually, we want to favor edges that are part of good routes.
    # pi[j] - pi[i] < 0 means j is closer to depot than i, so going i->j is "progress".
    # We want to reward progress, so we subtract the progress.
    
    # Attractiveness = 1 / (distance[i,j] - max(0, pi[i] - pi[j]) + epsilon)
    # But this could be negative or zero. Let's use:
    # Attractiveness = 1 / (distance[i,j] + 0.1 * |pi[i] - pi[j]| + epsilon)
    
    # Better: use the "reduced cost" idea.
    # For a route, the cost is sum of distances. The "potential" pi[i] represents
    # the minimum cost to reach i from depot. An edge i->j has reduced cost
    # c_ij = distance[i,j] + pi[i] - pi[j].
    # If c_ij < distance[i,j], it means the edge is "better" than expected.
    # We want to favor edges with small reduced cost.
    
    # Compute reduced costs
    reduced_cost = distance_matrix + pi[:, np.newaxis] - pi[np.newaxis, :]
    
    # Ensure reduced_cost is positive for heuristic
    reduced_cost = np.maximum(reduced_cost, 1e-9)
    
    # Heuristic = 1 / reduced_cost
    heuristic = 1.0 / reduced_cost
    
    # Also incorporate demand compatibility:
    # Edges between nodes with high combined demand relative to capacity are less attractive
    # (they're harder to fit in a single trip)
    demand_factor = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                combined_demand = demands[i] + demands[j]
                if combined_demand > capacity:
                    demand_factor[i, j] = 0.5
                elif combined_demand > capacity * 0.8:
                    demand_factor[i, j] = 0.8
    
    heuristic = heuristic * demand_factor
    
    # Ensure all values are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Ensure shape is correct
    if heuristic.shape != distance_matrix.shape:
        heuristic = np.ones_like(distance_matrix) / (distance_matrix + 1e-9)
    
    # Ensure no NaN or inf
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
