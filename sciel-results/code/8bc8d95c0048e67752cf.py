"""Geometric convex hull peeling heuristic for TSP construction."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unvisited = np.asarray(unvisited_nodes)
    if unvisited.size == 0:
        return int(current_node)
    
    # Safety check
    if time.time() > _DEADLINE:
        return int(unvisited[int(np.argmin(distance_matrix[current_node, unvisited]))])
    
    n = int(distance_matrix.shape[0])
    d = distance_matrix
    
    # Extract coordinates if possible, otherwise fall back to distance-based heuristic
    # Since we only have distance_matrix, we can't directly compute areas.
    # However, the prompt implies we might have access to points or we must infer geometry.
    # The contract says "distance_matrix" is provided. It does NOT provide points.
    # Therefore, we cannot compute the actual geometric area of the triangle (current, dest, candidate)
    # unless we reconstruct coordinates or use a proxy.
    
    # Wait, the inspiration says "geometric convex hull peeling... area of the triangle".
    # If we only have the distance matrix, we cannot compute the area directly.
    # However, for a Euclidean TSP, the distance matrix is derived from points.
    # Can we recover points? No, not uniquely without SVD/embedding which is expensive and noisy.
    
    # Let's re-read the task: "scored by mean closed-tour length on eight seeded uniform 50-city instances".
    # "generated Euclidean instances".
    # The function signature only receives `distance_matrix`.
    # Is there a way to get points? No.
    
    # Alternative interpretation: The "area" penalty can be approximated or replaced by a different geometric proxy.
    # Or, perhaps the "strategy" description is aspirational and I should implement a proxy for "non-crossing perimeter".
    # A common proxy for "keeping the tour tight/non-crossing" in a constructive heuristic is to look at the 
    # "regret" or the "angle" at the current node.
    
    # If we cannot compute area, we can use a "directional" heuristic.
    # Move that maintains the direction towards the destination but avoids "wasting" distance.
    
    # Let's try a different structural change that is computable with just the distance matrix:
    # "Lookahead" or "Regret" based selection.
    # But the directive specifically asks for: "minimizing the distance to the current node plus a penalty proportional to the area of the triangle..."
    
    # If I strictly follow the directive, I need area.
    # Can I estimate area from distances?
    # For a triangle with sides a, b, c, Area = 0.25 * sqrt((a+b+c)(-a+b+c)(a-b+c)(a+b-c)).
    # Here, the triangle is (current_node, destination_node, candidate).
    # Sides:
    # 1. dist(current, candidate) -> d[current, candidate]
    # 2. dist(current, destination) -> d[current, destination] (Constant for all candidates in this step)
    # 3. dist(destination, candidate) -> d[destination, candidate]
    
    # So YES, we can compute the area of the triangle formed by (current, destination, candidate) using Heron's formula on the distances!
    # This is a brilliant insight. The geometry is encoded in the distances.
    
    cur = int(current_node)
    dest = int(destination_node)
    
    # Get distances for the triangle (cur, dest, cand)
    # Side 1: d[cur, dest] (Constant)
    side_a = d[cur, dest]
    
    # Side 2: d[cur, cand] (Variable)
    side_b = d[cur, unvisited]
    
    # Side 3: d[dest, cand] (Variable)
    side_c = d[dest, unvisited]
    
    # Calculate Area for each candidate using Heron's Formula
    # s = (a + b + c) / 2
    # Area = sqrt(s * (s-a) * (s-b) * (s-c))
    
    # Vectorized calculation
    # a is scalar, b and c are vectors
    s = (side_a + side_b + side_c) / 2.0
    term1 = s
    term2 = s - side_a
    term3 = s - side_b
    term4 = s - side_c
    
    # Ensure non-negative due to floating point errors
    area_sq = term1 * term2 * term3 * term4
    area_sq = np.maximum(area_sq, 0.0)
    areas = np.sqrt(area_sq)
    
    # The strategy: Minimize distance + penalty * area
    # "minimizing the distance to the current node plus a penalty proportional to the area"
    # Cost = d[cur, cand] + lambda * Area
    
    # What is a good lambda?
    # Distances are O(1) for unit square. Areas are O(0.1-0.25).
    # A reasonable lambda might be around 1.0 to 5.0.
    # Let's start with lambda = 2.0 as a structural guess.
    # The current program uses a blend of greedy (dist to cur) and destination (dist to dest).
    # This new approach uses dist to cur + area penalty.
    
    # Let's try a few values of lambda if we can, or just pick a robust one.
    # Given the "structural change" directive, I will implement the mechanism with a fixed, well-reasoned lambda.
    # For 50 cities in [0,1]^2, average edge length is ~0.3-0.4. Average triangle area is ~0.05-0.1.
    # If lambda=1, area penalty is small compared to distance.
    # If lambda=5, area penalty is significant.
    # Let's use lambda = 3.0.
    
    lambda_penalty = 3.0
    costs = side_b + lambda_penalty * areas
    
    # Handle potential numerical issues
    costs = np.where(np.isnan(costs), np.inf, costs)
    
    idx = int(np.argmin(costs))
    return int(unvisited[idx])
# EVOLVE-BLOCK-END
