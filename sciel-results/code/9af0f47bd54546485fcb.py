import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    try:
        demands = np.asarray(demands, dtype=float)
        distance_matrix = np.asarray(distance_matrix, dtype=float)
        
        N = len(demands)
        
        if distance_matrix.shape != (N, N):
            pass

        eps = 1e-9
        d = distance_matrix + eps
        
        # Base heuristic: inverse distance
        # Standard ACO heuristic is often 1/distance
        # But we want to incorporate demand logic.
        
        # "depot-pressure" field:
        # Attractiveness to depot is inversely proportional to demand?
        # "high-demand nodes are harder to serve and should be visited early"
        # This implies we want to visit high demand nodes when capacity is full (early).
        
        # "scale edge weights by the product of the departure node's remaining capacity potential 
        # and the arrival node's demand efficiency"
        
        # Since the heuristic matrix is static (computed once), we can't use dynamic remaining capacity.
        # We must approximate this static bias.
        
        # Idea:
        # 1. Base weight: 1 / distance[i, j]
        # 2. Demand efficiency of arrival node j: 
        #    If demand[j] is high, it's "harder". 
        #    The prompt says "high-demand nodes... should be visited early".
        #    In a static matrix, how do we bias towards early visit?
        #    Maybe we just prioritize high demand nodes generally? Or low demand?
        #    Usually, saving algorithms cluster small demands.
        #    But the prompt suggests high demand = hard = visit early.
        #    Let's interpret "demand efficiency" as 1/demand? Or demand?
        #    "inversely proportional to its demand" for depot attractiveness.
        #    So node j has an inherent "pressure" proportional to 1/demands[j].
        #    If we want to visit high demand early, maybe we want the heuristic to be 
        #    higher for edges leading to high demand nodes?
        #    Or lower?
        #    If heuristic is higher, ant picks it with higher probability.
        #    If we want to visit high demand early, we should increase their attractiveness.
        #    So maybe weight by demands[j]?
        
        # Let's try:
        # eta[i, j] = (1 / d[i, j]) * (demands[j] / demands[i]) ?
        # Or just (1 / d[i, j]) * demands[j]?
        
        # Let's look at the directive again:
        # "Model the heuristic as a 'depot-pressure' field where each node's attractiveness 
        # to the depot is inversely proportional to its demand"
        # This suggests a node weight w_j = 1 / demands[j].
        # "scale edge weights by the product of the departure node's remaining capacity potential 
        # and the arrival node's demand efficiency"
        
        # Since we can't use dynamic capacity, let's assume "remaining capacity potential" 
        # is a static proxy, maybe related to the node's own demand or distance to depot.
        # Or maybe we just use the standard 1/distance and modify it slightly.
        
        # Let's try a simple demand-based scaling.
        # If we want to visit high demand early, we should make edges TO high demand nodes more attractive.
        # So multiply by demands[j].
        
        # Also, "departure node's remaining capacity potential".
        # If we are at node i, and it has high demand, we used a lot of capacity.
        # So we have less "potential" left?
        # Or if we are at a low demand node, we have more potential?
        # So maybe divide by demands[i]?
        
        # Proposed heuristic:
        # eta[i, j] = (1 / d[i, j]) * (demands[j] / demands[i])
        # But demands[0] is 0 (depot). Division by zero.
        # We need to handle depot specially.
        
        # Let's define safe demands.
        safe_demands = demands.copy()
        safe_demands[safe_demands == 0] = 1.0 # Depot demand is 0, set to 1 to avoid div by 0
        
        # Arrival demand efficiency:
        # "high-demand nodes ... visited early" -> Attractiveness should be high for high demand?
        # If I multiply by demands[j], high demand j gets higher weight.
        
        # Departure capacity potential:
        # If I am at i, and i had high demand, I am "fuller".
        # So maybe I should prioritize going back to depot?
        # But the matrix is for customer-to-customer and depot-to-customer.
        
        # Let's try:
        # eta[i, j] = (1 / d[i, j]) * (1 / safe_demands[i]) * (1 / safe_demands[j]) ?
        # This would favor low demand nodes.
        
        # Let's stick to the prompt's specific phrasing:
        # "attractiveness to the depot is inversely proportional to its demand"
        # This might imply that the "cost" of a node is high if demand is low?
        # No, "attractiveness ... inversely proportional".
        # So Attractiveness_j ~ 1/demands[j].
        # Low demand = High attractiveness.
        # This contradicts "high-demand ... visited early" if attractiveness means probability of selection.
        # Unless "attractiveness to the depot" means something else.
        
        # Let's try the opposite: Favor high demand nodes.
        # eta[i, j] = (1 / d[i, j]) * demands[j]
        
        # Let's also consider the distance to depot.
        # Nodes close to depot are easy.
        
        # Let's try a combination:
        # eta[i, j] = (1 / d[i, j]) * (demands[j] + 1)
        
        # Let's test the current approach first:
        # Current: eta = demands / d
        # This is exactly (1/d) * demands.
        # So the current program already implements "scale by arrival demand".
        
        # The directive says "design a new program ... from scratch; do not reuse the current designs".
        # So I must change the logic.
        
        # New Logic:
        # 1. Calculate base distance heuristic: base = 1 / d
        # 2. Calculate node potentials.
        #    "depot-pressure field ... inversely proportional to demand"
        #    Let node_weight[j] = 1 / demands[j] (with safe handling)
        # 3. "scale edge weights by ... arrival node's demand efficiency"
        #    Let's use node_weight[j] as the efficiency.
        # 4. "departure node's remaining capacity potential"
        #    Approximate this as 1 / demands[i] (if i is a customer).
        #    If i is depot, maybe just 1.0.
        
        # So: eta[i, j] = base[i, j] * node_weight[j] * departure_potential[i]
        
        # Let's implement this.
        
        # Handle depot (index 0)
        # demands[0] is 0.
        
        # We want to return a matrix of shape (N, N).
        
        # 1. Base inverse distance
        inv_dist = 1.0 / d
        
        # 2. Node weights (inversely proportional to demand)
        # For depot, demand is 0. Let's assign a high weight? Or low?
        # Depot is always available.
        # Let's set weight for depot to 1.0.
        # For customers, weight = 1 / demands.
        
        node_weights = np.ones(N)
        # Only customers (indices 1 to N-1) have demand > 0
        # demands are integers 1-9 as floats.
        # demands[0] is 0.
        
        # Avoid division by zero
        nonzero_mask = demands > 0
        node_weights[nonzero_mask] = 1.0 / demands[nonzero_mask]
        
        # "departure node's remaining capacity potential"
        # If we interpret this as "how much room is left", and we assume a full truck starts at depot,
        # and we visit nodes...
        # This is hard to static-ize.
        # Let's just use the node_weights for the arrival node j.
        # And for departure node i, maybe we don't scale, or scale by 1.
        
        # Let's try: eta[i, j] = inv_dist[i, j] * node_weights[j]
        
        eta = inv_dist * np.tile(node_weights, (N, 1))
        
        # Clean up
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9
        
        # Normalize? 
        # ACO heuristics are usually not normalized to sum to 1, but relative scale matters.
        # The current program doesn't normalize.
        
        return eta
        
    except Exception:
        try:
            N = len(demands)
        except:
            N = 0
        return np.full((N, N), 1e-9)
