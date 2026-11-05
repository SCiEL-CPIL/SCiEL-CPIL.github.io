import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    Monte Carlo estimation of vertex utility.
    
    1. The greedy decoder is deterministic given an ordering.
    2. We want to find an ordering that maximizes the size of the resulting Independent Set.
    3. Static scores (degree, centrality) assume a fixed "value" for each node.
    4. However, the value of picking a node depends on who else is picked (interference).
    5. We approximate the expected contribution of each vertex by simulating the greedy
       process many times with random tie-breaking (or random perturbations of the score).
    
    Algorithm:
    1. Compute a base score (e.g., -degree) to initialize the ordering.
    2. Run T simulations. In each simulation:
       a. Perturb the base scores with random noise (Gaussian).
       b. Run the greedy decoder to get an Independent Set S.
       c. For each vertex v in S, increment a counter `count[v]`.
       d. Optionally, for vertices not in S, we could estimate "regret" or just ignore them.
          Actually, we want the score to reflect "how often this node is selected when
          competing with others". 
    3. The final score for vertex v is proportional to count[v] / T.
       Wait, the decoder picks in descending score order.
       If we assign score = count[v], the decoder will pick the most "frequently selected" nodes first.
       This creates a feedback loop. 
       
    Alternative Interpretation:
    We want to assign a score s[v] such that sorting by s[v] descending yields a good IS.
    Let's estimate the "marginal gain" of including v.
    
    Better Monte Carlo Strategy:
    1. Generate a random permutation (or random scores) to simulate the "environment".
    2. Actually, the standard approach for this "learned ordering" is:
       - We have a fixed decoder.
       - We want to optimize the input scores.
       - We can use the scores as parameters.
       - Let's use a simple heuristic for the initial score: -degree.
       - Then, we simulate the greedy process. 
       
    Let's refine the "Stochastic Game Value" idea:
    Value(v) = E[ 1 if v is in IS ] under random perturbations of the priority.
    
    If we set priority(v) = Value(v), does it improve?
    If v is often in the IS when priorities are random, it means v is "robust" or "low conflict".
    High conflict nodes might be in the IS only if they get a very high random priority.
    Low conflict nodes might be in the IS even with lower priority.
    
    So, high Value(v) -> High Priority.
    This seems counter-intuitive to "Low Degree First" if high degree nodes are also high value?
    No, in ER graphs, high degree nodes block many others. They are only in the IS if they are picked very early.
    Low degree nodes are in the IS more frequently across random orderings because they block fewer nodes.
    Therefore, E[1 if v in IS] should be higher for low-degree nodes.
    Thus, the Monte Carlo estimate should naturally recover a "Low Degree First" like behavior, but with global corrections.
    
    Implementation Details:
    - n = 200.
    - T = 500 simulations?
    - Each simulation:
      - Generate random noise vector z ~ N(0, 1).
      - Score = base_score + lambda * z. (Base score can be 0 or -degree).
      - Sort indices by Score descending.
      - Greedy select:
        - Maintain a boolean array `selected` and `available`.
        - Iterate through sorted indices.
        - If available[i], select it. Mark neighbors as unavailable.
      - Count selections.
    - Final Score = counts / T.
    
    Complexity:
    - Sorting 200 items: ~200 log 200 ~ 1600 ops.
    - Greedy selection: O(n^2) worst case if we check all neighbors?
      With bitsets or boolean arrays:
      - For each selected node, we mark neighbors.
      - Total work per simulation is roughly sum of degrees of selected nodes.
      - Avg degree ~ 10. IS size ~ 100. Work ~ 1000.
      - 500 simulations * 1000 ops = 500,000 ops.
      - NumPy overhead might make this slower. 
      - 150s budget is huge. We can do more simulations.
      - Let's try T=1000.
      
    Optimization:
    - Use NumPy for the greedy step?
      - Sorting is hard to vectorize for the greedy step because the "available" set changes dynamically.
      - Pure Python loop for greedy step might be slow for 1000 sims.
      - n=200. 
      - Python loop:
        for idx in sorted_indices:
            if available[idx]:
                selected.append(idx)
                # mark neighbors
                neighbors = np.where(adj[idx, :])[0]
                available[neighbors] = False
                available[idx] = False
      - `np.where` is fast. 
      - 1000 sims * 200 steps = 200,000 Python iterations.
      - Each iteration has some NumPy calls.
      - 200,000 * ~10us = 2 seconds. Very safe.
      
    Let's add a "base score" to guide the randomness.
    If we use pure random, the estimate is unbiased for the "random ordering" strategy.
    But we know LDF is good.
    Maybe we should simulate perturbations around the LDF score?
    Score = -degree + noise.
    This explores the neighborhood of the LDF solution.
    This is a local search in score space, but using MC to evaluate the objective (IS size) indirectly via counts.
    
    Actually, simpler:
    Just run the MC to get the "average frequency of selection".
    Then use that frequency as the priority.
    
    Let's test if `count[v]` correlates with `-degree`.
    If yes, it's a refined LDF.
    If no, it's something new.
    
    One risk: The greedy decoder is sensitive to ties.
    Random noise breaks ties.
    
    Let's implement.
    """
    start_time = time.time()
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
        
    # Precompute neighbors for faster access?
    # adj is bool matrix. 
    # We can keep it as is.
    
    # Base score: -degree. This biases the MC towards LDF-like solutions.
    deg = adj.sum(axis=1).astype(float)
    base_scores = -deg
    
    # Number of simulations
    # We have ~150s. 
    # Let's aim for 2000 simulations.
    # If it's too slow, we reduce.
    T = 2000
    
    counts = np.zeros(n, dtype=float)
    
    # We need a random number generator.
    # Using np.random.default_rng for better performance and reproducibility if needed (though not required).
    rng = np.random.default_rng(42)
    
    # To speed up the greedy step, we can pre-extract neighbor lists as arrays of indices.
    # This avoids calling np.where inside the loop.
    neighbor_lists = []
    for i in range(n):
        neighbor_lists.append(np.where(adj[i, :])[0])
    
    # Simulation Loop
    for t in range(T):
        # Time check
        if t % 100 == 0 and time.time() - start_time > 140:
            break
            
        # Generate perturbed scores
        # Noise scale: 
        # Degrees are ~10. 
        # If noise is too small, we always pick the same order (LDF).
        # If noise is too large, we pick random order.
        # We want to explore "nearby" good orderings.
        # Standard deviation of degree is ~3. 
        # Let's use sigma = 2.0.
        noise = rng.normal(0, 2.0, n)
        scores = base_scores + noise
        
        # Sort indices by descending score
        # np.argsort returns indices that would sort the array.
        # We want descending, so we reverse or use -scores.
        order = np.argsort(-scores)
        
        # Greedy Selection
        available = np.ones(n, dtype=bool)
        
        # We need to track which nodes were selected to increment counts.
        # Instead of building a list, we can just increment counts directly?
        # Yes, but we need to ensure we only increment if selected.
        
        for idx in order:
            if not available[idx]:
                continue
                
            # Select idx
            available[idx] = False
            counts[idx] += 1.0
            
            # Mark neighbors as unavailable
            # neighbors = neighbor_lists[idx]
            # available[neighbors] = False
            
            # Vectorized update
            # This is faster than a loop over neighbors
            available[neighbor_lists[idx]] = False
            
    # The score for the final decoder should be the estimated probability of selection.
    # Higher probability -> Higher priority.
    final_scores = counts / T
    
    # Safety check
    if time.time() - start_time > 145:
        # Fallback to base scores if MC was interrupted or something went wrong
        return base_scores
        
    return final_scores
