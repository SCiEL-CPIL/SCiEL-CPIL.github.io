import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Local Search Optimization of Vertex Ordering
    
    Instead of using static structural features (like degree or core number) to assign scores,
    this strategy treats the assignment of scores as an optimization problem. 
    We define a surrogate objective function that approximates the size of the independent set
    produced by the greedy decoder. By using a smooth, differentiable surrogate (based on soft-min/soft-max
    or simply simulating the greedy process with a continuous relaxation of the selection order),
    we can perform gradient-based or local search to find an ordering that maximizes the expected IS size.
    
    However, since the greedy decoder is discrete and non-differentiable, a more practical "local search"
    approach within the family of "optimizing the ordering" is to:
    1. Start with a good initial ordering (e.g., LDF: Low Degree First).
    2. Simulate the greedy decoder to get the IS size.
    3. Identify "regret" or "conflict" points where a different ordering might have yielded a larger set.
    4. Adjust scores to prioritize vertices that were "blocked" but had high potential, or deprioritize
       vertices that blocked many others.
       
    A simpler and more robust method within this family is **Hill Climbing on the Score Vector**:
    - Initialize scores with a strong heuristic (e.g., -degree).
    - Simulate greedy.
    - For each vertex, try swapping its position with a neighbor or a random vertex, or adjusting its score slightly.
    - If the simulated IS size improves, keep the change.
    
    Given the 120s budget and N=200, we can afford a limited number of iterations.
    But simulating greedy is O(N^2) in the worst case (or O(N+E) with efficient data structures).
    With N=200, N^2 = 40,000. Even 1000 iterations would be 40M ops, which is fast in Python if vectorized,
    but pure Python loops might be slow.
    
    Let's use a vectorized simulation of the greedy process.
    
    Algorithm:
    1. Init scores: score = -degree (LDF).
    2. Repeat for T iterations:
       a. Sort vertices by score descending.
       b. Simulate greedy selection:
          - Maintain a boolean mask `selected` (False initially) and `blocked` (False initially).
          - Iterate through sorted vertices.
          - If vertex v is not selected and not blocked:
            - Select v (selected[v] = True)
            - Block neighbors (blocked[neighbors] = True)
       c. Compute IS size = sum(selected).
       d. If IS size improves, keep the new scores.
       e. Propose a new set of scores. How?
          - We can use a "regret-based" update.
          - Vertices that were blocked early but had low degree might have been better picked earlier.
          - Vertices that were picked but blocked many high-potential neighbors might be "bad" picks.
          
    Alternative simpler strategy within this family: **Simulated Annealing on the Ordering**.
    - Represent the ordering as a permutation.
    - Objective: IS size.
    - Moves: Swap two elements in the permutation.
    - Accept if better or with probability exp(-delta/T).
    
    This is a standard local search. Let's implement this.
    
    Steps:
    1. Initial permutation: sort by -degree (LDF).
    2. Current IS size = simulate(permutation).
    3. Loop while time < deadline:
       - Pick two random indices i, j.
       - Swap perm[i], perm[j].
       - New IS size = simulate(new_perm).
       - If new > current: accept, update current.
       - Else: accept with probability exp((new - current) / temperature).
       - Cool down temperature.
    
    Simulation cost: O(N^2) per swap.
    N=200.
    150s budget.
    Python loop overhead is significant.
    Let's estimate: 1 simulation ~ 200*200 = 40k ops.
    If we can do ~1000 simulations, that's 40M ops.
    In Python, 40M simple ops might take ~2-5 seconds.
    So we can do many more iterations.
    
    Let's implement efficient simulation.
    
    def simulate(perm, adj):
        n = len(perm)
        selected = np.zeros(n, dtype=bool)
        blocked = np.zeros(n, dtype=bool)
        count = 0
        for v in perm:
            if not selected[v] and not blocked[v]:
                selected[v] = True
                count += 1
                # Block neighbors
                # adj[v] is a boolean array
                blocked[adj[v]] = True
        return count
    
    This loop is O(N * E_v) where E_v is avg degree. Total O(N*E) ~ O(N^2) for dense, O(N) for sparse.
    For ER graphs with p=0.5, E ~ N^2/4. So O(N^2).
    
    To speed up, we can use numpy for the blocking step.
    
    Let's refine the local search to be more effective than random swaps.
    **Greedy Local Search (2-opt style)**:
    - Try swapping each pair (i, j) where i < j.
    - Keep the best swap.
    - Repeat until no improvement.
    
    This is O(N^2) swaps * O(N^2) sim = O(N^4) per iteration. Too slow.
    
    **First Improvement Local Search**:
    - Iterate i from 0 to N-1.
    - For each i, try swapping with j > i.
    - If improvement, swap and restart from i=0 (or i=0 to N).
    
    Still potentially slow if many restarts.
    
    **Random Restart Hill Climbing**:
    - Run hill climbing with random swaps for a fixed number of steps.
    - Keep the best solution found.
    
    Let's go with Random Restart Hill Climbing with a time guard.
    
    1. Init perm with LDF.
    2. best_perm = perm, best_size = simulate(perm).
    3. While time < deadline:
       - perm = best_perm.copy()
       - current_size = best_size
       - for step in range(1000):
           - i, j = random indices
           - swap perm[i], perm[j]
           - new_size = simulate(perm)
           - if new_size > current_size:
               - current_size = new_size
               - best_perm = perm.copy()
               - best_size = current_size
           - else:
               - swap back
       - (Optionally, perturb best_perm to start next restart)
    
    This is a solid local search strategy.
    
    Let's add a "smart" initial perturbation:
    - Instead of random swaps, try swapping a vertex that was "blocked" with a vertex that was "selected".
    - This targets the specific failure mode of the greedy decoder.
    
    Implementation details:
    - Use `time.time()` for deadline.
    - Deadline = 110s (leave 10s margin).
    - Use `np.random.default_rng()` for reproducibility if needed, but random is fine.
    
    Let's write the code.
    
    Note: The task asks for `priority(adjacency)` returning scores.
    The local search optimizes the *ordering*.
    The scores can be the inverse of the rank in the best ordering found.
    score[v] = -rank[v]
    This ensures the greedy decoder picks in the order of the best permutation.
    
    Wait, the greedy decoder picks by descending score.
    If score[v] = -rank[v], and rank 0 is first, then score 0 is highest.
    So score[v] = -rank[v] works.
    
    Let's implement.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # 1. Initial Ordering: LDF (Low Degree First)
    deg = adj.sum(axis=1)
    init_order = np.argsort(deg)  # Ascending degree -> Low degree first
    # The decoder picks highest score first.
    # If we want Low Degree First, the low degree vertices should have high scores.
    # So score[v] should be high for low deg.
    # In our permutation `perm`, `perm[0]` is the first picked.
    # So `perm[0]` should be the lowest degree vertex.
    # `init_order` is sorted by degree ascending. So `init_order[0]` is lowest degree.
    # This is the correct order for the decoder.
    
    perm = init_order.copy()
    
    # Helper: Simulate greedy decoder given a permutation (order of selection)
    def simulate(order):
        selected = np.zeros(n, dtype=bool)
        blocked = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v] and not blocked[v]:
                selected[v] = True
                count += 1
                blocked[adj[v]] = True
        return count
    
    best_perm = perm.copy()
    best_size = simulate(best_perm)
    
    rng = np.random.default_rng(42)
    
    # Local Search: Random Restart Hill Climbing
    while time.time() < deadline:
        current_perm = best_perm.copy()
        current_size = best_size
        
        # Perform a number of swaps
        # We can do a fixed number of steps per restart, or until no improvement for a while.
        # Let's do 500 steps per restart.
        no_improve_count = 0
        max_no_improve = 50
        
        for step in range(500):
            if time.time() > deadline:
                break
                
            # Pick two random indices to swap
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                continue
                
            # Swap
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            
            # Evaluate
            new_size = simulate(current_perm)
            
            if new_size > current_size:
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_perm = current_perm.copy()
                no_improve_count = 0
            else:
                # Swap back
                current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
    
    # Convert best permutation to scores
    # The decoder picks in descending score order.
    # We want the decoder to pick in the order of `best_perm`.
    # So the vertex at `best_perm[0]` should have the highest score.
    # The vertex at `best_perm[1]` should have the second highest score.
    # ...
    # The vertex at `best_perm[n-1]` should have the lowest score.
    
    scores = np.zeros(n)
    for rank, v in enumerate(best_perm):
        scores[v] = -rank  # Higher score for lower rank (earlier in permutation)
        
    return scores
