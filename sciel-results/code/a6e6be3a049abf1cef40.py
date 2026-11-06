import numpy as np
import time
import random

def priority(instance):
    """
    Returns a priority score for each activity to be used in a Serial SGS.
    
    Strategy: 
    This approach treats the scheduling problem as finding an optimal topological ordering 
    of activities. It uses Simulated Annealing to explore the space of valid permutations.
    The energy function is the makespan produced by the Serial Schedule Generation Scheme (SGS) 
    when activities are processed in that order.
    
    The returned priority vector is the inverse rank of the best found topological order.
    This ensures that the static priority list exactly replicates the best found ordering 
    in the SGS, effectively converting a combinatorial search over orders into the required 
    static score interface.
    """
    start_time = time.time()
    # Safety margin: 145s out of 150s limit
    deadline = start_time + 145.0

    # 1. Extract and process instance data
    N = len(instance["durations"])
    durations = np.array(instance["durations"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)  # Shape (N, R)
    capacities = np.array(instance["capacities"], dtype=float) # Shape (R,)
    R = len(capacities)
    
    # Handle predecessors
    preds = instance.get("predecessors", [[] for _ in range(N)])
    if isinstance(preds, np.ndarray):
        if preds.ndim == 2:
             preds = [np.where(preds[i] > 0)[0].tolist() for i in range(N)]
        else:
             preds = [[] for _ in range(N)]
    elif isinstance(preds, list):
         normalized_preds = []
         for p in preds:
             if isinstance(p, (list, tuple, np.ndarray)):
                 normalized_preds.append(list(map(int, p)))
             else:
                 normalized_preds.append([])
         preds = normalized_preds
    else:
         preds = [[] for _ in range(N)]

    # Build successors
    succs = [[] for _ in range(N)]
    for i in range(N):
        for p in preds[i]:
            succs[p].append(i)
            
    # 2. Define Fast SGS Simulation
    # This simulator takes a priority vector and returns the makespan.
    # It is optimized for speed as it will be called thousands of times.
    
    def simulate_sgs(priorities):
        scheduled = [False] * N
        # active: list of (finish_time, idx)
        active = []
        remaining_res = capacities.copy()
        time_current = 0.0
        makespan = 0.0
        
        # Pre-compute predecessor lists for faster access? 
        # preds is already list of lists.
        
        while True:
            # Find eligible
            eligible = []
            for i in range(N):
                if not scheduled[i]:
                    is_eligible = True
                    for p in preds[i]:
                        if not scheduled[p]:
                            is_eligible = False
                            break
                    if is_eligible:
                        eligible.append(i)
                        
            if not eligible:
                # Wait for next event
                if not active:
                    break # Should not happen if valid
                min_finish = min(f for f, _ in active)
                time_current = min_finish
                new_active = []
                for f, idx in active:
                    if f > time_current:
                        new_active.append((f, idx))
                    else:
                        remaining_res += demands[idx]
                        makespan = max(makespan, f)
                active = new_active
                continue
                
            # Sort eligible by priority (descending)
            eligible.sort(key=lambda x: priorities[x], reverse=True)
            
            # Attempt to schedule
            any_scheduled = False
            for i in eligible:
                # Check resources
                if np.all(remaining_res >= demands[i]):
                    scheduled[i] = True
                    remaining_res -= demands[i]
                    finish_t = time_current + durations[i]
                    active.append((finish_t, i))
                    makespan = max(makespan, finish_t)
                    any_scheduled = True
                    
            if not any_scheduled:
                # Wait
                if not active:
                    break
                min_finish = min(f for f, _ in active)
                time_current = min_finish
                new_active = []
                for f, idx in active:
                    if f > time_current:
                        new_active.append((f, idx))
                    else:
                        remaining_res += demands[idx]
                        makespan = max(makespan, f)
                active = new_active
                
        return makespan

    # 3. Generate Initial Topological Order
    # Use a greedy heuristic to start: Critical Path Length (CPL)
    in_degree = [len(p) for p in preds]
    queue = [i for i in range(N) if in_degree[i] == 0]
    topo_order = []
    indeg_temp = in_degree[:]
    q = queue[:]
    
    # Compute CPL for initial ordering
    cpl = np.zeros(N)
    # First, get a standard topo order to compute CPL backwards
    temp_topo = []
    temp_indeg = in_degree[:]
    temp_q = queue[:]
    while temp_q:
        u = temp_q.pop(0)
        temp_topo.append(u)
        for v in succs[u]:
            temp_indeg[v] -= 1
            if temp_indeg[v] == 0:
                temp_q.append(v)
                
    for i in range(len(temp_topo) - 1, -1, -1):
        u = temp_topo[i]
        max_succ_cpl = 0
        for v in succs[u]:
            if cpl[v] > max_succ_cpl:
                max_succ_cpl = cpl[v]
        cpl[u] = durations[u] + max_succ_cpl

    # Initial order: Sort by CPL descending, breaking ties by index
    # We need to generate a valid topological order.
    # We can use Kahn's algorithm but pick the node with highest CPL among available.
    in_deg = in_degree[:]
    available = [i for i in range(N) if in_deg[i] == 0]
    current_order = []
    
    # For speed, we'll maintain a list of available nodes.
    # Since N=30, sorting the available list each step is cheap.
    while available:
        # Pick node with highest CPL
        available.sort(key=lambda x: cpl[x], reverse=True)
        node = available.pop(0)
        current_order.append(node)
        for v in succs[node]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                available.append(v)
                
    # Convert order to priority vector
    # Higher priority = earlier in order.
    # priority[i] = N - rank(i)
    def order_to_priority(order):
        pri = np.zeros(N)
        for rank, node in enumerate(order):
            pri[node] = N - rank
        return pri

    current_priority = order_to_priority(current_order)
    current_makespan = simulate_sgs(current_priority)
    
    best_priority = current_priority.copy()
    best_makespan = current_makespan
    best_order = current_order.copy()

    # 4. Simulated Annealing
    # Move: Swap two adjacent elements in the topological order? 
    # No, swapping adjacent elements might violate precedence if the later element is a predecessor of the earlier one (which is impossible in a topo order, but swapping i and i+1 where i+1 depends on i is invalid).
    # Actually, in a topological order, if i is before j, j cannot be a predecessor of i. 
    # So swapping any two elements i, j in the list results in a new list. 
    # Is it a valid topological order? 
    # If we swap positions of A and B in a valid topo order:
    # If B depends on A (A is pred of B), then A must come before B. Swapping puts B before A -> Invalid.
    # If A depends on B, B must come before A. Swapping puts A before B -> Invalid.
    # If no dependency, swapping is valid.
    
    # So, a valid move is swapping two nodes that have NO precedence relationship between them.
    # To find such pairs efficiently:
    # Precompute a matrix or set of "incomparable" pairs? 
    # Or, simply try a random swap and check validity? 
    # Checking validity of a full order is O(N^2) or O(N+M). 
    # N=30, M is small. Checking validity is fast.
    
    def is_valid_topo(order):
        pos = {node: idx for idx, node in enumerate(order)}
        for i in range(N):
            for p in preds[i]:
                if pos[p] > pos[i]:
                    return False
        return True

    # Optimization: Instead of checking full validity, we can just try to swap and see if SGS works?
    # No, SGS assumes valid precedence. If we give it an invalid order, it might schedule a node before its predecessor, which is incorrect for the "Serial SGS" definition (which strictly respects precedence by only scheduling eligible nodes).
    # Wait, the SGS *itself* handles precedence by checking eligibility. 
    # The "order" in my SA is just the *priority ranking*. 
    # The SGS will sort eligible nodes by this ranking. 
    # If the ranking is "invalid" (i.e., implies a cycle or wrong order for dependencies), the SGS still works correctly because it *only* considers eligible nodes (those with all preds scheduled). 
    # So, ANY permutation is a valid input to the SGS. The SGS will ignore the relative order of non-eligible nodes. 
    # Therefore, we can swap ANY two nodes in the priority vector. We don't need to maintain a valid topological order explicitly; the SGS enforces the constraints.
    # This simplifies the move generation significantly: just swap priorities of any two nodes.
    
    # However, to be more effective, we should probably focus on swaps that might change the SGS behavior.
    # Swapping two nodes that are never eligible at the same time is useless.
    # But detecting that is hard. Let's just do random swaps.
    
    T = 100.0 # Initial temperature
    cooling_rate = 0.99
    min_T = 0.1
    
    # Determine number of iterations based on time
    # Let's do a fixed number of iterations that fit in the time budget, or loop until deadline.
    
    random.seed(42)
    
    # To make the search more effective, we can use a "reheat" strategy or just run until time runs out.
    # Given N=30, the search space is huge. SA might get stuck.
    # Alternative: Use the SGS makespan as a guide.
    
    # Let's try a different move: "Insertion" or "Shift".
    # But simple swaps are robust.
    
    # To improve convergence, we can use a "steepest descent" style acceptance or just standard SA.
    
    while time.time() < deadline:
        if T < min_T:
            # Reheat or stop? Let's reheat slightly to escape local minima
            T = min_T * 2
            # Or just break if we've been stuck?
            # Let's just keep going with min_T for now, or increase cooling rate.
            cooling_rate = 0.995
            
        i, j = random.sample(range(N), 2)
        
        # Swap priorities
        temp_pri = current_priority.copy()
        temp_pri[i], temp_pri[j] = temp_pri[j], temp_pri[i]
        
        new_makespan = simulate_sgs(temp_pri)
        
        delta = new_makespan - current_makespan
        
        if delta < 0:
            # Improvement
            current_priority = temp_pri
            current_makespan = new_makespan
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_priority = current_priority.copy()
        else:
            # Accept with probability exp(-delta / T)
            prob = np.exp(-delta / T)
            if random.random() < prob:
                current_priority = temp_pri
                current_makespan = new_makespan
        
        T *= cooling_rate
        
    return best_priority

