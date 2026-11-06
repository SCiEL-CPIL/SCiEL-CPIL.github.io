import numpy as np
import time
import random

def priority(instance):
    """
    Returns a priority score for each activity to be used in a Schedule Generation Scheme.
    Strategy: List Schedule Generation Scheme (LSGS) with 2-opt local search on the priority list.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin below 150s limit
    
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
            
    # Compute Topological Order
    in_degree = [len(p) for p in preds]
    queue = [i for i in range(N) if in_degree[i] == 0]
    topo_order = []
    indeg_temp = in_degree[:]
    q = queue[:]
    while q:
        u = q.pop(0)
        topo_order.append(u)
        for v in succs[u]:
            indeg_temp[v] -= 1
            if indeg_temp[v] == 0:
                q.append(v)
                
    # Compute Critical Path Length (CPL) backwards
    cpl = np.zeros(N)
    for i in range(len(topo_order) - 1, -1, -1):
        u = topo_order[i]
        max_succ_cpl = 0
        for v in succs[u]:
            if cpl[v] > max_succ_cpl:
                max_succ_cpl = cpl[v]
        cpl[u] = durations[u] + max_succ_cpl
        
    # Compute Successor Count (Total descendants)
    succ_count = np.zeros(N, dtype=int)
    for i in range(N):
        visited = set()
        stack = [i]
        while stack:
            node = stack.pop()
            for s in succs[node]:
                if s not in visited:
                    visited.add(s)
                    stack.append(s)
        succ_count[i] = len(visited)

    # 2. Define LSGS Simulation Function
    def simulate_lsgs(priority_list):
        """
        Simulates List Schedule Generation Scheme.
        priority_list: A list of activity indices representing the priority order.
        Returns makespan.
        """
        # State
        start_times = [-1] * N
        finish_times = [-1] * N
        scheduled = [False] * N
        remaining_res = capacities.copy()
        
        # We need to track active activities to know when resources are released.
        # Active: list of (finish_time, activity_index)
        active = []
        current_time = 0.0
        
        # Index into priority_list to pick next activity
        # Actually, LSGS works by iterating through the priority list.
        # For each activity in the list, if it's eligible (preds done), schedule at earliest feasible time.
        
        # However, the "earliest feasible time" depends on resources held by *other* activities 
        # that might be scheduled *later* in the list but start *earlier*?
        # No, in LSGS, we process the list sequentially. 
        # Wait, standard LSGS:
        # 1. Determine priority list (permutation).
        # 2. For each activity in the list:
        #    a. Determine earliest start time based on predecessors.
        #    b. Check if resources are available at that time.
        #    c. If not, wait until resources are available.
        #    d. Schedule activity.
        
        # This implies we must know the resource profile of the entire schedule? 
        # No, we build it incrementally. 
        # But if we schedule Activity A at t=10, and later in the list we have Activity B 
        # which could have started at t=5 but is behind A in the list, B is forced to start later?
        # Yes, that is the nature of LSGS. It is a heuristic.
        
        # Implementation detail: 
        # We need to efficiently find the earliest time t >= ES_i such that resources are sufficient.
        
        # To do this, we maintain a list of "resource events" or just scan through active tasks?
        # Since N=30, scanning is fine.
        
        for i in priority_list:
            if scheduled[i]:
                continue
                
            # Check if predecessors are scheduled
            # In LSGS, we usually only consider an activity eligible if its predecessors are *completed*.
            # But since we are building the list sequentially, predecessors *must* appear earlier in the list?
            # Not necessarily. If the priority list is not topologically sorted, we might encounter a node before its parent.
            # Standard LSGS assumes the priority list is a valid topological order or handles ineligible nodes by skipping them?
            # Actually, if we skip, we might never schedule it if we don't re-visit.
            # The task says "serial schedule-generation scheme starts the highest-priority eligible activity".
            # This implies the *input* to the SGS is a priority *score*, and the SGS picks the best eligible one at each step.
            # My `priority` function returns scores. The evaluator runs the SGS.
            # BUT, the task says "Write a Python priority(instance) function that scores each activity; a serial schedule-generation scheme starts the highest-priority eligible activity".
            # This means *I* do not run the SGS. *The evaluator* runs the SGS using my scores.
            # HOWEVER, the current program simulates the SGS to optimize the scores.
            # So I need a simulation that *exactly* mimics the evaluator's SGS to optimize my scores.
            
            # The evaluator's SGS:
            # 1. Time t = 0.
            # 2. Find eligible activities (unscheduled, preds scheduled).
            # 3. Pick the one with highest priority score.
            # 4. If it fits resources, schedule at t.
            # 5. If it doesn't fit, does it wait? Or does it pick the next best?
            # "starts the highest-priority eligible activity under ... limits"
            # Usually, if the highest priority one doesn't fit, the SGS might:
            # a) Wait until it fits (blocking others).
            # b) Skip it and try the next highest priority one that fits.
            # Standard Serial SGS (Klein): "Select the eligible activity with the highest priority that can be scheduled without violating resource constraints."
            # This implies (b). It picks the *first* one in the priority order that *fits*.
            
            # So my simulation `simulate_sgs_correct` in the current code is likely correct in logic:
            # Sort eligible by priority. Loop through them. Schedule if fits.
            
            # Why switch to LSGS?
            # LSGS is different. LSGS fixes the order. SGS re-evaluates at each time step.
            # If the evaluator uses SGS, I MUST simulate SGS to optimize correctly.
            # If I simulate LSGS, I might optimize for a different schedule than the evaluator produces.
            # Therefore, I MUST stick to simulating the SGS.
            
            # The structural change should be in the *Search* and *Heuristics*, not the simulation model, 
            # unless I am confident the evaluator uses LSGS. 
            # The prompt says "serial schedule-generation scheme". This is the standard name for SGS.
            # So I will stick to SGS simulation.
            
            pass # Dummy to prevent execution of incorrect LSGS logic above
            
        # Since I must simulate SGS, let's refine the SGS simulation for speed and correctness.
        # The current SGS simulation is O(N^2) per step because it scans all N activities to find eligible ones.
        # With N=30, this is fine.
        
        # Let's rewrite the simulation to be as fast as possible.
        
        scheduled = [False] * N
        active = []  # (finish_time, index)
        remaining_res = capacities.copy()
        t = 0.0
        makespan = 0.0
        
        while True:
            if all(scheduled):
                break
                
            # Find eligible
            eligible = []
            for i in range(N):
                if not scheduled[i]:
                    if all(scheduled[p] for p in preds[i]):
                        eligible.append(i)
                        
            if not eligible:
                if not active:
                    break
                min_f = min(f for f, _ in active)
                t = min_f
                new_active = []
                for f, idx in active:
                    if f > t:
                        new_active.append((f, idx))
                    else:
                        remaining_res += demands[idx]
                active = new_active
                continue
                
            # Sort eligible by priority score (higher is better)
            # priority_list is passed as scores array
            eligible.sort(key=lambda x: priority_list[x], reverse=True)
            
            scheduled_any = False
            for i in eligible:
                if all(remaining_res[r] >= demands[i][r] for r in range(R)):
                    scheduled[i] = True
                    remaining_res -= demands[i]
                    f = t + durations[i]
                    active.append((f, i))
                    makespan = max(makespan, f)
                    scheduled_any = True
            # Note: In standard SGS, we schedule *one* activity per time step decision?
            # Or all that fit?
            # "starts the highest-priority eligible activity"
            # Usually, after scheduling one, the set of eligible activities changes (successors become eligible) 
            # and resources change. 
            # Does the SGS restart the selection process immediately?
            # Yes, typically. It loops: Find eligible -> Pick best that fits -> Schedule -> Repeat.
            # My current loop `for i in eligible` schedules multiple activities at the same time t 
            # if they fit. This is a valid variation (often called "serial SGS with multiple scheduling per time point").
            # However, if scheduling A makes B ineligible (resource-wise) but B was higher priority, 
            # the "Pick best that fits" logic handles this by trying A first, then B.
            # If A fits, it is scheduled. Then we continue the loop? 
            # If we continue the loop with the *old* eligible list, we might schedule B even if A's scheduling 
            # changed the resource availability for B? 
            # My code checks `remaining_res` which is updated. So it is safe.
            # However, new successors of A are not in `eligible`. 
            # Should we re-calculate eligible after each scheduling?
            # Standard SGS: 
            # 1. Determine eligible set E.
            # 2. Order E by priority.
            # 3. For each j in E (in order):
            #    If j can be scheduled at t, schedule it.
            # 4. If any scheduled, restart from 1? Or continue to next time step?
            # If we don't restart, we might miss high-priority successors of A that could have started at t.
            # Example: A and B are eligible. A is high priority. B is low. A finishes instantly? No, duration > 0.
            # Successor C of A becomes eligible only after A is *scheduled* (in serial SGS, precedence is satisfied upon scheduling? 
            # No, usually upon *completion*? 
            # Wait. In Serial SGS, an activity is eligible if all predecessors are *scheduled* (started).
            # The resource constraint is the main bottleneck.
            # If A starts at t, C (succ of A) is eligible at t.
            # If we don't re-scan, C won't be considered at t.
            # So, the most robust SGS is:
            # While there are unscheduled activities:
            #   Find all eligible (preds scheduled).
            #   Sort by priority.
            #   Schedule the first one that fits resources at current time t.
            #   If none fit, advance t to next event.
            
            # Let's implement this "Single Scheduling per Time Step" version. It is slower but more standard.
            # Given N=30, the overhead is small.
            
            if not scheduled_any:
                if not active:
                    break
                min_f = min(f for f, _ in active)
                t = min_f
                new_active = []
                for f, idx in active:
                    if f > t:
                        new_active.append((f, idx))
                    else:
                        remaining_res += demands[idx]
                active = new_active
                
        return makespan

    # 3. Generate Initial Priority Scores
    # Use CPL as base
    base_scores = cpl.copy()
    
    # Normalize to 0..1 or keep raw? Raw is fine.
    
    best_scores = base_scores.copy()
    best_makespan = simulate_sgs_correct(base_scores)
    
    # 4. Local Search: 2-Opt on the *implied* priority list?
    # The input is scores. The SGS sorts by scores.
    # Changing scores is equivalent to changing the order.
    # A structural change: Instead of random swaps of scores, 
    # we can generate a random permutation of the topological order, 
    # assign scores based on that permutation (e.g., rank), 
    # and then do 2-opt swaps on the *permutation* to see if it improves.
    # This ensures we explore the space of valid topological orderings more systematically.
    
    # However, the output must be *scores*. 
    # If I find a good permutation P, I can return scores where score[P[i]] = N-i.
    # This guarantees the SGS will process them in order P (mostly, unless ties or resource blocks change effective order).
    
    # Let's try a different search: 
    # Start with CPL.
    # Perform hill climbing by swapping the *scores* of two activities.
    # But instead of random pairs, use a "Best Improvement" strategy on a smaller neighborhood?
    # Or, use Simulated Annealing?
    
    # Let's try a simple but effective approach: 
    # 1. Start with CPL.
    # 2. For each activity i, try to move it up in the priority list by swapping with its predecessor in the sorted order?
    # This is complex because "predecessor in sorted order" changes.
    
    # Alternative Structural Change: 
    # The current program uses `random.seed(42)`. 
    # Let's use a deterministic but diverse set of initial heuristics and a more aggressive local search.
    
    # Heuristics:
    # 1. CPL
    # 2. LPT
    # 3. CPL + SuccCount
    # 4. Random Topological Order
    
    candidates = []
    candidates.append(cpl.copy())
    candidates.append(durations.copy())
    candidates.append(cpl + succ_count.astype(float))
    
    # Add a few random topological sorts
    random.seed(123)
    for _ in range(5):
        # Generate random topological order
        in_deg = [len(p) for p in preds]
        avail = [i for i in range(N) if in_deg[i] == 0]
        order = []
        temp_in_deg = in_deg[:]
        temp_succs = [s[:] for s in succs]
        while avail:
            # Pick random from avail
            idx = random.randint(0, len(avail)-1)
            node = avail.pop(idx)
            order.append(node)
            for s in temp_succs[node]:
                temp_in_deg[s] -= 1
                if temp_in_deg[s] == 0:
                    avail.append(s)
        
        # Convert order to scores (higher score = earlier in list)
        scores = np.zeros(N)
        for rank, node in enumerate(order):
            scores[node] = N - rank
        candidates.append(scores)
        
    # Evaluate all
    best_scores = candidates[0]
    best_makespan = float('inf')
    for cand in candidates:
        ms = simulate_sgs_correct(cand)
        if ms < best_makespan:
            best_makespan = ms
            best_scores = cand.copy()
            
    # Local Search: Hill Climbing with Restart
    # Neighborhood: Swap scores of two random activities.
    # To make it structural, let's use "Swap Adjacent in Current Priority Order".
    # 1. Get current priority order (sorted indices).
    # 2. Pick two adjacent indices in this order.
    # 3. Swap their scores.
    # 4. Evaluate.
    
    current_scores = best_scores.copy()
    current_ms = best_makespan
    
    max_iters = 2000
    iter_count = 0
    
    while iter_count < max_iters and time.time() < deadline:
        if time.time() > deadline:
            break
            
        # Get current order
        order = np.argsort(current_scores)[::-1]
        
        # Pick a random pair of adjacent elements in the order
        if len(order) > 1:
            pos = random.randint(0, len(order) - 2)
            i = order[pos]
            j = order[pos + 1]
            
            # Swap scores
            temp = current_scores[i]
            current_scores[i] = current_scores[j]
            current_scores[j] = temp
            
            new_ms = simulate_sgs_correct(current_scores)
            
            if new_ms < current_ms:
                current_ms = new_ms
                best_scores = current_scores.copy()
                best_makespan = current_ms
            else:
                # Swap back
                current_scores[j] = current_scores[i]
                current_scores[i] = temp
        else:
            break
            
        iter_count += 1
        
    return best_scores

# Helper function defined inside priority to avoid global scope issues if needed, 
# but here it's local.

# Note: The function `simulate_sgs_correct` used in the code above is defined *inside* the `priority` function 
# in the draft, but I need to make sure it's accessible. 
# In the draft above, I defined `simulate_lsgs` then `pass`ed, then re-defined logic in comments. 
# I need to actually define `simulate_sgs_correct` properly in the final code.

# Let's re-assemble the final code cleanly.

