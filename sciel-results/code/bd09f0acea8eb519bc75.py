import numpy as np
import time

def priority(instance):
    """
    Calculates priority scores using a dynamic Resource-Aware Heuristic.
    
    Strategy:
    1. Perform a full serial schedule simulation to determine which activities 
       are "critical" under resource constraints.
    2. Prioritize activities that:
       a) Have high precedence tightness (low slack in the simulated schedule).
       b) Consume high proportions of scarce resources (high demand/capacity ratio).
    3. Return scores such that higher score = higher priority.
    
    This approach adapts to the specific resource bottlenecks of the instance,
    unlike static CPM which ignores resource availability.
    """
    start_time = time.time()
    
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    demands = np.asarray(instance["demands"], dtype=float) # shape (n, m)
    capacities = np.asarray(instance["capacities"], dtype=float) # shape (m,)
    
    n = len(durations)
    m = len(capacities)
    
    # Handle edge cases
    if n == 0:
        return np.array([])
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            if i is not None:
                successors[i].append(j)
            
    # Topological Sort (Kahn's Algorithm)
    in_degree = [len(predecessors[j]) for j in range(n)]
    queue = [j for j in range(n) if in_degree[j] == 0]
    topo_order = []
    
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) != n:
        # Fallback: simple duration-based priority if cycle detected
        return durations.copy()
    
    # --- Dynamic Simulation to Compute Priorities ---
    
    # We simulate the serial schedule generation scheme.
    # At each step, we pick the "best" activity from eligible ones.
    # To determine "best" for priority scoring, we simulate a greedy schedule 
    # using a base heuristic (e.g., Min Slack or Min Latest Start) to see 
    # which activities end up on the critical path or cause delays.
    
    # First, compute static CPM slack to guide the simulation's initial choices
    ES = [0.0] * n
    EF = [0.0] * n
    for j in topo_order:
        if predecessors[j]:
            ES[j] = max(EF[i] for i in predecessors[j])
        else:
            ES[j] = 0.0
        EF[j] = ES[j] + durations[j]
        
    makespan_cpm = max(EF)
    
    LF = [makespan_cpm] * n
    LS = [makespan_cpm] * n
    for j in reversed(topo_order):
        if successors[j]:
            LF[j] = min(LS[k] for k in successors[j])
        else:
            LF[j] = makespan_cpm
        LS[j] = LF[j] - durations[j]
        
    static_slack = [LS[j] - ES[j] for j in range(n)]
    
    # Now, run a serial schedule simulation to identify resource-critical activities.
    # We use a greedy strategy based on static slack to generate a schedule.
    # The "priority" for an activity in the final score will be based on:
    # 1. Its actual start time in the simulated schedule vs its static LS (delay).
    # 2. Its resource intensity.
    
    # Simulation State
    current_time = 0.0
    remaining_in_degree = in_degree.copy()
    finished = [False] * n
    start_times = [0.0] * n
    end_times = [0.0] * n
    
    # Track resource usage over time is complex, but for priority scoring,
    # we can track which activities were "delayed" the most or were critical.
    
    # A simpler, robust dynamic metric: 
    # Simulate the schedule. For each activity, calculate its "impact" on makespan.
    # Since we can't easily do full sensitivity analysis in 150s for 30 acts, 
    # we use a proxy: Activities that start later than their static LS are "resource blocked".
    # Activities with high demand/capacity ratios are "resource heavy".
    
    # Let's simulate a schedule using Min Slack as the tie-breaker (standard serial SGS).
    # We record the actual start time.
    
    eligible = [j for j in range(n) if remaining_in_degree[j] == 0]
    
    # We need to simulate time steps or event-driven.
    # Since n=30, we can do a simple event-driven simulation.
    
    # Resource availability is constant. We just need to know when resources free up.
    # However, in Serial SGS, we pick ONE activity at a time.
    # The start time of activity j is max(ES_j, max(end_time of predecessors), time when resources are available).
    # In Serial SGS, resources are available at the current "schedule time" if we are scheduling sequentially?
    # No, Serial SGS schedules one activity at a time. The start time of the next activity is determined by:
    # 1. Precedence constraints (max end time of predecessors).
    # 2. Resource availability: The activity can start at the earliest time t >= max_pred_end where all required resources are available.
    
    # To find the earliest feasible start time, we might need to check resource profiles.
    # With 30 activities, we can pre-compute resource profiles or just check intervals.
    
    # Let's implement a helper to find earliest start time given current schedule.
    
    def get_earliest_start(j, current_schedule):
        """
        current_schedule: list of (start, end, activity_id) for already scheduled activities.
        Returns the earliest time activity j can start given resources and precedence.
        """
        # Precedence constraint
        prec_constraint = 0.0
        for i in predecessors[j]:
            if i in scheduled_ids:
                prec_constraint = max(prec_constraint, scheduled_end_times[i])
        
        # Resource constraint
        # We need to find the smallest t >= prec_constraint such that for all resources r:
        # sum(demands[k][r] for k in scheduled if start[k] <= t < end[k]) + demands[j][r] <= capacities[r]
        
        # Candidate start times: prec_constraint, and end times of scheduled activities.
        candidates = [prec_constraint]
        for st, en, _ in current_schedule:
            if en > prec_constraint:
                candidates.append(en)
        
        # Sort candidates
        candidates = sorted(set(candidates))
        
        for t in candidates:
            feasible = True
            for r in range(m):
                usage = 0.0
                for st, en, k in current_schedule:
                    if st <= t < en:
                        usage += demands[k][r]
                if usage + demands[j][r] > capacities[r]:
                    feasible = False
                    break
            if feasible:
                return t
        # If no candidate works (shouldn't happen if prec_constraint is valid and resources eventually free),
        # fallback to a large number or just prec_constraint (which might be invalid, but we assume feasible instances)
        return candidates[-1] if candidates else prec_constraint

    # Simulation
    scheduled_ids = set()
    scheduled_end_times = [0.0] * n
    current_schedule = [] # list of (start, end, id)
    
    # We will simulate the schedule using a priority rule based on Static Slack.
    # The goal is to identify which activities are "bottlenecks".
    
    # To make the priority function more robust, we simulate the schedule and 
    # assign higher priority to activities that are "late" (high actual start vs static LS)
    # and "heavy" (high resource demand).
    
    # Actually, the `priority` function is called ONCE to return scores for ALL activities.
    # The evaluator then uses these scores to generate the schedule.
    # So, we can't simulate the schedule *using* the scores we're trying to compute in a feedback loop 
    # within the single `priority` call unless we simulate a *different* schedule to derive the scores.
    
    # Strategy: 
    # 1. Simulate a schedule using a simple heuristic (e.g., Min Static Slack).
    # 2. From this simulation, identify activities that have high "regret" or are critical.
    # 3. Return scores that favor these activities.
    
    # Let's simulate using Min Static Slack.
    remaining_in_degree = in_degree.copy()
    eligible = [j for j in range(n) if remaining_in_degree[j] == 0]
    scheduled = [] # list of scheduled activity ids
    start_times = [0.0] * n
    end_times = [0.0] * n
    
    # We need to track resource usage efficiently.
    # Since n is small (30), we can just check all scheduled activities for each new start.
    
    while len(scheduled) < n:
        # Select best eligible activity based on Min Static Slack
        # Tie-breaker: Min Static LS
        best_j = None
        best_score = float('inf')
        for j in eligible:
            # Heuristic for simulation: Min Slack
            score = static_slack[j]
            if score < best_score:
                best_score = score
                best_j = j
        
        if best_j is None:
            break # Should not happen
            
        # Find earliest start time for best_j
        # Precedence
        prec_end = 0.0
        for i in predecessors[best_j]:
            if end_times[i] > prec_end:
                prec_end = end_times[i]
                
        # Resource check
        # Candidate times: prec_end and end times of all scheduled activities
        candidates = [prec_end]
        for k in scheduled:
            if end_times[k] > prec_end:
                candidates.append(end_times[k])
        candidates = sorted(set(candidates))
        
        start_t = prec_end
        for t in candidates:
            feasible = True
            for r in range(m):
                usage = 0.0
                for k in scheduled:
                    if start_times[k] <= t < end_times[k]:
                        usage += demands[k][r]
                if usage + demands[best_j][r] > capacities[r]:
                    feasible = False
                    break
            if feasible:
                start_t = t
                break
        
        start_times[best_j] = start_t
        end_times[best_j] = start_t + durations[best_j]
        scheduled.append(best_j)
        scheduled_ids.add(best_j)
        
        # Update eligibility
        for v in successors[best_j]:
            remaining_in_degree[v] -= 1
            if remaining_in_degree[v] == 0:
                eligible.append(v)
                
        # Remove best_j from eligible
        eligible.remove(best_j)
        
        # Time check
        if time.time() - start_time > 140:
            break
            
    # Now, calculate final priority scores.
    # We want to prioritize activities that:
    # 1. Have low static slack (critical path).
    # 2. Were delayed in the simulation (actual start > static LS) -> indicates resource conflict.
    # 3. Have high resource intensity.
    
    scores = np.zeros(n)
    
    # Normalize components
    min_slack = min(static_slack) if static_slack else 0
    max_slack = max(static_slack) if static_slack else 1
    slack_range = max_slack - min_slack if max_slack > min_slack else 1
    
    max_delay = 0.0
    delays = [0.0] * n
    for j in range(n):
        delay = max(0, start_times[j] - LS[j])
        delays[j] = delay
        if delay > max_delay:
            max_delay = delay
    
    max_int = 0.0
    intensities = [0.0] * n
    for j in range(n):
        # Intensity: max ratio of demand to capacity
        if m > 0:
            ratios = []
            for r in range(m):
                if capacities[r] > 0:
                    ratios.append(demands[j][r] / capacities[r])
            intensities[j] = max(ratios) if ratios else 0
        if intensities[j] > max_int:
            max_int = intensities[j]
    
    for j in range(n):
        # Component 1: Inverse Slack (lower slack = higher score)
        c1 = 1.0 - (static_slack[j] - min_slack) / slack_range
        
        # Component 2: Delay (higher delay = higher priority to fix conflicts?)
        # Actually, if an activity is delayed, it might be better to schedule it earlier in the next run.
        # So high delay = high priority.
        c2 = delays[j] / max_delay if max_delay > 0 else 0
        
        # Component 3: Intensity (higher intensity = higher priority to secure resources)
        c3 = intensities[j] / max_int if max_int > 0 else 0
        
        # Weighted sum
        # Weights: Slack is most important, then Delay, then Intensity
        scores[j] = 10.0 * c1 + 5.0 * c2 + 2.0 * c3
        
    return scores
