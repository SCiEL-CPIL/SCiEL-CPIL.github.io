"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

# Deadline guard: kill after 145s to stay within 150s limit
_DEADLINE = 145.0
_start_time = time.time()


def _topological_order(predecessors, n):
    """Return a topological ordering of activities."""
    in_degree = [0] * n
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If not all nodes are in order (cycle), append remaining
    if len(order) < n:
        remaining = set(range(n)) - set(order)
        order.extend(sorted(remaining))
    
    return order


def _forward_pass(durations, predecessors, order, n):
    """Compute earliest start (ES) and earliest finish (EF)."""
    es = np.zeros(n)
    ef = np.zeros(n)
    for u in order:
        if predecessors[u]:
            es[u] = max(ef[p] for p in predecessors[u])
        ef[u] = es[u] + durations[u]
    return es, ef


def _backward_pass(durations, predecessors, successors, order, project_end, n):
    """Compute latest start (LS) and latest finish (LF)."""
    ls = np.zeros(n)
    lf = np.zeros(n)
    for u in reversed(order):
        if successors[u]:
            lf[u] = min(ls[s] for s in successors[u])
        else:
            lf[u] = project_end
        ls[u] = lf[u] - durations[u]
    return ls, lf


def _simulate_greedy_schedule(durations, demands, capacities, predecessors, successors, n, n_res):
    """
    Simulate a serial schedule-generation scheme using a composite priority that
    explicitly includes current resource slack. Returns per-activity priority scores
    derived from how each activity ranks during the simulation.
    
    Strategy: During the greedy simulation, when selecting the next eligible activity,
    we compute a dynamic score for each eligible activity that penalizes activities
    which would cause immediate resource bottlenecks (low remaining slack relative to
    their demand) and favors those that fit comfortably.
    
    We accumulate a "priority evidence" for each activity based on how often it would
    be selected early in the greedy process and how well it fits into the resource
    envelope at its selection time.
    """
    # Track selection order and dynamic scores
    selection_order = []
    dynamic_scores = np.zeros(n)
    
    # Activity state
    status = np.zeros(n, dtype=int)  # 0=not started, 1=in progress, 2=completed
    start_times = np.full(n, np.inf)
    end_times = np.full(n, np.inf)
    remaining_preds = [len(predecessors[i]) for i in range(n)]
    
    # Resource usage over time (we track active activities)
    current_time = 0.0
    active = []  # list of (end_time, activity_id)
    
    # Pre-compute static features for tie-breaking
    # Forward/backward pass for slack
    order = _topological_order(predecessors, n)
    es, ef = _forward_pass(durations, predecessors, order, n)
    project_end = np.max(ef) if project_end > 0 else np.max(ef)
    ls, lf = _backward_pass(durations, predecessors, successors, order, project_end, n)
    slack = ls - es
    max_slack = np.max(slack) if np.max(slack) > 0 else 1.0
    norm_slack = slack / max_slack
    
    # Resource scarcity
    total_demand = np.sum(demands, axis=0)
    scarcity = total_demand / (capacities + 1e-10)
    scarcity_contribution = np.sum(demands * scarcity, axis=1)
    max_scarc = np.max(scarcity_contribution) if np.max(scarcity_contribution) > 0 else 1.0
    norm_scarc = scarcity_contribution / max_scarc
    
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    norm_dur = durations / max_dur
    
    # Number of successors (out-degree) - activities with more successors are more critical
    out_degree = np.array([len(successors[i]) for i in range(n)], dtype=float)
    max_out = np.max(out_degree) if np.max(out_degree) > 0 else 1.0
    norm_out = out_degree / max_out
    
    # Run the greedy simulation
    scheduled = set()
    while len(scheduled) < n:
        # Advance time to next event if no eligible activities
        eligible = []
        for i in range(n):
            if i in scheduled:
                continue
            if all(p in scheduled for p in predecessors[i]):
                eligible.append(i)
        
        if not eligible:
            # Advance to next completion
            if not active:
                break
            next_end = min(t for t, _ in active)
            current_time = next_end
            # Remove completed
            active = [(t, a) for t, a in active if t > current_time]
            continue
        
        # Compute dynamic priority for each eligible activity
        # Current resource usage
        current_usage = np.zeros(n_res)
        for t, a in active:
            current_usage += demands[a]
        current_slack = capacities - current_usage  # remaining capacity for each resource
        
        # For each eligible activity, compute a composite dynamic score
        # Key idea: penalize activities that demand resources with low remaining slack
        best_score = -np.inf
        best_activity = None
        
        for i in eligible:
            # Dynamic resource fit score:
            # For each resource, compute how tight the fit is
            # If demand[i, r] > current_slack[r], it's infeasible (but we assume feasible instances)
            # Score based on minimum slack ratio across resources used
            
            # Resource fit: how much slack remains after scheduling this activity
            # We want to favor activities that leave the most "breathing room"
            # For each resource r where demand[i,r] > 0:
            #   slack_after = current_slack[r] - demand[i,r]
            #   tightness = demand[i,r] / (current_slack[r] + 1e-10)
            # We want LOW tightness (loose fit)
            
            tightness_values = []
            for r in range(n_res):
                if demands[i, r] > 1e-10:
                    tightness_values.append(demands[i, r] / (current_slack[r] + 1e-10))
            
            if tightness_values:
                # Use max tightness (worst-case resource constraint)
                max_tightness = max(tightness_values)
            else:
                max_tightness = 0.0
            
            # Dynamic score components:
            # 1. Low resource tightness (loose fit) - high priority
            # 2. Low slack (urgency from CPM) - high priority  
            # 3. High scarcity contribution - high priority
            # 4. Long duration - high priority (blocks resources)
            # 5. High out-degree (more successors depend on it) - high priority
            
            # Normalize tightness: lower is better
            # We'll combine using a weighted sum where lower tightness gives higher score
            
            # Static priority component
            static_score = (0.35 * (1.0 - norm_slack[i]) + 
                           0.25 * norm_scarc[i] + 
                           0.15 * norm_dur[i] +
                           0.15 * norm_out[i])
            
            # Dynamic resource fit component:
            # Convert tightness to a score: lower tightness -> higher score
            # Use exponential decay to make it smooth
            # tightness_score = exp(-alpha * max_tightness)
            # With alpha=1, tightness=0 -> 1.0, tightness=1 -> 0.37, tightness=2 -> 0.14
            alpha = 1.0
            tightness_score = np.exp(-alpha * min(max_tightness, 5.0))
            
            # Combined dynamic score
            dynamic_score = 0.6 * static_score + 0.4 * tightness_score
            
            if dynamic_score > best_score:
                best_score = dynamic_score
                best_activity = i
        
        if best_activity is None:
            break
        
        # Schedule the best activity
        scheduled.add(best_activity)
        start_times[best_activity] = current_time
        end_times[best_activity] = current_time + durations[best_activity]
        active.append((end_times[best_activity], best_activity))
        selection_order.append(best_activity)
        dynamic_scores[best_activity] = best_score
        
        # Record position in selection order (earlier = higher priority)
    
    # Now compute final priority scores based on:
    # 1. Selection order (earlier selected = higher priority)
    # 2. Dynamic score from the simulation
    
    # Selection order rank: earlier = higher priority
    order_rank = np.zeros(n)
    for idx, act in enumerate(selection_order):
        order_rank[act] = idx
    
    max_rank = max(n - 1, 1)
    norm_rank = order_rank / max_rank  # 0 = first selected, 1 = last selected
    
    # Combine: selection order is the primary signal, dynamic score as secondary
    # Invert rank: earlier selection -> higher priority
    priority_from_order = 1.0 - norm_rank
    
    # Normalize dynamic scores
    max_dyn = np.max(dynamic_scores) if np.max(dynamic_scores) > 0 else 1.0
    min_dyn = np.min(dynamic_scores) if np.min(dynamic_scores) < 1 else 0.0
    if max_dyn - min_dyn > 1e-10:
        norm_dyn = (dynamic_scores - min_dyn) / (max_dyn - min_dyn)
    else:
        norm_dyn = np.ones(n)
    
    # Final priority: blend selection order with dynamic score
    priority_score = 0.7 * priority_from_order + 0.3 * norm_dyn
    
    return priority_score


def priority(instance):
    global _start_time
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: return durations as priority
        return np.asarray(instance["durations"], dtype=float)
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # (n, n_res)
    capacities = np.asarray(instance["capacities"], dtype=float)  # (n_res,)
    predecessors = instance["predecessors"]  # list of lists
    
    n = len(durations)
    n_res = len(capacities)
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Check deadline before heavy computation
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5:
        # Fallback: simple CPM-based priority
        order = _topological_order(predecessors, n)
        es, ef = _forward_pass(durations, predecessors, order, n)
        project_end = np.max(ef)
        ls, lf = _backward_pass(durations, predecessors, successors, order, project_end, n)
        slack = ls - es
        max_slack = np.max(slack) if np.max(slack) > 0 else 1.0
        return 1.0 - slack / max_slack
    
    # Run the greedy simulation to get dynamic priority scores
    try:
        priority_score = _simulate_greedy_schedule(
            durations, demands, capacities, predecessors, successors, n, n_res
        )
    except Exception:
        # Fallback on error
        order = _topological_order(predecessors, n)
        es, ef = _forward_pass(durations, predecessors, order, n)
        project_end = np.max(ef)
        ls, lf = _backward_pass(durations, predecessors, successors, order, project_end, n)
        slack = ls - es
        max_slack = np.max(slack) if np.max(slack) > 0 else 1.0
        priority_score = 1.0 - slack / max_slack
    
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 2:
        # Return quickly
        return priority_score
    
    return priority_score
# EVOLVE-BLOCK-END
