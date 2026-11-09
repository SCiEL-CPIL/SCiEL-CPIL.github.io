"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Deadline guard
_START_TIME = None
_DEADLINE = 150.0

# UCB state for machine arms
# Key: machine_id, Value: dict with 'pulls' (count), 'total_reward' (sum of costs)
_UCB_STATE = {}

def _fast_dispatch(ready_operations, time):
    """Extremely fast fallback: Longest Remaining Work (LRW) / MWR."""
    if not ready_operations:
        return 0
    max_rw = -1
    best_idx = 0
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        if rw > max_rw:
            max_rw = rw
            best_idx = i
    return best_idx

def _estimate_machine_loads(ready_operations):
    """
    Estimate machine loads by simulating a greedy SPT assignment.
    This provides a better proxy for future congestion than static weights.
    """
    machine_load = {}
    
    # Sort operations by min duration (SPT heuristic for load estimation)
    ops_sorted = []
    for op in ready_operations:
        mo = op.get("machine_options", {})
        if mo:
            min_d = min(mo.values())
            ops_sorted.append((min_d, op))
    
    ops_sorted.sort(key=lambda x: x[0])
    
    # Greedily assign each op to the machine with the least current load
    # among its options, breaking ties by duration
    for min_d, op in ops_sorted:
        mo = op.get("machine_options", {})
        if not mo:
            continue
        
        # Find machine with min (load + duration)
        best_machine = None
        best_cost = float('inf')
        for m, d in mo.items():
            load = machine_load.get(m, 0)
            cost = load + d
            if cost < best_cost:
                best_cost = cost
                best_machine = m
        
        if best_machine is not None:
            d = mo[best_machine]
            machine_load[best_machine] = machine_load.get(best_machine, 0) + d
    
    return machine_load

def dispatch(ready_operations, time):
    global _START_TIME
    global _UCB_STATE
    
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE - 10.0:
        return _fast_dispatch(ready_operations, time)
    
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    if n_ops == 1:
        return 0
    
    # 1. Calculate normalization constants
    max_rw = 0
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        if rw > max_rw:
            max_rw = rw
    if max_rw <= 0:
        max_rw = 1.0

    # 2. Estimate machine loads using greedy SPT simulation
    machine_load = _estimate_machine_loads(ready_operations)
    
    if not machine_load:
        # Fallback if no machine options provided
        return _fast_dispatch(ready_operations, time)

    avg_load = sum(machine_load.values()) / len(machine_load)
    if avg_load <= 0:
        avg_load = 1.0

    # Weights - tuned for better makespan performance
    # Emphasize remaining work (critical path) and efficiency (SPT + load balancing)
    w_rw = 0.40
    w_spt = 0.25
    w_ucb = 0.35  # Weight for the UCB-based machine selection component
    
    best_score = -float('inf')
    best_idx = 0

    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        mo = op.get("machine_options", {})
        
        if not mo:
            min_d = 1.0
            ucb_score = 0.0
        else:
            durations = list(mo.values())
            min_d = min(durations)
            
            # Calculate UCB score for the best machine option
            # We want to minimize the "cost" (duration + congestion penalty)
            # UCB formula for minimization: 
            # Score = min_cost + sqrt(2 * ln(total_pulls) / pulls)
            # Here, we invert the logic: we want to maximize a score that favors
            # low estimated cost and high exploration of underused machines.
            
            # Let's define the "reward" as negative cost (so higher is better)
            # Or simpler: Calculate the UCB value for each machine and pick the one
            # with the highest UCB score (where UCB score = -min_cost + exploration_bonus)
            
            total_pulls = 0
            for m in mo.keys():
                if m in _UCB_STATE:
                    total_pulls += _UCB_STATE[m]['pulls']
            
            # If no pulls yet, exploration bonus is high
            if total_pulls == 0:
                total_pulls = 1
            
            best_ucb_val = -float('inf')
            for m, d in mo.items():
                load = machine_load.get(m, 0)
                load_factor = load / avg_load
                # Base cost: duration penalized by load
                base_cost = d * (1.0 + 1.5 * load_factor)
                
                # Get UCB state for this machine
                if m not in _UCB_STATE:
                    _UCB_STATE[m] = {'pulls': 0, 'total_reward': 0.0}
                
                pulls = _UCB_STATE[m]['pulls']
                if pulls == 0:
                    # Unexplored arm: give it a high bonus
                    ucb_component = 100.0  # Large bonus to force exploration
                else:
                    # Average cost so far (lower is better)
                    avg_cost = _UCB_STATE[m]['total_reward'] / pulls
                    # Exploration bonus
                    exploration = math.sqrt(2.0 * math.log(total_pulls + 1) / pulls)
                    # UCB score for minimization:
                    # We want to minimize avg_cost + exploration?
                    # No, standard UCB maximizes reward.
                    # Let's treat "reward" as -base_cost.
                    # But base_cost changes every time. 
                    # Instead, let's just use the current base_cost as the "exploit" term
                    # and add an exploration bonus based on how few times we've routed to this machine.
                    
                    # Exploit: lower base_cost is better. So we use -base_cost.
                    # Explore: higher bonus if pulls is low.
                    # Score = -base_cost + exploration_bonus
                    # To make it dimensionless, scale base_cost?
                    # Let's just use the raw values.
                    # If base_cost is 10, score = -10 + bonus.
                    # If base_cost is 5, score = -5 + bonus.
                    # We want to pick the machine with the highest score.
                    
                    # The exploration term should be significant.
                    # Let's scale exploration by avg_load to make it comparable to costs.
                    exploration_bonus = exploration * avg_load
                    
                    ucb_component = -base_cost + exploration_bonus
                
                if ucb_component > best_ucb_val:
                    best_ucb_val = ucb_component
            
            # Normalize UCB score for the final combination
            # The UCB score can be negative (if cost is high) or positive (if exploration is high).
            # We want to map this to a 0..1 range roughly.
            # Let's use a sigmoid or simple scaling.
            # Since we don't know the range, let's just use it directly but weighted.
            # Or normalize by avg_load?
            norm_ucb = best_ucb_val / (avg_load + 1.0)
            
            # SPT component: shorter is better
            norm_spt = 1.0 / (min_d + 1.0)

        # RW component
        norm_rw = rw / max_rw
        
        # Combined Score
        # We want to maximize the score.
        # High RW is good.
        # High SPT (short time) is good.
        # High UCB score (good machine choice) is good.
        
        score = (w_rw * norm_rw + 
                 w_spt * norm_spt + 
                 w_ucb * norm_ucb)
        
        if score > best_score:
            best_score = score
            best_idx = i
        elif abs(score - best_score) < 1e-9:
            # Tie breaker: prefer shorter processing time
            prev_op = ready_operations[best_idx]
            prev_mo = prev_op.get("machine_options", {})
            prev_min_d = min(prev_mo.values()) if prev_mo else 1.0
            
            cur_min_d = min(mo.values()) if mo else 1.0
            
            if cur_min_d < prev_min_d:
                best_idx = i

    # Update UCB state with the chosen operation's best machine
    # We need to identify which machine was "selected" for the UCB update.
    # The dispatch function returns an index, not a machine.
    # However, the simulation will assign the operation to a machine.
    # Usually, the simulation picks the machine with the minimum duration or the one suggested.
    # Since we don't return the machine, we assume the simulation will pick the "best" machine
    # based on some rule. To keep the UCB state consistent, we should update the state
    # for the machine that *would* be chosen by our heuristic (the one with max UCB score).
    
    if best_idx is not None and ready_operations[best_idx].get("machine_options"):
        best_op = ready_operations[best_idx]
        mo = best_op["machine_options"]
        
        # Recalculate the best machine for this op using the same logic as above
        machine_load = _estimate_machine_loads(ready_operations) # Re-estimate? Or use cached?
        # To save time, we can just pick the machine with min duration for the update,
        # or the one with the highest UCB score calculated in the loop.
        # Let's just pick the machine with the minimum duration for simplicity in update,
        # as that's a common default if machine isn't specified.
        # Or better: pick the machine that maximized the UCB score in the loop.
        
        total_pulls = 0
        for m in mo.keys():
            if m in _UCB_STATE:
                total_pulls += _UCB_STATE[m]['pulls']
        if total_pulls == 0:
            total_pulls = 1

        best_ucb_val = -float('inf')
        selected_machine = None
        for m, d in mo.items():
            load = machine_load.get(m, 0)
            load_factor = load / avg_load
            base_cost = d * (1.0 + 1.5 * load_factor)
            
            if m not in _UCB_STATE:
                _UCB_STATE[m] = {'pulls': 0, 'total_reward': 0.0}
            
            pulls = _UCB_STATE[m]['pulls']
            if pulls == 0:
                ucb_component = 100.0
            else:
                avg_cost = _UCB_STATE[m]['total_reward'] / pulls
                exploration = math.sqrt(2.0 * math.log(total_pulls + 1) / pulls)
                exploration_bonus = exploration * avg_load
                ucb_component = -base_cost + exploration_bonus
            
            if ucb_component > best_ucb_val:
                best_ucb_val = ucb_component
                selected_machine = m

        if selected_machine is not None:
            # Update the state for the selected machine
            # The "reward" is the negative of the actual cost incurred.
            # But we don't know the actual cost until the simulation runs.
            # For now, we update the pull count and add the estimated base_cost to total_reward.
            # This is an approximation.
            
            load = machine_load.get(selected_machine, 0)
            load_factor = load / avg_load
            d = mo[selected_machine]
            base_cost = d * (1.0 + 1.5 * load_factor)
            
            _UCB_STATE[selected_machine]['pulls'] += 1
            _UCB_STATE[selected_machine]['total_reward'] += base_cost

    return best_idx

# EVOLVE-BLOCK-END
