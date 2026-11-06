"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random
import math

_START_TIME = _time.time()
_DEADLINE = 120.0
_SEED = 42

def _simulate_sequence(ready_ops, machine_loads, num_steps=5):
    """
    Simulate a short lookahead sequence to estimate future bottleneck.
    Uses a greedy heuristic (min load) to assign operations to machines
    in the simulation, updating load estimates.
    Returns the maximum machine load after the simulation steps.
    """
    # Copy current loads to avoid side effects
    sim_loads = dict(machine_loads)
    # Create a pool of operations to simulate
    # We prioritize operations with fewer machine options (more constrained) in simulation
    # to see if they force high loads
    pool = list(ready_ops)
    
    # Sort by number of options (ascending) - constrained first
    pool.sort(key=lambda x: len(x['machine_options']))
    
    max_load = 0.0
    for _ in range(min(num_steps, len(pool))):
        if not pool:
            break
        # Pick the most constrained operation (first in sorted list)
        # In a real lookahead, we'd pick based on current state, but here we just
        # simulate a greedy assignment to estimate load distribution
        op = pool.pop(0)
        
        # Find the machine with minimum current load in simulation
        best_machine = None
        min_load = float('inf')
        for m, dur in op['machine_options'].items():
            load = sim_loads.get(m, 0.0)
            if load < min_load:
                min_load = load
                best_machine = m
        
        if best_machine is not None:
            # Assign to this machine
            new_load = min_load + op['machine_options'][best_machine]
            sim_loads[best_machine] = new_load
            if new_load > max_load:
                max_load = new_load
            # Remove this op's contribution from other machines' "potential" loads?
            # No, we are estimating the *realized* load if this op goes to best_machine.
            # However, other ops might still be scheduled.
            # To keep it simple, we just track the max load of assigned ops.
            
    return max_load

def dispatch(ready_operations, time):
    # Deadline guard
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE - 2.0:
        # Fallback: random or simple rule to avoid timeout
        return int(len(ready_operations) * random.random())
    
    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0
    if n_ops == 1:
        return 0

    # 1. Estimate current machine loads based on all ready operations
    # We assume that if an op is ready, it *will* be scheduled soon.
    # A simple estimate: for each machine, sum the durations of all ready ops
    # that can use it, weighted by 1/num_options (probability of assignment?)
    # Or simpler: just sum durations. This is a "potential load".
    machine_loads = {}
    all_machines = set()
    for op in ready_operations:
        for m in op['machine_options']:
            all_machines.add(m)
    
    for m in all_machines:
        load = 0.0
        for op in ready_operations:
            if m in op['machine_options']:
                load += op['machine_options'][m]
        machine_loads[m] = load

    # 2. For each candidate operation, estimate the "regret" or "impact"
    # Strategy: Pick the operation that, if scheduled now on its best machine,
    # results in the lowest *projected* maximum load.
    
    # To make this a "search" method, we run a mini-simulation for each candidate.
    # For candidate i:
    #   - Assign op i to its best machine (min load + duration)
    #   - Run a short greedy lookahead for the remaining ops
    #   - Record the max load
    # Select the candidate with the min max load.
    
    # Optimization: We don't want to run full simulations for every op if it's slow.
    # But n_ops is small (10 jobs, so maybe 5-10 ready ops).
    
    best_index = -1
    best_max_load = float('inf')
    
    # Pre-sort operations by remaining work to break ties in simulation
    # and to provide a deterministic order for the greedy part of simulation
    
    for i in range(n_ops):
        op = ready_operations[i]
        
        # Determine the "best" machine for this op given current loads
        best_machine = None
        min_cost = float('inf')
        for m, dur in op['machine_options'].items():
            # Cost: current load + duration
            cost = machine_loads.get(m, 0.0) + dur
            if cost < min_cost:
                min_cost = cost
                best_machine = m
        
        if best_machine is None:
            continue
            
        # Simulate the rest
        # Create a copy of loads with this op assigned
        sim_loads = dict(machine_loads)
        sim_loads[best_machine] = sim_loads.get(best_machine, 0.0) + op['machine_options'][best_machine]
        
        # Build a list of other ops to simulate
        other_ops = [op_j for j, op_j in enumerate(ready_operations) if j != i]
        
        # Run a simple greedy simulation for the other ops
        # We will assign them to their least loaded machine
        max_load_in_sim = sim_loads.get(best_machine, 0.0)
        
        # To speed up, we don't simulate all, just a few steps or all if small
        # Let's simulate all other ready ops in a greedy manner
        # Sort by remaining work descending (urgent first)
        other_ops.sort(key=lambda x: x['remaining_work'], reverse=True)
        
        for op_o in other_ops:
            best_m_o = None
            min_load_o = float('inf')
            for m, dur in op_o['machine_options'].items():
                load = sim_loads.get(m, 0.0)
                if load < min_load_o:
                    min_load_o = load
                    best_m_o = m
            
            if best_m_o is not None:
                new_load = min_load_o + op_o['machine_options'][best_m_o]
                sim_loads[best_m_o] = new_load
                if new_load > max_load_in_sim:
                    max_load_in_sim = new_load
        
        if max_load_in_sim < best_max_load:
            best_max_load = max_load_in_sim
            best_index = i

    if best_index == -1:
        # Fallback
        return 0
        
    return best_index
# EVOLVE-BLOCK-END
