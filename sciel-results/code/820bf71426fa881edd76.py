"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0

def _deadline_exceeded():
    return _time.time() >= _DEADLINE

def _soft_min(values, temperature=0.5):
    """Smooth surrogate for min using negative log-sum-exp."""
    if not values:
        return 0.0
    min_val = min(values)
    if min_val <= 0:
        min_val = 1e-9
    # Shift to avoid overflow
    shifted = [v - min_val for v in values]
    log_sum = math.log(sum(math.exp(v / temperature) for v in shifted))
    return min_val - temperature * log_sum

def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Strategy:
    - Priority: Maximize critical path urgency with congestion awareness.
      Score = remaining_work / soft_min_durations + 0.15 * remaining_ops
      - soft_min smooths the minimum duration to consider all machine options
      - Congestion penalty: subtract a small term for heavily loaded machines
    - Machine Selection: Use soft-min to pick machine balancing speed and load.
    """
    if _deadline_exceeded():
        return 0
    
    if not ready_operations:
        return 0
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = None
    
    # Compute average duration across all ready ops to normalize congestion
    all_durations = []
    for op in ready_operations:
        mo = op.get("machine_options", {})
        if mo:
            all_durations.extend(mo.values())
    
    avg_duration = sum(all_durations) / len(all_durations) if all_durations else 1.0
    
    for idx, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue
        
        durations = list(machine_options.values())
        machines = list(machine_options.keys())
        
        if not durations:
            continue
        
        # Use soft-min to smooth the machine selection cost
        sm_min = _soft_min(durations, temperature=0.5)
        
        if sm_min <= 0:
            sm_min = 1e-9
            
        # Core score: Urgency / Cost with smooth surrogate
        score = remaining_work / sm_min
        
        # Secondary objective: Prefer more remaining ops (critical path length)
        score += 0.15 * remaining_ops
        
        # Congestion penalty: penalize if the best machine is slow relative to average
        # This helps avoid bottleneck machines
        min_dur = min(durations)
        if min_dur > avg_duration * 1.5:
            score -= 0.05 * (min_dur - avg_duration) / avg_duration
        
        if score > best_score:
            best_score = score
            best_idx = idx
            
            # Machine selection: use soft-min ranking
            # Pick the machine that balances speed and congestion
            # Score each machine: lower is better (faster and less congested)
            best_m_score = float('inf')
            for m in machines:
                d = machine_options[m]
                # Machine cost: duration + small congestion penalty
                m_cost = d
                # Penalize machines significantly slower than the fastest option
                if d > min_dur * 1.3:
                    m_cost += 0.1 * (d - min_dur)
                
                if m_cost < best_m_score:
                    best_m_score = m_cost
                    best_machine = m
            
    if best_machine is not None:
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
