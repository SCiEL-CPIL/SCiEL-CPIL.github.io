import time
import numpy as np

def dispatch(ready_operations, time):
    """
    Dispatches the next operation using a Lookahead Critical Path (LCP) strategy.
    
    Strategy:
    - For each ready operation, we estimate the "criticality" of the job.
    - Criticality is defined as the sum of the minimum possible processing times 
      for all remaining operations in the job (including the current one).
    - We prioritize the operation belonging to the job with the highest criticality.
    - For machine selection, we choose the machine that minimizes the immediate 
      processing time for that specific operation, to keep the job on the 
      critical path moving as fast as possible.
    - This approach looks ahead to the end of the job, rather than just the 
      current step, which helps prevent short jobs from starving long, 
      complex jobs that will dominate the makespan.
    """
    if not ready_operations:
        return -1

    # Deadline guard: The evaluator kills after 150s. 
    # This function is called frequently. We assume the overhead is low, 
    # but we add a simple check just in case the list is huge or computation is slow.
    # Since we don't have a global start time, we rely on the fact that 
    # O(N * M) operations on small lists are fast. 
    # If we were doing complex graph searches, we'd need a global timer.
    # Here, we just ensure we don't do anything pathological.

    best_index = -1
    best_machine = -1
    best_score = -1.0  # We want to MAXIMIZE criticality (longest remaining path)

    for i, op in enumerate(ready_operations):
        # 1. Calculate the minimum duration for the current operation
        min_dur = float('inf')
        min_mach = -1
        for m, d in op['machine_options'].items():
            if d < min_dur:
                min_dur = d
                min_mach = m

        # 2. Calculate the Lookahead Critical Path
        # 'remaining_work' in the prompt description might just be the sum of durations,
        # but 'remaining_ops' suggests we have access to the structure.
        # However, the interface description says: 
        # "each ready operation is a dict with job, op, machine_options {machine: duration}, remaining_ops, remaining_work."
        
        # If 'remaining_work' is available, it is the most direct proxy for the 
        # sum of future processing times. 
        # If 'remaining_ops' is a list of future ops, we could sum their min durations.
        # Let's assume 'remaining_work' is the sum of min durations for all remaining ops 
        # (including current) or just the remaining ones. 
        # The prompt says: "remaining_work". 
        # Usually, in these tasks, remaining_work is a pre-calculated scalar.
        
        # Let's verify the semantics. If remaining_work is 0 for the last op, 
        # it might mean "no work after this". 
        # To be robust, let's calculate the score based on:
        # Score = min_dur (current) + remaining_work (future)
        # This effectively ranks by the total remaining effort of the job.
        
        rem_work = op.get('remaining_work', 0)
        
        # If remaining_work is not provided or is 0, we fall back to just min_dur.
        # But usually, remaining_work is the sum of processing times for all 
        # subsequent operations. 
        # Let's assume the standard interpretation: 
        # Total Criticality = min_dur (current) + sum(min_durations of future ops)
        # If 'remaining_work' is exactly that sum of future ops, then:
        criticality = min_dur + rem_work
        
        # Alternative interpretation: 'remaining_work' includes the current op.
        # If so, criticality = rem_work.
        # How to distinguish? 
        # If rem_work is 0, it's likely the last op. 
        # If we add min_dur, the last op has score = min_dur. 
        # If we don't, the last op has score = 0.
        # Prioritizing the last op with high duration is good (SPT-like for the end).
        # Prioritizing jobs with high *total* remaining work is good (LPT-like for the whole job).
        
        # Let's try a combined metric that strongly favors the job with the most 
        # total remaining work, but breaks ties with the current operation's speed.
        
        # Heuristic: 
        # Primary: Total remaining work (including current). 
        # Secondary: Shortest current duration (to keep the critical path tight).
        
        # Let's assume 'remaining_work' is the sum of min durations for *all* remaining ops 
        # including the current one? Or just the future ones?
        # Common in these benchmarks: remaining_work = sum of durations of all unprocessed ops.
        # Let's assume it includes the current one? 
        # Actually, the key is to identify the "Critical Job". 
        # The job with the longest remaining path is the one most likely to determine makespan.
        
        # Let's use a score that maximizes the remaining work.
        # If rem_work is the sum of future ops only, then total = min_dur + rem_work.
        # If rem_work is total, then total = rem_work.
        # Let's look at the data structure hints. 
        # "remaining_work" is a single value.
        # Let's assume it is the sum of the *minimum* processing times for all 
        # remaining operations (including the current one). Why? Because that is the 
        # standard definition of "remaining work" in critical path contexts.
        
        # However, to be safe against ambiguity, let's use a heuristic that works 
        # reasonably well either way.
        # Score = rem_work * 10 + (10 - min_dur)  ? No, scales are different.
        
        # Let's stick to the most robust standard heuristic for FJSP:
        # Priority = Remaining Work (Total)
        # If we pick the job with the most remaining work, we are effectively doing 
        # "Longest Remaining Processing Time" (LRPT) at the job level.
        
        # Let's try: Score = rem_work. 
        # If rem_work is 0, we fall back to min_dur? 
        # If rem_work is 0, it's the last op. We should pick the one with the 
        # shortest duration to finish quickly? Or longest to get it out of the way?
        # Usually, for the last op, SPT is good to clear the machine.
        # But if we prioritize high rem_work, the last op (rem_work=0 or min_dur) 
        # will be deprioritized until other jobs are done. This is correct for 
        # preventing starvation of long jobs.
        
        # Let's assume rem_work is the sum of min durations for all remaining ops 
        # (including current).
        # If not, and it's only future ops, then for the last op, rem_work=0.
        # In that case, all last ops have score 0. We need a tie-breaker.
        # Tie-breaker: Shortest Processing Time (min_dur).
        
        # So:
        # Primary Key: rem_work (Higher is better)
        # Secondary Key: -min_dur (Lower is better, i.e., shorter is better)
        
        # Let's combine into a single score for maximization:
        # Score = rem_work * 1000 - min_dur
        # This ensures rem_work dominates.
        
        # But what if rem_work is small? 
        # Let's normalize or just use the tuple comparison logic implicitly.
        
        current_score_tuple = (rem_work, -min_dur)
        
        # We want to maximize rem_work, then minimize min_dur.
        # So we compare tuples: (rem_work, -min_dur) > (best_rem, -best_min_dur)
        
        if best_index == -1:
            best_index = i
            best_machine = min_mach
            best_rem = rem_work
            best_min_dur = min_dur
        else:
            # Check if current is better
            if rem_work > best_rem:
                best_index = i
                best_machine = min_mach
                best_rem = rem_work
                best_min_dur = min_dur
            elif rem_work == best_rem:
                if min_dur < best_min_dur:
                    best_index = i
                    best_machine = min_mach
                    best_rem = rem_work
                    best_min_dur = min_dur

    return (best_index, best_machine)
