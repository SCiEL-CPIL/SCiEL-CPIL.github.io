"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global deadline guard
_START_TIME = _time.time()
_DEADLINE = 140.0  # 10s safety margin before 150s

def dispatch(ready_operations, time):
    # Check deadline
    if _time.time() > _DEADLINE:
        # Fallback: Return first index to prevent crash/timeout score 0
        return 0

    if not ready_operations:
        return 0

    n_ops = len(ready_operations)
    
    # Strategy: Regret-based Dispatching
    # 1. For each ready operation, estimate the completion time if scheduled on its best machine vs second-best machine.
    # 2. The "Regret" is the difference: Regret = Est_Comp(second_best) - Est_Comp(best).
    #    A high regret means that delaying this operation (or forcing it onto a worse machine)
    #    would significantly hurt the schedule. Therefore, we prioritize operations with high regret.
    # 3. This implicitly handles machine selection: we pick the machine that corresponds to the 
    #    "best" estimate in the regret calculation.
    # 4. To incorporate job criticality (since makespan is max over jobs), we can weight the regret
    #    or simply use it as the primary key. Pure regret is a strong heuristic for FJSP.
    
    best_idx = -1
    best_regret = -float('inf')
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        job_id = op["job"]
        remaining_work = op["remaining_work"]
        
        # Get all machine options and durations
        options = op["machine_options"]
        if not options:
            continue
            
        # Sort machines by duration (ascending)
        sorted_machines = sorted(options.items(), key=lambda x: x[1])
        
        # We need at least 2 machines to calculate a meaningful regret in the strict sense.
        # If only 1 machine, regret is 0 (or undefined), but we might still want to schedule it
        # based on other criteria. However, in FJSP, flexibility is key.
        # Let's define Est_Comp for a specific machine m:
        # Est_Comp(m) = time + duration(m) + (remaining_work - duration(m)) 
        # Wait, remaining_work usually includes the current op.
        # If remaining_work is the sum of durations of all remaining ops in the job (including current),
        # then remaining_work_after_current = remaining_work - duration(m).
        # So Est_Comp(m) = time + duration(m) + remaining_work - duration(m) = time + remaining_work.
        # This implies Est_Comp is independent of machine choice if remaining_work is a fixed sum.
        # This makes regret 0 for all ops if we use simple remaining_work.
        
        # This suggests we need a better estimate of "future work" that depends on machine choices 
        # or we need to consider machine load. Since we don't have machine load, 
        # maybe we should look at the *variance* or just use the duration difference?
        
        # Alternative interpretation of Regret in this context without machine load:
        # If we assume all machines are equally available (or we are just picking the fastest),
        # the "regret" of not picking the fastest machine is the time lost.
        # Regret = duration(second_best) - duration(best).
        # This prioritizes operations where the difference between best and second-best machine is largest.
        # This encourages using the most flexible/fastest path.
        
        if len(sorted_machines) < 2:
            # If only one machine, regret is 0. 
            # But we should still consider it. 
            # To make it comparable, we can assign a small negative regret or 0.
            regret = 0.0
            best_m = sorted_machines[0][0]
            best_d = sorted_machines[0][1]
        else:
            best_m, best_d = sorted_machines[0]
            second_m, second_d = sorted_machines[1]
            regret = second_d - best_d
            
            # We might also want to factor in the job's remaining work to ensure 
            # we don't just pick small ops with high flexibility.
            # A common weighted regret: Regret * (1 / (1 + remaining_work)) ? 
            # Or just pure regret?
            # Let's try pure regret first, but tie-break with remaining_work (higher is better for makespan? 
            # No, lower remaining work means job is closer to done, but makespan is max. 
            # Actually, for makespan, we want to keep critical jobs moving. 
            # High remaining work = critical job. 
            # So maybe Regret * remaining_work? 
            # Or Regret + alpha * remaining_work?
            
            # Let's stick to a simple weighted regret: 
            # Score = regret + 0.1 * remaining_work
            # This ensures that if regrets are similar, we pick the job with more work (critical path).
            
            score = regret + 0.1 * remaining_work
            if score > best_regret:
                best_regret = score
                best_idx = i
                best_machine = best_m
            continue # skip the generic update below

        # Update for single machine case (or if we didn't continue)
        # Actually, let's refactor to always calculate a score.
        # If single machine, regret=0. Score = 0.1 * remaining_work.
        if len(sorted_machines) == 1:
             score = 0.1 * remaining_work
             if score > best_regret:
                 best_regret = score
                 best_idx = i
                 best_machine = sorted_machines[0][0]

    if best_idx == -1:
        return 0

    return (int(best_idx), best_machine)

# EVOLVE-BLOCK-END
