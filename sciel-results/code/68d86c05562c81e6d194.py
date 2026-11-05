<<<<<<< SEARCH
    trial_count = 0
    max_trials = 50000 # Safety cap, time will limit us sooner
    
    while trial_count < max_trials:
        if time.time() > DEADLINE:
            break
=======
    trial_count = 0
    max_trials = 100000 # Increased cap, speedup from bitsets allows more trials
    
    while trial_count < max_trials:
        if time.time() > DEADLINE - 2.0:
            break
>>>>>>> REPLACE

Let's refine the local search moves. The current swap is random. A more effective move might be to swap a picked vertex with a non-picked neighbor or a non-picked vertex with a picked neighbor. However, implementing that requires knowing the picked set from the last evaluation, which adds state. Given the constraints, random swaps + perturbations are robust. I will slightly increase the noise in perturbations to help escape local optima.

