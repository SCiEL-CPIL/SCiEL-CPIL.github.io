"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard: keep a safety margin before the 150s wall-clock kill.
_DEADLINE = _time.monotonic() + 145.0


def _time_left():
    return _DEADLINE - _time.monotonic()


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an open-shop simulation.
    
    Strategy: combine job criticality (remaining work) with machine load balancing.
    For each ready operation, compute a score that balances:
      1. Job criticality: prefer operations from jobs with more remaining work
         (clears critical paths early).
      2. Machine load: prefer operations that go to less-loaded machines
         (spreads work to avoid bottlenecks).
    
    We use a weighted combination. Since we don't have direct access to machine
    loads in the interface, we approximate by considering the duration on the
    chosen machine relative to other options, and use remaining_work as the
    primary signal.
    
    Additionally, we prefer shorter operations on busy machines to reduce
    blocking, and longer operations on idle machines to make progress.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Primary: remaining work per job (critical path clearing)
    # Secondary: prefer operations with shorter duration on their best machine
    #            (reduces blocking time on any machine)
    # Tertiary: spread across machines by preferring ops whose machine has
    #            fewer concurrent options in the ready set

    # Count how many ready ops are targeting each machine (to estimate contention)
    machine_count = {}
    for op in ready_operations:
        mo = op["machine_options"]
        # The machine this op will use is ambiguous; use the one with min duration
        # as a proxy for "preferred machine"
        best_machine = min(mo, key=lambda m: mo[m])
        machine_count[best_machine] = machine_count.get(best_machine, 0) + 1

    scores = np.zeros(n)
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        mo = op["machine_options"]
        # Min duration on any machine (shortest possible processing)
        min_dur = min(mo.values())
        # Mean duration across machines
        mean_dur = np.mean(list(mo.values()))
        # Number of machine options (more options = more flexibility)
        num_opts = len(mo)
        # Contention: how many other ready ops target the same preferred machine
        best_machine = min(mo, key=lambda m: mo[m])
        contention = machine_count.get(best_machine, 1)

        # Score: high remaining work is good, low duration is good,
        # low contention is good, more flexibility is good
        score = rw - 0.5 * min_dur - 0.3 * (contention - 1) + 0.1 * num_opts
        scores[i] = score

    # Safety: if we're running out of time, just return the argmax
    if _time_left() < 1.0:
        return int(np.argmax(scores))

    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
