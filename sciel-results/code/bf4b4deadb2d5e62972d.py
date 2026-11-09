"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _clock
import math

_T0 = _clock.time()
_KILL = _T0 + 110.0


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Composite heuristic:
    - Machine selection: minimize effective completion time considering
      local congestion (how many ready ops target the same machine) and
      a soft load-balancing penalty.
    - Operation selection: maximize urgency (remaining work / processing time)
      with a tail-clearing boost in the late phase, and a critical-path
      penalty for jobs with many remaining ops.
    """
    if _clock.time() > _KILL:
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0

    # Collect features
    rw_list = []
    ro_list = []
    mo_list = []
    min_dur_list = []

    # Track per-machine demand among ready ops
    mach_demand = {}
    mach_dur_sum = {}

    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        mo = op.get("machine_options", {})
        rw_list.append(rw)
        ro_list.append(ro)
        mo_list.append(mo)

        if mo:
            bd = min(mo.values())
            min_dur_list.append(bd)
            for m, d in mo.items():
                mach_demand[m] = mach_demand.get(m, 0) + 1
                mach_dur_sum[m] = mach_dur_sum.get(m, 0) + d
        else:
            min_dur_list.append(0)

    if n == 1:
        mo = mo_list[0]
        if mo:
            best_m = min(mo, key=mo.get)
            return (0, best_m)
        return 0

    # Phase detection: smooth measure of how "late" we are
    avg_rw = sum(rw_list) / n
    max_rw = max(rw_list) if rw_list else 1
    # phase in [0,1]: 0 = early (lots of work left), 1 = late (little work left)
    # Use ratio of avg to max as a smooth proxy
    phase = 1.0 - (avg_rw / (max_rw + 1e-9)) if max_rw > 0 else 0.5
    # Clamp to [0.1, 0.9] to avoid extreme weights
    phase = max(0.1, min(0.9, phase))

    # Global load estimate for normalization
    total_load = sum(mach_dur_sum.values()) if mach_dur_sum else 1
    if total_load == 0:
        total_load = 1

    scores = []
    chosen_machines = []

    for i in range(n):
        rw = rw_list[i]
        ro = ro_list[i]
        mo = mo_list[i]

        if not mo:
            chosen_machines.append(None)
            scores.append(0.0)
            continue

        # --- Machine selection ---
        # Use a smooth soft-min over machine costs to pick the best machine.
        # Cost = duration * (1 + congestion_weight) + load_balance_penalty
        # Congestion weight grows smoothly with how many ready ops target same machine
        best_m = None
        best_cost = float("inf")
        best_dur = float("inf")

        for m, d in mo.items():
            cnt = mach_demand.get(m, 0)
            # Smooth congestion: use 1 - exp(-alpha * cnt) which saturates nicely
            cong_factor = 1.0 - math.exp(-0.6 * cnt)
            # Load balance: penalize machines with high ready-work demand
            load_frac = mach_dur_sum.get(m, 0) / total_load
            load_pen = 0.2 * load_frac
            # Effective cost: duration inflated by congestion, plus load penalty
            cost = d * (1.0 + 0.5 * cong_factor) + load_pen

            if cost < best_cost:
                best_cost = cost
                best_m = m
                best_dur = d

        chosen_machines.append(best_m)

        # --- Operation scoring ---
        # Urgency: remaining work scaled by inverse processing time
        urgency = rw / (best_dur + 0.5)

        # Criticality: jobs with many remaining ops benefit from early start
        # Use a smooth increasing function
        crit_bonus = 1.0 + 0.2 * (1.0 - math.exp(-0.3 * ro))

        # Phase-aware weighting:
        # Early phase (phase near 0): prioritize long remaining work (LRPT-like)
        # Late phase (phase near 1): prioritize short durations to clear tails
        # Smooth transition using phase as a blend parameter
        short_dur_bonus = 1.0 + 0.4 * max(0.0, 1.0 - best_dur / (avg_rw + 1.0))
        long_work_bonus = 1.0 + 0.3 * (rw / (avg_rw + 1.0))

        # Blend: early -> long_work_bonus, late -> short_dur_bonus
        phase_w = phase  # 0=early, 1=late
        blend_bonus = (1.0 - phase_w) * long_work_bonus + phase_w * short_dur_bonus

        # Congestion penalty on the chosen machine (smooth)
        cnt = mach_demand.get(best_m, 0)
        cont_pen = math.exp(-0.4 * cnt)

        score = urgency * crit_bonus * blend_bonus * cont_pen
        scores.append(score)

    # Pick best
    best_idx = 0
    best_s = scores[0]
    for i in range(1, n):
        if scores[i] > best_s:
            best_s = scores[i]
            best_idx = i

    bm = chosen_machines[best_idx]
    if bm is not None:
        return (best_idx, bm)
    return best_idx
# EVOLVE-BLOCK-END
