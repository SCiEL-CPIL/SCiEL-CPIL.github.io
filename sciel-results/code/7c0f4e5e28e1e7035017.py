"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _clock
import math

_T0 = _clock.time()
_LIMIT = 135.0


def _elapsed():
    return _clock.time() - _T0


def _logsumexp_soft_min(opts):
    """Smooth min via log-sum-exp (negative trick) with temperature."""
    if not opts:
        return 0.0
    vals = list(opts.values())
    if len(vals) == 1:
        return vals[0]
    # soft min = -logsumexp(-x_i / T) * T
    T = 2.0
    negs = [-v / T for v in vals]
    m = max(negs)
    if m == float('-inf'):
        return 0.0
    s = sum(math.exp(n - m) for n in negs)
    return -(m + math.log(s)) * T


def _logsumexp_soft_max(opts):
    """Smooth max via log-sum-exp with temperature."""
    if not opts:
        return 0.0
    vals = list(opts.values())
    if len(vals) == 1:
        return vals[0]
    T = 2.0
    scaled = [v / T for v in vals]
    m = max(scaled)
    if m == float('-inf'):
        return 0.0
    s = sum(math.exp(v - m) for v in scaled)
    return (m + math.log(s)) * T


def dispatch(ready_operations, time):
    if _elapsed() > _LIMIT:
        return 0
    n_ops = len(ready_operations)
    if n_ops <= 1:
        return 0

    def _priority(op, idx):
        rem = op["remaining_work"]
        opts = op["machine_options"]
        
        # Smooth min and max using log-sum-exp
        s_min = _logsumexp_soft_min(opts)
        s_max = _logsumexp_soft_max(opts)
        
        # Flexibility: number of machine options
        flexibility = len(opts) if opts else 0
        
        # Remaining ops count
        rem_ops = op.get("remaining_ops", 0)
        
        # Machine concentration: how concentrated is the workload across machines
        # Higher concentration = less flexible = worse
        if opts:
            total_dur = sum(opts.values())
            if total_dur > 0:
                # Normalized entropy-like measure: higher = more spread out
                probs = [v / total_dur for v in opts.values()]
                entropy = -sum(p * math.log(p + 1e-15) for p in probs)
                # Normalize by log of max possible
                max_entropy = math.log(len(opts)) if len(opts) > 1 else 1.0
                spread = entropy / max_entropy if max_entropy > 0 else 0.0
            else:
                spread = 0.0
        else:
            spread = 0.0
        
        # Composite score:
        # 1. Maximize remaining work (LRPT) - primary
        # 2. Minimize soft-min duration (prefer ops that can be done fast somewhere)
        # 3. Maximize spread (workload distributed across machines = less bottleneck)
        # 4. Maximize flexibility (more machine options = less bottlenecks)
        # 5. Maximize soft-max (long tasks shouldn't be starved)
        # 6. Minimize remaining ops (finish jobs that are almost done)
        # 7. Tie-break by index
        score = (
            rem,              # maximize
            -s_min,           # minimize soft-min (i.e., maximize -s_min)
            spread,           # maximize spread
            flexibility,      # maximize
            s_max,            # maximize
            -rem_ops,         # minimize
            -idx              # tie-break
        )
        return score

    ranked = [(_priority(op, i), i) for i, op in enumerate(ready_operations)]
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1]
# EVOLVE-BLOCK-END
