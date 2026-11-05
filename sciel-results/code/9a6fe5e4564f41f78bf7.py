"""Rewritten dispatch rule for job-shop scheduling (event-driven)."""
# EVOLVE-BLOCK-START
import time as _time


def dispatch(ready_operations, time):
    """Pick the ready operation to start next.

    Strategy: NSPT-like score combining machine-freeing (short processing time)
    with job-progress guarantee (remaining work). We scale the minimum available
    duration by a factor derived from remaining work, so that:
      - Among operations with similar job urgency, shorter processing times are
        preferred (frees machines sooner).
      - Jobs with more remaining work get a slight boost to avoid starvation.

    Score per ready operation i:
        min_dur_i = min(duration over machine_options)
        rem_i     = remaining_work[i]

    We use a composite score:
        score_i = min_dur_i * (1 + alpha * (rem_i / max_rem))
    and pick the operation with the SMALLEST score (prefer short ops, but
    penalize long-remaining jobs slightly less than short-remaining ones).

    In practice, we want:
      - Prefer short min_dur (SPT benefit).
      - Among comparable min_dur, prefer jobs with MORE remaining work
        (MWRT benefit to avoid starvation).

    So we sort by (min_dur, -remaining_work) primarily, but use a weighted
    scalar to keep it a single comparison. We'll use:
        score_i = min_dur_i - beta * (rem_i / max_rem)
    and pick the smallest score. A small beta ensures SPT dominates, but
    breaks ties in favor of jobs with more remaining work.

    We set beta adaptively: beta = 0.1 * max(min_dur) so the tie-break is
    meaningful but doesn't override the SPT preference.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features
    min_durs = []
    rem_works = []
    for op in ready_operations:
        mo = op.get("machine_options", {})
        if mo:
            md = min(mo.values())
        else:
            md = 1.0
        min_durs.append(md)
        rem_works.append(op.get("remaining_work", 0))

    max_rem = max(rem_works) if rem_works else 1.0
    max_md = max(min_durs) if min_durs else 1.0

    # Adaptive beta: small enough that SPT dominates, but provides tie-breaking
    # in favor of jobs with more remaining work.
    beta = 0.05 * max_md

    best_idx = 0
    best_score = float('inf')
    for i in range(n):
        # Normalize remaining work to [0, 1]
        norm_rem = rem_works[i] / max_rem if max_rem > 0 else 0.0
        # Score: lower is better. SPT component minus a small bonus for more remaining work.
        score = min_durs[i] - beta * norm_rem
        if score < best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
