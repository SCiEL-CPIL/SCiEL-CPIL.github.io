"""Job-shop dispatch rule: SEARCH approach.
At import time we generate many candidate dispatch heuristics (varied weights/rules),
evaluate each on small synthetic job-shop instances via an event-driven simulation,
and keep the best one within a wall-clock budget. dispatch() then just applies the
chosen rule with a cheap deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
import random
import heapq

_DEADLINE = 130.0        # Safety margin before 150s kill
_START_TIME = _time.time()
_BUDGET_FOR_SEARCH = 115.0  # leave margin for evaluation overhead

# ---------------------------------------------------------------------------
# Candidate rule definitions.
# Each rule is a function of (op, time, n_ops) -> score (higher = more urgent).
# op has keys: job, op, machine_options {machine: duration}, remaining_ops, remaining_work
# ---------------------------------------------------------------------------
def _min_dur(op):
    return min(op["machine_options"].values())

def _mean_dur(op):
    v = op["machine_options"].values()
    return sum(v) / len(v)

def _max_dur(op):
    return max(op["machine_options"].values())

def _num_machines(op):
    return len(op["machine_options"])

def _remaining_work(op):
    return op["remaining_work"]

def _remaining_ops(op):
    return op["remaining_ops"]

def _avg_remaining(op):
    ro = op["remaining_ops"]
    return op["remaining_work"] / ro if ro > 0 else op["remaining_work"]

# Rule families with tunable weights (a, b, c, d, e):
# score = a*eff + b*remaining_work - c*remaining_ops + d*mean_duration + e*num_machines
# where eff = remaining_work / min_duration  (efficiency)

def make_rule(a, b, c, d, e):
    def rule(op, time, n_ops):
        md = _min_dur(op)
        eff = _remaining_work(op) / md if md > 0 else _remaining_work(op) + 1000.0
        return (a * eff + b * _remaining_work(op) - c * _remaining_ops(op)
                + d * _mean_dur(op) + e * _num_machines(op))
    return rule

# SPT rule: minimize min duration -> score = -min_duration
def rule_spt(op, time, n_ops):
    return -_min_dur(op)

# Longest remaining work
def rule_lrw(op, time, n_ops):
    return _remaining_work(op)

# Most remaining ops
def rule_mro(op, time, n_ops):
    return _remaining_ops(op)

# Fewest remaining ops
def rule_fro(op, time, n_ops):
    return -_remaining_ops(op)

# Min remaining work
def rule_mrw(op, time, n_ops):
    return -_remaining_work(op)

# Mean duration (SPT on mean)
def rule_spt_mean(op, time, n_ops):
    return -_mean_dur(op)

# Ratio: remaining_work / num_machines
def rule_ratio(op, time, n_ops):
    nm = _num_machines(op)
    return _remaining_work(op) / nm if nm > 0 else _remaining_work(op)

# ATC: Apparent Tardiness Cost - prioritize jobs with more remaining work relative to time
def rule_atc(op, time, n_ops):
    rw = _remaining_work(op)
    md = _min_dur(op)
    if md <= 0:
        return rw
    # Higher score for ops with more remaining work that can be done quickly
    return rw / (md + 1.0)

# CR: Critical Ratio - remaining_work / (time_limit - current_time)
# We don't have a time limit, so use a normalized version
def rule_cr(op, time, n_ops):
    rw = _remaining_work(op)
    ro = _remaining_ops(op)
    if ro <= 0:
        return rw
    avg_remaining = rw / ro
    # Prioritize jobs where remaining work is large relative to average
    return rw - avg_remaining * 0.5

# EDD-like: prioritize jobs with fewer remaining ops (closer to completion)
def rule_edd(op, time, n_ops):
    return -_remaining_ops(op) * 0.1 - _min_dur(op) * 0.1

# Composite: combine SPT with remaining work
def rule_spt_rw(op, time, n_ops):
    return -_min_dur(op) * 0.5 + _remaining_work(op) * 0.5

# Machine utilization aware: prefer ops on machines with fewer options (bottleneck)
def rule_bottleneck(op, time, n_ops):
    nm = _num_machines(op)
    md = _min_dur(op)
    # Fewer machine options = more constrained = higher priority
    return (1.0 / (nm + 1.0)) * _remaining_work(op) - md * 0.1

# Machine-contention avoidance: score by ratio of min duration to total duration
# across all available machines. Higher ratio means the op can be done quickly
# relative to the total workload, reducing contention.
def rule_contention(op, time, n_ops):
    mo = op["machine_options"]
    if not mo:
        return 0.0
    md = min(mo.values())
    total = sum(mo.values())
    if total <= 0:
        return 0.0
    # Inverse contention: higher when min is large relative to total (concentrated)
    # But we want to avoid contention, so prefer ops where min is small relative to total
    # (spread out, less likely to block). Actually, let's think: if min/total is high,
    # the op is concentrated on one machine -> more contention. We want to AVOID that.
    # So score should be LOW for high concentration. Use 1 - (md/total) or similar.
    # But also consider remaining work: jobs with more remaining work should be prioritized.
    rw = _remaining_work(op)
    concentration = md / total
    # Lower concentration is better for avoiding contention
    return rw * (1.0 - concentration) + (1.0 / (md + 1.0)) * 0.1

# Soft-min based rule: use log-sum-exp style soft-min for duration
def rule_soft_min(op, time, n_ops):
    mo = op["machine_options"]
    if not mo:
        return 0.0
    vals = list(mo.values())
    if len(vals) == 1:
        md = vals[0]
    else:
        # Soft-min: -log(sum(exp(-v/tau))) * tau, with small tau
        tau = 0.5
        min_val = min(vals)
        # Numerically stable soft-min
        shifted = [v - min_val for v in vals]
        log_sum = np.log(np.sum(np.exp(-np.array(shifted) / tau)))
        md = min_val + tau * log_sum
    rw = _remaining_work(op)
    return rw / (md + 0.1)

# Hybrid contention-remaining: combine contention score with remaining ops
def rule_hybrid_contention(op, time, n_ops):
    mo = op["machine_options"]
    if not mo:
        return 0.0
    md = min(mo.values())
    total = sum(mo.values())
    rw = _remaining_work(op)
    ro = _remaining_ops(op)
    concentration = md / total if total > 0 else 0.0
    # Prioritize: more remaining work, lower concentration, fewer remaining ops
    return rw * (1.0 - concentration) - 0.05 * ro + 0.1 / (md + 1.0)

# ---------------------------------------------------------------------------
# Event-driven job-shop simulator.
# ready_operations: list of dicts for the ready ops at each event.
# The simulator maintains, for each job, its operation sequence and machine
# availability, and uses the dispatch rule to pick which ready op to start.
# ---------------------------------------------------------------------------
class JobShopInstance:
    def __init__(self, jobs_ops):
        # jobs_ops: list of jobs, each a list of (machine, duration)
        self.jobs = jobs_ops
        self.n_jobs = len(jobs_ops)
        self.machines = set()
        for job in jobs_ops:
            for m, d in job:
                self.machines.add(m)
        self.machines = sorted(self.machines)
        self.n_machines = len(self.machines)

    def total_work(self):
        return sum(d for job in self.jobs for _, d in job)

def simulate(instance, rule, max_steps=100000):
    """Run event-driven simulation. Returns makespan."""
    n_jobs = instance.n_jobs
    n_machines = instance.n_machines

    # job[i] = (current_op_index, [ops])
    job_cur_op = [0] * n_jobs
    job_remaining_ops = [len(instance.jobs[i]) for i in range(n_jobs)]
    job_remaining_work = [sum(d for _, d in instance.jobs[i]) for i in range(n_jobs)]

    machine_avail = [0] * n_machines
    machine_to_idx = {m: idx for idx, m in enumerate(instance.machines)}

    # Track completed ops per job to build ready_operations dynamically
    completed = [0] * n_jobs

    # We'll simulate by repeatedly: find all ready ops, dispatch picks one,
    # schedule it on its (single) machine, update availability, repeat.
    # In a 1-machine-per-op job-shop, each op has exactly one machine option.
    # But the interface allows machine_options dict; for generated instances we
    # use single machine per op (classic job-shop). The dispatch receives
    # machine_options as {machine: duration}.

    t = 0.0
    finished = 0
    steps = 0
    max_t = 0.0

    while finished < n_jobs and steps < max_steps:
        steps += 1
        # Build ready operations list
        ready = []
        for i in range(n_jobs):
            if job_cur_op[i] < len(instance.jobs[i]):
                m, d = instance.jobs[i][job_cur_op[i]]
                ready.append({
                    "job": i,
                    "op": job_cur_op[i],
                    "machine_options": {m: d},
                    "remaining_ops": job_remaining_ops[i],
                    "remaining_work": job_remaining_work[i],
                })
        if not ready:
            break

        # Dispatch
        if len(ready) == 1:
            idx = 0
        else:
            scores = [rule(op, t, len(ready)) for op in ready]
            idx = int(np.argmax(scores))

        op = ready[idx]
        i = op["job"]
        m, d = instance.jobs[i][job_cur_op[i]]
        m_idx = machine_to_idx[m]

        start = max(t, machine_avail[m_idx])
        end = start + d
        machine_avail[m_idx] = end

        # Update job state
        job_cur_op[i] += 1
        job_remaining_ops[i] -= 1
        job_remaining_work[i] -= d
        completed[i] += 1

        t = end
        if t > max_t:
            max_t = t
        if job_cur_op[i] >= len(instance.jobs[i]):
            finished += 1

    return max_t

# ---------------------------------------------------------------------------
# Instance generator
# ---------------------------------------------------------------------------
def generate_instance(n_jobs=6, n_machines=4, seed=0):
    rng = random.Random(seed)
    jobs = []
    for j in range(n_jobs):
        ops = []
        # Each job visits a random permutation of machines (classic job-shop)
        ms = list(range(n_machines))
        rng.shuffle(ms)
        for m in ms:
            d = rng.randint(1, 10)
            ops.append((m, d))
        jobs.append(ops)
    return JobShopInstance(jobs)

# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------
def generate_candidates():
    candidates = []
    # Base rules
    candidates.append(("spt", rule_spt))
    candidates.append(("lrw", rule_lrw))
    candidates.append(("mro", rule_mro))
    candidates.append(("fro", rule_fro))
    candidates.append(("mrw", rule_mrw))
    candidates.append(("spt_mean", rule_spt_mean))
    candidates.append(("ratio", rule_ratio))
    candidates.append(("atc", rule_atc))
    candidates.append(("cr", rule_cr))
    candidates.append(("edd", rule_edd))
    candidates.append(("spt_rw", rule_spt_rw))
    candidates.append(("bottleneck", rule_bottleneck))
    candidates.append(("contention", rule_contention))
    candidates.append(("soft_min", rule_soft_min))
    candidates.append(("hybrid_contention", rule_hybrid_contention))

    # Additional weighted combinations focusing on contention-aware features
    # score = a*rw*(1-md/total) + b/rw_eff + c*(-ro) + d*(1/nm)
    for a in [0.5, 1.0, 1.5]:
        for b in [0.0, 0.1, 0.2]:
            for c in [0.0, 0.05]:
                for d in [0.0, 0.1]:
                    def make_contention_rule(a=a, b=b, c=c, d=d):
                        def rule(op, time, n_ops):
                            mo = op["machine_options"]
                            if not mo:
                                return 0.0
                            md = min(mo.values())
                            total = sum(mo.values())
                            rw = _remaining_work(op)
                            ro = _remaining_ops(op)
                            nm = len(mo)
                            concentration = md / total if total > 0 else 0.0
                            return (a * rw * (1.0 - concentration)
                                    + b * (1.0 / (md + 1.0))
                                    - c * ro
                                    + d * (1.0 / (nm + 1.0)))
                        return rule
                    rule = make_contention_rule()
                    name = f"cont_{a}_{b}_{c}_{d}"
                    candidates.append((name, rule))

    # Weighted hybrid rules
    # a: efficiency weight, b: remaining_work weight, c: remaining_ops weight,
    # d: mean_duration weight, e: num_machines weight
    grid_a = [0.0, 0.5, 1.0, 2.0]
    grid_b = [0.0, 0.1, 0.5]
    grid_c = [0.0, 0.05, 0.1]
    grid_d = [-0.1, 0.0, 0.1]
    grid_e = [-0.05, 0.0, 0.05]

    for a in grid_a:
        for b in grid_b:
            for c in grid_c:
                for d in grid_d:
                    for e in grid_e:
                        rule = make_rule(a, b, c, d, e)
                        name = f"hyb_{a}_{b}_{c}_{d}_{e}"
                        candidates.append((name, rule))

    # Additional: combine SPT with remaining work in finer grid
    for spt_w in [0.3, 0.5, 0.7, 0.9]:
        for rw_w in [0.1, 0.3, 0.5, 0.7]:
            def make_spt_rw_rule(spt_w=spt_w, rw_w=rw_w):
                def rule(op, time, n_ops):
                    return -spt_w * _min_dur(op) + rw_w * _remaining_work(op)
                return rule
            rule = make_spt_rw_rule()
            name = f"sptrw_{spt_w}_{rw_w}"
            candidates.append((name, rule))

    return candidates

# ---------------------------------------------------------------------------
# Tournament: evaluate each candidate on a set of instances, pick best
# ---------------------------------------------------------------------------
def run_tournament():
    candidates = generate_candidates()
    rng = random.Random(42)
    # Generate test instances closer to actual evaluation size (10 jobs, 5 machines)
    # Use a mix of sizes for robustness, but weight primary instances more heavily
    test_instances = []
    # Primary: 10 jobs, 5 machines (matches evaluation) - more instances
    for s in range(8):
        test_instances.append(generate_instance(n_jobs=10, n_machines=5, seed=s))
    # Secondary: smaller instances for diversity
    for s in range(3):
        test_instances.append(generate_instance(n_jobs=8, n_machines=5, seed=s + 100))
    for s in range(2):
        test_instances.append(generate_instance(n_jobs=6, n_machines=4, seed=s + 200))

    results = []
    for name, rule in candidates:
        if _time.time() - _START_TIME > _BUDGET_FOR_SEARCH:
            break
        total = 0.0
        for inst in test_instances:
            ms = simulate(inst, rule)
            total += ms
        mean_ms = total / len(test_instances)
        results.append((mean_ms, name, rule))

    if not results:
        return rule_spt  # fallback
    results.sort(key=lambda x: x[0])
    best_rule = results[0][2]
    return best_rule

# ---------------------------------------------------------------------------
# Run search at import time
# ---------------------------------------------------------------------------
_best_rule = None
if _time.time() - _START_TIME < _BUDGET_FOR_SEARCH:
    try:
        _best_rule = run_tournament()
    except Exception:
        _best_rule = None
if _best_rule is None:
    _best_rule = rule_spt  # safe fallback

# ---------------------------------------------------------------------------
# Public dispatch function
# ---------------------------------------------------------------------------
def dispatch(ready_operations, time):
    """Dispatch rule for job-shop scheduling.
    Uses the best rule found by the import-time tournament search."""
    if not ready_operations:
        return 0
    # Deadline guard
    if _time.time() - _START_TIME > _DEADLINE:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Compute scores using the chosen rule
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = _best_rule(ready_operations[i], time, n)

    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END
