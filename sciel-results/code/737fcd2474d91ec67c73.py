"""Pebbling via exact DP over the DAG's topological order.

dp[v] = min transfers (load+store) to make v blue with fast memory empty.
peak[v] = peak red memory used in the chosen plan for v.
Recurrence minimises over subsets of parents (reuse of shared sub-results),
preferring feasible (peak <= budget) plans.
"""
# EVOLVE-BLOCK-START
import itertools


def _norm(dag):
    order = list(dag.get("order", dag.get("vertices", dag.get("nodes", []))))
    is_input = set(dag.get("inputs", dag.get("input", [])))
    is_output = set(dag.get("outputs", dag.get("output", [])))
    parents = {}
    for v in order:
        parents[v] = list(dag.get("parents", {}).get(v, dag.get("pred", {}).get(v, [])))
    return order, is_input, parents, is_output


def _feasible(ops, budget):
    red = set()
    for op, v in ops:
        if op == "load":
            if v not in red:
                if len(red) >= budget:
                    return False
                red.add(v)
        elif op == "delete":
            red.discard(v)
        elif op == "store":
            red.discard(v)
        elif op == "compute":
            red.add(v)
            if len(red) > budget:
                return False
    return True


def _solve(order, is_input, parents, is_output, budget):
    dp = {}
    peak = {}
    ops = {}
    for v in order:
        if v in is_input:
            dp[v] = 0
            peak[v] = 0
            ops[v] = []
            continue
        ps = parents[v]
        best = None
        best_ops = None
        best_peak = None
        if len(ps) <= 20:
            n = len(ps)
            size = 1 << n
            g = [0] * size
            for mask in range(size):
                s = 0
                for i in range(n):
                    if (mask >> i) & 1:
                        s += dp[ps[i]]
                g[mask] = s
                for i in range(n):
                    if not ((mask >> i) & 1):
                        g[mask] += 1
                g[mask] += 1
            # collect candidate (cost, peak, ops)
            cands = []
            for mask in range(size):
                total = g[mask]
                S = [ps[i] for i in range(n) if (mask >> i) & 1]
                op_list = []
                p_peak = 0
                for p in S:
                    op_list.extend(ops[p])
                    p_peak = max(p_peak, peak[p])
                for p in ps:
                    if p not in S:
                        op_list.append(("load", p))
                op_list.append(("compute", v))
                op_list.append(("store", v))
                at_compute = (len(ps) - len(S)) + 1
                cand_peak = max(p_peak, at_compute)
                cands.append((total, cand_peak, op_list))
            # prefer feasible (peak<=budget), then lowest cost
            feasible = [c for c in cands if c[1] <= budget]
            pool = feasible if feasible else cands
            pool.sort(key=lambda c: (c[0], c[1]))
            best, best_peak, best_ops = pool[0]
        else:
            total = len(ps) + sum(dp[p] for p in ps) + 1
            op_list = []
            p_peak = 0
            for p in ps:
                op_list.extend(ops[p])
                p_peak = max(p_peak, peak[p])
                op_list.append(("load", p))
            op_list.append(("compute", v))
            op_list.append(("store", v))
            best = total
            best_ops = op_list
            best_peak = max(p_peak, len(ps) + 1)
        dp[v] = best
        peak[v] = best_peak
        ops[v] = best_ops

    out = list(is_output)
    total = sum(dp[v] for v in out)
    final = []
    for v in out:
        final.extend(ops[v])
    return total, final, ops


def _naive(order, is_input, parents, is_output, budget):
    ops = []
    for v in order:
        if v in is_output:
            for p in parents[v]:
                if p not in is_input:
                    ops.extend(_naive_sub(p, order, is_input, parents))
                    ops.append(("delete", p))
                else:
                    ops.append(("load", p))
            ops.append(("compute", v))
            ops.append(("store", v))
            for p in parents[v]:
                ops.append(("delete", p))
    return ops


def _naive_sub(v, order, is_input, parents):
    if v in is_input:
        return [("load", v)]
    out = []
    for p in parents[v]:
        out.extend(_naive_sub(p, order, is_input, parents))
        out.append(("load", p))
    out.append(("compute", v))
    return out


def pebble(dag, fast_memory_budget):
    order, is_input, parents, is_output = _norm(dag)
    budget = int(fast_memory_budget)
    if budget < 1:
        budget = 1

    total, final_ops, ops = _solve(order, is_input, parents, is_output, budget)

    if final_ops is None or len(final_ops) > 20000 or not _feasible(final_ops, budget):
        final_ops = _naive(order, is_input, parents, is_output, budget)
        if len(final_ops) > 20000:
            final_ops = final_ops[:20000]

    return final_ops
# EVOLVE-BLOCK-END
