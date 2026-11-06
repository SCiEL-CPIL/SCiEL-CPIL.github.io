# EVOLVE-BLOCK-START
import random
import time
from collections import defaultdict


def _build_masks(clauses, n):
    pos = []
    neg = []
    for c in clauses:
        p = 0
        q = 0
        for l in c:
            if l > 0:
                p |= (1 << (l - 1))
            else:
                q |= (1 << (-l - 1))
        pos.append(p)
        neg.append(q)
    return pos, neg


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 295.0  # safety margin before 330s kill
    pos, neg = _build_masks(clauses, n)
    pos = tuple(pos)
    neg = tuple(neg)

    # Precompute clause literal index lists for fast flip
    clause_vars = []
    for c in clauses:
        clause_vars.append([(abs(l) - 1, l > 0) for l in c])

    # Build variable occurrence masks: for each variable, which clauses contain it positively/negatively
    var_pos_clauses = [0] * n  # bit i set in var_pos_clauses[v] means clause i has literal +v
    var_neg_clauses = [0] * n  # bit i set in var_neg_clauses[v] means clause i has literal -v
    for i, c in enumerate(clauses):
        for l in c:
            if l > 0:
                var_pos_clauses[l-1] |= (1 << i)
            else:
                var_neg_clauses[-l-1] |= (1 << i)

    best_assign = None
    best_sat = -1

    def sat_count_fast(v):
        """Count satisfied clauses using bit operations."""
        # A clause i is satisfied if:
        # - some positive literal is true: (v & pos[i]) != 0
        # - some negative literal is false: ((~v) & neg[i]) != 0
        cnt = 0
        for i in range(m):
            if (v & pos[i]) or ((~v) & neg[i]):
                cnt += 1
        return cnt

    def assign_to_int(assign):
        v = 0
        for i, b in enumerate(assign):
            if b:
                v |= (1 << i)
        return v

    def int_to_assign(v):
        return [bool(v & (1 << i)) for i in range(n)]

    def compute_unsat_mask(v):
        """Return a bitmask of unsatisfied clauses."""
        unsat_mask = 0
        for i in range(m):
            if not ((v & pos[i]) or ((~v) & neg[i])):
                unsat_mask |= (1 << i)
        return unsat_mask

    def get_unsat_indices(unsat_mask):
        """Convert unsat_mask to list of clause indices."""
        indices = []
        while unsat_mask:
            lsb = unsat_mask & (-unsat_mask)
            idx = lsb.bit_length() - 1
            indices.append(idx)
            unsat_mask ^= lsb
        return indices

    rng = random.Random(12345)

    # Phase 0: quick pure-literal / unit-style greedy initializations
    for attempt_seed in range(8):
        r = random.Random(attempt_seed)
        assign = [r.random() < 0.5 for _ in range(n)]
        v = assign_to_int(assign)
        s = sat_count_fast(v)
        if s > best_sat:
            best_sat = s
            best_assign = assign
        if s == m:
            return assign
        if time.time() > deadline:
            return int_to_assign(assign_to_int(best_assign)) if best_assign else [False]*n

    # Phase 1: Multi-restart WalkSAT with improved local search
    for restart in range(4000):
        if time.time() > deadline:
            break
        r = random.Random(restart)
        assign = [r.random() < 0.5 for _ in range(n)]
        v = assign_to_int(assign)
        
        # Track current unsat mask
        unsat_mask = compute_unsat_mask(v)
        sat_count = m - bin(unsat_mask).count('1')
        
        steps_budget = 4000
        for step in range(steps_budget):
            if time.time() > deadline:
                break
            
            if unsat_mask == 0:
                best_assign = int_to_assign(v)
                best_sat = m
                return best_assign
            
            # Get unsatisfied clause indices
            unsat_indices = get_unsat_indices(unsat_mask)
            
            # Pick a clause: use min-conflicts heuristic
            # Sample up to 8 unsat clauses for speed
            if len(unsat_indices) > 8:
                sample = r.sample(unsat_indices, 8)
            else:
                sample = unsat_indices
            
            # For each sampled clause, compute help count
            best_ci = None
            best_help = -1
            for ci in sample:
                help_cnt = 0
                for var, is_pos in clause_vars[ci]:
                    cur = bool(v & (1 << var))
                    if is_pos and not cur:
                        help_cnt += 1
                    elif (not is_pos) and cur:
                        help_cnt += 1
                if help_cnt > best_help:
                    best_help = help_cnt
                    best_ci = ci
            
            if best_ci is None:
                best_ci = sample[0]
            
            # Pick literal: use weighted random selection
            cv = clause_vars[best_ci]
            
            # Compute weights: prefer literals that help
            weights = []
            total_weight = 0
            for var, is_pos in cv:
                cur = bool(v & (1 << var))
                if (is_pos and not cur) or ((not is_pos) and cur):
                    w = 3.0  # helps
                else:
                    w = 1.0  # hurts
                weights.append(w)
                total_weight += w
            
            # Weighted random selection
            r_val = r.random() * total_weight
            cumulative = 0
            lit_idx = 0
            for k, w in enumerate(weights):
                cumulative += w
                if r_val <= cumulative:
                    lit_idx = k
                    break
            
            var, is_pos = cv[lit_idx]
            bit = 1 << var
            
            # Update v
            old_v = v
            v ^= bit
            
            # Update unsat mask efficiently
            # Clauses that might change: those containing this variable
            affected_mask = var_pos_clauses[var] | var_neg_clauses[var]
            
            # Recompute satisfaction for affected clauses only
            new_unsat_mask = unsat_mask
            for ci in range(m):
                if affected_mask & (1 << ci):
                    # Check if clause ci is now satisfied
                    if (v & pos[ci]) or ((~v) & neg[ci]):
                        new_unsat_mask &= ~(1 << ci)
                    else:
                        new_unsat_mask |= (1 << ci)
            
            unsat_mask = new_unsat_mask
            sat_count = m - bin(unsat_mask).count('1')
            
            # Track best
            if sat_count > best_sat:
                best_sat = sat_count
                best_assign = int_to_assign(v)
            
            if best_sat == m:
                return best_assign
        
        # after inner loop, track best
        if best_sat == m:
            return best_assign
        if time.time() > deadline:
            break

    if best_assign is None:
        return [False] * n
    return best_assign
# EVOLVE-BLOCK-END
