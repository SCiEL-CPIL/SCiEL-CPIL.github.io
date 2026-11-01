"""Combined covering-code constructor: bitmask greedy + pruning + simulated annealing.

Strategy:
  * Bitmask-based sphere representation for fast coverage tracking (inspiration 2).
  * Multiple greedy restarts with random ordering for diversity.
  * Pruning pass to remove redundant codewords.
  * Simulated annealing refinement: binary state vector, minimize |uncovered| + |code|.
  * Always-valid baseline as guaranteed fallback.
"""

import time
import itertools
import random
import numpy as np

_DEADLINE = 140.0


def _direct_sum(q, n, radius):
    """Always-valid baseline: direct sum of radius-1 perfect code with remaining coords."""
    base = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    return [h + list(tail) for h in base
            for tail in itertools.product(range(q), repeat=n - 4)]


def _covers_all(cw, q, n, radius):
    """Exact coverage check over the whole space."""
    total = q ** n
    if total > 10**7:
        return False
    covered = np.zeros(total, dtype=bool)
    for c in cw:
        for d in range(radius + 1):
            for pos in itertools.combinations(range(n), d):
                for vals in itertools.product(range(1, q), repeat=d):
                    w = list(c)
                    for p, v in zip(pos, vals):
                        w[p] = v
                    idx = 0
                    for x in w:
                        idx = idx * q + x
                    covered[idx] = True
    return covered.all()


def _build_spheres_bitmask(q, n, radius):
    """Build bitmask spheres for all words. Returns list of integers (bitmasks)."""
    total = q ** n
    powers = [q ** i for i in range(n)]
    spheres = [0] * total
    for x in range(total):
        mask = 1 << x
        if radius >= 1:
            for i in range(n):
                digit = (x // powers[i]) % q
                for val in range(q):
                    if val == digit:
                        continue
                    nb = x + (val - digit) * powers[i]
                    mask |= (1 << nb)
        spheres[x] = mask
    return spheres


def _prune_bitmask(code_indices, spheres, full_mask):
    """Remove redundant codewords using coverage counts."""
    current = list(code_indices)
    if not current:
        return current
    
    # Build coverage count: how many spheres cover each bit
    # We use the bitmask approach: for each bit in full_mask, count coverage
    # Efficient: iterate and track
    n_spheres = len(spheres)
    changed = True
    while changed:
        changed = False
        # Compute coverage counts
        cov_count = np.zeros(1, dtype=np.int32)  # placeholder, we'll use bitmask
        # Actually for bitmask approach, check if removing codeword i still covers all
        for i in range(len(current)):
            union_others = 0
            for j in range(len(current)):
                if i == j:
                    continue
                union_others |= spheres[current[j]]
            if union_others == full_mask:
                current.pop(i)
                changed = True
                break
    return current


def _greedy_bitmask(spheres, full_mask, time_budget, start_time, random_order=None):
    """Greedy set-cover using bitmask operations."""
    if random_order is not None:
        candidates = list(range(len(spheres)))
        random.shuffle(candidates)
    else:
        candidates = list(range(len(spheres)))
    
    covered = 0
    chosen = []
    while covered != full_mask:
        if time.time() - start_time > time_budget:
            break
        best_gain = -1
        best_idx = -1
        for idx in candidates:
            new_covered = spheres[idx] & ~covered
            gain = new_covered.bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_gain <= 0:
            break
        chosen.append(best_idx)
        covered |= spheres[best_idx]
    return chosen


def _simulated_annealing(code_indices, spheres, full_mask, time_budget, start_time):
    """Simulated annealing with incremental coverage tracking.
    
    Objective: minimize |code| + 10 * |uncovered|.
    Uses a coverage-count array for O(1) delta computation.
    """
    total = len(spheres)
    current_code = list(code_indices)
    
    # Build coverage count array: cov[w] = number of codewords whose sphere contains w
    cov = np.zeros(total, dtype=np.int32)
    for idx in current_code:
        mask = spheres[idx]
        # Count bits in mask and add to cov
        # Efficient: iterate set bits
        m = mask
        while m:
            bit = (m & -m).bit_length() - 1
            cov[bit] += 1
            m &= m - 1
    
    covered_count = (cov > 0).sum()
    uncovered_count = total - covered_count
    
    def _objective(code_len, unc):
        return code_len + 10 * unc
    
    current_obj = _objective(len(current_code), uncovered_count)
    best_code = list(current_code)
    best_obj = current_obj
    
    code_set = set(current_code)
    
    t0 = 5.0
    t_end = 0.01
    n_steps = 80000
    deadline = start_time + time_budget
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        progress = step / n_steps
        temperature = t0 * (t_end / t0) ** progress
        
        move_type = random.random()
        
        if move_type < 0.35:
            # Try removing a codeword
            if len(current_code) == 0:
                continue
            idx_to_remove = random.choice(current_code)
            
            # Compute delta: how many words become uncovered?
            # Words only covered by this codeword
            # Need to check: for each bit in spheres[idx_to_remove], if cov[bit] == 1, it becomes uncovered
            mask = spheres[idx_to_remove]
            newly_uncovered = 0
            m = mask
            while m:
                bit = (m & -m).bit_length() - 1
                if cov[bit] == 1:
                    newly_uncovered += 1
                m &= m - 1
            
            delta = 1 + 10 * newly_uncovered  # +1 for code size, +10*newly_uncovered for coverage loss
            # Wait: removing reduces code by 1, increases uncovered by newly_uncovered
            # delta_obj = (len-1 + 10*(unc+newly_uncovered)) - (len + 10*unc) = -1 + 10*newly_uncovered
            delta = -1 + 10 * newly_uncovered
            
            if delta <= 0 or random.random() < np.exp(-delta / max(temperature, 1e-10)):
                # Accept: update cov
                m = mask
                while m:
                    bit = (m & -m).bit_length() - 1
                    cov[bit] -= 1
                    m &= m - 1
                current_code.remove(idx_to_remove)
                code_set.discard(idx_to_remove)
                uncovered_count += newly_uncovered
                current_obj = _objective(len(current_code), uncovered_count)
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_code = list(current_code)
                    
        elif move_type < 0.60:
            # Try adding a random codeword
            new_idx = random.randint(0, total - 1)
            if new_idx in code_set:
                continue
            
            mask = spheres[new_idx]
            # How many newly covered words?
            newly_covered = 0
            m = mask
            while m:
                bit = (m & -m).bit_length() - 1
                if cov[bit] == 0:
                    newly_covered += 1
                m &= m - 1
            
            # delta = +1 (code) - 10 * newly_covered
            delta = 1 - 10 * newly_covered
            
            if delta <= 0 or random.random() < np.exp(-delta / max(temperature, 1e-10)):
                # Accept
                m = mask
                while m:
                    bit = (m & -m).bit_length() - 1
                    cov[bit] += 1
                    m &= m - 1
                current_code.append(new_idx)
                code_set.add(new_idx)
                uncovered_count -= newly_covered
                current_obj = _objective(len(current_code), uncovered_count)
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_code = list(current_code)
                    
        else:
            # Try swapping: remove one, add another
            if len(current_code) == 0:
                continue
            idx_to_remove = random.choice(current_code)
            new_idx = random.randint(0, total - 1)
            if new_idx == idx_to_remove or new_idx in code_set:
                continue
            
            mask_remove = spheres[idx_to_remove]
            mask_add = spheres[new_idx]
            
            # Compute net change in uncovered
            # Words that become uncovered: in mask_remove but not in mask_add, and cov==1
            # Words that become covered: in mask_add but not in mask_remove, and cov==0
            # This is complex; approximate by computing both
            newly_uncovered = 0
            m = mask_remove & ~mask_add
            while m:
                bit = (m & -m).bit_length() - 1
                if cov[bit] == 1:
                    newly_uncovered += 1
                m &= m - 1
            
            newly_covered = 0
            m = mask_add & ~mask_remove
            while m:
                bit = (m & -m).bit_length() - 1
                if cov[bit] == 0:
                    newly_covered += 1
                m &= m - 1
            
            # Code size unchanged, so delta = 10 * (newly_uncovered - newly_covered)
            delta = 10 * (newly_uncovered - newly_covered)
            
            if delta <= 0 or random.random() < np.exp(-delta / max(temperature, 1e-10)):
                # Accept: update cov
                m = mask_remove
                while m:
                    bit = (m & -m).bit_length() - 1
                    cov[bit] -= 1
                    m &= m - 1
                m = mask_add
                while m:
                    bit = (m & -m).bit_length() - 1
                    cov[bit] += 1
                    m &= m - 1
                current_code[current_code.index(idx_to_remove)] = new_idx
                code_set.discard(idx_to_remove)
                code_set.add(new_idx)
                uncovered_count += newly_uncovered - newly_covered
                current_obj = _objective(len(current_code), uncovered_count)
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_code = list(current_code)
    
    # Final local search: try all single swaps to improve
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i, idx_i in enumerate(best_code):
                if time.time() > deadline - 2:
                    break
                mask_i = spheres[idx_i]
                for new_idx in range(total):
                    if new_idx in best_code:
                        continue
                    mask_new = spheres[new_idx]
                    # Quick check: does removing idx_i and adding new_idx help?
                    # Only consider if new_idx covers words that idx_i doesn't
                    if (mask_new & ~mask_i) == 0:
                        continue
                    # Compute net uncovered change
                    # Rebuild cov for best_code without idx_i
                    cov_tmp = np.zeros(total, dtype=np.int32)
                    for idx in best_code:
                        if idx == idx_i:
                            continue
                        m = spheres[idx]
                        while m:
                            bit = (m & -m).bit_length() - 1
                            cov_tmp[bit] += 1
                            m &= m - 1
                    # Check if removing idx_i leaves gaps
                    m = mask_i
                    gaps = 0
                    while m:
                        bit = (m & -m).bit_length() - 1
                        if cov_tmp[bit] == 0:
                            gaps += 1
                        m &= m - 1
                    if gaps == 0:
                        # Can remove idx_i entirely
                        best_code.pop(i)
                        improved = True
                        break
                    # Check if swap helps
                    m = mask_i & ~mask_new
                    newly_unc = 0
                    while m:
                        bit = (m & -m).bit_length() - 1
                        if cov_tmp[bit] == 0:
                            newly_unc += 1
                        m &= m - 1
                    m = mask_new & ~mask_i
                    newly_cov = 0
                    while m:
                        bit = (m & -m).bit_length() - 1
                        if cov_tmp[bit] == 0:
                            newly_cov += 1
                        m &= m - 1
                    if newly_cov > newly_unc:
                        best_code[i] = new_idx
                        improved = True
                        break
                if improved:
                    break
    
    return best_code


def construct(q, n, radius):
    start = time.time()
    
    # Always-valid baseline
    if n >= 4 and radius >= 1:
        baseline = _direct_sum(q, n, radius)
    else:
        baseline = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    best = baseline
    best_len = len(baseline)
    
    # Only proceed with advanced search for tractable cases
    total = q ** n
    if total > 3 * 10**6 or best_len <= 1:
        return baseline
    
    # Build bitmask spheres
    spheres = _build_spheres_bitmask(q, n, radius)
    full_mask = (1 << total) - 1
    
    # Build word lookup
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    # Initialize with tetracode product and prune
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    init_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            w = tuple(h + list(tail))
            idx = 0
            for x in w:
                idx = idx * q + x
            init_code.append(idx)
    
    pruned_init = _prune_bitmask(init_code, spheres, full_mask)
    if len(pruned_init) < best_len:
        best_len = len(pruned_init)
        best = [all_words[i] for i in pruned_init]
    
    # Multiple greedy restarts
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() - start > _DEADLINE - 15:
            break
        
        random.seed(r)
        candidates = list(range(total))
        random.shuffle(candidates)
        
        covered = 0
        chosen = []
        while covered != full_mask:
            if time.time() - start > _DEADLINE - 15:
                break
            best_gain = -1
            best_idx = -1
            for idx in candidates:
                new_covered = spheres[idx] & ~covered
                gain = new_covered.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            if best_gain <= 0:
                break
            chosen.append(best_idx)
            covered |= spheres[best_idx]
        
        # Prune the greedy result
        pruned_greedy = _prune_bitmask(chosen, spheres, full_mask)
        
        # Try simulated annealing on the pruned result
        if time.time() - start < _DEADLINE - 10:
            sa_time = max(1.0, _DEADLINE - 10 - (time.time() - start))
            sa_result = _simulated_annealing(pruned_greedy, spheres, full_mask, 
                                            sa_time, start)
            if len(sa_result) < len(pruned_greedy):
                pruned_greedy = sa_result
        
        if len(pruned_greedy) < best_len:
            best_len = len(pruned_greedy)
            best = [all_words[i] for i in pruned_greedy]
    
    # Final safety check
    if time.time() - start > _DEADLINE - 5:
        return baseline
    
    # Verify the best solution
    if not _covers_all(best, q, n, radius):
        return baseline
    
    return best
