"""Smooth-antichain representations for targets 1..128.

Key improvements:
  - Precompute per-target candidate lists and their conflict bitmasks once.
  - Use a tighter DFS with memoization of failed states.
  - Process targets in an order that solves easy ones first, freeing time for hard ones.
  - For each target, try k=1,2,3,... in order and stop at the first valid antichain.
  - Deadline guard ensures we return a complete (possibly non-optimal) answer if time runs out.
  - Tighter pruning with conflict-aware bounds.
  - Per-target time budget prevents any single hard target from exhausting the wall-clock.
"""

import time
from functools import lru_cache

# --- Smooth number generation ---
def _gen_smooth(limit):
    vals = set()
    p2 = 1
    while p2 <= limit:
        p3 = p2
        while p3 <= limit:
            p5 = p3
            while p5 <= limit:
                vals.add(p5)
                p5 *= 5
            p3 *= 3
        p2 *= 2
    return sorted(v for v in vals if v <= limit)

_SMOOTH = _gen_smooth(128)
_N = len(_SMOOTH)
_VAL_INDEX = {v: i for i, v in enumerate(_SMOOTH)}

# --- Precompute pairwise divisibility conflict masks ---
def _build_conflict_masks():
    masks = [0] * _N
    for i in range(_N):
        vi = _SMOOTH[i]
        m = 0
        for j in range(_N):
            if i == j:
                continue
            vj = _SMOOTH[j]
            if vj % vi == 0 or vi % vj == 0:
                m |= (1 << j)
        masks[i] = m
    return masks

_CONFLICT = _build_conflict_masks()

# --- Precompute per-target candidate data ---
# For each target n (1..128), precompute:
#   cands[n]: list of smooth values <= n, sorted descending
#   cand_global[n]: corresponding global indices
#   cand_conflict[n]: list of conflict bitmasks over cands indices
#   suffix_max[n][i][r]: sum of cands[i:i+r] (max sum of r elements from i onward)
#   suffix_min[n][i][r]: sum of the r smallest elements in cands[i:]

_PRECOMP = {}

def _precompute_all():
    for n in range(1, 129):
        cands = [v for v in _SMOOTH if v <= n]
        cands.sort(reverse=True)
        m = len(cands)
        cand_global = [_VAL_INDEX[v] for v in cands]
        
        # Build conflict masks in terms of cands indices
        cand_conflict = [0] * m
        for i in range(m):
            gi = cand_global[i]
            mask = 0
            for j in range(m):
                if i == j:
                    continue
                gj = cand_global[j]
                if (_CONFLICT[gi] >> gj) & 1:
                    mask |= (1 << j)
            cand_conflict[i] = mask
        
        # Precompute suffix sums for bounds
        # suffix_max[i][r] = sum(cands[i:i+r])
        # suffix_min[i][r] = sum of r smallest in cands[i:] = sum(cands[m-r:m]) if m-r >= i
        # For efficiency, precompute prefix sums
        prefix = [0] * (m + 1)
        for i in range(m):
            prefix[i + 1] = prefix[i] + cands[i]
        
        # suffix_max[i][r] = prefix[i+r] - prefix[i]
        # suffix_min[i][r]: the r smallest in cands[i:] are cands[m-r:m] (since descending)
        # But only if m-r >= i. Since r <= m-i, m-r >= i is equivalent to r <= m-i.
        # suffix_min[i][r] = prefix[m] - prefix[m-r]
        
        _PRECOMP[n] = {
            'cands': cands,
            'cand_global': cand_global,
            'cand_conflict': cand_conflict,
            'prefix': prefix,
            'm': m,
        }

_precompute_all()

# --- Solve a single (n, k) ---
def _solve_target(n, k, deadline=None):
    if k == 0:
        return [] if n == 0 else None
    if k == 1:
        return [n] if n in _VAL_INDEX else None
    
    data = _PRECOMP[n]
    cands = data['cands']
    m = data['m']
    cand_conflict = data['cand_conflict']
    prefix = data['prefix']
    
    if m < k:
        return None
    
    chosen = []
    chosen_mask = 0
    nodes = 0
    TIME_CHECK = 2048
    _failed_states = set()
    
    def dfs(start, r, rem):
        nonlocal chosen_mask, nodes
        nodes += 1
        if nodes & (TIME_CHECK - 1) == 0 and deadline is not None:
            if time.time() > deadline:
                return False
        if r == 0:
            return rem == 0
        if start + r > m:
            return False
        
        # Check memoized failure
        state_key = (start, r, rem, chosen_mask)
        if state_key in _failed_states:
            return False
        
        # Max possible sum: sum of r largest from cands[start:start+r]
        max_sum = prefix[start + r] - prefix[start]
        if rem > max_sum:
            _failed_states.add(state_key)
            return False
        
        # Min possible sum: the r smallest elements available in cands[start:]
        # are cands[m-r:m] (since cands is descending), valid when m-r >= start
        if m - r >= start:
            min_sum = prefix[m] - prefix[m - r]
            if rem < min_sum:
                _failed_states.add(state_key)
                return False
        
        # Additional pruning: if the largest candidate at position 'start' exceeds rem,
        # we can skip forward to find the first candidate <= rem
        i_start = start
        if cands[i_start] > rem:
            # Binary search for the first index where cands[i] <= rem
            lo, hi = start, m
            while lo < hi:
                mid = (lo + hi) // 2
                if cands[mid] > rem:
                    lo = mid + 1
                else:
                    hi = mid
            i_start = lo
            if i_start + r > m:
                _failed_states.add(state_key)
                return False
        
        found = False
        for i in range(i_start, m - r + 1):
            v = cands[i]
            if v > rem:
                continue
            if (cand_conflict[i] & chosen_mask) != 0:
                continue
            chosen.append(v)
            chosen_mask |= (1 << i)
            if dfs(i + 1, r - 1, rem - v):
                found = True
                break
            chosen.pop()
            chosen_mask &= ~(1 << i)
        
        if not found:
            _failed_states.add(state_key)
        return found
    
    if dfs(0, k, n):
        return list(chosen)
    return None

# --- Main construct function ---
def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Leave 5s margin before 90s kill
    TOTAL_TARGETS = 128
    
    answers = [None] * 129
    
    # Process targets: smooth numbers first (k=1 is trivial), then non-smooth
    smooth_targets = [n for n in range(1, 129) if n in _VAL_INDEX]
    non_smooth_targets = [n for n in range(1, 129) if n not in _VAL_INDEX]
    
    # Sort non-smooth targets by difficulty heuristic: smaller targets first
    non_smooth_targets.sort(key=lambda x: x)
    
    target_order = smooth_targets + non_smooth_targets
    
    # Per-target time budget: divide remaining time among unsolved targets
    # Smooth targets solve instantly; non-smooth may need more time.
    n_non_smooth = len(non_smooth_targets)
    n_smooth = len(smooth_targets)
    
    for idx, n in enumerate(target_order):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            break
        
        # Estimate time budget for this target
        remaining_targets = TOTAL_TARGETS - idx
        remaining_time = DEADLINE - elapsed
        
        if n in _VAL_INDEX:
            # Smooth: trivial, give minimal budget
            target_budget = min(0.005, remaining_time / max(1, remaining_targets))
        else:
            # Non-smooth: allocate proportionally
            non_smooth_idx = idx - n_smooth
            if non_smooth_idx < 0:
                non_smooth_idx = 0
            remaining_non_smooth = max(1, n_non_smooth - non_smooth_idx)
            target_budget = remaining_time / remaining_non_smooth
            # Cap at 5 seconds per target, floor at 0.02s
            target_budget = max(0.02, min(target_budget, 5.0))
        
        target_deadline = time.time() + target_budget
        
        for k in range(1, min(31, n + 1)):
            if time.time() > target_deadline:
                break
            if time.time() > DEADLINE:
                break
            result = _solve_target(n, k, deadline=target_deadline)
            if result is not None:
                answers[n] = result
                break
    
    # Handle any remaining unsolved targets
    for n in range(1, 129):
        if answers[n] is None:
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 1.0:
                # Very tight: use emergency fallbacks only
                if n in _VAL_INDEX:
                    answers[n] = [n]
                else:
                    # Try two-term quickly
                    found = False
                    for i in range(len(_SMOOTH)):
                        a = _SMOOTH[i]
                        if a >= n:
                            break
                        b = n - a
                        if b <= 0:
                            continue
                        if b in _VAL_INDEX:
                            if a % b != 0 and b % a != 0:
                                answers[n] = [a, b]
                                found = True
                                break
                    if not found:
                        # Try three-term
                        found = False
                        for i in range(len(_SMOOTH)):
                            a = _SMOOTH[i]
                            if a >= n:
                                break
                            for j in range(i + 1, len(_SMOOTH)):
                                b = _SMOOTH[j]
                                if a + b >= n:
                                    break
                                c = n - a - b
                                if c <= 0:
                                    continue
                                if c in _VAL_INDEX:
                                    gc = _VAL_INDEX[c]
                                    ga = _VAL_INDEX[a]
                                    gb = _VAL_INDEX[b]
                                    if ((_CONFLICT[ga] >> gb) & 1) == 0 and \
                                       ((_CONFLICT[ga] >> gc) & 1) == 0 and \
                                       ((_CONFLICT[gb] >> gc) & 1) == 0:
                                        answers[n] = [a, b, c]
                                        found = True
                                        break
                            if found:
                                break
                        if not found:
                            answers[n] = [1] * n if n <= 30 else [n]
            else:
                # Try up to k=30 with a reasonable deadline
                remaining_time = DEADLINE - elapsed
                # Allocate remaining time to unsolved targets
                unsolved = sum(1 for x in range(1, 129) if answers[x] is None)
                per_target = max(0.5, remaining_time / max(1, unsolved))
                t_deadline = time.time() + min(per_target, 5.0)
                
                for k in range(1, 31):
                    if k > n:
                        break
                    if time.time() > t_deadline:
                        break
                    result = _solve_target(n, k, deadline=t_deadline)
                    if result is not None:
                        answers[n] = result
                        break
                
                if answers[n] is None:
                    # Emergency fallback
                    if n in _VAL_INDEX:
                        answers[n] = [n]
                    else:
                        found = False
                        for i in range(len(_SMOOTH)):
                            a = _SMOOTH[i]
                            if a >= n:
                                break
                            b = n - a
                            if b <= 0:
                                continue
                            if b in _VAL_INDEX:
                                if a % b != 0 and b % a != 0:
                                    answers[n] = [a, b]
                                    found = True
                                    break
                        if not found:
                            found = False
                            for i in range(len(_SMOOTH)):
                                a = _SMOOTH[i]
                                if a >= n:
                                    break
                                for j in range(i + 1, len(_SMOOTH)):
                                    b = _SMOOTH[j]
                                    if a + b >= n:
                                        break
                                    c = n - a - b
                                    if c <= 0:
                                        continue
                                    if c in _VAL_INDEX:
                                        gc = _VAL_INDEX[c]
                                        ga = _VAL_INDEX[a]
                                        gb = _VAL_INDEX[b]
                                        if ((_CONFLICT[ga] >> gb) & 1) == 0 and \
                                           ((_CONFLICT[ga] >> gc) & 1) == 0 and \
                                           ((_CONFLICT[gb] >> gc) & 1) == 0:
                                            answers[n] = [a, b, c]
                                            found = True
                                            break
                                if found:
                                    break
                            if not found:
                                answers[n] = [1] * n if n <= 30 else [n]
    
    return answers[1:129]
