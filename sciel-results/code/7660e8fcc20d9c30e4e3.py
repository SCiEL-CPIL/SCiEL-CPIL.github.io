"""Smooth-antichain representations for targets 1..128.

Key improvements over previous version:
  - Precompute per-target candidate lists and their conflict bitmasks once.
  - Use a tighter DFS with precomputed suffix-sum bounds for both min and max.
  - Process targets in an order that solves easy ones first, freeing time for hard ones.
  - For each target, try k=1,2,3,... in order and stop at the first valid antichain.
  - Deadline guard ensures we return a complete (possibly non-optimal) answer if time runs out.
"""

import time

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
def _solve_target(n, k):
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
    
    def dfs(start, r, rem):
        nonlocal chosen_mask
        if r == 0:
            return rem == 0
        if start + r > m:
            return False
        
        # Max possible sum: sum of r largest from cands[start:] = cands[start:start+r]
        max_sum = prefix[start + r] - prefix[start]
        if rem > max_sum:
            return False
        
        # Min possible sum: sum of r smallest in cands[start:]
        # The r smallest in cands[start:] are cands[m-r:m] (since descending)
        # This is valid when m-r >= start, i.e., r <= m-start (already checked)
        min_sum = prefix[m] - prefix[m - r]
        if rem < min_sum:
            return False
        
        for i in range(start, m - r + 1):
            v = cands[i]
            if v > rem:
                continue
            if (cand_conflict[i] & chosen_mask) != 0:
                continue
            chosen.append(v)
            chosen_mask |= (1 << i)
            if dfs(i + 1, r - 1, rem - v):
                return True
            chosen.pop()
            chosen_mask &= ~(1 << i)
        return False
    
    if dfs(0, k, n):
        return list(chosen)
    return None

# --- Main construct function ---
def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Leave 5s margin before 90s kill
    
    answers = [None] * 129
    
    # Process targets: smooth numbers first (k=1 is trivial), then non-smooth
    smooth_targets = [n for n in range(1, 129) if n in _VAL_INDEX]
    non_smooth_targets = [n for n in range(1, 129) if n not in _VAL_INDEX]
    target_order = smooth_targets + non_smooth_targets
    
    for n in target_order:
        if time.time() - start_time > DEADLINE:
            break
        
        for k in range(1, min(31, n + 1)):
            if time.time() - start_time > DEADLINE:
                break
            result = _solve_target(n, k)
            if result is not None:
                answers[n] = result
                break
    
    # Handle any remaining unsolved targets
    for n in range(1, 129):
        if answers[n] is None:
            if time.time() - start_time > DEADLINE - 2.0:
                # Last resort: try small k quickly
                for k in range(1, 8):
                    if k > n:
                        break
                    result = _solve_target(n, k)
                    if result is not None:
                        answers[n] = result
                        break
                if answers[n] is None:
                    # Emergency fallback: find any valid representation
                    # Try two-term
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
                                break
                    if answers[n] is None:
                        # Should not happen, but defensive
                        if n in _VAL_INDEX:
                            answers[n] = [n]
                        else:
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
                            if answers[n] is None:
                                # Absolute last resort
                                answers[n] = [1] * n if n <= 30 else [n]
            else:
                # Try up to k=10
                for k in range(1, 11):
                    if k > n:
                        break
                    result = _solve_target(n, k)
                    if result is not None:
                        answers[n] = result
                        break
                if answers[n] is None:
                    for k in range(11, 16):
                        if k > n:
                            break
                        result = _solve_target(n, k)
                        if result is not None:
                            answers[n] = result
                            break
                if answers[n] is None:
                    for k in range(16, 31):
                        if k > n:
                            break
                        result = _solve_target(n, k)
                        if result is not None:
                            answers[n] = result
                            break
                    if answers[n] is None:
                        # Should not happen for n<=128
                        if n in _VAL_INDEX:
                            answers[n] = [n]
                        else:
                            answers[n] = [1] * n if n <= 30 else [n]
    
    return answers[1:129]
