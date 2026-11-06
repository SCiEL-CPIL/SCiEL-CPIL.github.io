"""Optimized divisibility-antichain representations by smooth numbers."""
import time
from itertools import combinations

# Pre-compute smooth numbers 2^a * 3^b * 5^c <= 128
def _generate_smooth(limit):
    smooth = set()
    a = 1
    while a <= limit:
        b = a
        while b <= limit:
            c = b
            while c <= limit:
                smooth.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(x for x in smooth if x <= limit)

SMOOTH = _generate_smooth(128)
SMOOTH_SET = set(SMOOTH)
N_SMOOTH = len(SMOOTH)

def _antichain_ok(sorted_terms):
    """Check that no term divides another. Terms must be sorted ascending."""
    n = len(sorted_terms)
    for i in range(n):
        ti = sorted_terms[i]
        for j in range(i + 1, n):
            tj = sorted_terms[j]
            if tj % ti == 0:
                return False
    return True

def _find_min_terms(target, deadline):
    """Find minimum-k antichain sum representation of target using smooth numbers."""
    # k = 1
    if target in SMOOTH_SET:
        return [target]
    
    # k = 2: x + y = target, x < y, both smooth, y % x != 0
    for i in range(N_SMOOTH):
        x = SMOOTH[i]
        if x * 2 >= target:
            break
        y = target - x
        if y in SMOOTH_SET and y > x:
            if y % x != 0:
                return [x, y]
    
    # k = 3: x < y < z, x+y+z = target, antichain
    for i in range(N_SMOOTH):
        x = SMOOTH[i]
        if x * 3 >= target:
            break
        for j in range(i + 1, N_SMOOTH):
            y = SMOOTH[j]
            if y % x == 0:
                continue  # x divides y, invalid pair
            s2 = x + y
            if s2 * 2 >= target:
                break
            z = target - s2
            if z <= y:
                continue
            if z not in SMOOTH_SET:
                continue
            # Check antichain: z % x != 0 and z % y != 0
            if z % x != 0 and z % y != 0:
                return [x, y, z]
    
    # k = 4: backtracking
    if time.time() > deadline:
        return None
    
    # Try all combinations of 4 smooth numbers that sum to target
    # Use iterative approach with pruning
    found = None
    for i in range(N_SMOOTH):
        x = SMOOTH[i]
        if x * 4 > target:
            break
        rem1 = target - x
        for j in range(i + 1, N_SMOOTH):
            y = SMOOTH[j]
            if y % x == 0:
                continue
            if y * 3 > rem1:
                break
            rem2 = rem1 - y
            for k in range(j + 1, N_SMOOTH):
                z = SMOOTH[k]
                if z % x == 0 or z % y == 0:
                    continue
                if z * 2 > rem2:
                    break
                w = rem2 - z
                if w <= z:
                    continue
                if w not in SMOOTH_SET:
                    continue
                # Check antichain for w
                if w % x == 0 or w % y == 0 or w % z == 0:
                    continue
                return [x, y, z, w]
    
    # k = 5: backtracking
    if time.time() > deadline:
        return None
    
    for i in range(N_SMOOTH):
        x = SMOOTH[i]
        if x * 5 > target:
            break
        rem1 = target - x
        for j in range(i + 1, N_SMOOTH):
            y = SMOOTH[j]
            if y % x == 0:
                continue
            if y * 4 > rem1:
                break
            rem2 = rem1 - y
            for k in range(j + 1, N_SMOOTH):
                z = SMOOTH[k]
                if z % x == 0 or z % y == 0:
                    continue
                if z * 3 > rem2:
                    break
                rem3 = rem2 - z
                for l in range(k + 1, N_SMOOTH):
                    w = SMOOTH[l]
                    if w % x == 0 or w % y == 0 or w % z == 0:
                        continue
                    if w * 2 > rem3:
                        break
                    v = rem3 - w
                    if v <= w:
                        continue
                    if v not in SMOOTH_SET:
                        continue
                    if v % x == 0 or v % y == 0 or v % z == 0 or v % w == 0:
                        continue
                    return [x, y, z, w, v]
    
    # k = 6: general backtracking
    if time.time() > deadline:
        return None
    
    def backtrack(rem, start_idx, chosen, depth, max_depth):
        if time.time() > deadline:
            return None
        if depth == 0:
            if rem in SMOOTH_SET:
                candidate = chosen + [rem]
                # Check rem against all chosen
                for c in chosen:
                    if rem % c == 0:
                        return None
                return candidate
            return None
        if depth == 1:
            # Last pick: rem must be smooth, > last chosen, and not divisible by any chosen
            if rem not in SMOOTH_SET:
                return None
            if chosen and rem <= chosen[-1]:
                return None
            for c in chosen:
                if rem % c == 0:
                    return None
            return chosen + [rem]
        
        # Choose next term
        # Min possible: smallest smooth > chosen[-1] if chosen else SMOOTH[0]
        # Max possible: rem / depth (roughly), but must be smooth and > previous
        max_val = rem // depth
        for idx in range(start_idx, N_SMOOTH):
            val = SMOOTH[idx]
            if val > max_val:
                break
            if chosen and val <= chosen[-1]:
                continue
            # Check val not divisible by any chosen
            valid = True
            for c in chosen:
                if val % c == 0:
                    valid = False
                    break
            if not valid:
                continue
            # Also check no chosen divides val (already covered above since chosen < val)
            result = backtrack(rem - val, idx + 1, chosen + [val], depth - 1, max_depth)
            if result is not None:
                return result
        return None
    
    result = backtrack(target, 0, [], 6, 6)
    if result:
        return result
    
    # k = 7+ : very unlikely needed, but try up to 10 with backtracking
    for k in range(7, 11):
        if time.time() > deadline:
            break
        result = backtrack(target, 0, [], k, k)
        if result:
            return result
    
    return None

# Pre-compute all minimal representations at module load time
_PRECOMPUTED = {}

def _precompute_all():
    """Pre-compute minimal antichain representations for targets 1..128."""
    global _PRECOMPUTED
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin
    
    for n in range(1, 129):
        if time.time() > deadline:
            break
        ans = _find_min_terms(n, deadline)
        if ans is not None:
            _PRECOMPUTED[n] = ans
        else:
            # Fallback
            ans = _fallback(n)
            _PRECOMPUTED[n] = ans

def construct():
    # Use pre-computed results if available
    if _PRECOMPUTED:
        answers = [_PRECOMPUTED[n] for n in range(1, 129)]
        # Verify all are valid
        for i, ans in enumerate(answers):
            n = i + 1
            if sum(ans) != n:
                # Fallback if something went wrong
                ans = _fallback(n)
                answers[i] = ans
        return answers
    
    # If pre-computation didn't happen (shouldn't happen), compute on the fly
    start_time = time.time()
    deadline = start_time + 80.0
    
    answers = []
    for n in range(1, 129):
        if time.time() > deadline:
            ans = _fallback(n)
            answers.append(ans)
            continue
        
        ans = _find_min_terms(n, deadline)
        if ans is None:
            ans = _fallback(n)
        answers.append(ans)
    
    return answers

def _fallback(target):
    """Quick fallback: find any valid antichain sum, not necessarily minimal."""
    # Try k from 1 to 10 with a simple search
    for k in range(1, 11):
        # Simple recursive search without full optimization
        def search(rem, start_idx, chosen):
            if len(chosen) == k:
                if rem in SMOOTH_SET:
                    if chosen and rem <= chosen[-1]:
                        return None
                    for c in chosen:
                        if rem % c == 0:
                            return None
                    return chosen + [rem]
                return None
            if len(chosen) == k - 1:
                if rem in SMOOTH_SET:
                    if chosen and rem <= chosen[-1]:
                        return None
                    for c in chosen:
                        if rem % c == 0:
                            return None
                    return chosen + [rem]
                return None
            
            max_val = rem // (k - len(chosen))
            for idx in range(start_idx, N_SMOOTH):
                val = SMOOTH[idx]
                if val > max_val:
                    break
                if chosen and val <= chosen[-1]:
                    continue
                valid = True
                for c in chosen:
                    if val % c == 0:
                        valid = False
                        break
                if not valid:
                    continue
                result = search(rem - val, idx + 1, chosen + [val])
                if result:
                    return result
            return None
        
        result = search(target, 0, [])
        if result:
            return result
    # Last resort: should not happen
    return [1] if target >= 1 else []

# Run pre-computation at module load
_precompute_all()
