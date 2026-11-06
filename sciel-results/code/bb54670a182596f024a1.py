import time
import itertools

def construct():
    deadline = time.time() + 85.0
    
    # 1. Generate the Base-3 AP-free set
    powers = []
    p = 1
    while p <= 512:
        powers.append(p)
        p *= 3
    # powers: [1, 3, 9, 27, 81, 243]
    
    candidates = []
    for mask in range(1, 1 << len(powers)):
        val = 0
        for i in range(len(powers)):
            if mask & (1 << i):
                val += powers[i]
        if 1 <= val <= 512:
            candidates.append(val)
            
    candidates.sort()
    S = set(candidates)
    
    def get_violations(S):
        violations = {}
        S_list = sorted(S)
        n = len(S_list)
        if n < 3:
            return violations
            
        # Check sizes 3 and 4
        # Size 3: 3x = a+b+c
        # Size 4: 4x = a+b+c+d
        
        # Optimization: Precompute sums?
        # Just iterate combinations.
        
        for k in range(3, 5):
            for comb in itertools.combinations(S_list, k):
                s = sum(comb)
                if s % k == 0:
                    mean = s // k
                    if mean in S and mean not in comb:
                        violations[mean] = violations.get(mean, 0) + 1
        
        return violations

    # Iterative Pruning
    for _ in range(100):
        if time.time() > deadline - 5.0:
            break
            
        viol = get_violations(S)
        if not viol:
            break
            
        worst = max(viol, key=viol.get)
        S.remove(worst)
        
    final_set = sorted(S)
    
    # Full Verification
    def full_verify(arr):
        m = len(arr)
        if m < 3:
            return True
        
        max_sum = 48 * 512
        
        def build_dp(a):
            mm = len(a)
            dpp = [0] * (mm + 1)
            dpp[0] = 1
            for val in a:
                for k in range(mm, 0, -1):
                    dpp[k] |= (dpp[k-1] << val)
            return dpp
        
        for i in range(m):
            x = arr[i]
            rest = arr[:i] + arr[i+1:]
            dp_rest = build_dp(rest)
            
            # Check if x is mean of subset of rest (size >= 2)
            # Since AP-free, size 2 is impossible. Start k=3.
            for k in range(3, m):
                target = k * x
                if target > max_sum:
                    continue
                if (dp_rest[k] >> target) & 1:
                    return False
            
            # Check if y is mean of subset of arr containing x
            # Subset size = j+1. j from rest.
            # If j=1, size 2, implies 3-term AP. Impossible.
            # So start j=2.
            for j in range(2, m):
                for y in arr:
                    if y == x:
                        continue
                    target = (j + 1) * y - x
                    if target < 0 or target > max_sum:
                        continue
                    if (dp_rest[j] >> target) & 1:
                        return False
        return True

    if not full_verify(final_set):
        # If it fails, try to remove more elements
        # This is a safety net
        for _ in range(10):
            if time.time() > deadline - 1.0:
                break
            # Find a violation and remove
            # Re-run get_violations
            viol = get_violations(S)
            if not viol:
                break
            worst = max(viol, key=viol.get)
            S.remove(worst)
            final_set = sorted(S)
            if full_verify(final_set):
                break

    return final_set
