import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Target differences: 1..30
    # Bitmask: bit i (0-indexed) corresponds to difference i+1.
    FULL_MASK = (1 << 30) - 1

    def check_solution(A):
        diffs = set()
        for i in range(len(A)):
            for j in range(i):
                d = A[i] - A[j]
                if d in diffs:
                    return False
                if 1 <= d <= 30:
                    diffs.add(d)
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True

    def solve(M_limit):
        """
        Finds a Sidon set with max element <= M_limit covering 1..30.
        Returns the set if found, else None.
        """
        def dfs(A, covered_mask, all_diffs):
            if time.time() > deadline:
                return None
            
            if (covered_mask & FULL_MASK) == FULL_MASK:
                return A[:]
            
            if len(A) > 32:
                return None
                
            missing_count = 30 - bin(covered_mask & FULL_MASK).count("1")
            if missing_count == 0:
                return A[:]
                
            slots_left = 32 - len(A)
            if slots_left <= 0:
                return None
                
            # Pruning: Max new diffs possible
            max_possible_new_diffs = slots_left * len(A) + slots_left * (slots_left - 1) // 2
            if missing_count > max_possible_new_diffs:
                return None

            current_max = A[-1] if A else -1
            
            candidates = []
            for x in range(current_max + 1, M_limit + 1):
                # Check Sidon property
                valid = True
                new_diffs = []
                for a in A:
                    d = x - a
                    if d in all_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                new_mask = covered_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                        
                # Heuristic: Prioritize candidates that cover more missing diffs
                # If a candidate covers no new diffs, it's likely suboptimal for minimal diameter
                # but we keep it as a last resort if no progress is made?
                # Actually, if we don't make progress, we use a slot.
                # Let's only keep candidates that make progress to keep tree small.
                if new_mask == covered_mask:
                    continue
                
                # Score: number of new bits set
                score = (new_mask & FULL_MASK).bit_count() - (covered_mask & FULL_MASK).bit_count()
                candidates.append((x, new_mask, new_diffs, score))
            
            # Sort by score descending
            candidates.sort(key=lambda item: -item[3])
            
            for x, new_mask, new_diffs, _ in candidates:
                new_A = A + [x]
                new_all_diffs = all_diffs | set(new_diffs)
                
                res = dfs(new_A, new_mask, new_all_diffs)
                if res is not None:
                    return res
                    
            return None

        return dfs([0], 0, set())

    # Search for minimal M
    for M in range(30, 200):
        if time.time() > deadline - 5:
            break
        res = solve(M)
        if res is not None:
            if check_solution(res):
                return res
                
    # Fallback: Greedy construction
    a = [0]
    def get_diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    
    for d in range(1, 31):
        if d in get_diffs(sorted(a)):
            continue
        found = False
        candidates = set()
        for v in a:
            candidates.add(v + d)
            if v - d >= 0:
                candidates.add(v - d)
        
        for x in sorted(candidates):
            if x in a:
                continue
            trial = sorted(a + [x])
            diffs = get_diffs(trial)
            if len(set(diffs)) == len(diffs):
                a = trial
                found = True
                break
        
        if not found:
            y = 2 * max(a) + 100
            while True:
                trial = sorted(a + [y])
                diffs = get_diffs(trial)
                if len(set(diffs)) == len(diffs):
                    a = trial
                    break
                y += 1
    return a

if __name__ == "__main__":
    res = construct()
    print(res)
    print(len(res))
    print(max(res))
