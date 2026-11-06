"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    n = 256
    
    def is_valid_set(S):
        """Check if set S satisfies the property."""
        S_list = sorted(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                for k in range(j, m):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def count_conflicts(candidate, S):
        """Count how many elements in S conflict with adding candidate.
        Returns list of conflicting elements."""
        conflicts = []
        for a in S:
            if a == candidate:
                continue
            if a < candidate:
                # a is the smaller one, need to check a | (candidate + c) for c >= candidate
                for c in S:
                    if c < candidate:
                        continue
                    if c == candidate:
                        continue
                    if (candidate + c) % a == 0:
                        conflicts.append(a)
                        break
            else:
                # candidate < a, need to check candidate | (a + c) for c >= a
                for c in S:
                    if c < a:
                        continue
                    if c == a:
                        continue
                    if (a + c) % candidate == 0:
                        conflicts.append(a)
                        break
        return conflicts
    
    # Phase 1: Greedy from largest to smallest
    S = set()
    for num in range(n, 0, -1):
        if time.time() > deadline - 5:
            break
        # Check if adding num to S is valid
        valid = True
        for a in S:
            if a > num:
                # num < a, check num | (a + c) for all c in S with c >= a
                for c in S:
                    if c >= a and c != a:
                        if (a + c) % num == 0:
                            valid = False
                            break
                    elif c == a:
                        if (a + a) % num == 0:
                            valid = False
                            break
                if not valid:
                    break
            else:
                # a < num, check a | (num + c) for all c in S with c >= num
                for c in S:
                    if c >= num and c != num:
                        if (num + c) % a == 0:
                            valid = False
                            break
                    elif c == num:
                        pass  # can't happen since we're adding num
                if not valid:
                    break
        if valid:
            S.add(num)
    
    best_set = S.copy()
    best_size = len(S)
    
    # Phase 2: Try to add excluded elements
    excluded = [x for x in range(1, n + 1) if x not in S]
    for num in sorted(excluded):
        if time.time() > deadline - 5:
            break
        # Try adding num
        valid = True
        for a in S:
            if a > num:
                for c in S:
                    if c >= a:
                        if c == a:
                            if (a + a) % num == 0:
                                valid = False
                                break
                        else:
                            if (a + c) % num == 0:
                                valid = False
                                break
                if not valid:
                    break
            elif a < num:
                for c in S:
                    if c >= num:
                        if (num + c) % a == 0:
                            valid = False
                            break
                if not valid:
                    break
        if valid:
            S.add(num)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
    
    # Phase 3: Local search - try swaps
    if time.time() < deadline - 5:
        S = best_set.copy()
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            in_set = sorted(S)
            out_set = sorted([x for x in range(1, n + 1) if x not in S])
            for out_num in out_set:
                if time.time() > deadline - 5:
                    break
                for in_num in in_set:
                    if time.time() > deadline - 5:
                        break
                    # Try swapping: remove in_num, add out_num
                    if out_num == in_num:
                        continue
                    S_new = S - {in_num}
                    S_new.add(out_num)
                    # Quick check: is S_new valid?
                    valid = True
                    S_list = sorted(S_new)
                    m = len(S_list)
                    for i in range(m):
                        a = S_list[i]
                        for j in range(i + 1, m):
                            b = S_list[j]
                            for k in range(j, m):
                                c = S_list[k]
                                if (b + c) % a == 0:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid and len(S_new) > len(S):
                        S = S_new
                        improved = True
                        break
                if improved:
                    break
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    return list(best_set)
# EVOLVE-BLOCK-END
