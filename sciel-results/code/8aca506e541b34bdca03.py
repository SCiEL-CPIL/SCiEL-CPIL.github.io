import time
import random

def construct():
    start = time.time()
    DEADLINE = start + 85  # Safety margin before 90s limit

    def compute_block_sums(s):
        """Return set of all consecutive block sums."""
        sums = set()
        n = len(s)
        for i in range(n):
            acc = 0
            for j in range(i, n):
                acc += s[j]
                sums.add(acc)
        return sums

    def is_valid(s):
        """Check if sequence is valid: strictly increasing, values in [1,256], distinct block sums."""
        n = len(s)
        if n == 0:
            return True
        for i in range(n - 1):
            if s[i] >= s[i + 1]:
                return False
        for v in s:
            if v < 1 or v > 256:
                return False
        sums = compute_block_sums(s)
        expected = n * (n + 1) // 2
        return len(sums) == expected

    def greedy_build(direction='smallest'):
        """Build a sequence greedily."""
        values = []
        sums = set()
        if direction == 'smallest':
            candidates = range(1, 257)
        else:
            candidates = range(256, 0, -1)
        
        for x in candidates:
            if time.time() > DEADLINE - 2:
                break
            t = 0
            ok = True
            for a in reversed(values):
                t += a
                if t in sums:
                    ok = False
                    break
            if not ok:
                continue
            if x in sums:
                continue
            values.append(x)
            sums.add(x)
            t = x
            for a in reversed(values[:-1]):
                t += a
                sums.add(t)
        
        return sorted(values)

    def multi_start_local_search(num_restarts=3):
        """Multiple restarts with local search: try insertions and swaps to grow sequence."""
        best_overall = []
        
        for r in range(num_restarts):
            if time.time() > DEADLINE - 10:
                break
            
            # Different initial sequences
            if r == 0:
                init = greedy_build('smallest')
            elif r == 1:
                init = greedy_build('largest')
            else:
                # Random initialization
                init = []
                sums = set()
                used = set()
                for _ in range(random.randint(5, 10)):
                    x = random.randint(1, 256)
                    if x in used:
                        continue
                    if init and x <= init[-1]:
                        continue
                    t = 0
                    ok = True
                    for a in reversed(init):
                        t += a
                        if t in sums:
                            ok = False
                            break
                    if not ok:
                        continue
                    if x in sums:
                        continue
                    init.append(x)
                    used.add(x)
                    sums.add(x)
                    t = x
                    for a in reversed(init[:-1]):
                        t += a
                        sums.add(t)
            
            # Local search improvement
            current = list(init)
            current_set = set(current)
            improved = True
            while improved and time.time() < DEADLINE - 5:
                improved = False
                n = len(current)
                
                # Try all insertions
                for x in range(1, 257):
                    if x in current_set:
                        continue
                    for pos in range(n + 1):
                        new_s = current[:pos] + [x] + current[pos:]
                        valid_inc = all(new_s[k] < new_s[k+1] for k in range(len(new_s)-1))
                        if not valid_inc:
                            continue
                        if is_valid(new_s):
                            current = new_s
                            current_set = set(current)
                            improved = True
                            break
                    if improved:
                        break
                
                # If no insertion helped, try swaps to enable future insertions
                if not improved and n >= 2:
                    for pos1 in range(n):
                        for pos2 in range(pos1 + 1, n):
                            new_s = list(current)
                            new_s[pos1], new_s[pos2] = new_s[pos2], new_s[pos1]
                            valid_inc = all(new_s[k] < new_s[k+1] for k in range(len(new_s)-1))
                            if valid_inc and is_valid(new_s):
                                # Check if this swap enables an insertion
                                new_set = set(new_s)
                                for x in range(1, 257):
                                    if x in new_set:
                                        continue
                                    for pos in range(len(new_s) + 1):
                                        ins_s = new_s[:pos] + [x] + new_s[pos:]
                                        valid_inc2 = all(ins_s[k] < ins_s[k+1] for k in range(len(ins_s)-1))
                                        if valid_inc2 and is_valid(ins_s):
                                            current = ins_s
                                            current_set = set(current)
                                            improved = True
                                            break
                                    if improved:
                                        break
                                if improved:
                                    break
                        if improved:
                            break
            
            if len(current) > len(best_overall):
                best_overall = list(current)
        
        return best_overall

    # Run multi-start local search
    best = multi_start_local_search(num_restarts=3)
    
    # Final verification
    if not is_valid(best):
        n = len(best)
        for i in range(n):
            if is_valid(best[:i + 1]):
                best = best[:i + 1]
            else:
                break
        else:
            best = best[:i]
    
    return best
